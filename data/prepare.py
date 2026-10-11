"""Stream TinyStories -> uint16 .bin shards. RAM-safe for 8GB, resumable per shard.

- Val = official TinyStories validation split (overrides doc-holdout; see decisions.md).
- Appends <|endoftext|> (id 256) after every document.
- Shards written with buffered tofile() appends, never a growing memmap.
- Resumable: data/meta.json tracks docs_done per split + finished shards;
  reruns skip finished docs/shards. Partial last shard is appended to.
- --max_docs caps train docs (smoke test); val docs capped at max(100, max_docs//5).
  Full run: omit --max_docs (Kaggle, internet on, save out_dir as dataset).

Output: data/train_00000.bin ... data/val_00000.bin (uint16) + data/meta.json.
Report: vocab size, tokens/char, train/val token totals, shard sizes.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
from tqdm import tqdm

from tokenizer.bpe import EOT_ID, BPETokenizer

FLUSH_DOCS = 500       # encode this many docs, then tofile() once
FLUSH_TOKENS = 1_000_000  # or flush when buffer exceeds this many tokens


def load_or_init_meta(out_dir):
    path = os.path.join(out_dir, "meta.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), path
    meta = {"train_docs_done": 0, "val_docs_done": 0,
            "shards": [], "tokenizer": "", "vocab_size": 0}
    return meta, path


def save_meta(meta, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)


def current_shard(meta, split):
    """Return (shard_index, tokens_in_shard) for the open last shard of split."""
    for s in reversed(meta["shards"]):
        if s["split"] == split:
            return s["index"], s["tokens"]
    return 0, 0


def encode_stream(tok, split, skip, limit, append_eot=True):
    """Yield (ids, nchars) per doc from official split, skipping `skip` docs."""
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    it = itertools.islice(ds, skip, None if limit is None else skip + limit)
    for ex in it:
        text = ex["text"]
        ids = tok.encode(text)
        if append_eot:
            ids.append(EOT_ID)
        yield ids, len(text)


def write_split(tok, split, out_dir, meta, meta_path, shard_tokens, max_docs):
    skip = meta.get(f"{split}_docs_done", 0)
    # max_docs is a TOTAL cap (idempotent reruns). None = whole split.
    if max_docs is not None and skip >= max_docs:
        print(f"[prepare:{split}] already done ({skip} docs >= cap {max_docs})")
        return skip
    todo = None if max_docs is None else max_docs - skip
    print(f"[prepare:{split}] resume skip={skip} todo={todo}")

    shard_idx, shard_count = current_shard(meta, split)
    # Find existing record for the open shard (if any).
    open_rec = None
    for s in meta["shards"]:
        if s["split"] == split and s["index"] == shard_idx and s["tokens"] > 0:
            open_rec = s
    if shard_count == 0 and open_rec is None:
        pass  # first shard, nothing open

    def shard_path(i):
        return os.path.join(out_dir, f"{split}_{i:05d}.bin")

    buf, buf_tokens = [], 0
    docs_done = skip
    total_tokens = sum(s["tokens"] for s in meta["shards"] if s["split"] == split)
    total_chars = meta.get(f"{split}_chars", 0)
    t0 = time.time()

    # Open current shard file in append-binary mode (buffered appends).
    fh = open(shard_path(shard_idx), "ab")

    def roll_shard():
        nonlocal shard_idx, shard_count, fh, open_rec
        fh.close()
        shard_idx += 1
        shard_count = 0
        fh = open(shard_path(shard_idx), "ab")
        open_rec = None

    def flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        arr = np.array([t for doc in buf for t in doc], dtype=np.uint16)
        assert arr.max() < 65536, "vocab exceeds uint16"
        arr.tofile(fh)
        fh.flush()
        buf, buf_tokens = [], 0

    stream = encode_stream(tok, split, skip, todo)
    if todo is not None:
        stream = tqdm(stream, total=todo, desc=f"{split}", unit="doc")
    else:
        stream = tqdm(stream, desc=f"{split}", unit="doc")

    pending_docs = 0
    for ids, nchars in stream:
        buf.append(ids)
        buf_tokens += len(ids)
        pending_docs += 1
        total_chars += nchars
        # Would this overflow the shard? Flush first, then roll if needed.
        if buf_tokens >= FLUSH_TOKENS or pending_docs >= FLUSH_DOCS:
            # If flushing would overflow shard, roll before writing.
            cur = shard_count + buf_tokens
            if cur > shard_tokens and shard_count > 0:
                # Write part? Simpler: flush current buffer to old shard if it
                # fits, else roll then flush. Here buffer may straddle; accept
                # slight overshoot only when a single flush exceeds shard size.
                roll_shard()
            flush()
            # Update counts after flush: recompute from buffer is gone, so
            # track via docs. Instead track tokens added now:
            shard_count = os.path.getsize(shard_path(shard_idx)) // 2
            docs_done += pending_docs
            pending_docs = 0
            total_tokens = sum(
                s["tokens"] for s in meta["shards"] if s["split"] == split
            ) + shard_count + sum(len(d) for d in buf)
            meta[f"{split}_docs_done"] = docs_done
            meta[f"{split}_chars"] = total_chars
            # Upsert open shard record.
            if open_rec is None:
                open_rec = {"split": split, "index": shard_idx,
                            "file": os.path.basename(shard_path(shard_idx)),
                            "tokens": shard_count}
                meta["shards"].append(open_rec)
            else:
                open_rec["tokens"] = shard_count
            save_meta(meta, meta_path)
            if shard_count >= shard_tokens:
                roll_shard()

    # Final flush.
    if buf:
        if shard_count + buf_tokens > shard_tokens and shard_count > 0:
            roll_shard()
        flush()
        shard_count = os.path.getsize(shard_path(shard_idx)) // 2
        docs_done += pending_docs
        if open_rec is None:
            meta["shards"].append({"split": split, "index": shard_idx,
                                   "file": os.path.basename(shard_path(shard_idx)),
                                   "tokens": shard_count})
        else:
            if open_rec["index"] == shard_idx:
                open_rec["tokens"] = shard_count
            else:
                meta["shards"].append(
                    {"split": split, "index": shard_idx,
                     "file": os.path.basename(shard_path(shard_idx)),
                     "tokens": shard_count})
        meta[f"{split}_docs_done"] = docs_done
        meta[f"{split}_chars"] = total_chars
        save_meta(meta, meta_path)
    fh.close()
    dt = time.time() - t0
    print(f"[prepare:{split}] docs={docs_done} tokens~={shard_count} "
          f"this-shard time={dt:.1f}s")
    return docs_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/tok8192.json")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--max_docs", type=int, default=None,
                    help="cap train docs (smoke test); val capped at max(100, max_docs//5)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = BPETokenizer.load(args.tokenizer)
    print(f"[prepare] vocab={tok.vocab_size}")

    meta, meta_path = load_or_init_meta(args.out_dir)
    meta["tokenizer"] = args.tokenizer
    meta["vocab_size"] = tok.vocab_size
    save_meta(meta, meta_path)

    train_cap = args.max_docs
    val_cap = None if args.max_docs is None else max(100, args.max_docs // 5)
    write_split(tok, "train", args.out_dir, meta, meta_path,
                args.shard_tokens, train_cap)
    # Reload meta (train updated it) then val.
    meta, _ = load_or_init_meta(args.out_dir)
    write_split(tok, "validation", args.out_dir, meta, meta_path,
                args.shard_tokens, val_cap)

    meta, _ = load_or_init_meta(args.out_dir)
    # Final report.
    import glob as _glob
    print("\n==== prepare report ====")
    print(f"vocab size: {meta['vocab_size']}")
    for split in ("train", "validation"):
        toks = sum(s["tokens"] for s in meta["shards"] if s["split"] == split)
        chars = meta.get(f"{split}_chars", 0)
        tpc = toks / chars if chars else 0
        files = sorted(_glob.glob(os.path.join(args.out_dir, f"{split}_*.bin")))
        sizes = [os.path.getsize(f) / 1024 / 1024 for f in files]
        print(f"{split}: docs={meta.get(split+'_docs_done')} tokens={toks} "
              f"tokens/char={tpc:.3f}")
        for f, mb in zip(files, sizes):
            print(f"  {f} {mb:.1f}MB")


if __name__ == "__main__":
    main()
