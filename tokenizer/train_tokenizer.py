"""Train tok8192.json on a ~50-100MB TinyStories subset. Local-first.

Streams train split in order (no shuffle, deterministic), stops once
corpus_mb of raw text is collected, trains BPE, saves single JSON.
Reports training time (amendment 2).
"""
import argparse
import time

from tokenizer.bpe import BPETokenizer, DEFAULT_VOCAB_SIZE


def stream_texts(limit_bytes):
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    total, out = 0, []
    for ex in ds:
        t = ex["text"]
        b = len(t.encode("utf-8"))
        out.append(t)
        total += b
        if total >= limit_bytes:
            break
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_mb", type=float, default=75)
    ap.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--out", default="tokenizer/tok8192.json")
    args = ap.parse_args()

    limit = int(args.corpus_mb * 1024 * 1024)
    print(f"[train_tok] collecting {args.corpus_mb}MB of TinyStories train...")
    texts, nbytes = stream_texts(limit)
    print(f"[train_tok] docs={len(texts)} bytes={nbytes} "
          f"({nbytes/1024/1024:.1f}MB)")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(texts, vocab_size=args.vocab_size, verbose=True)
    dt = time.time() - t0
    print(f"[train_tok] training time: {dt:.1f}s on {nbytes/1024/1024:.1f}MB")
    tok.save(args.out)


if __name__ == "__main__":
    main()
