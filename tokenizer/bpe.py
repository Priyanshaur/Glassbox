"""Byte-level BPE tokenizer, from scratch.

ID layout (fixed, see docs/decisions.md):
  0-255   : single bytes
  256     : <|endoftext|> (EOT, appended after every doc in prepare.py)
  257     : <|pad|>
  258..   : learned merges, total vocab 8192 including specials.

Pre-tokenization: a GPT-2 style regex splits text into short chunks
(words, numbers, punctuation, whitespace runs). Merges never cross a
chunk boundary. Why: pure-Python pair counting on 50-100MB is only
tractable on short chunks, and it avoids junk merges spanning words.

Training speed trick: count each UNIQUE chunk once with a Counter
weighted by frequency, then train on the unique set. TinyStories repeats
the same words ("once", "upon", "the") millions of times, so unique
chunks (~tens of thousands) are far fewer than total chunks (millions).
Pair counts are updated incrementally: each merge only revisits chunks
containing that pair (via a pair -> chunks index), not the whole corpus.
"""
import json
from collections import Counter

import regex

# GPT-2 style pattern, simplified. Requires the `regex` package for \p{L}.
# Splits into: contractions, letters, numbers, punctuation, whitespace.
DEFAULT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

EOT = "<|endoftext|>"
PAD = "<|pad|>"
BASE_VOCAB = 256
EOT_ID = 256
PAD_ID = 257
MERGE_START = 258
DEFAULT_VOCAB_SIZE = 8192


class BPETokenizer:
    def __init__(self, merges=None, pattern=DEFAULT_PATTERN):
        self.pattern = pattern
        self._rx = regex.compile(pattern)
        self.specials = {EOT: EOT_ID, PAD: PAD_ID}
        self._special_re = regex.compile(
            "(" + "|".join(regex.escape(s) for s in self.specials) + ")"
        )
        # merges: list of (bytes_a, bytes_b) in learned order.
        self.merges = list(merges) if merges else []
        self._rebuild()

    # -- internal tables --
    def _rebuild(self):
        """Rebuild vocab + rank tables from merges. Called after train/load."""
        self.vocab = {}          # bytes -> id
        self.id_to_bytes = {}    # id -> bytes (None for specials)
        for i in range(256):
            b = bytes([i])
            self.vocab[b] = i
            self.id_to_bytes[i] = b
        self.id_to_bytes[EOT_ID] = None
        self.id_to_bytes[PAD_ID] = None
        for rank, (a, b) in enumerate(self.merges):
            new = a + b
            if new not in self.vocab:
                # Merges are unique by construction, so id = MERGE_START + rank.
                self.vocab[new] = MERGE_START + rank
                self.id_to_bytes[MERGE_START + rank] = new
        # rank lookup for encode: (bytes, bytes) -> merge order
        self.ranks = {pair: r for r, pair in enumerate(self.merges)}
        # chunk string -> list[int] cache for encode speed
        self._cache = {}

    @property
    def vocab_size(self):
        return MERGE_START + len(self.merges)

    # -- text splitting --
    def _split_specials(self, text):
        """Split text on special tokens, keeping them. Returns [(is_special, str)]."""
        parts = self._special_re.split(text)
        out = []
        for p in parts:
            if not p:
                continue
            out.append((p in self.specials, p))
        return out

    def _chunks(self, segment):
        return [m.group(0) for m in self._rx.finditer(segment)]

    # -- training --
    def train(self, texts, vocab_size=DEFAULT_VOCAB_SIZE, verbose=True):
        """Learn merges from an iterable of strings.

        texts: iterable of documents (already capped to ~50-100MB by caller).
        Builds Counter(chunk -> freq), then incrementally merges.
        Deterministic: ties broken by sorted byte order.
        """
        # 1. Count unique chunks (weighted). Specials split out first so a
        # literal "<|endoftext|>" inside training text never merges across.
        counts = Counter()
        for text in texts:
            for is_special, seg in self._split_specials(text):
                if is_special:
                    continue
                for ch in self._chunks(seg):
                    counts[ch] += 1
        if verbose:
            print(f"[bpe] unique chunks: {len(counts)} "
                  f"total chunks: {sum(counts.values())}")

        # 2. Current split per unique chunk: list of single-byte tokens.
        splits = {ch: [bytes([b]) for b in ch.encode("utf-8")] for ch in counts}
        # Global pair counts weighted by chunk frequency, plus index.
        pair_counts = Counter()
        pair_to_chunks = {}
        for ch, freq in counts.items():
            toks = splits[ch]
            seen = set()
            for a, b in zip(toks, toks[1:]):
                pair_counts[(a, b)] += freq
                if (a, b) not in seen:
                    seen.add((a, b))
                    pair_to_chunks.setdefault((a, b), set()).add(ch)

        n_merges = vocab_size - MERGE_START
        merges = []
        for i in range(n_merges):
            if not pair_counts:
                break
            # Deterministic best: max count, tie -> smallest bytes pair.
            best_count = max(pair_counts.values())
            cands = [p for p, c in pair_counts.items() if c == best_count]
            best = sorted(cands)[0]
            a, b = best
            new_tok = a + b
            merges.append((a, b))
            if verbose and (i < 5 or (i + 1) % 1000 == 0):
                print(f"[bpe] merge {i+1}/{n_merges}: "
                      f"{a!r}+{b!r} count={best_count}")
            # All chunks containing best pair get merged.
            affected = list(pair_to_chunks.get(best, ()))
            for ch in affected:
                freq = counts[ch]
                old = splits[ch]
                # Merge all non-overlapping occurrences left-to-right.
                new_split, j = [], 0
                while j < len(old):
                    if j < len(old) - 1 and old[j] == a and old[j + 1] == b:
                        new_split.append(new_tok)
                        j += 2
                    else:
                        new_split.append(old[j])
                        j += 1
                if len(new_split) == len(old):
                    continue
                old_pairs = Counter(zip(old, old[1:]))
                new_pairs = Counter(zip(new_split, new_split[1:]))
                # Apply deltas weighted by freq.
                for p in set(old_pairs) | set(new_pairs):
                    delta = (new_pairs.get(p, 0) - old_pairs.get(p, 0)) * freq
                    if delta:
                        pair_counts[p] += delta
                        if pair_counts[p] <= 0:
                            del pair_counts[p]
                    # Maintain index.
                    in_old = p in old_pairs
                    in_new = p in new_pairs
                    if in_new and not in_old:
                        pair_to_chunks.setdefault(p, set()).add(ch)
                    elif in_old and not in_new:
                        s = pair_to_chunks.get(p)
                        if s is not None:
                            s.discard(ch)
                            if not s:
                                pair_to_chunks.pop(p, None)
                splits[ch] = new_split
            # Best pair fully merged out of the running counts.
            pair_counts.pop(best, None)
            pair_to_chunks.pop(best, None)

        self.merges = merges
        self._rebuild()
        if verbose:
            print(f"[bpe] done: vocab={self.vocab_size} merges={len(merges)}")
        return self

    # -- encode / decode --
    def _encode_chunk(self, ch):
        """Encode one regex chunk to ids, using cache."""
        hit = self._cache.get(ch)
        if hit is not None:
            return hit
        toks = [bytes([b]) for b in ch.encode("utf-8")]
        # Greedily merge lowest-rank pair until none mergeable.
        while len(toks) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(toks) - 1):
                r = self.ranks.get((toks[i], toks[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = toks[best_i], toks[best_i + 1]
            toks = toks[:best_i] + [a + b] + toks[best_i + 2:]
        ids = [self.vocab[t] for t in toks]
        self._cache[ch] = ids
        return ids

    def encode(self, text):
        """Text -> list[int]. Special-token strings become their ids."""
        ids = []
        for is_special, seg in self._split_specials(text):
            if is_special:
                ids.append(self.specials[seg])
                continue
            for ch in self._chunks(seg):
                ids.extend(self._encode_chunk(ch))
        return ids

    def decode(self, ids):
        """ids -> str. Byte buffer flushed at specials so EOT/PAD roundtrip."""
        parts, buf = [], bytearray()
        inv_special = {v: k for k, v in self.specials.items()}
        for i in ids:
            if i in inv_special:
                if buf:
                    parts.append(bytes(buf).decode("utf-8", errors="replace"))
                    buf.clear()
                parts.append(inv_special[i])
                continue
            b = self.id_to_bytes.get(i)
            if b is None:
                raise ValueError(f"unknown id {i}")
            buf.extend(b)
        if buf:
            parts.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(parts)

    # -- persistence: single JSON file --
    def save(self, path):
        """Save merges as latin1 strings (byte-exact) + pattern + specials."""
        data = {
            "vocab_size": self.vocab_size,
            "pattern": self.pattern,
            "specials": self.specials,
            "merges": [[a.decode("latin1"), b.decode("latin1")]
                       for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"[bpe] saved {self.vocab_size} vocab to {path}")

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [(a.encode("latin1"), b.encode("latin1"))
                  for a, b in data["merges"]]
        tok = cls(merges=merges, pattern=data.get("pattern", DEFAULT_PATTERN))
        return tok
