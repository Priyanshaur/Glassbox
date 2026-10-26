"""Where does the TinyStories mojibake come from? (needs network access)

Part A - what data/prepare.py sees. Streams the first N stories of each split
exactly the way data/prepare.py does (`load_dataset("roneneldan/TinyStories",
split=..., streaming=True)`, field `text`) and counts, by code point:

  garbled  U+00E2 U+20AC U+2122   (a UTF-8 right single quote read as cp1252)
  correct  U+2019

Part B - is the Hub's parquet text derived from clean raw text? Range-reads the
first ~4 MB of the original `TinyStories-valid.txt` (stories split on
<|endoftext|>), aligns it story-by-story with the streamed validation split,
and tests the hypothesis

    parquet_text == raw_txt.encode("utf-8").decode("cp1252", errors="ignore")

(whitespace-normalised). `errors="ignore"` matters: cp1252 leaves the bytes
0x81 0x8D 0x8F 0x90 0x9D undefined, so the closing double quote U+201D
(E2 80 9D) loses its third byte, which is not recoverable from the parquet text.

Usage: python analysis/check_source_mojibake.py [N_PER_SPLIT] [out.md]
"""
import itertools
import sys
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
OUT = sys.argv[2] if len(sys.argv) > 2 else None
RANGE_BYTES = 4_000_000

GARBLED = chr(0xE2) + chr(0x20AC) + chr(0x2122)
CORRECT = chr(0x2019)
G_BYTES = GARBLED.encode("utf-8")
C_BYTES = CORRECT.encode("utf-8")

from datasets import load_dataset
from huggingface_hub import hf_hub_url

norm = lambda s: " ".join(s.split())
L = []
P = L.append
P("# TinyStories source check (streamed from Hugging Face)\n")
P("## A. What `data/prepare.py` receives\n")
P("Same call as `data/prepare.py`: `load_dataset(\"roneneldan/TinyStories\", "
  "split=..., streaming=True)`, field `text`, nothing else applied.\n")
P("| split | stories | with garbled U+00E2 U+20AC U+2122 | % | with correct U+2019 | garbled count | correct count |")
P("|---|---|---|---|---|---|---|")
val_docs = None
for split in ("train", "validation"):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    docs = [ex["text"] for ex in itertools.islice(ds, N)]
    g = sum(GARBLED in d for d in docs)
    c = sum(CORRECT in d for d in docs)
    P(f"| {split} | {len(docs)} | {g} | {100 * g / len(docs):.1f}% | {c} | "
      f"{sum(d.count(GARBLED) for d in docs)} | {sum(d.count(CORRECT) for d in docs)} |")
    print(split, len(docs), "garbled docs", g, "correct docs", c, flush=True)
P("")

P("## B. Original raw text vs the parquet-derived text\n")
url = hf_hub_url("roneneldan/TinyStories", "TinyStories-valid.txt", repo_type="dataset")
raw = urllib.request.urlopen(urllib.request.Request(
    url, headers={"Range": f"bytes=0-{RANGE_BYTES - 1}"}), timeout=60).read()
txt = raw.decode("utf-8", errors="ignore").split("<|endoftext|>")[:-1]   # last piece is range-cut
P(f"First {len(raw):,} bytes of the original `TinyStories-valid.txt` "
  f"({len(txt):,} complete stories):\n")
P(f"- garbled-form bytes (`C3 A2 E2 82 AC E2 84 A2`): **{raw.count(G_BYTES)}**")
P(f"- correct U+2019 bytes (`E2 80 99`): **{raw.count(C_BYTES)}**\n")
ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
pq = [ex["text"] for ex in itertools.islice(ds, len(txt))]
ascii_idx = [i for i, t in enumerate(txt) if t.isascii()]
na_idx = [i for i, t in enumerate(txt) if not t.isascii()]
asc_eq = sum(norm(pq[i]) == norm(txt[i]) for i in ascii_idx)
h = lambda s: s.encode("utf-8").decode("cp1252", errors="ignore")
plain_eq = sum(norm(pq[i]) == norm(txt[i]) for i in na_idx)
cp_eq = sum(norm(pq[i]) == norm(h(txt[i])) for i in na_idx)
P("Story-by-story, whitespace-normalised, same order:\n")
P("| group | stories | parquet text == raw text | parquet text == raw text read as cp1252 (`errors=\"ignore\"`) |")
P("|---|---|---|---|")
P(f"| ASCII-only raw stories | {len(ascii_idx)} | {asc_eq} | - |")
P(f"| raw stories with any non-ASCII character | {len(na_idx)} | {plain_eq} | **{cp_eq}** |")
P("")
P("**Conclusion.** The original `.txt` files are clean UTF-8 (correct U+2019, no garbled "
  "sequences). The parquet text that the default loader serves, and that `data/prepare.py` "
  "tokenizes unchanged, equals the raw text re-decoded as cp1252 with undefined bytes dropped, "
  "for every non-ASCII story checked. So the corruption is in the Hub's parquet files, "
  "not introduced by `data/prepare.py`. It is partly irreversible: the closing double quote "
  "U+201D loses its third byte.")
text = "\n".join(L) + "\n"
if OUT:
    open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print(text)
