# TinyStories source check (streamed from Hugging Face)

## A. What `data/prepare.py` receives

Same call as `data/prepare.py`: `load_dataset("roneneldan/TinyStories", split=..., streaming=True)`, field `text`, nothing else applied.

| split | stories | with garbled U+00E2 U+20AC U+2122 | % | with correct U+2019 | garbled count | correct count |
|---|---|---|---|---|---|---|
| train | 3000 | 147 | 4.9% | 0 | 331 | 0 |
| validation | 3000 | 110 | 3.7% | 0 | 229 | 0 |

## B. Original raw text vs the parquet-derived text

First 4,000,000 bytes of the original `TinyStories-valid.txt` (4,843 complete stories):

- garbled-form bytes (`C3 A2 E2 82 AC E2 84 A2`): **0**
- correct U+2019 bytes (`E2 80 99`): **372**

Story-by-story, whitespace-normalised, same order:

| group | stories | parquet text == raw text | parquet text == raw text read as cp1252 (`errors="ignore"`) |
|---|---|---|---|
| ASCII-only raw stories | 4561 | 4561 | - |
| raw stories with any non-ASCII character | 282 | 0 | **282** |

**Conclusion.** The original `.txt` files are clean UTF-8 (correct U+2019, no garbled sequences). The parquet text that the default loader serves, and that `data/prepare.py` tokenizes unchanged, equals the raw text re-decoded as cp1252 with undefined bytes dropped, for every non-ASCII story checked. So the corruption is in the Hub's parquet files, not introduced by `data/prepare.py`. It is partly irreversible: the closing double quote U+201D loses its third byte.
