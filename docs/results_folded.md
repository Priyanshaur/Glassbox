# Keyword coverage re-scored with garbled characters folded back

The corpus contains the garbled sequence U+00E2 U+20AC U+2122 instead of an apostrophe (`docs/source_mojibake_check.md`). U+00E2 is a word character, so a keyword directly next to it would fail the whole-word match. **Original** = text as generated (the numbers in `docs/results_final.md`, unchanged and still primary). **Apostrophe folded** = that sequence replaced by an ASCII apostrophe before matching. **All folded** is an extra check that also folds the garbled double-quote and dash sequences.

## How many samples are affected

| model | samples | with garbled apostrophe | with any garbled sequence | coverage changes (apostrophe fold) | coverage changes (all folded) | keyword hits gained (all folded) |
|---|---|---|---|---|---|---|
| sft | 600 | 10 (1.7%) | 16 (2.7%) | 0 | 0 | 0 |
| dpo_s0 | 600 | 20 (3.3%) | 31 (5.2%) | 0 | 1 | 1 |
| dpo_s1 | 600 | 14 (2.3%) | 28 (4.7%) | 0 | 0 | 0 |

## Mean keyword coverage (0-3): original vs folded

| model | original | apostrophe folded | all folded |
|---|---|---|---|
| sft | 0.8917 | 0.8917 | 0.8917 |
| dpo_s0 | 1.7067 | 1.7067 | 1.7083 |
| dpo_s1 | 1.7683 | 1.7683 | 1.7683 |

DPO minus SFT, mean keyword coverage (0-3) (95% CI over prompts, B=10,000):

| comparison | original diff [CI] | apostrophe folded diff [CI] | all folded diff [CI] |
|---|---|---|---|
| dpo_s0 minus sft | +0.8150 [+0.7100, +0.9167] | +0.8150 [+0.7100, +0.9167] | +0.8167 [+0.7133, +0.9183] |
| dpo_s1 minus sft | +0.8767 [+0.7733, +0.9800] | +0.8767 [+0.7733, +0.9800] | +0.8767 [+0.7733, +0.9800] |

## Fraction covering all 3: original vs folded

| model | original | apostrophe folded | all folded |
|---|---|---|---|
| sft | 0.0283 | 0.0283 | 0.0283 |
| dpo_s0 | 0.1700 | 0.1700 | 0.1700 |
| dpo_s1 | 0.2067 | 0.2067 | 0.2067 |

DPO minus SFT, fraction covering all 3 (95% CI over prompts, B=10,000):

| comparison | original diff [CI] | apostrophe folded diff [CI] | all folded diff [CI] |
|---|---|---|---|
| dpo_s0 minus sft | +0.1417 [+0.1083, +0.1767] | +0.1417 [+0.1083, +0.1767] | +0.1417 [+0.1083, +0.1767] |
| dpo_s1 minus sft | +0.1783 [+0.1350, +0.2233] | +0.1783 [+0.1350, +0.2233] | +0.1783 [+0.1350, +0.2233] |

## Reading

Folding the garbled apostrophe changed the coverage of **0** of 1800 samples; folding every garbled sequence changed **1**. The whole-word undercount is possible in principle (a keyword directly before or after a garbled character), but it did not occur in these samples, so the original numbers and the DPO-minus-SFT CIs are unaffected here.
