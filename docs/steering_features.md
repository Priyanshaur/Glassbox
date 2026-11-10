# Steering features - selection rules (fixed BEFORE any feature scan, labeling or steering run)

The pre-registration (`docs/phase6_preregistration.md`, section C) requires at least 5 features with valid labels whose
concept is a keyword set of at least 3 whole words, listed in this file before any steering run. This file fixes
how they may be chosen. It changes no threshold. **No feature has been chosen yet; the list is added below, and committed,
before the first steering run.**

## Eligibility: each of the 5 steering features must satisfy all of these

1. **Valid label:** passed the blind check of section B (accuracy >= 16 of 20), drawn from the accepted layer-4 SAE.
2. **Semantic, not token-level or punctuation features.** Judged on the feature's 24 top contexts from the train-distribution
   scan (position 0 skipped), using these checks, which are recorded for every candidate whether accepted or not:
   - no single token string accounts for more than 50% of the 24 top activating tokens;
   - at least 6 distinct token types among those 24;
   - the top tokens are not mostly punctuation, whitespace, digits, quotation marks, capitalisation or formatting tokens;
   - the label names a concept (an animal, weather, food, an emotion, a place or setting, an occupation, a kind of event)
     rather than a token, a spelling pattern, a position or a grammatical role.
3. **At least 5 distinct documents per feature.** The feature's activating contexts must come from at least 5 distinct
   documents, checked on both the 24 top train contexts and the 10 true-active contexts of the validation blind check. (The
   top-24 rule already takes at most one context per document; this requirement is stated so it is checked, not assumed.)
4. **Keyword set:** at least 3 whole content words that express the concept; no stop words, no inflection-only variants of one
   word counted twice, and none of the words that are also one of the steering prompt's three keywords (pre-registered rule).

## Amendment 1 (2026-11-05): seeded selection among eligible candidates

The pre-registration only requires at least 5 valid-label features. The seeded rule below (seed 20261102) is an addition
made by the assistant and approved by the user before any scan, labeling or steering run. It narrows the choice
and removes selection by outcome. It changes no threshold of the pre-registration.

## Choosing among eligible candidates (no steering results are used)

- All features that meet 1-4 are listed with their checks. If there are more than 5, the 5 are the first 5 in a fixed
  random order (seed 20261102) of that list. Features are never picked by how well they steered, because no steering run
  exists yet.
- If fewer than 5 are eligible, the shortfall is reported and the steering claim is not attempted on weaker features.
- Each chosen feature is recorded here with: feature index, label, the 5+ document ids behind rule 3, its check values, its
  keyword set and its `a_f` (90th percentile of non-zero activations on held-out data).

## Target model

SFT if A5 passes, otherwise base (decided by `analysis/a5_transfer.py`, thresholds unchanged).

## Chosen features (2026-11-07; fixed here BEFORE any steering run; no steering result exists)

Procedure as pre-announced: all 4096 features were put through the mechanical checks of rules 2 and 3
(`analysis/steering_candidates.py`; 509 passed), ordered with seed 20261102, and walked in that order. Each feature in the
order was read (top-24 train contexts, promoted tokens) and either rejected with a reason or labeled, blind-checked and
accepted; the walk stopped at the 5th accepted feature. The reading of "names a concept" (rule 2, last bullet) is a judgment
by the assistant, made without any steering result; the rejections are listed so it can be audited.

| rank in seeded order | feature | decision | reason |
|---|---|---|---|
| 1 | 84 | rejected | grammar: verb after "had to" |
| 2 | 3779 | rejected | grammar: pronoun after "I like X." |
| 3 | **1319** | accepted | foods and animals named in example lists |
| 4 | 756 | rejected | grammar: "it/them" after "use" |
| 5 | **3651** | accepted | household furniture (vase, bookcase, couch, ceiling) |
| 6 | 3328 | rejected | grammar: relative clause ("he had never been to") |
| 7 | 2461 | rejected | dialogue politeness tokens (OK / thank you), no concept |
| 8 | **1672** | accepted | clothing |
| 9 | **1131** | accepted | the night sky (stars, moon) |
| 10 | **3990** | accepted | animal as the subject of a sentence |

Not evaluated (after the 5th accepted): ranks 11 and later. Ranks 11-16 were read for the log only: 3347 (different
ways/tastes, abstract), 3694 (tidiness), 2816 (sentence-initial name), 2563 (He shows), 3589 (marriage), 485 ("didn't
look/be"); 3694 and 3589 were labeled and blind-checked as reserves (both 20/20) and are NOT used.

| feature | label (<= 12 words) | blind check | train24: docs / token types / top token share | val10: docs / top share | a_f (p90, validation scan) | keyword set (whole words) |
|---|---|---|---|---|---|---|
| 1319 | Plural food or animal nouns in lists of examples | 20/20 | 24 / 7 / 0.42 (apples) | 10 / 0.60 | 2.97 | apples, bananas, carrots, oranges, cheese, cows, pigs, chickens, horses |
| 3651 | Words after household furniture (vase, bookcase, couch, ceiling) | 20/20 | 24 / 11 / 0.21 (of) | 10 / 0.40 | 1.72 | vase, bookcase, bookshelf, couch, ceiling, shelf |
| 1672 | Clothing items | 20/20 | 24 / 6 / 0.42 (sweater) | 10 / 0.40 | 6.36 | shirt, coat, jacket, sweater, suit, pants |
| 1131 | The night sky: stars, moon, bright dots | 20/20 | 24 / 14 / 0.17 (they) | 10 / 0.30 | 2.26 | stars, moon, twinkling, dots |
| 3990 | An animal name right after "The" as the subject | 20/20 | 24 / 15 / 0.21 (dog) | 10 / 0.20 | 3.31 | crocodile, alligator, kangaroo, tiger, turtle, hippo, ostrich, bear, wolf, frog, spider |

- `a_f` is read from a log-binned histogram of the non-zero activations (bin width about 5.6%), so it is approximate.
- Honest limits: (1) the blind check is a screening check by the same model that wrote the labels, and it scored 60/60 at
  20/20 on the random sample too, because the control contexts are random positions and the labels name concrete tokens, so
  it can distinguish "this token" from "a random token" but says little about how well a label describes the concept;
  (2) 1131 and 3651 fire mostly on function words next to the concept (they, it, of, had), so steering them may raise
  concept words less directly than 1672, 1319 or 3990; (3) 1319 mixes two concepts (foods and farm animals).
- Words that are also one of a prompt's three keywords are excluded for that prompt (pre-registered rule).

## Result (kernel `priyanshgaur/glassbox-steering-full` v1, commit `4a8cddb`, 2026-11-10)

Pre-registered run, nothing changed after seeing data: SFT model, layer 4, 5 features, 50 validation prompts x n=8, max_new
256, T 0.8, top-k 40, scales {1,2,4,8}, 3 random same-norm controls, s=-2, ablation. 4550 cells, 11,678 s on a T4 (3.24 GPU h).
SHA-256 gates MATCH (SFT `d043156e...`, SAE `bd9ec5e3...`). Full numbers: `docs/phase6_steering_verdict.json`.

**Verdict: steering effect NOT established - 2 of 5 features succeed (need >= 3).**

| feature | concept | succeeded | working scale | why not (if failed) |
|---|---|---|---|---|
| 3651 | household furniture | yes | 4 | - |
| 1131 | night sky | yes | 2 | - |
| 1672 | clothing | **no** | none | huge effect, fluency rule (d) fails at every scale |
| 1319 | foods and animals | no | none | gain 0.076 at scale 4 (< 0.10), and fluency fails there |
| 3990 | animal as subject | no | none | no effect (gain <= 0.024) |

Concept-rate gain over the random control (feature minus random; baseline concept rate is 0.007-0.075):

| feature | s=1 | s=2 | s=4 | s=8 | s=-2 vs baseline | ablation vs baseline |
|---|---|---|---|---|---|---|
| 3651 | +0.022 | +0.074 | +0.149 | +0.399 | -0.010 | 0.000 |
| 1131 | +0.028 | +0.213 | +0.657 | +0.908 | -0.007 | -0.003 |
| 1672 | +0.658 | +0.935 | +0.999 | +1.000 | -0.015 | -0.007 |
| 1319 | +0.003 | +0.026 | +0.076 | -0.011 | -0.010 | 0.000 |
| 3990 | +0.011 | +0.024 | +0.016 | 0.000 | -0.030 | +0.003 |

Fluency at the working scale (baseline: eot 0.922, distinct-2 0.360, perplexity 2.41): 3651 at s=4: eot 0.945, distinct-2 0.318,
perplexity ratio 1.18; 1131 at s=2: eot 0.917, distinct-2 0.331, ratio 1.08.

- **Clothing (1672) is the clearest case of "works but breaks the text".** At s=1 the concept rate is 0.68 against 0.02 for the
  random control, but distinct-2 falls from 0.360 to 0.306 (limit: a fall of at most 0.05, missed by 0.004). At s=2 end-of-text rate is
  0.46 and the perplexity ratio 3.8. The pre-registered scale grid has nothing between 0 and 1, so no scale passes. We do not
  re-grade this feature as a success; the finer scale was not pre-registered.
- **Controls behave:** the random same-norm direction never raises the concept rate, s=-2 never raises it, and ablation changes it by
  at most 0.007. Ablation is nearly uninformative here because the baseline concept rate is about 1%.
- **Concept rate is a keyword count** (whole words, the prompt's own three keywords excluded), so it measures "these words appear",
  not that the story is about the concept. High scales produce word-salad that contains the words (see s=8 for 1672).
- Not tested: other layers, other prompt templates, features beyond these five, and the base model.
