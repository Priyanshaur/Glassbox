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

## Chosen features

(none yet)
