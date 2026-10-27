# Phase 6 pre-registration: sparse autoencoder, feature labeling, steering

Committed **before any SAE run** (the smoke runs on the debug model in S1-S3 test the
code and are not evaluated against these thresholds). Nothing below is changed after
results are seen. If something here turns out to be unworkable, the change is made in a
new dated section that says so, and the original rule and its outcome stay in the report.

## Scope fixed by the plan

- **Model for the SAE:** the pretrained base model (`weights_step12158.pt`), 8 blocks,
  d_model 512, context 512.
- **Layer:** index 4 only at first (the output of block 4, after both residual adds, as
  returned by `record_layers=[4]`). Layers 2 and 6 are started **only if layer 4 passes
  every rule A1-A4 below**.
- **SAE:** TopK, `d_in` 512, `F` = 4,096 features (8x), `k` = 32, about 100M training
  tokens from the train split. Training windows come from the deterministic shard
  loader; evaluation uses the validation split, which the SAE never trains on.
- **Labeling:** in-session by the assistant from text snippets; no external API.

## A. SAE acceptance (layer 4, base model)

Measured on held-out validation tokens (at least 500,000 tokens, i.e. 1,000 windows of
512, or all validation windows if fewer exist) unless stated. Definitions are the ones
implemented in `interp/eval_sae.py`.

| id | metric | definition | pass rule |
|---|---|---|---|
| A1 | fraction of variance unexplained (FVU) | `sum ||x - x_hat||^2 / sum ||x - mean(x)||^2` over tokens | **<= 0.15** |
| A2 | L0 | mean number of non-zero latents per token | **<= 32 and >= 8** |
| A3 | dead-feature fraction | fraction of the 4,096 features that never fire over **10M** fresh train-distribution tokens | **<= 0.10** |
| A4 | CE loss recovered | `1 - (CE_sae - CE_clean) / (CE_meanablate - CE_clean)`; CE_sae replaces the layer-4 residual by the SAE reconstruction, CE_meanablate replaces it by the dataset-mean residual | **>= 0.85** |
| A5 | SFT transfer | the **base-trained** SAE applied to the **SFT** model's layer-4 residual: FVU and CE recovered (with the SFT model's own CE_clean / mean-ablation), on plain validation windows **and** on template-formatted prompt+story sequences | **FVU <= 0.25 and CE recovered >= 0.80 on both** |

- **Layer 4 passes** iff A1-A4 all hold. A5 does not decide the layer; it decides which
  model steering targets (section C).
- **Remedies if layer 4 fails**, each tried at most once, thresholds unchanged: (R1) 2x
  the training tokens; (R2) F = 8,192 with k = 32. If it still fails, the failure is
  reported and layers 2 and 6 are not started.
- Reported alongside, without a pass rule: mean activation norm, mean decoder-column
  norm, the L0 histogram, FVU per position bucket (first 16 positions vs the rest).

## B. Feature labeling check

- **Sample:** 60 alive features (firing at least 20 times in a 10M-token scan) drawn
  uniformly at random with seed 20261101 from the accepted SAE. The features chosen for
  steering are labeled and checked separately by the same procedure.
- **Writing a label:** from the 24 highest-activating contexts (+-12 tokens, at most one
  per document) of a scan of the train distribution, plus the tokens the feature's
  decoder direction promotes and suppresses through the tied unembedding. The label is
  at most 12 words, or exactly **NO CLEAR LABEL** when no consistent pattern is visible.
- **Blind check, per labeled feature:** a shuffled file of 20 contexts from a *disjoint*
  validation-split scan: 10 true top-activating contexts and 10 control contexts
  (random positions where the feature is inactive). Given only the written label, the
  labeler marks the 10 contexts it predicts active. **Accuracy** = correct
  classifications / 20. The key lives in a separate git-ignored file and is opened only
  after scoring.
- **A label is valid iff accuracy >= 0.80** (16 of 20; probability 0.6% under chance).
- **The labeling procedure passes iff at least 70% of the labeled features are valid.**
  The **fraction of the sample with NO CLEAR LABEL is always reported** (no pass rule).
  If it exceeds 60%, the SAE is reported as largely uninterpretable at this size.
- **Limitation, stated up front:** the labeler and the scorer are the same model (the
  assistant), so this is a screening check, not independent validation. The user may
  spot-check a random 30 labels.

## C. Steering success

- **Target model:** the SFT model if A5 passes, otherwise the base model. Injection
  uses `inject` at layer 4, at **every position, in prefill and in every decode step**.
- **Features:** at least 5 features with valid labels (section B) whose concept can be
  written as a keyword set of at least 3 whole words. The features and keyword sets are
  fixed in `docs/steering_features.md` and committed **before any steering run**. A
  concept word that is also one of the prompt's three keywords is not counted.
- **Vector:** `s * a_f * d_f` expressed in raw residual units (`a_f` = 90th percentile
  of the feature's non-zero activations on held-out data, `d_f` its unit decoder
  direction, divided by the activation normalization scale).
- **Conditions per feature:**
  1. baseline (no steering);
  2. **dose-response**: scales **s in {1, 2, 4, 8}**;
  3. **random control**: at each scale, 3 random unit directions of the same norm
     (fixed seeds), averaged;
  4. **negative control**: s = **-2**;
  5. **ablation**: the feature's activation is removed (`x -= act_f(x) * d_f`, applied
     at every position) .
- **Samples:** 50 validation prompts (`train.eval_prompts.eval_prompt`) x n = 8, max_new
  256, temperature 0.8, top-k 40, seed = prompt idx, the same in every condition, the
  engine in its default (correct) mode.
- **Metrics:** concept rate = fraction of samples containing at least one concept word
  (whole word); `eot_rate`; distinct-2; perplexity ratio = mean NLL of the steered
  sample text scored by the **unsteered** target model, exponentiated, divided by the
  same for baseline samples.
- **A feature succeeds** iff there is a working scale `s*` in {1, 2, 4, 8} such that
  - (a) concept rate(feature, s*) - concept rate(random, s*) >= **0.10**, with the 95%
    prompt-bootstrap CI (paired by prompt, B = 10,000) excluding 0;
  - (b) dose-response: the same difference is non-decreasing (within 0.02) over the
    scales up to `s*`;
  - (c) controls: at s = -2 and under ablation the concept rate is not above baseline
    (the CI of the difference does not exclude 0 upward);
  - (d) fluency at `s*`: `eot_rate` >= baseline - 0.10, distinct-2 >= baseline - 0.05,
    perplexity ratio <= 1.25.
- **Phase 6 steering claim holds iff at least 3 of the >= 5 pre-chosen features succeed.**
  Otherwise the report says "steering effect not established". Every pre-chosen feature
  is reported in full, successes and failures.
- Also reported: the fraction of features with no clear label (section B), and for each
  succeeded feature the full dose-response table.
