# Human spot-check protocol (fill in by hand)

The rule-based win rate is circular: DPO was trained to satisfy the same
criterion. This blind check measures whether the wins look real to a person.

## Procedure

1. Generate the blind file (model identities hidden, order shuffled):
   `python bench/win_rate.py --config configs/base_30m.yaml ... --emit_spotcheck 40`
   This writes `win_rate.json.spotcheck.csv` (columns: pair_id, prompt, A, B,
   human_choice) plus a hidden `win_rate.json.spotcheck_map.json`. Do NOT open
   the mapping file until scoring is done.
2. For each of 30-50 rows, read A and B and write one of `A`, `B`, or `tie`
   into `human_choice`. Judge: (a) uses all prompt words naturally,
   (b) reads like a coherent story, (c) no repetition/gibberish.
3. Unblind with the mapping file and tally: human-DPO-agreement rate,
   i.e. fraction of non-tie human choices matching the DPO side, plus the
   rule-vs-human agreement on the same pairs.

## Fill in

- Rater:
- Date:
- Pairs judged: /40
- Human prefers DPO: , SFT: , ties:
- Rule-vs-human agreement:
- Notes (failure examples welcome; weak models fail interestingly):
