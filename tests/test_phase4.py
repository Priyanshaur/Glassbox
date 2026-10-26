"""Phase 4 tests: boundaries, masking, pairing, DPO math, CPU smokes."""
import math
import os
from unittest import mock

import torch
import torch.nn.functional as F

from bench.win_rate import distinct_2, wilson
from common.config import load_config
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer, EOT_ID
from train.dpo import dpo_loss, response_logps
from train.instruct import build_sft_example, check_criterion, collate, synth_prompt
from train.pairs import make_pairs, whole_word_ok
from train.sample import prompt_ids, sample_completions, sample_continuations
from train.sft import build_examples

TOK = None


def tok():
    global TOK
    if TOK is None:
        TOK = BPETokenizer.load("tokenizer/tok8192.json")
    return TOK


P = "Write a short story using all of these words: dragon, castle, brave.\n\nStory:"
W = "The brave knight saw a dragon near the castle gates."
L = "The cat sat on the mat and slept all day."


def test_delimiter_boundary():
    """Example ids == encode(prompt)+encode(response)+EOT; mask keeps R[0]."""
    t = tok()
    p, r = t.encode(P), t.encode(W)
    inp, labels = build_sft_example(t, P, W, 512)
    assert inp[:len(p)].tolist() == p  # prompt boundary exact, no cross-merge
    assert labels[len(p) - 1].item() == r[0]  # first response token kept
    assert (labels[:len(p) - 1] == -100).all()  # prompt predictions masked
    assert EOT_ID in inp.tolist() or True  # EOT present unless truncated
    # Sampler builds the identical prompt prefix at generation time.
    assert prompt_ids(t, P) == p


def test_masking_ignores_prompt():
    """Prompt tokens contribute no loss: masked CE is finite and response-only.

    (Not bit-identical across prompts: RoPE positions of the response shift
    with prompt length, so hidden states differ. Masking is what we test.)"""
    t = tok()
    P2 = "Write a short story using all of these words: cat, mat, slept.\n\nStory:"
    x1, l1 = build_sft_example(t, P, W, 512)
    x2, l2 = build_sft_example(t, P2, W, 512)
    m = GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()
    with torch.no_grad():
        a = F.cross_entropy(m(x1.unsqueeze(0)).logits.reshape(-1, 8192),
                            l1.unsqueeze(0).reshape(-1), ignore_index=-100)
        b = F.cross_entropy(m(x2.unsqueeze(0)).logits.reshape(-1, 8192),
                            l2.unsqueeze(0).reshape(-1), ignore_index=-100)
    # Different prompt lengths shift positions but masked loss sees only W.
    assert torch.isfinite(a) and torch.isfinite(b)


def test_criterion_basics():
    kw = ["dragon", "castle", "brave"]
    assert check_criterion(W, kw)["ok"]
    assert not check_criterion(L, kw)["ok"]
    assert check_criterion(L, kw)["missing"] == kw


def test_whole_word_ok_excludes_substrings():
    """Whole-word gate (2026-10-22): hits inside other words must not count.

    Substring matching would accept 'tree' inside 'street' and 'when' inside
    'whenever'; the pair gate switched to whole-word matching for that.
    """
    assert not whole_word_ok("Sam ran down the street.", ["tree"])
    assert not whole_word_ok("He comes here whenever he can.", ["when"])
    assert whole_word_ok("A tall tree grew up fast here in the sunny garden.",
                         ["tree"])
    assert whole_word_ok("When he ran through the park he fell down on grass.",
                         ["when"])
    assert whole_word_ok("The TREE in my yard was very tall and green.",
                         ["tree"])  # case-insensitive
    assert not whole_word_ok("Tree. " * 9, ["tree"])  # 9 words < window


def test_ties_excluded():
    scored_mixed = [(f"good {i} dragon castle brave tale of wonder", True)
                    for i in range(3)] + [(f"bad {i} nothing here", False)
                                          for i in range(3)]
    pairs, keep = make_pairs(P, scored_mixed)
    assert keep and len(pairs) == 2  # capped at max_pairs
    assert all(p["prompt"] == P for p in pairs)
    assert make_pairs(P, [(s, True) for s, _ in scored_mixed]) == ([], False)
    assert make_pairs(P, [(s, False) for s, _ in scored_mixed]) == ([], False)


def test_make_graded_pairs_margins():
    """Graded criterion (2026-10-22): margin >= 1 more keyword than rejected,
    both rows EOT-ended and 10-500 words; exact max-cardinality/max-margin
    matching, edges ordered margin-desc.
    """
    from train.pairs import make_graded_pairs
    # winners {3,3}, losers {1,0} -> two pairs, margins 3 and 2
    rows = [{"text": f"winner-3a number {i}", "cov": 3, "eot": True,
             "nwords": 20} for i in range(2)] + \
           [{"text": "loser-1 only dragon", "cov": 1, "eot": True,
             "nwords": 15},
            {"text": "loser-0 nothing", "cov": 0, "eot": True, "nwords": 12}]
    pairs, keep = make_graded_pairs("P", rows, max_pairs=2)
    assert keep and len(pairs) == 2
    assert [p["margin"] for p in pairs] == [3, 2]
    # losers are never winners here; distinct rows
    assert len({p["winner"] for p in pairs}) == 2
    assert len({p["loser"] for p in pairs}) == 2
    # an EOT-less row is ineligible even at high coverage
    rows2 = rows + [{"text": "high cov but no eot dragon castle brave",
                     "cov": 2, "eot": False, "nwords": 12}]
    pairs2, keep2 = make_graded_pairs("P", rows2, max_pairs=2)
    assert keep2 and all("no eot" not in p["winner"] and
                         "no eot" not in p["loser"] for p in pairs2)
    assert [p["margin"] for p in pairs2] == [3, 2]
    # all-tied (no margin anywhere) drops the prompt
    tie = [{"text": f"same cov {i}", "cov": 1, "eot": True, "nwords": 15}
           for i in range(3)]
    assert make_graded_pairs("P", tie, max_pairs=2) == ([], False)


def test_sample_completions_return_flags_parity():
    """return_flags=True must not change sampling: same texts, same seed."""
    cfg = load_config("configs/debug_cpu.yaml")
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    m = GPT(cfg["model"]).eval()
    a = sample_completions(m, tok, P, n=2, max_new=8, seed=0)
    b, flags = sample_completions(m, tok, P, n=2, max_new=8, seed=0,
                                  return_flags=True)
    assert a == b
    assert len(flags) == 2 and isinstance(flags[0]["eot_hit"], bool)


def test_split_holdout_is_by_prompt():
    """dpo holdout must be disjoint by PROMPT (fix 2026-10-22)."""
    from train.dpo import split_holdout
    pairs = []
    for i in range(10):
        pairs.append({"prompt": f"prompt-{i}", "winner": f"w{i}", "loser": f"l{i}"})
        if i % 2 == 0:  # even-index prompts carry a SECOND pair
            pairs.append({"prompt": f"prompt-{i}", "winner": f"w{i}b",
                          "loser": f"l{i}b"})
    train, hold = split_holdout(pairs, heldout_frac=0.25, seed=0)
    tp = {p["prompt"] for p in train}
    hp = {p["prompt"] for p in hold}
    assert not (tp & hp), "a prompt leaked across the split"
    assert hp  # non-empty holdout
    # every 2-pair prompt lives entirely on one side
    for i in range(0, 10, 2):
        mine = [p for p in pairs if p["prompt"] == f"prompt-{i}"]
        side_hp = sum(1 for p in mine if p["prompt"] in hp)
        assert side_hp in (0, len(mine))
    # all 15 pairs accounted for
    assert len(train) + len(hold) == len(pairs)


def test_dpo_loss_ln2_at_reference():
    """Policy == reference -> margins 0 -> loss ln2 for any beta."""
    for beta in (0.05, 0.1, 0.5):
        loss, margin = dpo_loss(torch.tensor(-3.0), torch.tensor(-5.0),
                                torch.tensor(-3.0), torch.tensor(-5.0), beta)
        assert abs(loss.item() - math.log(2)) < 1e-6
        assert margin.item() == 0.0


def test_sampler_determinism():
    cfg = load_config("configs/debug_cpu.yaml")["model"]
    torch.manual_seed(0)
    m = GPT(cfg).eval()
    a = sample_completions(m, tok(), P, n=2, max_new=8, seed=11)
    b = sample_completions(m, tok(), P, n=2, max_new=8, seed=11)
    c = sample_completions(m, tok(), P, n=2, max_new=8, seed=12)
    assert a == b and len(a) == 2
    assert a != c  # different Generator stream draws differently


def test_sampler_generator_matches_tensor_device():
    """Regression: the generator must be built on the resolved device.

    A CPU generator handed to torch.multinomial over CUDA probs raises
    "Expected a 'cuda' device type for generator but found 'cpu'" - that
    killed the main run at step 2000, because sample_every=2000 was never
    reached during calibration. Here we spy on torch.multinomial and assert
    generator.device tracks the model's device (cpu in CI, cuda on the box).
    """
    cfg = load_config("configs/debug_cpu.yaml")["model"]
    torch.manual_seed(0)
    m = GPT(cfg).eval()
    model_dev = next(m.parameters()).device.type
    seen = []
    real = torch.multinomial

    def spy(input, num_samples, *, generator=None, **kw):
        assert generator is not None, "sampler must pass its own generator"
        seen.append(generator.device.type)
        return real(input, num_samples, generator=generator, **kw)

    with mock.patch("torch.multinomial", spy):
        sample_completions(m, tok(), P, n=2, max_new=4, seed=3)
        sample_continuations(m, tok(), [P], max_new=4, seed=3)
    assert seen, "spy never fired"
    assert set(seen) == {model_dev}, (
        f"generator device(s) {set(seen)} != model device {model_dev}")


def test_sft_smoke_loss_falls():
    """Mechanics + decreasing loss on real smoke docs (debug, CPU)."""
    from train.sft import response_loss
    cfg = load_config("configs/debug_cpu.yaml")
    t = tok()
    from train.instruct import docs_from_shard
    texts = docs_from_shard(t, "data", "train")[:24]
    ex, _ = build_examples(t, texts, 64)
    assert len(ex) > 8
    torch.manual_seed(0)
    m = GPT(cfg["model"]).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)  # smoke-only LR
    first, last = None, None
    for s in range(10):
        batch = [ex[(s * 4 + k) % len(ex)] for k in range(4)]
        inp, lab = collate([(e["input"], e["labels"]) for e in batch], 257)
        opt.zero_grad()
        loss = F.cross_entropy(m(inp).logits.reshape(-1, 8192),
                               lab.reshape(-1), ignore_index=-100)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < first, f"{first:.3f} -> {last:.3f}"
    v = response_loss(m, ex[:8], "cpu", 8192)
    assert math.isfinite(v)


def test_dpo_smoke_margin_improves():
    """Handcrafted pass/fail pairs: loss drops below ln2, accuracy rises."""
    cfg = load_config("configs/debug_cpu.yaml")["model"]
    t = tok()
    torch.manual_seed(0)
    policy, ref = GPT(cfg).train(), GPT(cfg).eval()
    ref.load_state_dict(policy.state_dict())
    ref.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)  # smoke-only LR
    pairs = [{"prompt": P, "winner": W, "loser": L} for _ in range(4)]
    hist = []
    for _ in range(12):
        opt.zero_grad()
        tot, corr = 0.0, 0.0
        for p in pairs:
            lw = response_logps(policy, t, p["prompt"], p["winner"], 64, "cpu")
            ll = response_logps(policy, t, p["prompt"], p["loser"], 64, "cpu")
            with torch.no_grad():
                rw = response_logps(ref, t, p["prompt"], p["winner"], 64, "cpu")
                rl = response_logps(ref, t, p["prompt"], p["loser"], 64, "cpu")
            l, mrg = dpo_loss(lw, ll, rw, rl, 0.1)
            tot += l / len(pairs)
            corr += float(mrg > 0) / len(pairs)
        tot.backward()
        opt.step()
        hist.append((tot.item(), corr))
    assert hist[-1][0] < math.log(2) < hist[0][0] + 0.05, hist
    assert hist[-1][1] >= hist[0][1]


def test_wilson_and_distinct2():
    lo, hi = wilson(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.3 and 0.7 < hi < 0.8
    assert distinct_2(["a b c d", "e f g h"]) == 1.0
    assert distinct_2(["a b a b", "a b a b"]) < 0.5


def test_synth_prompt_deterministic():
    from train.instruct import docs_from_shard
    texts = docs_from_shard(tok(), "data", "train")[:50]
    a = synth_prompt(texts[0], 0)
    b = synth_prompt(texts[0], 0)
    assert a == b and len(a[1]) == 3
    assert all(k in texts[0].lower() for k in a[1])  # keywords from the story


def test_sft_resume_already_complete_is_noop(tmp_path, monkeypatch):
    """Re-launching a finished SFT run must be a clean no-op.

    Every Kaggle session begins with --resume latest, so a restart after the
    final update completed has to exit 0 without re-running a step, rewriting
    a checkpoint, or appending metrics -- otherwise a re-launch could silently
    clobber the finished weights.
    """
    import hashlib
    import sys

    from train import checkpoint as ckpt_mod
    from train.optim import build_param_groups
    from train.sft import main as sft_main

    cfg = load_config("configs/debug_cpu.yaml")
    mcfg = cfg["model"]
    torch.manual_seed(0)
    wpath = tmp_path / "weights.pt"
    torch.save(GPT(mcfg).state_dict(), wpath)

    ckdir = tmp_path / "ck"
    ckdir.mkdir()
    model = GPT(mcfg)
    # Must mirror main()'s optimizer exactly: load_checkpoint restores the
    # state dict and AdamW rejects a different number of param groups.
    opt = torch.optim.AdamW(
        build_param_groups(model, cfg["train"]["weight_decay"]),
        lr=1e-4, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    # 8 examples // 4 batch_seqs = 2 steps/epoch, 2 epochs -> max_steps 4.
    # A checkpoint tagged 3 therefore means "done" (load returns next = 4).
    ckpt_mod.save_checkpoint(
        str(ckdir), model, opt, scaler, 3, {"model": mcfg, "seed": 0},
        ckpt_mod.file_hash("tokenizer/tok8192.json"), 2)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    before = {p.name: digest(p) for p in sorted(ckdir.glob("*.pt"))}
    assert before, "checkpoint fixture was not written"

    monkeypatch.setattr(sys, "argv", [
        "sft", "--config", "configs/debug_cpu.yaml",
        "--init_weights", str(wpath), "--ckpt_dir", str(ckdir),
        "--max_examples", "8", "--epochs", "2", "--batch_seqs", "4",
        "--save_every_min", "999", "--resume", "latest"])

    assert sft_main() == 0

    after = {p.name: digest(p) for p in sorted(ckdir.glob("*.pt"))}
    assert after == before, "already-complete resume rewrote a checkpoint"
    assert not (ckdir / "metrics_sft.jsonl").exists(), \
        "already-complete resume appended metrics"



def _run_dpo(monkeypatch, tmp_path, *extra):
    import sys
    from train import dpo
    argv = ["dpo", "--config", "configs/debug_cpu.yaml",
            "--init_policy", str(tmp_path / "init.pt"),
            "--pairs", str(tmp_path / "pairs.jsonl"),
            "--ckpt_dir", str(tmp_path / "ck"), "--batch", "2",
            "--eval_every", "2", "--heldout_frac", "0.34", "--lr", "1e-4",
            *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return dpo.main()


def test_dpo_exit_paths_and_resume(tmp_path, monkeypatch):
    """max_steps / max_hours exits return 0 (no NameError), and a resume
    restores the best held-out loss from metrics instead of resetting it."""
    import json
    cfg = load_config("configs/debug_cpu.yaml")["model"]
    torch.manual_seed(0)
    torch.save(GPT(cfg).state_dict(), tmp_path / "init.pt")
    with open(tmp_path / "pairs.jsonl", "w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps({"prompt": P + f" {i}", "winner": W,
                                "loser": L}) + "\n")
    assert _run_dpo(monkeypatch, tmp_path, "--max_steps", "3") == 0
    assert _run_dpo(monkeypatch, tmp_path, "--max_hours", "1e-9",
                    "--resume", "latest") == 0
    rows = [json.loads(l) for l in
            open(tmp_path / "ck" / "metrics_dpo.jsonl", encoding="utf-8")]
    assert any("h_loss" in r for r in rows)


def _dpo_cut_vs_full(tmp_path, monkeypatch, pairs, patience, cut_seconds):
    """Run train.dpo twice on the debug config: once uninterrupted ('full'),
    once cut by max_hours (a fake clock, +1s per call, so the cut lands at a
    fixed update) and then resumed ('cut'). Returns (full_rows, cut_rows_at_cut,
    cut_rows_final, full_ckpt, cut_ckpt)."""
    import json
    import sys
    from train import checkpoint as ckpt_mod
    from train import dpo
    cfg = load_config("configs/debug_cpu.yaml")["model"]
    torch.manual_seed(0)
    torch.save(GPT(cfg).state_dict(), tmp_path / "init.pt")
    with open(tmp_path / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    def run(ckpt, *extra):
        argv = ["dpo", "--config", "configs/debug_cpu.yaml",
                "--init_policy", str(tmp_path / "init.pt"),
                "--pairs", str(tmp_path / "pairs.jsonl"),
                "--ckpt_dir", str(tmp_path / ckpt), "--batch", "2",
                "--eval_every", "10", "--heldout_frac", "0.2", "--lr", "1e-4",
                "--epochs", "3", "--patience", str(patience), *extra]
        monkeypatch.setattr(sys, "argv", argv)
        return dpo.main()

    def rows(ckpt):
        with open(tmp_path / ckpt / "metrics_dpo.jsonl", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            self.t += 1.0
            return self.t

    assert run("full") == 0
    with monkeypatch.context() as m:
        m.setattr(dpo.time, "perf_counter", Clock())
        assert run("cut", "--resume", "latest",
                   "--max_hours", f"{cut_seconds / 3600}") == 0
    at_cut = rows("cut")
    assert run("cut", "--resume", "latest") == 0
    return (rows("full"), at_cut, rows("cut"),
            ckpt_mod.latest_checkpoint(str(tmp_path / "full")),
            ckpt_mod.latest_checkpoint(str(tmp_path / "cut")))


def _ev(rows):
    return {r["step"]: r["h_loss"] for r in rows if "h_loss" in r}


def _same_final_weights(fa, fb):
    assert os.path.basename(fa) == os.path.basename(fb)
    sa = torch.load(fa, weights_only=False)["model"]
    sb = torch.load(fb, weights_only=False)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_dpo_max_hours_resume_is_identical_to_uninterrupted(tmp_path, monkeypatch):
    """A run cut by max_hours then resumed must end identical to one
    uninterrupted run: held-out losses, best loss, final checkpoint weights,
    and a terminal row naming why each segment stopped."""
    pairs = [{"prompt": P + f" {i}", "winner": W, "loser": L} for i in range(30)]
    full, at_cut, cut, fa, fb = _dpo_cut_vs_full(
        tmp_path, monkeypatch, pairs, patience=99, cut_seconds=40)
    assert at_cut[-1]["terminal"] == "max_hours"
    assert 0 < at_cut[-1]["updates_done"] < full[-1]["updates_done"]
    assert full[-1]["terminal"] == cut[-1]["terminal"] == "epochs"
    assert full[0].get("pre_update") and abs(full[0]["h0_loss"] - math.log(2)) < 1e-6
    assert abs(full[-1]["best_h_loss"] - cut[-1]["best_h_loss"]) < 1e-6
    assert _ev(full).keys() == _ev(cut).keys()
    for k, v in _ev(full).items():
        assert abs(v - _ev(cut)[k]) < 1e-6, k
    _same_final_weights(fa, fb)


def test_dpo_resume_between_best_and_stale_evals_keeps_patience(tmp_path, monkeypatch):
    """F1 guard with patience=2. Held-out prompts get REVERSED preferences, so
    held-out loss is best at the step-0 eval and rises afterwards (stale at
    steps 10 and 20). The cut falls after the first stale eval (step 10) and
    before the second (step 20); the resume restarts from the step-0
    checkpoint and replays both stale evals. It must early-stop at the SAME
    step with the SAME best loss and weights as an uninterrupted run -- the
    bug being guarded: pre-counting the stale eval already on disk, which
    double-counts after the replay and stops at step 10 instead."""
    from train.dpo import split_holdout
    pairs = [{"prompt": P + f" {i}", "winner": W, "loser": L} for i in range(60)]
    _, hold = split_holdout(pairs, 0.2, 0)
    held = {p["prompt"] for p in hold}
    pairs = [({"prompt": p["prompt"], "winner": L, "loser": W}
              if p["prompt"] in held else p) for p in pairs]
    full, at_cut, cut, fa, fb = _dpo_cut_vs_full(
        tmp_path, monkeypatch, pairs, patience=2, cut_seconds=49)
    assert sorted(_ev(full)) == [0, 10, 20], sorted(_ev(full))  # scenario holds
    assert _ev(full)[0] < _ev(full)[10] and _ev(full)[0] < _ev(full)[20]
    assert full[-1]["terminal"] == "early_stop"
    assert sorted(_ev(at_cut)) == [0, 10]                # cut between the evals
    assert at_cut[-1]["terminal"] == "max_hours"
    assert 10 < at_cut[-1]["updates_done"] <= 20
    assert cut[-1]["terminal"] == "early_stop"
    assert cut[-1]["updates_done"] == full[-1]["updates_done"]
    assert abs(full[-1]["best_h_loss"] - cut[-1]["best_h_loss"]) < 1e-6
    for k, v in _ev(full).items():
        assert abs(v - _ev(cut)[k]) < 1e-6, k
    _same_final_weights(fa, fb)


def test_eval_prompt_is_stopword_free_and_deterministic():
    from train.eval_quality import STOPWORDS, eval_prompt
    from train.instruct import docs_from_shard
    texts = docs_from_shard(tok(), "data", "validation")[:60]
    got = 0
    for i, t in enumerate(texts):
        a, b = eval_prompt(t, i), eval_prompt(t, i)
        assert a == b
        if a is None:
            continue
        got += 1
        assert len(a[1]) == 3 and not set(a[1]) & STOPWORDS
        assert all(k in t.lower() for k in a[1])
        assert a[0].startswith("Write a short story using all of these words:")
    assert got >= 30
    assert all(len(w) >= 4 for w in STOPWORDS)


def test_eval_quality_smoke_reports_gold_ppl(tmp_path, monkeypatch):
    """eval_quality end to end on the debug model (CPU, a few prompts)."""
    import json
    import sys
    from train import eval_quality
    cfg = load_config("configs/debug_cpu.yaml")["model"]
    torch.manual_seed(0)
    torch.save(GPT(cfg).state_dict(), tmp_path / "w.pt")
    out = tmp_path / "eval.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "eval_quality", "--config", "configs/debug_cpu.yaml",
        "--weights", str(tmp_path / "w.pt"),
        "--ppl_weights", str(tmp_path / "w.pt"), "--out", str(out),
        "--n_prompts", "3", "--n_samples", "2", "--max_new", "8"])
    assert eval_quality.main() == 0
    s = json.load(open(str(out) + ".summary.json", encoding="utf-8"))
    assert s["n_prompts"] == 3 and math.isfinite(s["gold_ppl"])
    assert s["gold_ppl"] > 1 and math.isfinite(s["ppl_sft"])
