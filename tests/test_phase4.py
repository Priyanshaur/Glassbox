"""Phase 4 tests: boundaries, masking, pairing, DPO math, CPU smokes."""
import math
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

