"""Steering hook of the generation engine: `inject` is opt-in; inject=None / a zero vector change nothing."""
import torch

from common.config import load_config
from inference.generate import Generator
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer

L = 1


def _gen():
    cfg = load_config("configs/debug_cpu.yaml")
    torch.manual_seed(4)
    model = GPT(cfg["model"]).eval()
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    return Generator(model, tok), model, tok, cfg["model"]["d_model"]


def test_inject_none_and_zero_vector_are_bit_identical_to_the_default():
    g, _, _, d = _gen()
    kw = dict(prompt="Once upon a time", n=3, max_new=12, temperature=0.8, top_k=20, seed=7, stop_at_eot=False)
    base = g.generate(**kw)
    none = g.generate(**kw, inject=None)
    zero = g.generate(**kw, inject={L: torch.zeros(d)})
    assert none.new_ids == base.new_ids and none.texts == base.texts
    assert zero.new_ids == base.new_ids                        # adding exact zeros changes no bit


def test_inject_matches_a_cacheless_full_forward_greedy_loop():
    """With top_k=1 the engine (prefill + cached decode, inject at every position and every step) must produce the
    tokens of a plain greedy loop that re-runs the whole sequence with the same inject each step."""
    g, model, tok, d = _gen()
    v = torch.randn(d, generator=torch.Generator().manual_seed(1)) * 3.0
    res = g.generate("Once upon a time", n=1, max_new=10, temperature=1.0, top_k=1, seed=0, stop_at_eot=False,
                     inject={L: v})
    ids = torch.tensor([tok.encode("Once upon a time")])
    for _ in range(10):
        with torch.no_grad():
            nxt = model(ids, inject={L: v}).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    assert res.new_ids[0] == ids[0, -10:].tolist()
    plain = g.generate("Once upon a time", n=1, max_new=10, temperature=1.0, top_k=1, seed=0, stop_at_eot=False)
    assert plain.new_ids != res.new_ids                         # the injection really changed the output


def test_inject_callable_is_forwarded():
    g, _, _, d = _gen()
    calls = []
    def hook(x):
        calls.append(x.shape[1])
        return torch.zeros_like(x)
    g.generate("Once upon a time", n=1, max_new=4, top_k=1, stop_at_eot=False, inject={L: hook})
    assert calls[0] > 1 and calls[1:] == [1, 1, 1]               # one prefill call, then one per decode step
