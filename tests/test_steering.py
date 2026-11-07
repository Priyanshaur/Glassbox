"""Phase 6 / C: the steering harness (vectors, controls, success rules) on synthetic data."""
import math

import numpy as np
import torch

from analysis import steering as T
from interp.sae import TopKSAE


def test_vectors_have_the_stated_norms_and_ablation_removes_the_features_contribution():
    sae = TopKSAE(16, 40, 4, seed=3)
    scale, a_f, f = 0.5, 2.0, 7
    v = T.feature_vector(sae, scale, f, a_f, 4)
    assert torch.allclose(v.norm(), torch.tensor(4 * a_f / scale), rtol=1e-5)          # |s a_f d_f / scale|, d_f unit
    assert torch.allclose(v * scale / (4 * a_f), sae.W_dec[f].detach(), atol=1e-6)     # direction = decoder row
    assert torch.allclose(T.feature_vector(sae, scale, f, a_f, -2), -v / 2)
    r = T.random_vector(float(v.norm()), 101, 16)
    assert torch.allclose(r.norm(), v.norm(), rtol=1e-5) and not torch.allclose(r, v)
    assert torch.equal(r, T.random_vector(float(v.norm()), 101, 16))                  # fixed seed
    x = torch.randn(3, 5, 16)
    acts, _ = sae.encode(x.reshape(-1, 16) * scale)
    delta = T.ablation_hook(sae, scale, f)(x)
    want = -(acts[:, f].reshape(3, 5, 1)) * sae.W_dec[f].detach() / scale
    assert torch.allclose(delta, want, atol=1e-6) and delta.shape == x.shape


def test_conditions_cover_the_preregistered_set():
    sae = TopKSAE(16, 40, 4, seed=3)
    names = [c[0] for c in T.conditions(sae, 0.5, 7, 2.0)]
    assert names[:4] == ["s1", "s2", "s4", "s8"] and "s-2" in names and "ablate" in names
    assert sum(n.startswith("rand_") for n in names) == 12                             # 4 scales x 3 seeds
    assert len(names) == 18


def test_bootstrap_ci_brackets_the_mean_and_excludes_zero_for_a_clear_effect():
    lo, hi = T.boot_ci([0.3] * 40 + [0.2] * 10)
    assert 0.2 < lo <= hi < 0.31
    lo, hi = T.boot_ci(list(np.random.default_rng(0).normal(0, 1, 50)))
    assert lo < 0 < hi


def _cells(effect_by_scale, fluent=True, neg_up=False):
    """Synthetic cells for one feature: concept words appear with probability p; random and baseline p=0.1."""
    rng = np.random.default_rng(0)
    concept = ["apples", "cheese", "carrots"]
    cells = {}

    def rows(p, eot=1.0, nll=2.0):
        out = []
        for idx in range(50):
            texts = [("apples and a story" if rng.random() < p else "a plain story about toys") + " the end" for _ in range(8)]
            out.append({"idx": idx, "keywords": ["toys", "red", "ball"], "texts": texts, "eot": [rng.random() < eot for _ in range(8)],
                        "nll": [nll + 0.01 * k for k in range(8)]})
        return out
    cells[(None, "baseline")] = rows(0.1)
    for s in T.SCALES:
        cells[(1, f"s{s}")] = rows(0.1 + effect_by_scale[s], eot=1.0 if fluent else 0.5, nll=2.0 if fluent else 4.0)
        for r in T.RANDOM_SEEDS:
            cells[(1, f"rand_s{s}_{r}")] = rows(0.1)
    cells[(1, "s-2")] = rows(0.1 + (0.4 if neg_up else 0.0))
    cells[(1, "ablate")] = rows(0.1)
    return cells, concept


def test_a_planted_dose_response_effect_succeeds():
    cells, concept = _cells({1: 0.05, 2: 0.2, 4: 0.4, 8: 0.5})
    v = T.judge_feature(cells, 1, concept)
    assert v["success"] and 2 in v["working_scales"] and 1 not in v["working_scales"]


def test_no_effect_negative_control_rise_or_poor_fluency_all_fail():
    cells, concept = _cells({1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0})
    assert not T.judge_feature(cells, 1, concept)["success"]
    cells, concept = _cells({1: 0.05, 2: 0.2, 4: 0.4, 8: 0.5}, neg_up=True)           # s=-2 also raises the concept
    assert not T.judge_feature(cells, 1, concept)["success"]
    cells, concept = _cells({1: 0.05, 2: 0.2, 4: 0.4, 8: 0.5}, fluent=False)           # text degraded
    assert not T.judge_feature(cells, 1, concept)["success"]


def test_keyword_words_of_the_prompt_are_not_counted_and_matching_is_whole_word():
    assert T.has_concept("She ate apples.", ["apples"], set())
    assert not T.has_concept("She ate pineapples.", ["apples"], set())
    assert not T.has_concept("She ate apples.", ["apples"], {"apples"})


def test_every_injection_vector_lives_on_the_saes_device():
    """Kaggle probe v2 failed with cuda/cpu tensors mixed: the random controls were built on the CPU. 'meta' stands in
    for cuda here, so any tensor created on the wrong device is caught without a GPU."""
    sae = TopKSAE(16, 40, 4, seed=3).to("meta")
    for name, _kind, _s, inj in T.conditions(sae, 0.5, 7, 2.0):
        for v in inj.values():
            if torch.is_tensor(v):
                assert v.device.type == "meta", name
