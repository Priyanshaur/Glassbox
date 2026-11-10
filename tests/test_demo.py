"""Local steering demo: offers only passed features, same seed for both stories, rejects others."""
import json

import torch
from starlette.testclient import TestClient

from common.config import load_config
from demo import app as D
from inference.generate import Generator
from interp.sae import TopKSAE
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer


def _client():
    cfg = load_config("configs/debug_cpu.yaml")
    torch.manual_seed(0)
    model = GPT(cfg["model"]).eval()
    gen = Generator(model, BPETokenizer.load(cfg["data"]["tokenizer_path"]))
    sae = TopKSAE(cfg["model"]["d_model"], 4096, 4, seed=1)
    verdict = json.load(open(D.VERDICT, encoding="utf-8"))
    return TestClient(D.make_app(gen, sae, 0.5, verdict))


def test_only_passed_features_are_offered_and_the_note_is_there():
    c = _client()
    j = c.get("/api/features").json()
    assert sorted(f["feature"] for f in j["offered"]) == [1131, 3651]
    assert sorted(j["not_offered"]) == [1319, 1672, 3990]
    assert "did not show a usable effect" in j["note"] and "NOT established" in j["claim"]
    page = c.get("/").text
    assert "Read this first" in page and "/api/features" in page
    assert {f["validated_scale"] for f in j["offered"]} == {2, 4}


def test_rejected_feature_and_bad_input_are_refused():
    c = _client()
    ok = dict(feature=1131, scale=1, words="dog, park, happy", max_new=4)
    assert c.post("/api/generate", json={**ok, "feature": 1672}).status_code == 422       # clothing failed: not offered
    assert c.post("/api/generate", json={**ok, "scale": 99}).status_code == 422
    assert c.post("/api/generate", json={**ok, "words": "only, two"}).status_code == 422
    assert c.post("/api/generate", json={"feature": "x"}).status_code == 422


def test_scale_zero_gives_identical_stories_and_scale_changes_the_steered_one(monkeypatch):
    monkeypatch.setattr(D.ST, "LAYER", 1)          # the debug model has fewer than 5 blocks
    c = _client()
    z = c.post("/api/generate", json=dict(feature=1131, scale=0, words="dog, park, happy", max_new=12, seed=3)).json()
    assert z["baseline"] == z["steered"] and not z["beyond_validated_range"]
    s = c.post("/api/generate", json=dict(feature=1131, scale=3, words="dog, park, happy", max_new=12, seed=3)).json()
    assert s["baseline"] == z["baseline"] and s["beyond_validated_range"]                # same seed: same unsteered story
    assert s["steered"] != s["baseline"]
