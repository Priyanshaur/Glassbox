"""Local steering demo (Phase 6). Only the features that PASSED the pre-registered steering test are offered.

    python -m demo.app                    # http://127.0.0.1:8000   (SFT model + SAE decoder row, CPU, one request at a time)

For a prompt (three words for the story template) it generates the same story twice with the same sampling seed: once
unsteered, once with `scale * a_f * d_f / act_scale` added to the layer-4 residual at every position (prefill and every
decode step), the exact intervention of the steering experiment (analysis/steering.py). The page shows a permanent note that the
other three tested features did not show a usable effect, and the measured numbers behind the two offered ones."""
import argparse
import json
import os
import re
import sys
import threading

import torch  # noqa: F401
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

sys.path.insert(0, os.getcwd())
from analysis import steering as ST  # noqa: E402
from train.instruct import TMPL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VERDICT = os.path.join(os.path.dirname(HERE), "docs", "phase6_steering_verdict.json")
NAMES = {1131: "The night sky (stars, moon)", 3651: "Household furniture (vase, bookcase, couch)"}
MAX_NEW_CAP = 200
OVERSHOOT = 1.5        # slider may go to 1.5x the validated scale; beyond the validated range the text degrades


def offered_features(verdict):
    """({feature: info} for the features whose steering test passed, [features that did not])."""
    ok, no = {}, []
    for v in verdict["features"]:
        f = v["feature"]
        if v["success"]:
            s = v["working_scales"][0]
            t = v["scales"][str(s)]
            ok[f] = {"feature": f, "name": NAMES[f], "validated_scale": s, "max_scale": s * OVERSHOOT,
                     "concept_words": ST.FEATURES[f][1], "a_f": ST.FEATURES[f][0],
                     "measured": {"concept_rate_steered": t["concept_feature"], "concept_rate_random_control": t["concept_random"],
                                  "gain": t["diff"], "gain_ci95": t["ci"], "baseline_concept_rate": v["baseline"]["concept"],
                                  "end_of_story_rate": t["eot"], "perplexity_ratio": t["ppl_ratio"]}}
        else:
            no.append(f)
    return ok, no


def concept_words_in(text, words):
    low = set(re.findall(r"[a-z']+", text.lower()))
    return sorted(w for w in words if w in low)


def make_app(generator, sae, act_scale, verdict):
    offered, rejected = offered_features(verdict)
    lock = threading.Lock()

    async def index(request):
        return FileResponse(os.path.join(HERE, "index.html"))

    async def features(request):
        return JSONResponse({"offered": list(offered.values()), "not_offered": rejected,
                             "note": "Only 2 of the 5 pre-registered features passed the steering test. The other three (clothing, "
                                     "foods and animals, an animal as a sentence subject) did not show a usable effect and are not offered.",
                             "claim": "steering effect NOT established (2 of 5 passed, 3 were needed)"})

    async def generate(request):
        try:
            body = await request.json()
            f = int(body["feature"])
            scale = float(body["scale"])
            words = [w.strip() for w in str(body["words"]).split(",") if w.strip()]
            seed = int(body.get("seed", 0))
            max_new = min(int(body.get("max_new", 160)), MAX_NEW_CAP)
        except (KeyError, ValueError, TypeError):
            return JSONResponse({"error": "bad request"}, status_code=422)
        if f not in offered:
            return JSONResponse({"error": "that feature did not pass the steering test and is not offered"}, status_code=422)
        info = offered[f]
        if not 0 <= scale <= info["max_scale"] or len(words) != 3:
            return JSONResponse({"error": f"scale must be in [0, {info['max_scale']}] and exactly 3 words are needed"},
                                status_code=422)
        prompt = TMPL.format(words=", ".join(words))
        inj = None if scale == 0 else {ST.LAYER: ST.feature_vector(sae, act_scale, f, info["a_f"], scale)}
        kw = dict(n=1, max_new=max_new, temperature=0.8, top_k=40, seed=seed)
        with lock:
            base = generator.generate(prompt, **kw)
            steered = generator.generate(prompt, **kw, inject=inj)
        return JSONResponse({
            "prompt": prompt, "scale": scale, "beyond_validated_range": scale > info["validated_scale"],
            "baseline": base.texts[0], "steered": steered.texts[0],
            "baseline_concept_words": concept_words_in(base.texts[0], info["concept_words"]),
            "steered_concept_words": concept_words_in(steered.texts[0], info["concept_words"])})

    return Starlette(routes=[Route("/", index), Route("/api/features", features),
                             Route("/api/generate", generate, methods=["POST"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_30m.yaml")
    ap.add_argument("--weights", default="checkpoints/glassbox-sft-final/sft_final_best.pt")
    ap.add_argument("--sae", default="checkpoints/glassbox-sae-l4/sae_latest.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    import uvicorn
    from analysis.a5_transfer import load_model
    from common.config import load_config
    from inference.generate import Generator
    from interp.sae import load_sae
    from tokenizer.bpe import BPETokenizer
    cfg = load_config(a.config)
    sae, meta = load_sae(a.sae)
    for p in sae.parameters():
        p.requires_grad_(False)
    gen = Generator(load_model(cfg, a.weights), BPETokenizer.load(cfg["data"]["tokenizer_path"]))
    verdict = json.load(open(VERDICT, encoding="utf-8"))
    print(f"demo on http://{a.host}:{a.port}  (loopback only; CPU generation takes several seconds)")
    uvicorn.run(make_app(gen, sae, meta["act_scale"], verdict), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
