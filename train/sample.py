"""Minimal temperature/top-k sampler for the training time-lapse.

NOT the Phase 5 inference engine: no KV cache, no batching, no streaming.
Writes prompt continuations to a file every sample_every steps.

Own torch.Generator (never the training RNG), eval mode + no_grad, and the
model's train mode is restored afterwards (amendment 5).
"""
import torch

from tokenizer.bpe import EOT, EOT_ID


def prompt_ids(tok, prompt):
    """Prompt encoding shared by SFT example building and generation.

    SFT concatenates encode(prompt) + encode(response); the sampler feeds
    encode(prompt) as the seed. Same ids both sides (tested in test_phase4).
    """
    return tok.encode(prompt)


@torch.no_grad()
def sample_continuations(model, tok, prompts, max_new=64, temperature=0.8,
                         top_k=50, seed=0, device=None):
    was_training = model.training
    model.eval()
    # Resolve the device BEFORE building the generator: torch.multinomial
    # rejects a CPU generator on a CUDA tensor ("Expected a 'cuda' device type
    # for generator but found 'cpu'"). That crashed the main run at step 2000.
    device = device or next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    outs = []
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], dtype=torch.long, device=device)
        for _ in range(max_new):
            logits = model(ids).logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k < logits.shape[-1]:
                thr, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < thr[:, -1:], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
            ids = torch.cat([ids, nxt], dim=1)
        outs.append(tok.decode(ids[0].tolist()))
    model.train(was_training)
    return outs


@torch.no_grad()
def sample_completions(model, tok, prompt, n=4, max_new=64, temperature=0.8,
                       top_k=50, seed=0, device=None, return_flags=False):
    """Batched KV-cache sampler: n completions of ONE prompt, one batch.

    Same-length rows (identical prompt) so no padding is needed. Full forward
    on the prompt, then single-token steps with past_kvs (exercises the Phase
    2 cache path). Own Generator, eval/no_grad, restores train mode.
    Returns RESPONSE texts (ASCII prompt prefix stripped, cut at EOT).
    With return_flags=True returns (texts, flags) where each flags row
    carries {'eot_hit': bool} — the flag is collected from the finished
    mask that already existed in the loop, so RNG consumption is identical
    either way (parity-tested).
    """
    was_training = model.training
    model.eval()
    # Device first, then the generator on that device (see sample_continuations).
    device = device or next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    ctx = model.cfg["context_len"]
    base = torch.tensor([prompt_ids(tok, prompt)] * n, dtype=torch.long,
                        device=device)
    assert base.shape[1] + max_new <= ctx, "prompt + max_new exceeds context"
    out = model(base)
    past, ids = out.new_kvs, base
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(max_new):
        step = model(ids[:, -1:], past_kvs=past, pos_offset=ids.shape[1] - 1)
        past = step.new_kvs
        logits = step.logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None and top_k < logits.shape[-1]:
            thr, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < thr[:, -1:], float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)
        ids = torch.cat([ids, nxt], dim=1)
        finished |= (nxt[:, 0] == EOT_ID)
        if bool(finished.all()):
            break
    model.train(was_training)
    outs = []
    for row in ids.tolist():
        # Prompt is pure ASCII -> strips cleanly off the decoded front.
        outs.append(tok.decode(row)[len(prompt):].split(EOT)[0])
    if return_flags:
        return outs, [{"eot_hit": bool(f)} for f in finished.tolist()]
    return outs
