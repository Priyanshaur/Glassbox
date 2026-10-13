"""Minimal temperature/top-k sampler for the training time-lapse.

NOT the Phase 5 inference engine: no KV cache, no batching, no streaming.
Writes prompt continuations to a file every sample_every steps.

Own torch.Generator (never the training RNG), eval mode + no_grad, and the
model's train mode is restored afterwards (amendment 5).
"""
import torch


@torch.no_grad()
def sample_continuations(model, tok, prompts, max_new=64, temperature=0.8,
                         top_k=50, seed=0, device=None):
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    device = device or next(model.parameters()).device
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
