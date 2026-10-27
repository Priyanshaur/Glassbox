"""Generation engine: static KV cache, temperature / top-k / top-p, streaming.

Generator.generate(prompt, n=...) samples n completions of ONE prompt as a batch
(identical rows, so no padding). Variable-length prompts in one batch are not
supported (documented stretch goal).

Parity with train.sample.sample_completions
-------------------------------------------
For the same seed the engine produces the same tokens as sample_completions when
`replay_last_prompt_token=True`. That flag reproduces a quirk of the old sampler:
after the prefill it feeds the LAST PROMPT TOKEN AGAIN as the first decode step
(`model(ids[:, -1:], past_kvs=past, pos_offset=P-1)`), so that token's K/V sit in
the cache twice and the first next-token distribution differs slightly from the
model's true one (measured on the SFT model: total variation 0.047 on a template
prompt). The engine's default (`replay_last_prompt_token=False`) takes the first
token's logits straight from the prefill, which is the correct distribution.

Other parity details kept on purpose: the sampler's op order (divide by
temperature, top-k mask via `logits < kth`, softmax, torch.multinomial with a
device Generator seeded once), and rows that already hit EOT keep being sampled
until every row is finished (so the RNG stream stays aligned across rows).
"""
import time
from dataclasses import dataclass, field

import torch

from inference.kv_cache import StaticKVCache
from tokenizer.bpe import EOT, EOT_ID


@dataclass
class GenerationResult:
    texts: list                  # completion text per row, cut at the first EOT
    eot_hit: list                # per row: did it sample EOT within the steps run
    new_ids: list                # per row: every sampled token id (raw, incl. after EOT)
    n_prompt: int                # prompt length in tokens
    n_steps: int                 # decode steps actually run
    meta: dict = field(default_factory=dict)


class Generator:
    def __init__(self, model, tok, device=None):
        self.model, self.tok = model, tok
        self.device = device or next(model.parameters()).device
        self.ctx = model.cfg["context_len"]

    # ------------------------------------------------------------------ sampling
    @staticmethod
    def _sample(logits, gen, temperature, top_k, top_p):
        """logits (B, V) -> (B, 1) token ids. Same op order as sample_completions."""
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k < logits.shape[-1]:
            thr, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < thr[:, -1:], float("-inf"))
        if top_p is not None and top_p < 1.0:
            srt, idx = torch.sort(logits, dim=-1, descending=True)
            cum = torch.softmax(srt, dim=-1).cumsum(dim=-1)
            drop = (cum - torch.softmax(srt, dim=-1)) > top_p   # keep the token that crosses p
            srt = srt.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)
        return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)

    # ------------------------------------------------------------------ core loop
    @torch.no_grad()
    def _iter(self, prompt, n, max_new, temperature, top_k, top_p, seed,
              stop_at_eot, replay_last_prompt_token):
        """Yield (step, nxt (n,1), finished (n,)) after each sampled token."""
        was_training = self.model.training
        self.model.eval()
        try:
            prompt_ids = self.tok.encode(prompt)
            P = len(prompt_ids)
            assert P + max_new <= self.ctx, "prompt + max_new exceeds context"
            gen = torch.Generator(device=self.device).manual_seed(seed)
            base = torch.tensor([prompt_ids] * n, dtype=torch.long, device=self.device)
            cache = StaticKVCache.from_model(self.model, n, P + max_new)
            out = self.model(base, past_kvs=cache.layers, pos_offset=0)
            logits = out.logits[:, -1, :]
            ids = base
            finished = torch.zeros(n, dtype=torch.bool, device=self.device)
            for step in range(max_new):
                if replay_last_prompt_token and step == 0:
                    # old-sampler quirk: re-feed the last prompt token (cache grows by one)
                    s = self.model(ids[:, -1:], past_kvs=cache.layers, pos_offset=P - 1)
                    logits = s.logits[:, -1, :]
                elif step > 0:
                    s = self.model(ids[:, -1:], past_kvs=cache.layers,
                                   pos_offset=ids.shape[1] - 1)
                    logits = s.logits[:, -1, :]
                nxt = self._sample(logits, gen, temperature, top_k, top_p)
                ids = torch.cat([ids, nxt], dim=1)
                finished |= (nxt[:, 0] == EOT_ID)
                yield step, nxt, finished
                if stop_at_eot and bool(finished.all()):
                    break
        finally:
            self.model.train(was_training)

    # ------------------------------------------------------------------ public API
    def generate(self, prompt, n=1, max_new=64, temperature=0.8, top_k=50, top_p=None,
                 seed=0, stop_at_eot=True, replay_last_prompt_token=False, should_stop=None):
        """n completions of one prompt. stop_at_eot=False always runs exactly
        max_new steps (fixed-length benchmarking). `should_stop` (a callable) is
        checked after every sampled token; when it returns True generation ends
        early (the server uses it for timeouts and cancellation) and
        `meta["stopped"]` is True. `meta` also carries `ttft_s` (generation start to
        the first sampled token: tokenization, cache allocation, prefill, first
        sample) and `decode_s` (first token to last token)."""
        cols, finished, steps, stopped = [], None, 0, False
        t0 = time.perf_counter()
        t_first = None
        for step, nxt, fin in self._iter(prompt, n, max_new, temperature, top_k, top_p, seed,
                                         stop_at_eot, replay_last_prompt_token):
            if t_first is None:
                t_first = time.perf_counter()
            cols.append(nxt)
            finished, steps = fin, step + 1
            if should_stop is not None and should_stop():
                stopped = True
                break
        t1 = time.perf_counter()
        new = torch.cat(cols, dim=1) if cols else torch.empty((n, 0), dtype=torch.long)
        rows = new.tolist()
        texts = [self.tok.decode(r).split(EOT)[0] for r in rows]
        return GenerationResult(texts=texts,
                                eot_hit=[bool(f) for f in finished.tolist()] if cols else [False] * n,
                                new_ids=rows, n_prompt=len(self.tok.encode(prompt)),
                                n_steps=steps,
                                meta={"ttft_s": (t_first - t0) if t_first is not None else None,
                                      "decode_s": (t1 - t_first) if t_first is not None else 0.0,
                                      "stopped": stopped})

    def stream(self, prompt, max_new=64, temperature=0.8, top_k=50, top_p=None, seed=0,
               should_stop=None, info=None):
        """Yield text pieces as they are generated (one sequence). Stops at EOT;
        a piece is held back while it ends in an incomplete UTF-8 character.
        `should_stop` is checked after every token (early end, `info["stopped"]`).
        `info`, if given, is filled in: n_new (tokens sampled, EOT excluded), eot,
        stopped, ttft_s."""
        info = info if info is not None else {}
        info.update(n_new=0, eot=False, stopped=False, ttft_s=None)
        ids, sent = [], ""
        t0 = time.perf_counter()
        for _, nxt, fin in self._iter(prompt, 1, max_new, temperature, top_k, top_p, seed,
                                      True, False):
            if info["ttft_s"] is None:
                info["ttft_s"] = time.perf_counter() - t0
            tok_id = int(nxt[0, 0])
            if tok_id == EOT_ID:
                info["eot"] = True
                break
            ids.append(tok_id)
            info["n_new"] = len(ids)
            text = self.tok.decode(ids)
            if not text.endswith("�") and len(text) > len(sent):
                yield text[len(sent):]
                sent = text
            if should_stop is not None and should_stop():
                info["stopped"] = True
                break
        final = self.tok.decode(ids)
        if len(final) > len(sent):
            yield final[len(sent):]
