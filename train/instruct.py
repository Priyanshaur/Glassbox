"""Instruct data without new special tokens: plain-text delimiters only.

Fallback decision (see decisions.md): the instruct-dataset ID is unconfirmed,
so prompts are synthesized deterministically from story text. Three content
words sampled from a story become its constraint; the ORIGINAL story passes
the criterion by construction, model generations may or may not.

Prompt format (prompt and response tokenized SEPARATELY, then concatenated --
never merged across the boundary, exactly what the sampler feeds at
generation time):

    Write a short story using all of these words: w1, w2, w3.

    Story:<response><|endoftext|>
"""
import json
import os
import re

import numpy as np
import torch

from tokenizer.bpe import EOT_ID

TMPL = "Write a short story using all of these words: {words}.\n\nStory:"
SALT = 12345


def content_words(text):
    """Lowercase alpha words len>=4, deduplicated. Deterministic order."""
    return sorted({w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4})


def synth_prompt(text, idx, n_words=3):
    """(prompt, keywords) or None if the story has too few content words."""
    words = content_words(text)
    if len(words) < n_words:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([SALT, idx]))
    kw = sorted(rng.choice(words, size=n_words, replace=False).tolist())
    return TMPL.format(words=", ".join(kw)), kw


def check_criterion(response, keywords):
    """Rule-based pass/fail. Deterministic; flaws documented in decisions.md."""
    missing = [k for k in keywords if k not in response.lower()]
    nwords = len(response.split())
    ok = not missing and 10 <= nwords <= 500
    return {"ok": ok, "missing": missing, "nwords": nwords}


def build_sft_example(tok, prompt, response, context_len):
    """input_ids + labels with -100 over prompt predictions.

    full = P + R + [EOT]; labels[i] predicts full[i+1]. Prompt occupies
    full[0..P-1], so labels[0..P-2] (== labels[:P-1]) are masked; labels[P-1]
    predicts the first response token and is KEPT. Truncates to context_len.
    """
    p = tok.encode(prompt)
    r = tok.encode(response)
    full = p + r + [EOT_ID]
    if len(full) > context_len:
        full = full[:context_len - 1] + [EOT_ID]
    P = min(len(p), len(full) - 1)
    inp = torch.tensor(full[:-1], dtype=torch.long)
    labels = torch.tensor(full[1:], dtype=torch.long)
    labels[:P - 1] = -100
    return inp, labels


def iter_shard_texts(tok, out_dir, split):
    """Yield a split's story texts ONE story at a time (streaming, not a list).

    Reads the shard as a uint16 array (~128 MB per 67M-token shard) and decodes
    story by story, cutting only at EOT ids so a story can never be split
    across a decode call. docs_from_shard used to .tolist() the whole file and
    decode it in one pass, which builds a multi-GB Python list on the real
    corpus; here memory tracks the array, and a caller that stops early (the
    --max_examples cap in train/sft.py) never pays for the rest of the split.
    Same texts, same order, and the trailing story still comes out even if a
    shard has no EOT after it.
    """
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    for rec in meta["shards"]:
        if rec["split"] != split:
            continue
        ids = np.fromfile(os.path.join(out_dir, rec["file"]), dtype=np.uint16)
        prev = -1
        for e in np.flatnonzero(ids == EOT_ID):
            text = tok.decode(ids[prev + 1:e].tolist()).strip()
            prev = int(e)
            if text:
                yield text
        tail = tok.decode(ids[prev + 1:].tolist()).strip()
        if tail:
            yield tail


def docs_from_shard(tok, out_dir, split):
    """Decode shard .bins to story texts (split on EOT). Deterministic order."""
    return list(iter_shard_texts(tok, out_dir, split))


def collate(examples, pad_id):
    """Right-pad (safe: causal positions never attend to trailing pads)."""
    L = max(len(e[0]) for e in examples)
    B = len(examples)
    inp = torch.full((B, L), pad_id, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        inp[i, :len(x)] = x
        lab[i, :len(y)] = y
    return inp, lab
