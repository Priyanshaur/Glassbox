"""Residual-stream activations for SAE training and evaluation.

Activations are the output of block `layer` (after both residual adds), as returned by
`GPT.forward(record_layers=[layer])`. Training batches are built the way the SAE
trainer needs them and are a pure function of (seed, buffer epoch):

  fill_buffer(...)      draws `buffer_tokens // ctx` windows from the shard loader
                        (deterministic in (seed, epoch)), runs the model in chunks,
                        and stacks every token's activation (scaled) into one tensor.
  epoch_batches(...)    visits every row of that buffer once, in a permutation fixed by
                        (seed, epoch), in minibatches of `batch_tokens`.

Because both are pure functions of (seed, epoch), training can resume exactly at a
buffer-epoch boundary. Activations are multiplied by `scale = sqrt(d) / mean_norm` so
the average token norm is sqrt(d); the scale is stored with the SAE.
"""
import numpy as np
import torch

from train.data_loader import ShardLoader  # noqa: F401  (re-exported for callers)


@torch.no_grad()
def record_windows(model, input_ids, layer, chunk=32, device=None):
    """(N, T) token ids -> (N, T, d) float32 activations at `layer` (on `device`/CPU)."""
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    outs = []
    for i in range(0, input_ids.shape[0], chunk):
        x = input_ids[i:i + chunk].to(device)
        # stop_after: blocks past `layer` and the output layer are not needed (bit-identical, tested)
        outs.append(model(x, record_layers=[layer], stop_after=layer).recorded[layer].float())
    model.train(was_training)
    return torch.cat(outs, dim=0)


def draw_windows(loader, seed, step, n_windows, ctx):
    """(n_windows, ctx) input ids, deterministic in (seed, step)."""
    return loader.batch_ids(seed, step, n_windows, ctx)[:, :ctx]


def norm_stats(model, loader, layer, seed, n_windows, ctx, chunk=32, device=None):
    """Token-norm statistics of the layer's activations (no scaling applied)."""
    acts = record_windows(model, draw_windows(loader, seed, 0, n_windows, ctx), layer, chunk, device)
    n = acts.norm(dim=-1)                                     # (N, T)
    flat = n.flatten()
    return {"tokens": int(flat.numel()), "d": int(acts.shape[-1]),
            "mean_norm": float(flat.mean()), "std_norm": float(flat.std()),
            "p01": float(flat.quantile(0.01)), "p99": float(flat.quantile(0.99)),
            "max_norm": float(flat.max()),
            "max_over_mean": float(flat.max() / flat.mean()),
            "pos0_mean_norm": float(n[:, 0].mean()), "rest_mean_norm": float(n[:, 1:].mean()),
            "mean_vector_norm": float(acts.reshape(-1, acts.shape[-1]).mean(0).norm())}


def estimate_norm_scale(model, loader, layer, seed, n_windows, ctx, chunk=32, device=None,
                        skip_first=1):
    """scale such that scale * mean_token_norm == sqrt(d), the mean taken over the positions
    the SAE trains on (window positions >= skip_first; position 0 is an outlier sink)."""
    acts = record_windows(model, draw_windows(loader, seed, 0, n_windows, ctx), layer, chunk, device)
    mean_norm = float(acts[:, skip_first:].norm(dim=-1).mean())
    return float(np.sqrt(acts.shape[-1]) / mean_norm)


@torch.no_grad()
def fill_buffer(model, loader, layer, seed, epoch, buffer_tokens, ctx, scale, chunk=32, device=None,
                skip_first=1):
    """(n_windows * (ctx - skip_first), d) scaled activations, a pure function of
    (seed, epoch). The first `skip_first` positions of every window (position 0: a
    ~2x-norm sink-like outlier) are left out of the training buffer."""
    n_windows = max(buffer_tokens // ctx, 1)
    ids = draw_windows(loader, seed, epoch, n_windows, ctx)
    acts = record_windows(model, ids, layer, chunk, device)[:, skip_first:]
    return (acts * scale).reshape(-1, acts.shape[-1])


def buffer_rows(buffer_tokens, ctx, skip_first=1):
    """Number of training rows a buffer holds."""
    return max(buffer_tokens // ctx, 1) * (ctx - skip_first)


def epoch_batches(buffer, batch_tokens, seed, epoch):
    """Yield (batch_tokens, d) minibatches covering a fixed permutation of the buffer
    (a trailing partial batch is dropped)."""
    n = buffer.shape[0]
    perm = torch.from_numpy(np.random.default_rng(np.random.SeedSequence([seed, epoch, 1])).permutation(n))
    for i in range(0, n - batch_tokens + 1, batch_tokens):
        yield buffer[perm[i:i + batch_tokens]]


def eval_activations(model, tok_ids_loader, layer, n_windows, ctx, scale, seed=999_983, chunk=32,
                     device=None):
    """Held-out activations: windows from a *validation* loader, fixed seed (comparable
    across runs). Returns (n_windows * ctx, d) scaled, plus the input ids."""
    ids = draw_windows(tok_ids_loader, seed, 0, n_windows, ctx)
    acts = record_windows(model, ids, layer, chunk, device)
    return (acts * scale).reshape(-1, acts.shape[-1]), ids
