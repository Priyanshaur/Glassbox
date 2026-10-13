"""AdamW param groups: decay on weight matrices only.

Decays: 2-D Linear weights EXCEPT the token embedding (frequent-token rows
would be over-regularized; spec excludes embeddings).
No decay: norms (1-D), embedding, (biases if any ever appear).
"""
def build_param_groups(model, weight_decay):
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and "wte" not in name and "embedding" not in name:
            decay.append(p)
        else:
            nodecay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0}]
