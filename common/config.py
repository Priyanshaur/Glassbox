"""Shared YAML config loader. Single source of truth for all phases."""
import copy

import yaml


def load_config(path):
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def with_overrides(config, **kwargs):
    """Return a copy with top-level keys overridden (for CLI flags).

    Example: with_overrides(cfg, seed=0)
    Only top-level keys allowed in Phase 0 to keep it obvious.
    """
    out = copy.deepcopy(config)
    for k, v in kwargs.items():
        if k not in out:
            raise KeyError(f"Unknown top-level config key: {k}")
        out[k] = v
    return out
