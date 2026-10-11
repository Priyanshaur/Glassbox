# MiniGPT-Stack + Brain Scanner (Phase 0 skeleton)
Repo root is this folder (`Portfolio Project/` acts as `minigpt-stack/` from the spec).

## Setup (Windows, Python 3.13, i3 + 8GB RAM)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `make` is not installed (common on Windows), use the direct commands:
```powershell
pip install -r requirements.txt
python -m pytest tests/ -v
```

With `make` (Git Bash / WSL):
```
make install
make test
```

## Quickstart
```powershell
python -m pytest tests/ -v
python -c "from common.config import load_config; print(load_config('configs/debug_cpu.yaml')['model'])"
```

## Structure
```
configs/        # one yaml per experiment (30M, 85M stub, debug CPU)
common/         # shared config loader (small deviation from spec, see docs/decisions.md)
tokenizer/ model/ data/ train/ inference/ interp/ demo/ bench/
tests/ docs/ notebooks/
```

## Checkpoints / save states
Every training config already carries a `checkpoint:` block (`dir`, `save_every`, `keep_last_n`).
`train/checkpoint.py` (Phase 3) will save model + optimizer + GradScaler + scheduler + RNG state + data position,
so a dead Kaggle session resumes with a continuous loss curve. See `docs/decisions.md`.
