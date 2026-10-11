# Windows note: `make` needs Git Bash/WSL. Otherwise run the commands directly (see README).
.PHONY: install test clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['__pycache__', '.pytest_cache']]; [shutil.rmtree(str(p), ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
