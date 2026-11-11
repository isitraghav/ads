"""Step 1: build the single source of truth from the raw sources.
Usage:  .venv/Scripts/python scripts/01_build_ssot.py   (about 3-4 minutes, ~5 GB RAM)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abd_demand import etl

t = time.time()
etl.run()
print(f"done in {time.time() - t:.0f}s")
