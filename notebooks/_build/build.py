"""Build and execute the project notebooks:  .venv/Scripts/python notebooks/_build/build.py [01 02 ...]"""
import importlib, sys, time
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
NOTEBOOKS = {"01": ("nb01_data", "01_data_pipeline_quality_eda.ipynb"),
             "02": ("nb02_hypotheses", "02_hypothesis_testing.ipynb"),
             "03": ("nb03_forecast", "03_forecast_model_and_decision.ipynb"),
             "04": ("nb04_responsible", "04_explainability_fairness_monitoring.ipynb"),
             "05": ("nb05_twenty", "05_twenty_hypotheses.ipynb")}
for key in (sys.argv[1:] or NOTEBOOKS):
    mod, out = NOTEBOOKS[key]
    nb = nbf.v4.new_notebook()
    nb.cells = importlib.import_module(mod).CELLS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    t = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE.parent)}}).execute()
    nbf.write(nb, HERE.parent / out)
    print(f"{out}: executed in {time.time() - t:.0f}s")
