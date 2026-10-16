"""Strip outputs and execution metadata from Jupyter notebooks before committing.

Usage:
    python scripts/strip_notebooks.py path/to/a.ipynb [more.ipynb ...]
    python scripts/strip_notebooks.py --check path/to/a.ipynb   # exit 1 if any notebook still has outputs

Why: notebook outputs can carry competition data (well IDs, logs, label values,
predictions), they bloat the repo, and they make every diff unreadable. The
numbers worth keeping from a run go into the stage README or docs/, not into
cell outputs.

Uses only the standard library, so it runs anywhere (Colab, Kaggle, laptop).
"""
import json
import sys
from pathlib import Path

# notebook-level metadata keys that are session noise, not content
DROP_NB_META = ("widgets", "accelerator", "colab", "kaggle", "language_info")
# cell-level metadata keys that are session noise
DROP_CELL_META = ("id", "outputId", "colab", "executionInfo", "execution", "papermill", "trusted")


def strip(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        md = cell.get("metadata", {})
        for k in DROP_CELL_META:
            md.pop(k, None)
    for k in DROP_NB_META:
        nb.get("metadata", {}).pop(k, None)
    return nb


def has_outputs(nb: dict) -> bool:
    return any(c.get("outputs") or c.get("execution_count") is not None
               for c in nb.get("cells", []) if c.get("cell_type") == "code")


def main(argv):
    check = "--check" in argv
    paths = [Path(a) for a in argv if a != "--check"]
    dirty = []
    for p in paths:
        nb = json.loads(p.read_text(encoding="utf-8"))
        if check:
            if has_outputs(nb):
                dirty.append(p)
            continue
        p.write_text(json.dumps(strip(nb), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"stripped {p}")
    if dirty:
        print("notebooks with outputs:\n  " + "\n  ".join(map(str, dirty)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
