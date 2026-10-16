"""Build the folder to upload as the Kaggle dataset that carries this repo's code.

    python scripts/make_kaggle_src.py [out_dir]     # default: dist/rogii-wellbore-src

Copies ``src/rogii_wellbore`` and ``configs/`` and writes a ``dataset-metadata.json``; then
``kaggle datasets create -p dist/rogii-wellbore-src`` (or ``version``) uploads it. The submit
driver finds the package anywhere under /kaggle/input.
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(out=None):
    out = Path(out or REPO / 'dist' / 'rogii-wellbore-src')
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(REPO / 'src' / 'rogii_wellbore', out / 'src' / 'rogii_wellbore',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(REPO / 'configs', out / 'configs')
    (out / 'dataset-metadata.json').write_text(json.dumps(
        {'title': 'rogii-wellbore-src', 'id': 'tuckerarrants/rogii-wellbore-src', 'licenses': [{'name': 'MIT'}]}, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
