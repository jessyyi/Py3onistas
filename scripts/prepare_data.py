"""Build the cleaned job table from the raw Kaggle download.

Usage (from the repository root):
    python scripts/prepare_data.py --input data/raw --output data/processed
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw", help="folder with the Kaggle CSV files")
    parser.add_argument("--output", default="data/processed", help="folder for the cleaned files")
    args = parser.parse_args()

    clean, log, audit = run_pipeline(args.input, args.output)

    print(json.dumps(log, indent=2, default=str))
    print()
    print(audit["salary"])
    print()
    print(audit["function_coverage"].head(10))


if __name__ == "__main__":
    main()
