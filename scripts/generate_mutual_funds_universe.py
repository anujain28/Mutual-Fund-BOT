"""
scripts/generate_mutual_funds_universe.py

Utility script (optional) to build `data/mutual_funds_universe.csv`
from any raw MF master you have (AMFI, Groww export, etc.).

Right now this is just a stub; customise as you like.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_FILE = DATA_DIR / "mutual_funds_universe.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Replace this with real source (AMFI / manual list / etc.)
    # For now we create a tiny dummy file as an example.
    df = pd.DataFrame(
        [
            {
                "SchemeCode": 1,
                "SchemeName": "Sample Nifty 50 Index Fund",
                "Category": "Index",
                "AMC": "Demo AMC",
            },
            {
                "SchemeCode": 2,
                "SchemeName": "Sample Flexi Cap Fund",
                "Category": "Flexi Cap",
                "AMC": "Demo AMC",
            },
        ]
    )

    df.to_csv(OUT_FILE, index=False)
    print(f"Written {OUT_FILE} with {len(df)} rows")


if __name__ == "__main__":
    main()

