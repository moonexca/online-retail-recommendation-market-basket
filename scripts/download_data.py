"""Download the official UCI Online Retail II workbook."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also create online_retail_II.csv for the historical notebook.",
    )
    args = parser.parse_args()

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "online-retail-ii.zip"
    workbook = destination / "online_retail_II.xlsx"

    if not archive.exists():
        print(f"Downloading {DATA_URL}")
        urlretrieve(DATA_URL, archive)
    if not workbook.exists():
        with ZipFile(archive) as compressed:
            source_name = compressed.namelist()[0]
            with compressed.open(source_name) as source, workbook.open("wb") as target:
                target.write(source.read())
    print(f"Workbook ready: {workbook}")

    if args.write_csv:
        sheets = pd.read_excel(workbook, sheet_name=None)
        csv_path = Path("online_retail_II.csv")
        pd.concat(sheets.values(), ignore_index=True).to_csv(csv_path, index=False)
        print(f"Historical-notebook CSV ready: {csv_path}")


if __name__ == "__main__":
    main()
