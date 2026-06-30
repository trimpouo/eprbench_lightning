"""Extract final test R@1/5/10/100 from a Lightning CSV log directory."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", type=Path, help="Path to CSV logger version dir containing metrics.csv")
    args = parser.parse_args()

    metrics_file = args.csv_dir / "metrics.csv"
    if not metrics_file.exists():
        print(f"No metrics.csv found in {args.csv_dir}")
        return

    rows = list(csv.DictReader(open(metrics_file)))
    # Find last row with test metrics
    for row in reversed(rows):
        if row.get("test/R@1") not in (None, "", "0"):
            print(f"test/R@1:  {float(row['test/R@1']):.2f}")
            print(f"test/R@5:  {float(row['test/R@5']):.2f}")
            print(f"test/R@10: {float(row['test/R@10']):.2f}")
            print(f"test/R@100:{float(row['test/R@100']):.2f}")
            return
    print("No test metrics found")


if __name__ == "__main__":
    main()
