"""
Prints raw totals for the three engineered features, reading in chunks
so the 100MB+ file never loads fully into memory.

Reports:
  - Total mismatches   -> count of rows where sender_mismatch == 1
  - Total links         -> sum of link_count across all rows
  - Total urgency score  -> sum of urgency_keyword_density across all rows

Usage:
    python totals.py
"""

import pandas as pd

INPUT_PATH = r"C:\Users\Prometheus\Desktop\Project Exhibition I\Review2\review2 output.csv"
CHUNK_SIZE = 50_000


def main():
    total_rows = 0
    mismatch_count = 0
    total_links = 0
    total_urgency_density = 0.0

    reader = pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)

    for chunk in reader:
        total_rows += len(chunk)
        mismatch_count += int((chunk["sender_mismatch"] == 1).sum())
        total_links += int(chunk["link_count"].sum())
        total_urgency_density += float(chunk["urgency_keyword_density"].sum())

    print(f"Total rows scanned:      {total_rows:,}")
    print(f"Total mismatches:        {mismatch_count:,}")
    print(f"Total links:             {total_links:,}")
    print(f"Total urgency density:   {total_urgency_density:.4f}")


if __name__ == "__main__":
    main()