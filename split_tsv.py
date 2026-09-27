"""
split_tsv.py

Splits question_train_with_trees.tsv into N roughly equal parts by row
count (default N=3), each written as an independently valid TSV file
(its own header row, original "id" values preserved -- not renumbered --
so every row stays traceable back to its line number in the original
question.train).

Row-count splitting rather than byte splitting: a TSV row here holds a full
JSON tree object; splitting on raw bytes could cut a row (and its JSON)
in half. Splitting on whole rows can't do that.

Usage:
    python3 split_tsv.py [--parts 3] [--input question_train_with_trees.tsv]
"""

import argparse
import csv
import os

DEFAULT_INPUT = "question_train_with_trees.tsv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--parts", type=int, default=3)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(here, args.input) if not os.path.isabs(args.input) else args.input

    with open(in_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)

    n = len(rows)
    base, extra = divmod(n, args.parts)
    sizes = [base + 1 if i < extra else base for i in range(args.parts)]  # spread remainder over first parts

    stem, ext = os.path.splitext(args.input)
    start = 0
    for i, size in enumerate(sizes, start=1):
        chunk = rows[start:start + size]
        start += size
        out_name = f"{stem}_part{i}{ext}"
        out_path = os.path.join(here, out_name)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(chunk)
        n_bytes = os.path.getsize(out_path)
        first_id = chunk[0][0] if chunk else "-"
        last_id = chunk[-1][0] if chunk else "-"
        print(f"{out_name}: {len(chunk)} rows (id {first_id}-{last_id}), "
              f"{n_bytes} bytes ({n_bytes / (1024*1024):.1f} MiB)")


if __name__ == "__main__":
    main()
