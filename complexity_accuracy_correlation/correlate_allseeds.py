"""
correlate_allseeds.py

Per-seed correlation between each construction's syntactic complexity and the
model's accuracy on that construction, for a single combined results file
such as results_allseeds_new_per_type.csv
(jacoblvovski/jail_lexicon_attitude_study), layout:

    checkpoint,type,acc,n

The seed is taken from the checkpoint name
(checkpoint_300000_seed228_embonly.pth -> 228). Metrics are the ones in
correlate_complexity_accuracy.py, computed on the declarative half of "type".

Output (in <out_dir>)
------
    <prefix>_per_type_metrics.csv
        one row per (seed, type): n, acc and every metric
    <prefix>_correlations.csv
        one row per (seed, metric): Pearson r and Spearman rho with
        two-sided p-values, unweighted (one point per construction)
    <prefix>_correlations.xlsx
        sheet "correlations": same as the correlations CSV
        one sheet per metric (sentence_length excluded, it is the control):
        per seed, Pearson r of accuracy with the metric, Pearson r with
        sentence_length, and diff = r(sentence_length) - r(metric).
        Same for Spearman in the columns to the right.

Usage
-----
    python3 correlate_allseeds.py <results_csv> [out_dir] [prefix]

Needs openpyxl for the xlsx file (pip install openpyxl).
"""

import csv
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correlate_complexity_accuracy import (  # noqa: E402
    METRICS,
    compute_metrics,
    corr_pvalue,
    pearson,
    spearman,
)
from dependency_distance_from_tokens import extract_declarative_tags  # noqa: E402

CONTROL = "sentence_length"


def seed_of(checkpoint):
    m = re.search(r"seed(\d+)", checkpoint)
    if not m:
        raise ValueError(f"no seed in checkpoint name: {checkpoint}")
    return int(m.group(1))


def nan_to_none(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    in_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    prefix = sys.argv[3] if len(sys.argv) > 3 else "allseeds_new"
    os.makedirs(out_dir, exist_ok=True)

    cache = {}
    by_seed = defaultdict(list)
    with open(in_path, newline="") as f:
        for r in csv.DictReader(f):
            t = r["type"].strip()
            if t not in cache:
                cache[t] = compute_metrics(t)
            row = {
                "seed": seed_of(r["checkpoint"]),
                "checkpoint": os.path.basename(r["checkpoint"]),
                "type": t,
                "declarative": " ".join(extract_declarative_tags(t)),
                "n": int(r["n"]),
                "acc": float(r["acc"]),
                **cache[t],
            }
            by_seed[row["seed"]].append(row)
    seeds = sorted(by_seed)

    # per-type metrics
    cols = ["seed", "checkpoint", "type", "declarative", "n", "acc"] + METRICS
    per_type_path = os.path.join(out_dir, f"{prefix}_per_type_metrics.csv")
    with open(per_type_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in seeds:
            for r in by_seed[s]:
                w.writerow({k: fmt(r[k]) for k in cols})

    # correlations
    corr = []
    lookup = {}
    for s in seeds:
        rows = by_seed[s]
        acc = [r["acc"] for r in rows]
        for m in METRICS:
            x = [r[m] for r in rows]
            pr, sr = pearson(x, acc), spearman(x, acc)
            rec = {
                "seed": s,
                "checkpoint": rows[0]["checkpoint"],
                "metric": m,
                "n_types": len(rows),
                "n_distinct_metric_values": len(set(x)),
                "mean_acc": sum(acc) / len(acc),
                "pearson_r": pr,
                "pearson_p": corr_pvalue(pr, len(rows)) if not math.isnan(pr) else float("nan"),
                "spearman_rho": sr,
                "spearman_p": corr_pvalue(sr, len(rows)) if not math.isnan(sr) else float("nan"),
            }
            corr.append(rec)
            lookup[(s, m)] = rec

    corr_cols = list(corr[0].keys())
    corr_path = os.path.join(out_dir, f"{prefix}_correlations.csv")
    with open(corr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=corr_cols)
        w.writeheader()
        for r in corr:
            w.writerow({k: fmt(v) for k, v in r.items()})

    # xlsx
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "correlations"
    ws.append(corr_cols)
    for r in corr:
        ws.append([nan_to_none(r[k]) for k in corr_cols])

    def diff(a, b):
        return None if (math.isnan(a) or math.isnan(b)) else a - b

    for m in METRICS:
        if m == CONTROL:
            continue
        sh = wb.create_sheet(m[:31])
        sh.append([
            "seed", "mean_acc",
            f"pearson_r({m})", f"pearson_r({CONTROL})",
            f"pearson_diff ({CONTROL} - {m})",
            f"spearman_rho({m})", f"spearman_rho({CONTROL})",
            f"spearman_diff ({CONTROL} - {m})",
        ])
        for s in seeds:
            rm, rl = lookup[(s, m)], lookup[(s, CONTROL)]
            sh.append([
                s, rm["mean_acc"],
                nan_to_none(rm["pearson_r"]), nan_to_none(rl["pearson_r"]),
                diff(rl["pearson_r"], rm["pearson_r"]),
                nan_to_none(rm["spearman_rho"]), nan_to_none(rl["spearman_rho"]),
                diff(rl["spearman_rho"], rm["spearman_rho"]),
            ])
        if lookup[(seeds[0], m)]["n_distinct_metric_values"] == 1:
            sh.append([])
            sh.append([f"{m} is constant across all constructions in this "
                       "test set, so its correlation is undefined (blank)."])

    for sh in wb.worksheets:
        for c in sh[1]:
            c.font = Font(bold=True)
        sh.freeze_panes = "A2"
        for i, col in enumerate(sh.iter_cols(min_row=1, max_row=1), 1):
            sh.column_dimensions[get_column_letter(i)].width = max(12, len(str(col[0].value)) + 2)
        for row in sh.iter_rows(min_row=2):
            for c in row:
                if isinstance(c.value, float):
                    c.number_format = "0.0000"

    xlsx_path = os.path.join(out_dir, f"{prefix}_correlations.xlsx")
    wb.save(xlsx_path)

    print(f"{sum(len(v) for v in by_seed.values())} rows, {len(cache)} types, {len(seeds)} seeds")
    for p in (per_type_path, corr_path, xlsx_path):
        print("wrote", p)


if __name__ == "__main__":
    main()
