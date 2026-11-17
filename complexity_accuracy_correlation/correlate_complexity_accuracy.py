"""
correlate_complexity_accuracy.py

Correlates the syntactic complexity of each construction (type-tag template)
with the per-type accuracy of the trained question-formation models.

Input
-----
A directory of *_per_type.csv files, e.g. results_per_type/ from
jacoblvovski/jail_lexicon_attitude_study. Two layouts are accepted:

    checkpoint,type,acc,n                    (generalization set, question.test)
    checkpoint,test_file,type,n,acc          (question_recursion_depthN.test)

Metrics (all computed on the declarative half of "type", see
dependency_distance_from_tokens.py):

    sentence_length          number of tags (control: most metrics grow with it)
    avg_dependency_distance  dependency_distance_from_tokens.py
    avg_branching_factor     branching_factor_from_tokens.py
    clauses_per_sentence     sentence_counts_from_tokens.py
    verbs_per_sentence       (always equal to clauses in this grammar)
    nps_per_sentence
    verb_noun_ratio

Output
------
    <out_dir>/per_type_complexity_accuracy.csv
        one row per (model, test_set, type): accuracy, n and every metric
    <out_dir>/complexity_accuracy_correlations.csv
        Pearson r and Spearman rho (with two-sided p-values) between each
        metric and accuracy, for these groupings:
          model x test_set   - one model on one test file
          model              - one model, all its test files pooled
          all                - every model and test file pooled
        Correlations are unweighted (one point per type). When a metric
        is constant within a group the correlation is undefined and left
        empty.

Usage
-----
    python3 correlate_complexity_accuracy.py <results_per_type_dir> [out_dir]

Standard library only (no pandas/scipy needed).
"""

import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from branching_factor_from_tokens import average_branching_factor  # noqa: E402
from dependency_distance_from_tokens import (  # noqa: E402
    dependency_distance,
    extract_declarative_tags,
)
from sentence_counts_from_tokens import sentence_counts  # noqa: E402

METRICS = [
    "sentence_length",
    "avg_dependency_distance",
    "avg_branching_factor",
    "clauses_per_sentence",
    "verbs_per_sentence",
    "nps_per_sentence",
    "verb_noun_ratio",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(type_string):
    counts = sentence_counts(type_string)
    return {
        "sentence_length": len(extract_declarative_tags(type_string)),
        "avg_dependency_distance": dependency_distance(type_string)[1],
        "avg_branching_factor": average_branching_factor(type_string),
        "clauses_per_sentence": counts["clauses_per_sentence"],
        "verbs_per_sentence": counts["verbs_per_sentence"],
        "nps_per_sentence": counts["nps_per_sentence"],
        "verb_noun_ratio": counts["verb_noun_ratio"],
    }


# ---------------------------------------------------------------------------
# Statistics (stdlib only)
# ---------------------------------------------------------------------------

def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def corr_pvalue(r, n):
    """Two-sided p-value for a correlation r on n points (t test, df=n-2)."""
    df = n - 2
    if df <= 0:
        return float("nan")
    if abs(r) >= 1.0:
        return 0.0
    t2 = r * r * df / (1.0 - r * r)
    return _betainc(df / 2.0, 0.5, df / (df + t2))


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


def rankdata(v):
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def model_name(checkpoint):
    """checkpoint_300000_seed228_embonly.pth -> seed228_embonly"""
    base = os.path.basename(checkpoint)
    base = re.sub(r"\.pth$", "", base)
    return re.sub(r"^checkpoint_\d+_", "", base)


def load_results(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                test_file = r.get("test_file") or "question.test"
                rows.append({
                    "source_file": os.path.basename(path),
                    "model": model_name(r["checkpoint"]),
                    "checkpoint": os.path.basename(r["checkpoint"]),
                    "test_set": re.sub(r"\.test$", "", os.path.basename(test_file)),
                    "type": r["type"].strip(),
                    "declarative": " ".join(extract_declarative_tags(r["type"])),
                    "n": int(r["n"]),
                    "acc": float(r["acc"]),
                })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def correlation_rows(group_kind, group_name, rows):
    out = []
    acc = [r["acc"] for r in rows]
    for m in METRICS:
        x = [r[m] for r in rows]
        pr, sr = pearson(x, acc), spearman(x, acc)
        out.append({
            "grouping": group_kind,
            "group": group_name,
            "metric": m,
            "n_types": len(rows),
            "n_distinct_metric_values": len(set(x)),
            "pearson_r": pr,
            "pearson_p": corr_pvalue(pr, len(rows)) if not math.isnan(pr) else float("nan"),
            "spearman_rho": sr,
            "spearman_p": corr_pvalue(sr, len(rows)) if not math.isnan(sr) else float("nan"),
        })
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    results_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)

    rows = load_results(results_dir)
    cache = {}
    for r in rows:
        if r["type"] not in cache:
            cache[r["type"]] = compute_metrics(r["type"])
        r.update(cache[r["type"]])

    per_type_cols = ["model", "checkpoint", "test_set", "source_file", "type",
                     "declarative", "n", "acc"] + METRICS
    per_type_path = os.path.join(out_dir, "per_type_complexity_accuracy.csv")
    with open(per_type_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_type_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) for k in per_type_cols})

    by_model_set = defaultdict(list)
    by_model = defaultdict(list)
    for r in rows:
        by_model_set[(r["model"], r["test_set"])].append(r)
        by_model[r["model"]].append(r)

    corr = []
    for (model, test_set), grp in sorted(by_model_set.items()):
        corr += correlation_rows("model x test_set", f"{model} | {test_set}", grp)
    for model, grp in sorted(by_model.items()):
        corr += correlation_rows("model", model, grp)
    corr += correlation_rows("all", "all models, all test sets", rows)

    corr_path = os.path.join(out_dir, "complexity_accuracy_correlations.csv")
    with open(corr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr[0].keys()))
        w.writeheader()
        for r in corr:
            w.writerow({k: fmt(v) for k, v in r.items()})

    print(f"{len(rows)} rows, {len(cache)} distinct types, {len(by_model)} models")
    print(f"wrote {per_type_path}")
    print(f"wrote {corr_path}")


if __name__ == "__main__":
    main()
