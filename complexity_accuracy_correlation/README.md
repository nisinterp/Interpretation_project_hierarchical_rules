# Complexity metrics vs. per-type accuracy

Correlates the syntactic complexity of each question-formation construction
(the `type` tag template) with model accuracy on that construction, using the
files in `data/results_per_type/` (copied from the `results_per_type/` folder
of jacoblvovski/jail_lexicon_attitude_study). Those files cover 6 models,
scored on `question.test` and on the depth-2 to depth-5 center-embedding
test sets in `../qf_data/test/`.

## Run

```bash
python3 correlate_complexity_accuracy.py data/results_per_type results
```

Uses only the Python standard library.

## Files

| File | What it does |
|---|---|
| `dependency_distance_from_tokens.py` | Parses a tag string into a dependency tree and computes the average dependency distance (Liu 2008) |
| `branching_factor_from_tokens.py` | Average number of dependents per head, on the same tree |
| `sentence_counts_from_tokens.py` | Clauses, verbs and NPs per sentence, and the verb/noun ratio |
| `add_avg_dependency_distance.py` | Older single-metric script: adds an `avg_dependency_distance` column to one `results_per_type.csv`-style file. It does not handle the recursion-depth files, so use the script below instead |
| `correlate_complexity_accuracy.py` | Loads every `*_per_type.csv`, computes all metrics for each type and writes the two result CSVs |
| `results/per_type_complexity_accuracy.csv` | One row per (model, test set, type): `acc`, `n` and every metric |
| `results/complexity_accuracy_correlations.csv` | Pearson r and Spearman rho with two-sided p-values for each metric against accuracy, grouped by model × test set, by model and pooled across everything |

All metrics are computed on the declarative half of `type`, the part before
`quest`. `sentence_length` (the number of tags) is included as a control.

## Main findings

- On `question.test`, average dependency distance correlates strongly and
  negatively with accuracy for seed228 (r = −0.85) and seed4 (r = −0.88), and
  positively for seed0 (r = +0.53), the model that did not generalize. Sentence
  length, branching factor and NP count do not correlate (|r| < 0.2).
- Within each recursion-depth file nothing correlates (|r| ≤ 0.15).
- With depths pooled, every metric correlates negatively with accuracy
  (r ≈ −0.5 to −0.63, strongest for clause count).

## Caveats

- In this grammar `clauses_per_sentence` always equals `verbs_per_sentence`.
- Within a single test file the metrics take very few distinct values (see
  `n_distinct_metric_values`). Clause count is constant within each file, so
  its correlation is left empty there.
- The per-model and pooled correlations mix recursion depths, so they mostly
  show that accuracy drops as depth increases.
- Types are not independent samples: many share most of their structure. Treat
  the p-values as descriptive.
