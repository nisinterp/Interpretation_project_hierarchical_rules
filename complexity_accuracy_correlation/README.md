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

## All-seeds run (`results_allseeds_new_per_type.csv`)

Input: `data/allseeds_new/results_allseeds_new_per_type.csv` (from
jacoblvovski/jail_lexicon_attitude_study), with 36 seeds × 56 `question.test`
constructions.

```bash
pip install openpyxl
python3 correlate_allseeds.py data/allseeds_new/results_allseeds_new_per_type.csv results allseeds_new
```

| File | Contents |
|---|---|
| `results/allseeds_new_per_type_metrics.csv` | One row per (seed, construction): `n`, `acc` and every metric |
| `results/allseeds_new_correlations.csv` | One row per (seed, metric): Pearson r and Spearman rho with p-values, across the 56 constructions |
| `results/allseeds_new_correlations.xlsx` | Sheet 1 is the same as the correlations CSV. Each other sheet covers one metric and has one row per seed: r(metric, acc), r(sentence_length, acc) and `diff = r(sentence_length) − r(metric)`, for both Pearson and Spearman |

Notes:
- 14 seeds have the same accuracy on every construction (13 at 100%, 1 at
  0%), so their correlations are undefined and left blank.
- `clauses_per_sentence` and `verbs_per_sentence` are always 2 on
  `question.test`, so their sheets are blank.
- Of the 22 seeds with defined correlations, average dependency distance is
  significantly negative (p < .05) for 14. Those are mostly seeds that
  generalize (mean acc ≥ 0.75). It is significantly positive for 4 seeds that
  do not generalize (mean acc ≤ 0.35). Sentence length, branching factor and
  NP count show no consistent effect (median |r| < 0.1).

## Longer-PP run (`results_new_allseeds_longer_pp_per_type.csv`)

Input: `data/allseeds_longer_pp/results_new_allseeds_longer_pp_per_type.csv`
(from jacoblvovski/jail_lexicon_attitude_study). It has 36 seeds × 836
constructions, scored on `../pp_depth_extension/data/question_extended_pp_d3.test`
(PP chains of depth 0–3 on the main subject, the noun inside the RC, or the
main object).

```bash
python3 correlate_allseeds.py data/allseeds_longer_pp/results_new_allseeds_longer_pp_per_type.csv results allseeds_longer_pp
```

Outputs are `results/allseeds_longer_pp_{per_type_metrics.csv,correlations.csv,correlations.xlsx}`.
They have the same layout as the all-seeds run above, and each sheet compares
the metric with `sentence_length`.

Parser change: in this test set a subject can carry a PP chain followed by an
RC that attaches to the head noun ("the peacock near the ravens who doesn't
sleep does giggle"). `parse_np` now allows `Det N PP* RC?`, where nouns inside a
PP take only further PPs. No earlier file contains a PP followed by an RC, and
all earlier result files are reproduced byte for byte.

Results (Pearson r, one point per construction, 836 per seed):

| metric | median r | seeds with significant r < 0 | seeds with significant r > 0 | seeds where \|r\| > \|r(sentence_length)\| |
|---|---|---|---|---|
| sentence_length | −0.08 | 20 | 2 | – |
| avg_dependency_distance | −0.27 | 28 | 7 | 34/36 |
| avg_branching_factor | −0.19 | 27 | 4 | 35/36 |
| nps_per_sentence | −0.05 | 13 | 2 | 5/36 |
| verb_noun_ratio | +0.09 | 3 | 22 | 20/36 |

- No seed is at ceiling on this set (mean accuracy 0.10–0.85), so every
  correlation is defined.
- Average dependency distance is positively correlated with accuracy for all 7
  seeds with mean accuracy ≤ 0.31 (r ≈ +0.44 to +0.56). The correlation is
  about 0 for seed0 (0.38) and negative for all 28 seeds with mean accuracy ≥ 0.47
  (r down to −0.60). This is the same split as on `question.test`, and here it
  covers every seed. Sentence length stays weak (|r| ≤ 0.16) for every seed.
- Pooled accuracy by where the PP chain attaches: no PP 0.79, object 0.79 (at
  every depth), main subject 0.56 / 0.50 / 0.49 (depth 1/2/3), inside the RC
  0.48 / 0.44 / 0.43. Most of the dependency-distance effect therefore comes
  from PP chains that lengthen the subject.
- `clauses_per_sentence` and `verbs_per_sentence` are always 2 here, so their
  sheets are blank. Per-type `n` is small (about 12 sentences), so per-type
  accuracies are noisy.
