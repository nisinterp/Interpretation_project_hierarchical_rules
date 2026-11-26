# Question-formation test data with variable-depth PP chains

Extends the paper's depth-1 generalisation test set (`qf_data/test_type/quest_test_type.csv`,
56 templates). The sentences keep one relative clause on the main subject. One noun
(main subject, the noun inside the relative clause, or main object) carries a
right-branching chain of 0–3 prepositional phrases:

    doesn't her newt with our salamanders near our vulture behind my xylophones who my yaks do amuse accept your quail ?

Only the paper's 64 words are used, so models trained on the original data can be
evaluated directly (e.g. with `training_and_evaluating_models/evaluate_qf_custom_data.py`).

Differences from the paper's test set:
- **Object relative clauses:** the noun inside the clause has the opposite number to the
  main subject ("the newt that the yaks **do** admire **doesn't** giggle"), so the two
  auxiliaries differ in number. `--orc_number same` reproduces the paper's choice.
- **Auxiliary words:** as in the paper, the relative-clause auxiliary is always a different
  word from the main auxiliary, so fronting the first auxiliary (linear rule) always gives
  a wrong first word.
- **The paper's 56 templates** are exactly the "PP depth 0, or depth 1 on the object"
  part of this set (object relatives with the inner noun's number flipped).

## Files

| File | What it is |
|---|---|
| `generate_qf_data_from_cfg_extended.py` | `--mode train`: the original `hier_gen/code/generate_qf_data_from_cfg.py`, unchanged (same output). `--mode extended`: PP-chain test sentences (`--max_pp`, `--orc_number`, `--num_samples`, `--seed`). |
| `generate_extended_test_set.sh` | Generates 10000 sentences into `data/` and validates them. Settings: `QF_N`, `QF_MAX_PP`, `QF_SEED`, `QF_ORC_NUMBER`, `QF_PYTHON`, `QF_TRAIN_FILE`. |
| `validate_extended_qf_data.py` | Independent parser-based checker: grammar and agreement, question = main-aux fronting, distinct aux words, vocabulary, coverage of all types, relation to the paper's 56 templates. |
| `data/question_extended_pp_d3.test` | 10000 test pairs, `<decl> . quest<TAB><question> ?` (same format as `question.test`). |
| `data/question_extended_pp_d3.test.type` | Tag sequences per line (same format as `question.test.type`). |
| `data/question_extended_pp_d3.test.meta.jsonl` | Per line: type, skeleton, PP depth, PP site, aux positions. |
| `data/quest_test_type_extended_pp_d3.csv` | All 836 possible types (`,type,number,share` like `quest_test_type.csv`, plus skeleton / rc / vp / pp_depth / pp_site / length / aux positions); `number` = count in the generated file. All 836 appear. |

## Regenerate

    QF_PYTHON=python bash pp_depth_extension/generate_extended_test_set.sh
    QF_MAX_PP=4 QF_N=20000 bash pp_depth_extension/generate_extended_test_set.sh
