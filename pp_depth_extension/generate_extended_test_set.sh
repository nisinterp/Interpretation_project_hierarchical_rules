#!/usr/bin/env bash
# Generate an extended question-formation test set: depth-1 RC sentences
# (as in the paper's test set) with a PP chain of depth 0..QF_MAX_PP on one
# noun, using only the paper's vocabulary. It also writes the list of all
# possible types and validates everything with an independent parser.
#
# Usage (from anywhere):
#   bash pp_depth_extension/generate_extended_test_set.sh                 # 10000 sentences, PP depth <= 3
#   QF_N=20000 QF_MAX_PP=4 QF_SEED=1 bash pp_depth_extension/generate_extended_test_set.sh
#   QF_PYTHON=~/miniforge3/envs/hiergenv/bin/python \
#   QF_TRAIN_FILE=~/emnlp2025-593/code/data_utils/question_formation_data/question.train \
#       bash pp_depth_extension/generate_extended_test_set.sh
#
# Settings (env vars): QF_N, QF_MAX_PP, QF_SEED, QF_ORC_NUMBER (different|same),
#   QF_NAME, QF_DATA_DIR, QF_TYPE_DIR, QF_PYTHON,
#   QF_TRAIN_FILE (optional: also check every word occurs in this training file).
# Extra generator options can be appended, e.g. "--quest_ratio 0.5".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PYTHON="${QF_PYTHON:-python}"
N="${QF_N:-10000}"
MAX_PP="${QF_MAX_PP:-3}"
SEED="${QF_SEED:-42}"
ORC_NUMBER="${QF_ORC_NUMBER:-different}"
NAME="${QF_NAME:-question_extended_pp_d${MAX_PP}}"
DATA_DIR="${QF_DATA_DIR:-$HERE/data}"
TYPE_DIR="${QF_TYPE_DIR:-$HERE/data}"
TYPES_CSV="$TYPE_DIR/quest_test_type_extended_pp_d${MAX_PP}.csv"

"$PYTHON" "$HERE/generate_qf_data_from_cfg_extended.py" --mode extended \
    --num_samples "$N" --max_pp "$MAX_PP" --seed "$SEED" --orc_number "$ORC_NUMBER" \
    --dataset_name "$NAME" --data_out_dir "$DATA_DIR" --type_out_dir "$TYPE_DIR" \
    --types_csv "$TYPES_CSV" "$@"

echo
echo "=== validation ==="
EXTRA=()
if [[ -n "${QF_TRAIN_FILE:-}" ]]; then EXTRA+=(--train_file "$QF_TRAIN_FILE"); fi
"$PYTHON" "$HERE/validate_extended_qf_data.py" \
    --data "$DATA_DIR/$NAME.test" --type "$TYPE_DIR/$NAME.test.type" \
    --meta "$TYPE_DIR/$NAME.test.meta.jsonl" --types_csv "$TYPES_CSV" \
    --orig_types "$REPO/qf_data/test_type/quest_test_type.csv" \
    --max_pp "$MAX_PP" --orc_number "$ORC_NUMBER" "${EXTRA[@]}"
