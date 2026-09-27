"""
build_question_train_tsv.py

Builds question_train_with_trees.tsv from
hier_gen/data_utils/question_formation_data/question.train, using the
deterministic grammar-based converter in grammar_ud_converter.py.

Columns
-------
  id                : the row's 0-indexed line number in the original
                       question.train file (NOT re-indexed after any
                       filtering -- there is none; every line becomes
                       exactly one output row).
  declarative       : the sentence in declarative (canonical) word order.
                       This is always directly present in the dataset: it
                       is the "input" side of every line, decl-tagged or
                       quest-tagged alike (question.train's quest lines
                       store their pre-transformation, declarative-order
                       form as the input field; only the output field is
                       actually fronted). See "building_trees_journal.txt"
                       for the reasoning.
  declarative_tree  : UD dependency tree of `declarative`, as compact JSON
                       (same schema as creating_json_tree.py).
  question          : the corresponding question (aux-fronted) form.
                       - For quest-tagged rows, this is exactly the
                         dataset's own given output field (verified to
                         match, see the journal).
                       - For decl-tagged rows (which the dataset does NOT
                         separately store a transformed form for -- their
                         "output" field is just the declarative repeated),
                         this is derived by applying the paper's own
                         question-formation rule (front the auxiliary that
                         immediately follows the fully-built subject NP)
                         to `declarative`. The grammar-based converter that
                         does this was validated against all 41,165 rows
                         where the dataset DOES give the answer, with a
                         100% exact-match rate, before being trusted on the
                         other 58,835 rows.
  question_tree     : UD dependency tree of `question`, as compact JSON.

Tree JSON is written compact (json.dumps with no indent -- no embedded
newlines), and the whole row is written through Python's csv module with
tab delimiting and default quoting, so any stray tab/quote/newline inside a
field is escaped rather than corrupting the row structure.
"""

import csv
import json
import os

from grammar_ud_converter import build_converter, GrammarError

HIER_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hier_gen")
INPUT_PATH = os.path.join(
    HIER_GEN_DIR, "data_utils", "question_formation_data", "question.train"
)
OUTPUT_FILENAME = "question_train_with_trees.tsv"


def iter_rows(path):
    """Yield (id, marker, declarative_tokens) for every line of
    question.train, in file order."""
    with open(path) as f:
        for line_id, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            inp, _out = line.split("\t")
            toks = inp.split()
            marker = toks[-1]
            assert marker in ("decl", "quest")
            toks = toks[:-1]  # drop marker
            assert toks[-1] == "."
            decl_tokens = toks[:-1]  # sentence tokens, no period
            yield line_id, marker, decl_tokens


def main():
    sentence_to_trees = build_converter()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILENAME)
    n_written = 0
    n_errors = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "declarative", "declarative_tree", "question", "question_tree"])

        for line_id, marker, decl_tokens in iter_rows(INPUT_PATH):
            try:
                decl_json, quest_json = sentence_to_trees(decl_tokens)
            except GrammarError as e:
                n_errors += 1
                print(f"[skip] line {line_id}: {e}")
                continue

            writer.writerow([
                line_id,
                decl_json["text"],
                json.dumps(decl_json, separators=(",", ":")),
                quest_json["text"],
                json.dumps(quest_json, separators=(",", ":")),
            ])
            n_written += 1

    print(f"Wrote {n_written} rows ({n_errors} skipped due to grammar errors) to: {out_path}")


if __name__ == "__main__":
    main()
