"""
Evaluate trained question formation (QF) checkpoints on custom test files.

Same evaluation as evaluate_qf_checkpoint.py (and the "aux" callback during training): the model reads
a declarative sentence followed by "quest" and must predict the auxiliary that starts the question,
choosing among do / does / don't / doesn't. Instead of the fixed question.val / question.test sets,
any file in the question formation data format can be used.

Test file format (same as data_utils/question_formation_data/question.*), one example per line:
    <declarative sentence> . quest<TAB><correct question> ?
e.g.
    my zebra that does read doesn't giggle . quest<TAB>doesn't my zebra that does read giggle ?
The correct answer is the first word of the question. Lines ending in "decl" (declaration copying)
are skipped. All words must be in the model's training vocabulary (the 69-word McCoy et al. vocabulary
for the provided datasets); files with unknown words are rejected with a list of those words.

Reported per test file:
    acc             accuracy (prediction == first word of the correct question)
    linear_agree    fraction of predictions equal to the first auxiliary of the declarative,
                    i.e. the answer of the linear rule
    acc_ambiguous   accuracy on examples where the linear rule also gives the correct answer
    acc_unambiguous accuracy on examples where only the hierarchical rule is correct (OOD-style)

Usage (from any directory):
    python evaluate_qf_custom.py CHECKPOINT.pth --test_file my_test.txt
    python evaluate_qf_custom.py runs/*/checkpoint_300000.pth --test_file a.txt --test_file b.txt --out results.csv
    python evaluate_qf_custom.py CHECKPOINT.pth --test_file my_test.txt --per_type --save_predictions preds.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
AUXS = ["doesn't", "does", "do", "don't"]


def read_test_file(path):
    """Return (prefixes, gold answers, linear-rule answers) for the question lines of a test file."""
    prefixes, gold, linear = [], [], []
    n_skipped = 0
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected '<declarative> . quest<TAB><question> ?', got: {line}")
            inp, out = parts[0].split(), parts[1].split()
            if inp[-1] == "decl":
                n_skipped += 1
                continue
            if inp[-1] != "quest":
                raise ValueError(f"{path}:{line_no}: input must end with 'quest' (or 'decl'), got: {parts[0]}")
            if out[0] not in AUXS:
                raise ValueError(f"{path}:{line_no}: question must start with one of {AUXS}, got: {parts[1]}")
            first_aux = next((w for w in inp if w in AUXS), None)
            if first_aux is None:
                raise ValueError(f"{path}:{line_no}: declarative contains no auxiliary: {parts[0]}")
            prefixes.append(" ".join(inp))
            gold.append(out[0])
            linear.append(first_aux)
    return prefixes, gold, linear, n_skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help=".pth checkpoint file(s)")
    parser.add_argument("--test_file", action="append", required=True,
                        help="test file in question formation format; repeat the option for several files")
    parser.add_argument("--args_json", default=None,
                        help="training args.json (default: the one next to each checkpoint)")
    parser.add_argument("--per_type", action="store_true",
                        help="also report accuracy per sentence type (part-of-speech template of the declarative)")
    parser.add_argument("--save_predictions", default=None, help="write every prediction to this CSV file")
    parser.add_argument("--out", default=None, help="write the summary to this CSV file")
    args = parser.parse_args()

    # Resolve user paths, then run from the code directory: the data loaders locate data_utils/ and cfgs/
    # relative to the working directory.
    args.checkpoints = [os.path.abspath(p) for p in args.checkpoints]
    args.test_file = [os.path.abspath(p) for p in args.test_file]
    for opt in ("args_json", "save_predictions", "out"):
        if getattr(args, opt):
            setattr(args, opt, os.path.abspath(getattr(args, opt)))
    os.chdir(CODE_DIR)
    sys.path.insert(0, CODE_DIR)

    import torch
    from evaluate_qf_orig_data import build_vocab, load_train_args
    from generate_qf_data_from_cfg import load_type_token_map, token_seq_to_type_seq
    from train_transformers import get_base_transformer_lm
    from util import test_continuations

    tests = {}
    for path in args.test_file:
        prefixes, gold, linear, n_skipped = read_test_file(path)
        if not prefixes:
            raise ValueError(f"{path}: no question examples found")
        tests[path] = (prefixes, gold, linear)
        print(f"{path}: {len(prefixes)} question examples" + (f" ({n_skipped} decl lines skipped)" if n_skipped else ""))

    if args.per_type:
        _, token2tag = load_type_token_map()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab_cache = {}
    rows, type_rows, pred_rows = [], [], []

    for ckpt in args.checkpoints:
        print(f"\n=== {ckpt}")
        train_args = load_train_args(ckpt, args.args_json)
        key = (train_args.dataset, train_args.exclude_identity)
        if key not in vocab_cache:
            vocab_cache[key] = build_vocab(train_args)
        in_vocab = vocab_cache[key]

        model, _ = get_base_transformer_lm(train_args, in_vocab, model_name=ckpt)
        model.to(device)
        model.eval()
        aux_idxs = [in_vocab[w] for w in AUXS]

        def tokenizer(s):
            return [model.encoder_sos] + in_vocab(s)

        for path, (prefixes, gold, linear) in tests.items():
            unknown = sorted({w for p in prefixes for w in p.split()} - set(in_vocab.words))
            if unknown:
                raise ValueError(f"{path}: words not in the model's training vocabulary: {unknown}")

            with torch.no_grad():
                probs = test_continuations(tokenizer, model, prefixes, 0)
            pred = [AUXS[i] for i in probs[:, aux_idxs].argmax(dim=1).tolist()]

            correct = [p == g for p, g in zip(pred, gold)]
            ambiguous = [l == g for l, g in zip(linear, gold)]
            amb = [c for c, a in zip(correct, ambiguous) if a]
            unamb = [c for c, a in zip(correct, ambiguous) if not a]
            row = {
                "checkpoint": ckpt,
                "seed": train_args.get("seed"),
                "test_file": path,
                "n": len(gold),
                "acc": sum(correct) / len(gold),
                "linear_agree": sum(p == l for p, l in zip(pred, linear)) / len(gold),
                "n_ambiguous": len(amb),
                "acc_ambiguous": sum(amb) / len(amb) if amb else None,
                "n_unambiguous": len(unamb),
                "acc_unambiguous": sum(unamb) / len(unamb) if unamb else None,
            }
            rows.append(row)

            if args.per_type:
                by_type = defaultdict(list)
                for p, c in zip(prefixes, correct):
                    by_type[" ".join(token_seq_to_type_seq(p, token2tag).split())].append(c)
                for t, cs in sorted(by_type.items()):
                    type_rows.append({"checkpoint": ckpt, "test_file": path, "type": t,
                                      "n": len(cs), "acc": sum(cs) / len(cs)})
            if args.save_predictions:
                for p, g, l, pr in zip(prefixes, gold, linear, pred):
                    pred_rows.append({"checkpoint": ckpt, "test_file": path, "input": p, "gold": g,
                                      "linear_rule_answer": l, "prediction": pr, "correct": pr == g})

    def fmt(x):
        return "  -  " if x is None else f"{x:.4f}"

    print("\n=== Results")
    print(f"{'acc':>7} {'linear':>7} {'acc_amb':>8} {'acc_unamb':>9} {'n':>6}  checkpoint | test file")
    for r in rows:
        print(f"{fmt(r['acc']):>7} {fmt(r['linear_agree']):>7} {fmt(r['acc_ambiguous']):>8} "
              f"{fmt(r['acc_unambiguous']):>9} {r['n']:>6}  {r['checkpoint']} | {r['test_file']}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.out}")
    if type_rows:
        if args.out:
            type_out = os.path.splitext(args.out)[0] + "_per_type.csv"
            with open(type_out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(type_rows[0].keys()))
                writer.writeheader()
                writer.writerows(type_rows)
            print(f"saved {type_out}")
        else:
            print("\n=== Accuracy per sentence type")
            for r in type_rows:
                print(f"{r['acc']:.3f}  (n={r['n']:4d})  {r['type']}")
    if args.save_predictions:
        with open(args.save_predictions, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pred_rows)
        print(f"saved {args.save_predictions}")


if __name__ == "__main__":
    main()
