"""
Evaluate trained question formation (QF) checkpoints (.pth files saved by train_transformers.py).

Uses the same evaluation as the training callback ("val_aux"/"test_aux" in the training logs):
the model reads a declarative sentence followed by "quest" and must predict the auxiliary that
starts the question. Accuracy on
  - val:  in-distribution (ambiguous) questions
  - test: the OOD generalization set (only the hierarchical rule gives the right answer),
          i.e. the paper's "OOD generalization accuracy"

The model architecture is read from the args.json file that training writes next to the
checkpoints; without it, the Section 2.3 QF settings are assumed (6 layers, 8 heads, 512 dims,
tied embeddings).

Usage:
    python evaluate_qf_checkpoint.py runs/qf_section2/checkpoints/qf_original_seed0/checkpoint_300000.pth
    python evaluate_qf_checkpoint.py runs/qf_section2/checkpoints/*/checkpoint_300000.pth --out results.csv
    python evaluate_qf_checkpoint.py path/to/checkpoint.pth --per_type   # OOD accuracy per sentence type
"""

import argparse
import csv
import json
import os

import torch

from data_utils import build_datasets_lm
from data_utils.lm_dataset_helpers import eval_lm_callback
from train_transformers import AttrDict, get_base_transformer_lm

# Section 2.3 QF settings, used for anything missing from args.json
DEFAULT_ARGS = {
    "dataset": "question_original",
    "vec_dim": 512,
    "n_heads": 8,
    "encoder_n_layers": 6,
    "mode": "enc_dec",
    "no_pos_enc": False,
    "pos_scale": 1.0,
    "gated_model": False,
    "dropout": 0.1,
    "tied_embedding": True,
    "label_smoothing": 0.0,
    "is_prefix_lm": False,
    "exclude_identity": False,
}


def load_train_args(checkpoint_path, args_json=None):
    path = args_json or os.path.join(os.path.dirname(checkpoint_path), "args.json")
    train_args = dict(DEFAULT_ARGS)
    if os.path.exists(path):
        with open(path) as f:
            train_args.update(json.load(f))
    else:
        print(f"no args.json found at {path}; assuming the Section 2.3 QF model settings")
    return AttrDict(train_args)


def build_vocab(train_args):
    # Rebuild the vocabulary exactly as training did (sorted word set of the dataset files).
    dataset = train_args.dataset
    if "question_D" in dataset or "question_S" in dataset:
        _, in_vocab, _ = build_datasets_lm(
            filename_prefix=dataset, test_filename_prefix="question",
            include_only_quest=train_args.exclude_identity,
        )
    else:
        _, in_vocab, _ = build_datasets_lm(include_only_quest=train_args.exclude_identity)
    return in_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help=".pth checkpoint file(s)")
    parser.add_argument("--args_json", default=None,
                        help="training args.json (default: the one next to each checkpoint)")
    parser.add_argument("--splits", default="val,test", help="comma-separated splits to evaluate")
    parser.add_argument("--per_type", action="store_true",
                        help="also report OOD (test) accuracy per sentence type (question.test.type)")
    parser.add_argument("--out", default=None, help="write results to this CSV file")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits = args.splits.split(",")
    vocab_cache = {}
    rows, type_rows = [], []

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

        row = {"checkpoint": ckpt, "seed": train_args.get("seed")}
        with torch.no_grad():
            for split in splits:
                if split == "test" and args.per_type:
                    acc, _, acc_per_type = eval_lm_callback(model, in_vocab, "test", return_output=True)
                    for sent_type, (type_acc, n) in sorted(acc_per_type.items()):
                        type_rows.append({"checkpoint": ckpt, "type": sent_type, "acc": type_acc, "n": n})
                else:
                    acc = eval_lm_callback(model, in_vocab, split)
                row[f"{split}_acc"] = float(acc)
        rows.append(row)

    print("\n=== Results (val = in-distribution accuracy, test = OOD generalization accuracy)")
    for row in rows:
        accs = "  ".join(f"{s}: {row[f'{s}_acc']:.4f}" for s in splits)
        print(f"{row['checkpoint']}  {accs}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.out}")
        if type_rows:
            type_out = os.path.splitext(args.out)[0] + "_per_type.csv"
            with open(type_out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["checkpoint", "type", "acc", "n"])
                writer.writeheader()
                writer.writerows(type_rows)
            print(f"saved {type_out}")
    elif type_rows:
        print("\n=== OOD accuracy per sentence type")
        for r in type_rows:
            print(f"{r['acc']:.3f}  (n={r['n']:4d})  {r['type']}")


if __name__ == "__main__":
    main()
