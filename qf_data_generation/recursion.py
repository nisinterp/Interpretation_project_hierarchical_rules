"""
recursion.py

Generate question-formation sentences with *recursive, multiply center-embedded*
relative clauses, to probe whether a model trained on Qin, Saphra & Alvarez-Melis
(EMNLP 2025, "Data Drives Unstable Hierarchical Generalization in LMs") has
learned a genuinely hierarchical / recursive rule for auxiliary fronting, or
only a shallow heuristic that happens to match it at the shallow depths seen
in the original training data.

--------------------------------------------------------------------------
Why this script exists
--------------------------------------------------------------------------
In the paper's own generator (`generate_qf_data_from_cfg.py`), every declarative
sentence is drawn from the *fixed set of sentence templates already present in
`question.train`*. Those templates never nest a relative clause inside another
relative clause: the CFG that produced them allows at most one RC on the main
subject NP and/or one RC on the main object NP (aux_count is capped at 3: one
main-clause aux + at most one aux from a subject RC + at most one aux from an
object RC). That is "depth 1" center embedding at best -- structurally
identical to the textbook single-embedding case, e.g.:

    the dog that the cat chased barked .

There is no sentence in the released data of the doubly-embedded form

    the dog that the cat that the mouse chased bit barked .
                 \\____________________________/
                  another relative clause nested
                  *inside* the first one (depth 2)

or the triply-embedded form (depth 3), analogous to the classic
psycholinguistic center-embedding examples (Chomsky & Miller, 1963) and to the
user-supplied example:

    "The dog whose owner is a man that you might have heard of chased the cat."
     \\_______________________________________________________/
      one head NP, modified by a chain of nested relative clauses

If a model appears to have learned "front the structurally main auxiliary,
not the linearly first one" (the paper's hierarchical rule) but has in fact
only learned a shallow proxy that works for the single level of embedding it
was trained on (e.g. "front the second aux" or "front the last aux before the
first non-aux content run"), that proxy will break as soon as embedding depth
is pushed past what training ever contained. This script generates such
out-of-distribution probes (and, if desired, sizeable training sets) at any
requested depth.

--------------------------------------------------------------------------
Grammar
--------------------------------------------------------------------------
Reuses the exact terminal vocabulary of the study (`cfgs/tag_token_map.txt`):
det, n_s/n_p, v_intrans, v_trans, aux_s/aux_p (do/does/don't/doesn't),
rel (who/that), prep.

Recursive subject NP (the construction that actually produces center
embedding -- the relative clause is *object-extracted*, so the embedded NP
sits in the *middle* of its clause, interrupting it, exactly like the human
center-embedding literature):

    NP(depth=0)  ->  det n
    NP(depth=d)  ->  det n rel NP(depth=d-1) aux v_trans      (d > 0)

Reading NP(depth=d) aloud: "<det> <n> <rel> [NP(depth=d-1)] <aux> <v_trans>"
means "<n>, which [NP(depth=d-1)] <aux> <v_trans>" -- the head noun is the
*object* of the embedded verb (gap = object), and the embedded NP is that
verb's *subject*, so its own relative clause (if any) is itself nested inside
this one. Recursing this way d times yields exactly d properly nested,
crossing-free center embeddings, matching the shape of the worked example
above.

The full sentence is:

    S  ->  NP(depth) AUX V_intrans .
    S  ->  NP(depth) AUX V_trans NP_obj .
    NP_obj -> det n [prep det n]?

AUX agrees in number with the *outermost* head noun of NP(depth) (the true
grammatical subject of the matrix clause); each embedded RC's own AUX agrees
with *its* embedded NP's head noun, exactly as in the paper's grammar. The
matrix AUX is always the auxiliary immediately following the fully-built
subject NP -- by construction, not by heuristic detection -- so the question
transformation (front that AUX, replace "." with "?") is always well defined,
regardless of depth.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    # A single depth-2 (double center-embedding) OOD test set, 2000 examples
    python recursion.py --depth 2 --num_samples 2000 --split test

    # Sweep several depths in one call (one file per depth) -- e.g. to
    # replicate the paper's depth-1 regime and extend it to depth 3
    python recursion.py --depths 0,1,2,3 --num_samples 20000 --split test

    # A sizeable *training* set at a fixed depth
    python recursion.py --depth 1 --num_samples 100000 --split train

Output mirrors the paper's file format exactly, so it loads with the
existing `data_utils/lm_dataset_helpers.py` / `train_transformers.py`
pipeline unchanged:

    <input tokens> . decl \t <input tokens> .
    <input tokens> . quest \t <aux> <input tokens minus that aux> ?

Two files are written per depth: the `.{split}` data file and a companion
`.{split}.type` file (token -> grammatical tag), analogous to
`question.test.type`, so downstream analysis scripts that key off type tags
keep working.
"""

import argparse
import os
from collections import defaultdict

import numpy as np

CFG_DIR = "cfgs/"
OUTPUT_DIR = "data_utils/question_formation_data/"


def load_type_token_map(cfg_dir=CFG_DIR):
    """Load the study's terminal vocabulary (tag -> list of surface tokens)."""
    type_to_token_map = defaultdict(list)
    with open(os.path.join(cfg_dir, "tag_token_map.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag, token = line.split("\t")
            type_to_token_map[tag].append(token)
    return type_to_token_map


def build_center_embedded_np(depth, type_map, rng):
    """
    Recursively build a subject NP with `depth` levels of true center
    embedding (object-extracted relative clauses nested inside one another).

    Returns (tokens: list[str], tags: list[str], number: 's'|'p') where
    `number` is the grammatical number of THIS NP's own head noun (used by
    whichever level embeds this NP to pick that level's aux agreement).
    """
    number = rng.choice(["s", "p"])
    det = rng.choice(type_map["det"])
    noun = rng.choice(type_map[f"n_{number}"])
    tokens = [det, noun]
    tags = ["det", f"n_{number}"]

    if depth <= 0:
        return tokens, tags, number

    rel = rng.choice(type_map["rel"])
    embedded_tokens, embedded_tags, embedded_number = build_center_embedded_np(
        depth - 1, type_map, rng
    )
    aux = rng.choice(type_map[f"aux_{embedded_number}"])
    verb = rng.choice(type_map["v_trans"])

    tokens = tokens + [rel] + embedded_tokens + [aux, verb]
    tags = tags + ["rel"] + embedded_tags + [f"aux_{embedded_number}", "v_trans"]
    return tokens, tags, number


def build_simple_np(type_map, rng, pp_prob=0.0):
    """A flat (non-recursive) NP, optionally with one PP modifier, for the
    matrix object position -- kept simple so that recursion depth measures
    exactly one thing: embedding under the matrix subject."""
    number = rng.choice(["s", "p"])
    det = rng.choice(type_map["det"])
    noun = rng.choice(type_map[f"n_{number}"])
    tokens = [det, noun]
    tags = ["det", f"n_{number}"]

    if rng.random() < pp_prob:
        prep = rng.choice(type_map["prep"])
        det2 = rng.choice(type_map["det"])
        num2 = rng.choice(["s", "p"])
        noun2 = rng.choice(type_map[f"n_{num2}"])
        tokens += [prep, det2, noun2]
        tags += ["prep", "det", f"n_{num2}"]

    return tokens, tags


def generate_sentence(depth, type_map, rng, p_transitive=0.6, pp_prob=0.2):
    """
    Build one declarative sentence with the matrix subject center-embedded
    to `depth` levels. Returns:
        decl_tokens: full surface sentence (list[str]), ending before "."
        decl_tags:   matching grammatical tags
        main_aux_idx: index into decl_tokens of the matrix auxiliary
                      (the one a "quest" transformation must front)
    """
    subj_tokens, subj_tags, subj_number = build_center_embedded_np(depth, type_map, rng)

    main_aux = rng.choice(type_map[f"aux_{subj_number}"])
    main_aux_idx = len(subj_tokens)

    if rng.random() < p_transitive:
        main_verb = rng.choice(type_map["v_trans"])
        obj_tokens, obj_tags = build_simple_np(type_map, rng, pp_prob=pp_prob)
        tail_tokens = [main_verb] + obj_tokens
        tail_tags = ["v_trans"] + obj_tags
    else:
        main_verb = rng.choice(type_map["v_intrans"])
        tail_tokens = [main_verb]
        tail_tags = ["v_intrans"]

    decl_tokens = subj_tokens + [main_aux] + tail_tokens
    decl_tags = subj_tags + [f"aux_{subj_number}"] + tail_tags
    return decl_tokens, decl_tags, main_aux_idx


def make_decl_pair(decl_tokens):
    sent = " ".join(decl_tokens)
    return f"{sent} . decl\t{sent} ."


def make_quest_pair(decl_tokens, main_aux_idx):
    main_aux = decl_tokens[main_aux_idx]
    rest = decl_tokens[:main_aux_idx] + decl_tokens[main_aux_idx + 1 :]
    inp = " ".join(decl_tokens)
    out = " ".join([main_aux] + rest)
    return f"{inp} . quest\t{out} ?"


def generate_dataset(depth, num_samples, type_map, rng, quest_ratio, p_transitive, pp_prob):
    pairs = []
    type_lines = []
    num_quest = int(round(num_samples * quest_ratio))

    for i in range(num_samples):
        decl_tokens, decl_tags, main_aux_idx = generate_sentence(
            depth, type_map, rng, p_transitive=p_transitive, pp_prob=pp_prob
        )
        if i < num_quest:
            pairs.append(make_quest_pair(decl_tokens, main_aux_idx))
            type_lines.append(" ".join(decl_tags) + " quest")
        else:
            pairs.append(make_decl_pair(decl_tokens))
            type_lines.append(" ".join(decl_tags) + " decl")

    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    type_lines = [type_lines[i] for i in order]
    return pairs, type_lines


def write_dataset(pairs, type_lines, out_path):
    with open(out_path, "w") as f:
        f.write("\n".join(pairs))
    with open(out_path + ".type", "w") as f:
        f.write("\n".join(type_lines))


def summarize(depth, pairs):
    lengths = [len(line.split("\t")[0].split()) for line in pairs]
    print(
        f"[depth={depth}] n={len(pairs)} "
        f"avg_len={np.mean(lengths):.1f} min_len={min(lengths)} max_len={max(lengths)} "
        f"(expected embedded auxs per subject NP = {depth})"
    )


def main(args):
    rng = np.random.default_rng(args.seed)
    type_map = load_type_token_map()

    os.makedirs(args.output_dir, exist_ok=True)

    depths = (
        [int(d) for d in args.depths.split(",")] if args.depths else [args.depth]
    )

    for depth in depths:
        if depth < 0:
            raise ValueError("depth must be >= 0")
        pairs, type_lines = generate_dataset(
            depth,
            args.num_samples,
            type_map,
            rng,
            quest_ratio=args.quest_ratio,
            p_transitive=args.p_transitive,
            pp_prob=args.pp_prob,
        )
        summarize(depth, pairs)

        out_name = f"question_recursion_depth{depth}"
        if args.dataset_name:
            out_name += f"_{args.dataset_name}"
        out_path = os.path.join(args.output_dir, f"{out_name}.{args.split}")
        write_dataset(pairs, type_lines, out_path)
        print(f"  -> wrote {out_path} (+ .type)")
        print(f"  -> use --dataset {out_name} --dataset_dir {args.output_dir} in train_transformers.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate recursively center-embedded question-formation sentences "
        "(depth 0 = no embedding, depth 1 = paper's single-RC case, depth >= 2 = "
        "double/triple/... center embedding not present in the original data)."
    )
    parser.add_argument("--depth", type=int, default=1, help="Center-embedding depth (single run).")
    parser.add_argument(
        "--depths",
        type=str,
        default=None,
        help="Comma-separated list of depths, e.g. '0,1,2,3'. Overrides --depth "
        "and writes one file per depth.",
    )
    parser.add_argument("--num_samples", type=int, default=20000, help="Examples per depth.")
    parser.add_argument("--quest_ratio", type=float, default=0.5, help="Fraction of examples that are 'quest' (aux-fronting) pairs; remainder are 'decl' identity pairs.")
    parser.add_argument("--p_transitive", type=float, default=0.6, help="Probability the matrix verb is transitive (adds a plain object NP).")
    parser.add_argument("--pp_prob", type=float, default=0.2, help="Probability the matrix object NP carries a PP modifier.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test", help="Suffix for the output file, e.g. 'train' or 'test'.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional extra tag appended to the output filename.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    args = parser.parse_args()
    main(args)
