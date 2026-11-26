"""
Generate Question Formation Data from CFG -- EXTENDED (variable-depth PP chains)

Extended version of hier_gen/code/generate_qf_data_from_cfg.py (Qin, Saphra &
Alvarez-Melis, EMNLP 2025).

Two modes:

  --mode train     The original generator, unchanged (training data from the
                   templates in question.train). Must be run from hier_gen/code.

  --mode extended  NEW. Generates depth-1 question-formation test sentences
                   (one relative clause on the main subject, as in the paper's
                   generalisation test set), in which one noun carries a
                   right-branching chain of 0..max_pp prepositional phrases:
                       "... the newt near the yaks behind our zebra above my quail ..."
                   Only the paper's 64 words are used (its 69-entry model
                   vocabulary = 64 words + . ? quest decl <pad>), so models
                   trained on the original data can be evaluated on the result. See the EXTENSION section below.

Outputs of --mode extended (same formats as the paper's files):
  <data_out_dir>/<name>.<split>             "<decl> . quest\t<question> ?"
  <type_out_dir>/<name>.<split>.type        "<input tags> quest\t<output tags>"
  <type_out_dir>/<name>.<split>.meta.jsonl  skeleton / PP depth / PP site per line
  <type_out_dir>/quest_<split>_type_<name>.csv  (or --types_csv)
                                            ALL possible types: ,type,number,share
                                            (like quest_test_type.csv) + skeleton,
                                            pp_depth, pp_site, length, ...

Usage:
    # 10000 test sentences with PP chains of depth 0..3
    python generate_qf_data_from_cfg_extended.py --mode extended --num_samples 10000 --max_pp 3

    # Object RCs with the same number as the main subject (paper's choice)
    python generate_qf_data_from_cfg_extended.py --mode extended --orc_number same

    # Original behaviour (training data), from hier_gen/code
    python generate_qf_data_from_cfg_extended.py --mode train --dataset_name my_data
"""

import os
import copy
from collections import defaultdict
import numpy as np
import pandas as pd
# WordVocabulary is imported inside main_train (only needed by --mode train)


CFG_DIR = "cfgs/"  # relative to hier_gen/code
QUES_FORM_DATA_DIR = "data_utils/question_formation_data"
DATA_DIR = "data_utils/"


def load_question_formation_data(split, include_decl=True, include_quest=True, filename=None):
    """Load question formation data from file."""
    if filename is None:
        filename = f"{QUES_FORM_DATA_DIR}/question.{split}"
    else:
        filename = f"{QUES_FORM_DATA_DIR}/{filename}.{split}"

    with open(filename) as f:
        lines = f.read().split("\n")
        if lines[-1] == "":
            lines = lines[:-1]

    in_sents, out_sents = [], []
    counts = {'quest': 0, 'decl': 0}

    for line in lines:
        in_sent, out_sent = line.split("\t")

        if 'quest' in line:
            counts['quest'] += 1
        else:
            counts['decl'] += 1

        if not include_decl and "decl" in in_sent:
            continue
        if not include_quest and "quest" in in_sent:
            continue

        in_sent = " ".join(in_sent.split()[:-1])
        in_sents.append(in_sent)
        out_sents.append(out_sent)

    print(f"{split}: {counts}")
    return pd.DataFrame({"input": in_sents, "output": out_sents})


def token_seq_to_type_seq(token_seq, token2tag):
    """Convert token sequence to type sequence."""
    return " ".join(
        [token2tag.get(token, token) for token in token_seq.split()]
    ).strip()


def count_num_auxs_template(text):
    """Count number of auxiliary verbs in a template."""
    assert isinstance(text, str)
    return text.count("aux")


def load_type_token_map():
    """Load type to token mapping from config file."""
    type_to_token_map = defaultdict(list)
    token2tag = {}

    with open(f"{CFG_DIR}/tag_token_map.txt") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            type_, token = line.split("\t")
            type_to_token_map[type_].append(token)
            if token not in token2tag:
                token2tag[token] = type_

    token2tag["."] = ""
    token2tag["?"] = ""

    return type_to_token_map, token2tag


def generate_decl_pair(inp_types, type_to_token_map):
    """Generate a declarative sentence pair by random token substitution."""
    inp_tokens = copy.copy(inp_types)
    for i, type_ in enumerate(inp_types):
        if type_ in type_to_token_map:
            inp_tokens[i] = np.random.choice(type_to_token_map[type_])
    return f"{' '.join(inp_tokens)} . decl\t{' '.join(inp_tokens)} ."


def generate_quest_pair(inp_types, out_types, type_to_token_map):
    """Generate a question sentence pair by random token substitution."""
    inp_tokens = copy.copy(inp_types)
    for i, type_ in enumerate(inp_types):
        if type_ in type_to_token_map:
            inp_tokens[i] = np.random.choice(type_to_token_map[type_])

    # Align output to input so same substitutions can be used
    out_id_to_inp_id = [0 for _ in range(len(out_types))]

    # First token is always the auxiliary, aligned to first differing token
    for idx, (inp_type, out_type) in enumerate(zip(inp_types, out_types[1:])):
        if inp_type != out_type:
            out_id_to_inp_id[0] = idx
            break
        else:
            out_id_to_inp_id[idx + 1] = idx

    # Align the rest of the tokens
    for idx in range(out_id_to_inp_id[0] + 1, len(out_types)):
        out_id_to_inp_id[idx] = idx

    out_tokens = [inp_tokens[out_id_to_inp_id[idx]] for idx in range(len(out_types))]
    return f"{' '.join(inp_tokens)} . quest\t{' '.join(out_tokens)} ?"


def main_train(args):
    from vocabulary import WordVocabulary
    np.random.seed(args.seed)

    # Load type-token mapping
    type_to_token_map, token2tag = load_type_token_map()
    token_seq_to_type_seq_fn = lambda token_seq: token_seq_to_type_seq(token_seq, token2tag)

    # Load training data
    train_quest_data = load_question_formation_data("train", include_decl=False, include_quest=True)
    train_decl_data = load_question_formation_data("train", include_decl=True, include_quest=False)

    # Convert to type sequences
    quest_input_types = train_quest_data['input'].apply(token_seq_to_type_seq_fn).values.tolist()
    quest_output_types = train_quest_data['output'].apply(token_seq_to_type_seq_fn).values.tolist()
    decl_input_types = np.array(train_decl_data['input'].apply(token_seq_to_type_seq_fn).values.tolist())

    # Determine unique quest input types (for hier/linear classification)
    unique_quest_input_types = set(np.unique(quest_input_types))

    print(f"Total quest types: {len(unique_quest_input_types)}")
    print(f"Total decl types: {len(np.unique(decl_input_types))}")

    # ==================== QUESTION GENERATION ====================
    quest_pairs = []

    # Copy original question pairs
    for i in range(len(train_quest_data['input'])):
        quest_pairs.append(f"{train_quest_data['input'][i]} quest\t{train_quest_data['output'][i]}")

    if args.quest_only:
        # Generate only questions up to num_samples
        while len(quest_pairs) < args.num_samples:
            for inp, out in zip(quest_input_types, quest_output_types):
                inp_types = inp.split()
                out_types = out.split()
                quest_pairs.append(generate_quest_pair(inp_types, out_types, type_to_token_map))
                if len(quest_pairs) >= args.num_samples:
                    break

        quest_pairs = quest_pairs[:args.num_samples]
        sentence_pairs = quest_pairs

    else:
        # ==================== DECLARATIVE GENERATION ====================

        # Determine number of decl samples
        if args.keep_quest_ratio:
            if args.decl_ratio is None:
                decl_ratio = len(train_decl_data) / (len(train_quest_data) + len(train_decl_data))
            else:
                decl_ratio = args.decl_ratio
            num_decl_samples = int(args.num_samples * decl_ratio)
        else:
            num_decl_samples = args.num_samples - len(train_quest_data)

        num_quest_samples = args.num_samples - num_decl_samples
        print(f"Target decl samples: {num_decl_samples}, quest samples: {num_quest_samples}")

        # Classify decl types into hierarchical and linear
        # aux_count=1: all linear, aux_count=2: mixed, aux_count=3: all hierarchical
        hier_decl_types = []
        linear_decl_types = []

        for inp in decl_input_types:
            aux_count = count_num_auxs_template(inp)
            if aux_count == 1:
                linear_decl_types.append(inp)
            elif aux_count == 2:
                if inp in unique_quest_input_types:
                    linear_decl_types.append(inp)
                else:
                    hier_decl_types.append(inp)
            elif aux_count == 3:
                hier_decl_types.append(inp)

        print(f"Hierarchical decl types: {len(hier_decl_types)}, Linear decl types: {len(linear_decl_types)}")

        # Apply hier_linear_decl_ratio if specified
        if args.hier_linear_decl_ratio is not None:
            ratio = [float(x) for x in args.hier_linear_decl_ratio.split(",")]
            assert len(ratio) == 2, "hier_linear_decl_ratio must have 2 values (hier,linear)"
            assert abs(sum(ratio) - 1.0) < 1e-6, "hier_linear_decl_ratio must sum to 1"

            num_hier = int(num_decl_samples * ratio[0])
            num_linear = num_decl_samples - num_hier
            print(f"Target hier decl: {num_hier}, linear decl: {num_linear}")
        else:
            # Use original distribution
            total_original = len(hier_decl_types) + len(linear_decl_types)
            num_hier = int(num_decl_samples * len(hier_decl_types) / total_original)
            num_linear = num_decl_samples - num_hier

        # Apply sbj/obj ratio for hierarchical decls (aux_count >= 2)
        if args.decl_sbj_obj_ratio is not None:
            sbj_obj_ratio = [float(x) for x in args.decl_sbj_obj_ratio.split(",")]
            assert len(sbj_obj_ratio) == 2, "decl_sbj_obj_ratio must have 2 values (sbj,obj)"
            assert abs(sum(sbj_obj_ratio) - 1.0) < 1e-6, "decl_sbj_obj_ratio must sum to 1"

            print(f"Applying sbj/obj ratio control: {sbj_obj_ratio}")

            # Split hier decls into sbj and obj types
            hier_sbj_types = []
            hier_obj_types = []

            for inp in hier_decl_types:
                tokens = inp.split()
                if len(tokens) > 3 and tokens[2] == 'rel':
                    if "aux" in tokens[3]:  # sbj-extracted
                        hier_sbj_types.append(inp)
                    elif tokens[3] == "det":  # obj-extracted
                        hier_obj_types.append(inp)
                    else:
                        hier_sbj_types.append(inp)  # default to sbj
                else:
                    hier_sbj_types.append(inp)  # default to sbj

            print(f"Original hier sbj: {len(hier_sbj_types)}, obj: {len(hier_obj_types)}")

            # Resize to target ratio
            num_sbj = int(num_hier * sbj_obj_ratio[0])
            num_obj = num_hier - num_sbj

            resized_sbj = []
            resized_obj = []

            while len(resized_sbj) < num_sbj:
                resized_sbj.extend(hier_sbj_types)
            while len(resized_obj) < num_obj:
                resized_obj.extend(hier_obj_types)

            hier_decl_types = resized_sbj[:num_sbj] + resized_obj[:num_obj]
            print(f"Resized hier sbj: {num_sbj}, obj: {num_obj}")

        # Apply num_types diversity control
        if args.num_types > 0:
            unique_hier = list(set(hier_decl_types))
            unique_linear = list(set(linear_decl_types))

            np.random.shuffle(unique_hier)
            np.random.shuffle(unique_linear)

            # Distribute num_types proportionally
            total_unique = len(unique_hier) + len(unique_linear)
            if total_unique > args.num_types:
                num_hier_types = int(args.num_types * len(unique_hier) / total_unique)
                num_linear_types = args.num_types - num_hier_types

                selected_hier = set(unique_hier[:num_hier_types])
                selected_linear = set(unique_linear[:num_linear_types])

                hier_decl_types = [t for t in hier_decl_types if t in selected_hier]
                linear_decl_types = [t for t in linear_decl_types if t in selected_linear]

                print(f"After diversity control - hier types: {len(selected_hier)}, linear types: {len(selected_linear)}")

        # Generate decl pairs
        decl_pairs = []

        # Generate hierarchical decls
        while len(decl_pairs) < num_hier:
            for inp in hier_decl_types:
                inp_types = inp.split()
                decl_pairs.append(generate_decl_pair(inp_types, type_to_token_map))
                if len(decl_pairs) >= num_hier:
                    break

        hier_pairs = decl_pairs[:num_hier]

        # Generate linear decls
        decl_pairs = []
        while len(decl_pairs) < num_linear:
            for inp in linear_decl_types:
                inp_types = inp.split()
                decl_pairs.append(generate_decl_pair(inp_types, type_to_token_map))
                if len(decl_pairs) >= num_linear:
                    break

        linear_pairs = decl_pairs[:num_linear]
        decl_pairs = hier_pairs + linear_pairs

        print(f"Generated decl pairs: {len(decl_pairs)} (hier: {len(hier_pairs)}, linear: {len(linear_pairs)})")

        # Generate additional quest pairs if needed
        while len(quest_pairs) < num_quest_samples:
            for inp, out in zip(quest_input_types, quest_output_types):
                inp_types = inp.split()
                out_types = out.split()
                quest_pairs.append(generate_quest_pair(inp_types, out_types, type_to_token_map))
                if len(quest_pairs) >= num_quest_samples:
                    break

        quest_pairs = quest_pairs[:num_quest_samples]
        print(f"Generated quest pairs: {len(quest_pairs)}")

        sentence_pairs = quest_pairs + decl_pairs

    # Check vocab
    in_vocab = WordVocabulary(sentence_pairs, split_punctuation=False)
    print(f"Total vocab: {len(in_vocab)}")

    # Ensure correct sample size
    train_pairs = sentence_pairs[:args.num_samples]
    print(f"Final training pairs: {len(train_pairs)}")

    # Save data
    data_dir = f"{DATA_DIR}/question_formation_data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    output_filename = f"{data_dir}/question_D_{args.dataset_name}.train"
    with open(output_filename, "w") as f:
        f.write("\n".join(train_pairs))
    print(f"Saved data: {output_filename}")
    print(f"Use --dataset question_D_{args.dataset_name} in train_transformers.py")



# =============================================================================
# ==========  EXTENSION: depth-1 sentences with PP chains of variable depth  ===
# =============================================================================
#
# Everything above this line is the original generator, kept as it was
# (only `main` was renamed to `main_train` and the WordVocabulary import was
# made lazy so the extension also runs outside hier_gen/code).
#
# Only the paper's own vocabulary is used (cfgs/tag_token_map.txt), so models
# trained on the original data can be evaluated on these sentences.
#
# ---------------------------------------------------------------------------
# 1. Skeletons
# ---------------------------------------------------------------------------
# The paper's generalisation test set (qf_data/test_type/quest_test_type.csv)
# has 56 input templates. All of them come from this grammar, with a relative
# clause (RC) on the main subject (embedding depth 1):
#
#   S      -> NP_S[n] RC[n] aux_n VP
#   RC[n]  -> rel aux_n v_intrans              (subject RC, intransitive)
#           | rel aux_n v_trans NP_R           (subject RC, transitive)
#           | rel NP_R[m] aux_m v_trans        (object RC)
#   VP     -> v_intrans | v_trans NP_O
#   NP     -> det n_s | det n_p
#
# That gives 24 skeletons (2 subject numbers x 4 RC variants x 3 VP variants).
# The other 32 of the 56 templates are the transitive skeletons with one PP
# on the main object, which here is simply "PP depth 1 on O".
#
# Object RCs: in the paper's test set the RC subject always has the same
# number as the main subject (m = n). Here, by default, it has the OPPOSITE
# number (m != n, --orc_number different). The RC aux then differs in number
# from the main aux ("the newt that the yaks do admire does giggle"), so
# fronting the wrong aux gives a visibly wrong question.
# --orc_number same reproduces the paper's choice.
# Subject RCs cannot do this: their aux agrees with the main subject itself.
#
# Aux tokens: as in the paper's test set (100% of its 10000 lines), the RC aux
# TOKEN is always different from the main aux token. For subject RCs (same
# number) this means opposite polarity, e.g. does / doesn't. The evaluation
# only looks at the first word of the question, so if the two tokens were equal
# the linear and the hierarchical rule would give the same answer.
#
# ---------------------------------------------------------------------------
# 2. PP chains (new)
# ---------------------------------------------------------------------------
# A right-branching chain of k PPs (k = 0..max_pp) is attached to ONE of the
# skeleton's noun slots: S = main subject head, R = noun inside the RC,
# O = main object.
#     ... n prep det n prep det n prep det n ...   ("in house in town in province")
# The nouns in the chain have free number (s/p). They never govern agreement.
# With a chain on S, the RC follows the whole chain but still agrees with
# the head of S:
#     "the peacock near the ravens who doesn't sleep does giggle"
# The question is formed as always: front the main-clause aux.
#
# The paper's 56 test templates are the {k=0} + {k=1 on O} part of this space
# (with --orc_number same). The whole space is small enough to list every
# concrete tag sequence: 836 types for max_pp=3.
# =============================================================================

import itertools
import json

# Copy of hier_gen/code/cfgs/tag_token_map.txt restricted to the tags this
# grammar uses. It is used when that file is not available (e.g. when running
# inside this repo).
BUILTIN_BASE_TYPE_TOKENS = {
    "det": ["the", "some", "my", "your", "our", "her"],
    "n_s": ["newt", "orangutan", "peacock", "quail", "raven", "salamander", "tyrannosaurus",
            "unicorn", "vulture", "walrus", "xylophone", "yak", "zebra"],
    "n_p": ["newts", "orangutans", "peacocks", "quails", "ravens", "salamanders", "tyrannosauruses",
            "unicorns", "vultures", "walruses", "xylophones", "yaks", "zebras"],
    "aux_s": ["does", "doesn't"],
    "aux_p": ["do", "don't"],
    "v_trans": ["entertain", "amuse", "high_five", "applaud", "confuse", "admire", "accept",
                "remember", "comfort"],
    "v_intrans": ["giggle", "smile", "sleep", "swim", "wait", "move", "change", "read", "eat"],
    "rel": ["who", "that"],
    "prep": ["around", "near", "with", "upon", "by", "behind", "above", "below"],
}

RC_VARIANTS = ("src_intrans", "src_trans", "orc")   # subject RC (intr./tr.), object RC
VP_VARIANTS = ("intrans", "trans")
FLIP = {"s": "p", "p": "s"}


def load_base_type_token_map(cfg_dir=None):
    """tag -> tokens, from cfg_dir/tag_token_map.txt if it exists (as in the
    original generator), otherwise from the built-in copy."""
    path = os.path.join(cfg_dir or CFG_DIR, "tag_token_map.txt")
    if not os.path.exists(path):
        return {k: list(v) for k, v in BUILTIN_BASE_TYPE_TOKENS.items()}
    type_map = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tag, tok = line.split("\t")
                type_map[tag].append(tok)
    return {k: list(v) for k, v in type_map.items()}


def enumerate_skeletons(orc_number="different"):
    """The 24 depth-1 skeletons, as dicts {name, num: {slot: s|p}, rc, vp}."""
    skels = []
    for n_s in ("s", "p"):
        for rc in RC_VARIANTS:
            if rc == "src_intrans":
                r_nums = [None]
            elif rc == "src_trans":
                r_nums = ["s", "p"]
            else:
                r_nums = [FLIP[n_s]] if orc_number == "different" else [n_s]
            for n_r in r_nums:
                for vp in VP_VARIANTS:
                    for n_o in ([None] if vp == "intrans" else ["s", "p"]):
                        num = {"S": n_s}
                        if n_r is not None:
                            num["R"] = n_r
                        if n_o is not None:
                            num["O"] = n_o
                        name = f"S={n_s}|rc={rc}" + (f":R={n_r}" if n_r else "") + \
                               f"|vp={vp}" + (f":O={n_o}" if n_o else "")
                        skels.append({"name": name, "num": num, "rc": rc, "vp": vp})
    return skels


def skeleton_slots(skel):
    return [s for s in ("S", "R", "O") if s in skel["num"]]


def render_template(skel, pp_depth, pp_site, chain_numbers):
    """Input tags for a skeleton with a PP chain of `pp_depth` PPs on `pp_site`
    (chain_numbers: list of s/p, one per chain noun).
    Returns (tags, main_aux_index, rc_aux_index)."""
    num = skel["num"]

    def np_(slot):
        tags = ["det", f"n_{num[slot]}"]
        if slot == pp_site:
            for cn in chain_numbers[:pp_depth]:
                tags += ["prep", "det", f"n_{cn}"]
        return tags

    n = num["S"]
    tags = np_("S") + ["rel"]
    if skel["rc"] == "src_intrans":
        rc_aux = len(tags)
        tags += [f"aux_{n}", "v_intrans"]
    elif skel["rc"] == "src_trans":
        rc_aux = len(tags)
        tags += [f"aux_{n}", "v_trans"] + np_("R")
    else:
        tags += np_("R")
        rc_aux = len(tags)
        tags += [f"aux_{num['R']}", "v_trans"]
    main_aux = len(tags)
    tags.append(f"aux_{n}")
    tags += ["v_intrans"] if skel["vp"] == "intrans" else ["v_trans"] + np_("O")
    return tags, main_aux, rc_aux


def enumerate_construction_space(max_pp=3, orc_number="different"):
    """Every concrete type (skeleton x PP depth x PP site x chain-noun numbers)."""
    for skel in enumerate_skeletons(orc_number):
        specs = [(0, None)] + [(k, s) for k in range(1, max_pp + 1) for s in skeleton_slots(skel)]
        for k, site in specs:
            for chain_numbers in itertools.product("sp", repeat=k):
                tags, main_aux, rc_aux = render_template(skel, k, site, list(chain_numbers))
                yield {"type": " ".join(tags), "skeleton": skel["name"], "rc": skel["rc"],
                       "vp": skel["vp"], "pp_depth": k, "pp_site": site or "",
                       "length": len(tags), "main_aux_index": main_aux, "rc_aux_index": rc_aux,
                       "_skel": skel, "_chain": list(chain_numbers)}


def sample_construction(skels, rng, max_pp):
    """Skeleton uniform, PP depth uniform in 0..max_pp, site uniform over the
    skeleton's slots, chain-noun numbers uniform."""
    skel = skels[rng.integers(len(skels))]
    k = int(rng.integers(0, max_pp + 1))
    slots = skeleton_slots(skel)
    site = slots[rng.integers(len(slots))] if k else None
    chain = [["s", "p"][rng.integers(2)] for _ in range(k)]
    return skel, k, site, chain


def fill_tokens(tags, type_map, rng, main_aux, rc_aux):
    """Random token for every tag. The RC aux token is always different from
    the main aux token (as in the paper's test set)."""
    toks = [type_map[t][rng.integers(len(type_map[t]))] for t in tags]
    if toks[rc_aux] == toks[main_aux]:
        choices = [a for a in type_map[tags[rc_aux]] if a != toks[main_aux]]
        toks[rc_aux] = choices[rng.integers(len(choices))]
    return toks


def generate_extended_examples(num_samples, type_map, rng, max_pp=3, orc_number="different",
                               quest_ratio=1.0, cover_all=True):
    """List of example dicts. With cover_all (and num_samples >= the number of
    possible types), every possible type is generated at least once and the
    rest is sampled with sample_construction."""
    skels = enumerate_skeletons(orc_number)
    plan = []
    if cover_all:
        space = list(enumerate_construction_space(max_pp, orc_number))
        if num_samples >= len(space):
            plan = [(r["_skel"], r["pp_depth"], r["pp_site"] or None, r["_chain"]) for r in space]
    while len(plan) < num_samples:
        plan.append(sample_construction(skels, rng, max_pp))

    num_quest = int(round(num_samples * quest_ratio))
    quest_flags = [True] * num_quest + [False] * (num_samples - num_quest)
    quest_flags = [quest_flags[i] for i in rng.permutation(num_samples)]

    out = []
    for (skel, k, site, chain), is_quest in zip(plan, quest_flags):
        tags, main_aux, rc_aux = render_template(skel, k, site, chain)
        toks = fill_tokens(tags, type_map, rng, main_aux, rc_aux)
        inp = " ".join(toks)
        q_toks = [toks[main_aux]] + toks[:main_aux] + toks[main_aux + 1:]
        q_tags = [tags[main_aux]] + tags[:main_aux] + tags[main_aux + 1:]
        if is_quest:
            line = f"{inp} . quest\t{' '.join(q_toks)} ?"
            type_line = f"{' '.join(tags)} quest\t{' '.join(q_tags)}"
        else:
            line = f"{inp} . decl\t{inp} ."
            type_line = f"{' '.join(tags)} decl\t{' '.join(tags)}"
        out.append({"line": line, "type_line": type_line, "type": " ".join(tags),
                    "skeleton": skel["name"], "pp_depth": k, "pp_site": site or "",
                    "main_aux_index": main_aux, "rc_aux_index": rc_aux, "quest": is_quest})
    order = rng.permutation(len(out))
    return [out[j] for j in order]


def main_extended(args):
    rng = np.random.default_rng(args.seed)
    type_map = load_base_type_token_map(args.cfg_dir)
    if args.max_pp < 0:
        raise SystemExit("--max_pp must be >= 0")
    name = args.dataset_name or f"question_extended_pp_d{args.max_pp}"

    examples = generate_extended_examples(args.num_samples, type_map, rng, max_pp=args.max_pp,
                                          orc_number=args.orc_number, quest_ratio=args.quest_ratio,
                                          cover_all=not args.no_cover_all)

    os.makedirs(args.data_out_dir, exist_ok=True)
    os.makedirs(args.type_out_dir, exist_ok=True)
    data_path = os.path.join(args.data_out_dir, f"{name}.{args.split}")
    type_path = os.path.join(args.type_out_dir, f"{name}.{args.split}.type")
    meta_path = os.path.join(args.type_out_dir, f"{name}.{args.split}.meta.jsonl")
    with open(data_path, "w") as f:
        f.write("\n".join(e["line"] for e in examples))
    with open(type_path, "w") as f:
        f.write("\n".join(e["type_line"] for e in examples))
    with open(meta_path, "w") as f:
        for e in examples:
            f.write(json.dumps({k: v for k, v in e.items() if k not in ("line", "type_line")}) + "\n")

    # List of ALL possible types, in the format of quest_test_type.csv
    # (,type,number,share) plus descriptive columns; number = count among the
    # generated quest examples.
    space = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                          for r in enumerate_construction_space(args.max_pp, args.orc_number)])
    assert space["type"].is_unique
    counts = pd.Series([e["type"] for e in examples if e["quest"]]).value_counts()
    space.insert(1, "number", space["type"].map(counts).fillna(0).astype(int))
    space.insert(2, "share", space["number"] / max(1, space["number"].sum()))
    list_path = args.types_csv or os.path.join(args.type_out_dir, f"quest_{args.split}_type_{name}.csv")
    space.to_csv(list_path)

    lengths = [len(e["type"].split()) for e in examples]
    print(f"max_pp={args.max_pp} orc_number={args.orc_number}  samples={len(examples)}  "
          f"possible types={len(space)}  generated types={int((space['number'] > 0).sum())}  "
          f"length min/mean/max={min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}")
    print(f"wrote {data_path}\n      {type_path}\n      {meta_path}\n      {list_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate question formation data from CFG (extended)")
    parser.add_argument("--mode", choices=["train", "extended"], default="train",
                        help="'train' = original generator; 'extended' = depth-1 sentences with PP chains")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Output name. Required for --mode train; "
                             "default 'question_extended_pp_d<max_pp>' for --mode extended")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Total number of samples (default: 100000 for train, 10000 for extended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # ---- original (--mode train) options ----
    parser.add_argument("--num_types", type=int, default=-1,
                        help="[train] Limit diversity of sentence types. -1 = no limit")
    parser.add_argument("--quest_only", action="store_true",
                        help="[train] Only generate question pairs (no declaratives)")
    parser.add_argument("--keep_quest_ratio", action="store_true",
                        help="[train] Maintain original quest/decl ratio instead of filling with decls")
    parser.add_argument("--decl_ratio", type=float, default=None,
                        help="[train] Explicit ratio of decl samples (e.g., 0.5). Used with --keep_quest_ratio")
    parser.add_argument("--hier_linear_decl_ratio", type=str, default=None,
                        help="[train] Ratio of hierarchical vs linear decls, comma-separated (e.g., '0.5,0.5')")
    parser.add_argument("--decl_sbj_obj_ratio", type=str, default=None,
                        help="[train] Ratio of subject vs object relative clauses for hier decls (e.g., '0.5,0.5')")

    # ---- extension (--mode extended) options ----
    parser.add_argument("--max_pp", type=int, default=3,
                        help="[extended] Maximum depth of the PP chain (0..max_pp, uniform)")
    parser.add_argument("--orc_number", choices=["different", "same"], default="different",
                        help="[extended] Number of the RC subject in object RCs relative to the main "
                             "subject: 'different' (default; RC aux differs in number from the main aux) "
                             "or 'same' (as in the paper's test set)")
    parser.add_argument("--no_cover_all", action="store_true",
                        help="[extended] Do not force every possible type to appear at least once")
    parser.add_argument("--quest_ratio", type=float, default=1.0,
                        help="[extended] Fraction of quest pairs; the rest are decl (identity) pairs. "
                             "The paper's test set is all quest (1.0)")
    parser.add_argument("--split", type=str, default="test", help="[extended] File suffix, e.g. test/train")
    parser.add_argument("--cfg_dir", type=str, default=None,
                        help="[extended] Directory with tag_token_map.txt (default: cfgs/ if present, "
                             "else a built-in copy of the paper's vocabulary)")
    parser.add_argument("--data_out_dir", type=str, default="data_utils/question_formation_data/",
                        help="[extended] Where the data file goes")
    parser.add_argument("--type_out_dir", type=str, default="data_utils/question_formation_data/",
                        help="[extended] Where the .type / .meta.jsonl / types csv go")
    parser.add_argument("--types_csv", type=str, default=None,
                        help="[extended] Path of the list of all possible types "
                             "(default: <type_out_dir>/quest_<split>_type_<name>.csv)")

    args = parser.parse_args()
    if args.mode == "train":
        if not args.dataset_name:
            parser.error("--dataset_name is required for --mode train")
        if args.num_samples is None:
            args.num_samples = 100000
        main_train(args)
    else:
        if args.num_samples is None:
            args.num_samples = 10000
        main_extended(args)
