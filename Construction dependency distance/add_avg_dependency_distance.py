"""
add_avg_dependency_distance.py

Reads results_per_type.csv (from jacoblvovski/jail_lexicon_attitude_study),
computes the average dependency distance of each row's CONSTRUCTION, and
writes out result_per_type_add.csv: every original column, plus one new
column "avg_dependency_distance".

What "construction" means in this file
------------------------------------------
results_per_type.csv's "type" column is not a sentence -- it's a
grammatical TYPE-TAG TEMPLATE, in exactly the tag vocabulary used
throughout this project's hier_gen grammar (det, n_s/n_p, v_trans/
v_intrans, aux_s/aux_p, rel, prep -- the same tags as
hier_gen/cfgs/tag_token_map.txt). Each row looks like:

    det n_p rel aux_p v_intrans aux_p v_intrans quest aux_p det n_p rel aux_p v_intrans v_intrans

i.e. "<declarative-order tag sequence> quest <question-order tag
sequence>" -- the same two-part shape as hier_gen's own question.test.type
file. "acc" and "n" are, per row, the model's accuracy and the number of
actual test sentences that were generated from this one template (with
different lexical fillings -- different nouns/verbs/determiners each time).

Which half is used, and why
--------------------------------
Dependency distance is computed from the DECLARATIVE half only (the part
before " quest "). That's the half that defines the construction's own
syntactic complexity (subject-embedding depth, RC type, PP presence) --
the same declarative-side structure the paper's own hier/linear
classification is defined from (see hier_gen/data_utils/
lm_dataset_helpers.py's determine_decltype, and this project's own
recursion.py/grammar_ud_converter.py). The question-order half would give
different distances (fronting the aux changes several words' linear
offsets) and isn't what "the construction" refers to here.

Key point this relies on: dependency distance only depends on which
GRAMMATICAL CATEGORY sits at each position (det/noun/verb/aux/rel/prep),
never on which specific lexeme is chosen -- e.g. a determiner is always
exactly 1 position from the noun it modifies, whether that determiner is
"the" or "my". Since a "type" row already IS the category sequence, this
script needs no lexicon at all, and its result is exact: every one of the
`n` actual sentences that share a given row's template has the IDENTICAL
average dependency distance, because they all have the identical tree
shape and differ only in which words fill each slot. So one computation
per row/template is both correct and complete -- there is nothing to
average over multiple concrete sentences.

Grammar and dependency-distance definition
-----------------------------------------------
Structural parser (parse_np/parse_sentence) is the same grammar validated
in grammar_ud_converter.py against all 100,000 rows of question.train (see
"tree building journal.txt" / "building_trees_journal.txt" in this
project) -- reproduced here in a trimmed, tag-only form (no lexicon, no
UD-label rendering, since only token POSITIONS and their HEADS are needed
for a distance calculation, not deprel labels or surface words):

    S    -> NP AUX V .              (V = v_intrans)
    S    -> NP AUX V NP .           (V = v_trans)
    NP   -> Det N
    NP   -> Det N RC-subj
    NP   -> Det N RC-obj
    NP   -> Det N PP
    RC-subj -> rel AUX V (NP)?
    RC-obj  -> rel NP AUX V
    PP   -> prep NP

Dependency distance, per Liu (2008) (same definition used in this
project's compute_dependency_distance.py): for a word at position `id`
whose head is at position `head`, its distance is |id - head|. The
sentence root (main verb) has no head and is excluded. Average dependency
distance for a construction = mean of these distances over every word in
its declarative-order template.
"""

import argparse
import csv
import os


class GrammarError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Structural parser: identical grammar/branching logic to
# grammar_ud_converter.py's parse_np/parse_sentence, operating directly on
# tag strings (no surface tokens needed -- see module docstring for why).
# ---------------------------------------------------------------------------

def parse_np(tags, idx):
    if idx >= len(tags) or tags[idx] != "det":
        raise GrammarError(f"expected det at {idx}, got {tags[idx:idx+1]}")
    det_idx = idx
    idx += 1
    if idx >= len(tags) or not tags[idx].startswith("n_"):
        raise GrammarError(f"expected noun at {idx}, got {tags[idx:idx+1]}")
    noun_idx = idx
    idx += 1

    mod = None
    if idx < len(tags) and tags[idx] == "rel":
        rel_idx = idx
        idx += 1
        if idx < len(tags) and tags[idx].startswith("aux_"):
            # subject-extracted RC: rel AUX V (NP)?
            aux_idx = idx
            idx += 1
            if idx >= len(tags) or tags[idx] not in ("v_trans", "v_intrans"):
                raise GrammarError(f"expected verb at {idx}")
            verb_idx = idx
            v_type = tags[idx]
            idx += 1
            obj_node = None
            if v_type == "v_trans":
                obj_node, idx = parse_np(tags, idx)
            mod = dict(type="rc_subj", rel_idx=rel_idx, aux_idx=aux_idx, verb_idx=verb_idx, obj=obj_node)
        elif idx < len(tags) and tags[idx] == "det":
            # object-extracted RC: rel NP AUX V   (gap = object)
            embedded_node, idx = parse_np(tags, idx)
            if idx >= len(tags) or not tags[idx].startswith("aux_"):
                raise GrammarError(f"expected aux at {idx}")
            aux_idx = idx
            idx += 1
            if idx >= len(tags) or tags[idx] != "v_trans":
                raise GrammarError(f"expected v_trans at {idx}")
            verb_idx = idx
            idx += 1
            mod = dict(type="rc_obj", rel_idx=rel_idx, embedded=embedded_node, aux_idx=aux_idx, verb_idx=verb_idx)
        else:
            raise GrammarError(f"unexpected token after 'rel' at {idx}: {tags[idx:idx+1]}")
    elif idx < len(tags) and tags[idx] == "prep":
        prep_idx = idx
        idx += 1
        pp_np, idx = parse_np(tags, idx)
        mod = dict(type="pp", prep_idx=prep_idx, np=pp_np)

    return dict(det_idx=det_idx, noun_idx=noun_idx, mod=mod), idx


def parse_sentence(tags):
    """Parse a declarative-order tag sequence (no trailing period tag --
    this file's "type" templates don't include one). Returns
    (subject_node, main_aux_idx, main_verb_idx, object_node_or_None)."""
    subject_node, idx = parse_np(tags, 0)
    if idx >= len(tags) or not tags[idx].startswith("aux_"):
        raise GrammarError(f"expected main aux at {idx}")
    main_aux_idx = idx
    idx += 1
    if idx >= len(tags) or tags[idx] not in ("v_trans", "v_intrans"):
        raise GrammarError(f"expected main verb at {idx}")
    main_verb_idx = idx
    v_type = tags[idx]
    idx += 1
    object_node = None
    if v_type == "v_trans":
        object_node, idx = parse_np(tags, idx)
    if idx != len(tags):
        raise GrammarError(f"leftover tags after parsing: {idx} != {len(tags)}")
    return subject_node, main_aux_idx, main_verb_idx, object_node


# ---------------------------------------------------------------------------
# Head map: position -> head position (None for the root). Deliberately
# omits deprel labels and any lexical/token information -- both irrelevant
# to a pure distance calculation (see module docstring).
# ---------------------------------------------------------------------------

def build_head_map(subject_node, main_aux_idx, main_verb_idx, object_node):
    heads = {}

    def emit_np(node, parent_idx):
        noun_idx = node["noun_idx"]
        det_idx = node["det_idx"]
        heads[noun_idx] = parent_idx
        heads[det_idx] = noun_idx

        mod = node["mod"]
        if mod is None:
            return
        if mod["type"] == "pp":
            heads[mod["prep_idx"]] = mod["np"]["noun_idx"]
            emit_np(mod["np"], noun_idx)
        elif mod["type"] == "rc_subj":
            heads[mod["rel_idx"]] = mod["verb_idx"]
            heads[mod["aux_idx"]] = mod["verb_idx"]
            heads[mod["verb_idx"]] = noun_idx
            if mod["obj"] is not None:
                emit_np(mod["obj"], mod["verb_idx"])
        elif mod["type"] == "rc_obj":
            heads[mod["rel_idx"]] = mod["verb_idx"]
            emit_np(mod["embedded"], mod["verb_idx"])
            heads[mod["aux_idx"]] = mod["verb_idx"]
            heads[mod["verb_idx"]] = noun_idx
        else:
            raise GrammarError(f"unknown mod type {mod['type']}")

    heads[main_verb_idx] = None  # root: no head
    emit_np(subject_node, main_verb_idx)
    heads[main_aux_idx] = main_verb_idx
    if object_node is not None:
        emit_np(object_node, main_verb_idx)

    return heads


def extract_declarative_tags(type_field):
    """Pull out just the declarative-order tag sequence from a "type"
    field. Every row in results_per_type.csv is quest-tagged (checked
    before writing this script), but handle a possible "... decl" row too,
    for robustness if this script is ever pointed at a file that has some."""
    if " quest " in type_field:
        return type_field.split(" quest ")[0].split()
    if type_field.endswith(" decl"):
        return type_field[: -len(" decl")].split()
    raise GrammarError(f"can't find declarative half of type field: {type_field!r}")


def average_dependency_distance(type_field):
    """Mean dependency distance (Liu, 2008) of the construction named by
    one results_per_type.csv "type" field, i.e. mean(|id - head|) over
    every non-root position in its declarative-order tree."""
    tags = extract_declarative_tags(type_field)
    subject_node, main_aux_idx, main_verb_idx, object_node = parse_sentence(tags)
    heads = build_head_map(subject_node, main_aux_idx, main_verb_idx, object_node)
    distances = [abs(idx - head) for idx, head in heads.items() if head is not None]
    return sum(distances) / len(distances) if distances else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results_per_type.csv")
    parser.add_argument("--output", default="result_per_type_add.csv")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    in_path = args.input if os.path.isabs(args.input) else os.path.join(here, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(here, args.output)

    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["avg_dependency_distance"]
        rows = list(reader)

    n_ok = 0
    n_err = 0
    out_rows = []
    for row in rows:
        try:
            mdd = average_dependency_distance(row["type"])
            n_ok += 1
        except GrammarError as e:
            # Don't silently drop a row we couldn't parse -- keep it, with
            # an empty value, and report it so it's visible, not hidden.
            mdd = ""
            n_err += 1
            print(f"[warn] could not parse type field, leaving blank: {e}")
        row["avg_dependency_distance"] = mdd
        out_rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Read {len(rows)} rows from {in_path}")
    print(f"Computed avg_dependency_distance for {n_ok} rows ({n_err} could not be parsed)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
