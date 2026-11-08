"""
dependency_distance_from_tokens.py

Standalone version of the dependency-distance calculation used in
add_avg_dependency_distance.py (the script that produced
result_per_type_add.csv / result_per_type_seed228embonly_add.csv), pulled
out of the CSV pipeline so it can be run on a single tag string directly --
no CSV file, no "type"/"acc"/"n" columns, just the tokens themselves.

What counts as "the token string"
--------------------------------------
A whitespace-separated sequence of this project's grammar tags -- the same
vocabulary as hier_gen/cfgs/tag_token_map.txt and the "type" column of
results_per_type.csv: det, n_s/n_p, v_trans/v_intrans, aux_s/aux_p, rel,
prep. For example, the declarative side of a construction:

    det n_p rel aux_p v_intrans aux_p v_intrans

You can also pass a full results_per_type.csv-style "type" field (the
declarative half, " quest ", and the question half all together) -- this
script finds and uses only the declarative half automatically, same as
add_avg_dependency_distance.py does, since that's the half that defines the
construction's own structure (see that script's docstring for why).

Grammar, and why no lexicon/actual words are needed
---------------------------------------------------------
This is the same grammar validated against all 100,000 declaratives in
question.train (see "tree building journal.txt" / "building_trees_journal.
txt") and already used, in this exact trimmed tag-only form, in
add_avg_dependency_distance.py:

    S    -> NP AUX V .              (V = v_intrans)
    S    -> NP AUX V NP .           (V = v_trans)
    NP   -> Det N
    NP   -> Det N RC-subj
    NP   -> Det N RC-obj
    NP   -> Det N PP
    RC-subj -> rel AUX V (NP)?
    RC-obj  -> rel NP AUX V
    PP   -> prep NP

Dependency distance only depends on which grammatical CATEGORY sits at
each position (det/noun/verb/aux/rel/prep) -- never on which specific word
is chosen there (a determiner is always exactly 1 position from the noun
it modifies, whether it's "the" or "my"). So a tag string like the one
above is already everything this calculation needs; no vocabulary/lexicon
lookup, and no actual sentence, is required.

Dependency-distance definition: Liu (2008) -- for a token at position `id`
whose head is at position `head`, distance = |id - head|. The sentence
root (main verb) has no head and is excluded. Average dependency distance
= mean of these distances over every token in the string.

Usage
-----
    python3 dependency_distance_from_tokens.py "det n_p rel aux_p v_intrans aux_p v_intrans"

    # also accepts a full results_per_type.csv "type" field directly:
    python3 dependency_distance_from_tokens.py "det n_p rel aux_p v_intrans aux_p v_intrans quest aux_p det n_p rel aux_p v_intrans v_intrans"

    # or read the string from stdin:
    echo "det n_p aux_p v_intrans" | python3 dependency_distance_from_tokens.py
"""

import sys


class GrammarError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Structural parser -- identical grammar/branching logic to
# grammar_ud_converter.py / add_avg_dependency_distance.py, operating
# directly on tag strings (see module docstring for why no lexicon or
# actual tokens are needed).
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
    """Parse a declarative-order tag sequence (no trailing period tag).
    Returns (subject_node, main_aux_idx, main_verb_idx, object_node_or_None)."""
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
# Head map: position -> head position (None for the root). No deprel
# labels, no lexical/token information -- both irrelevant to distance.
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


def extract_declarative_tags(token_string):
    """If given a full results_per_type.csv-style "type" field (declarative
    half + " quest "/" decl" + question half), return just the declarative
    half's tags. If given a bare declarative tag string, return it as-is."""
    if " quest " in token_string:
        return token_string.split(" quest ")[0].split()
    if token_string.endswith(" decl"):
        return token_string[: -len(" decl")].split()
    return token_string.split()


def dependency_distance(token_string):
    """
    Returns (per_token_distances, average) for the declarative-order tag
    string `token_string`:
      per_token_distances: list of (position, tag, head_position_or_None,
                                     distance_or_None) for every token
      average: mean dependency distance (Liu, 2008) over all non-root
                tokens
    """
    tags = extract_declarative_tags(token_string)
    subject_node, main_aux_idx, main_verb_idx, object_node = parse_sentence(tags)
    heads = build_head_map(subject_node, main_aux_idx, main_verb_idx, object_node)

    per_token = []
    distances = []
    for i, tag in enumerate(tags):
        head = heads[i]
        dist = None if head is None else abs(i - head)
        per_token.append((i, tag, head, dist))
        if dist is not None:
            distances.append(dist)

    average = sum(distances) / len(distances) if distances else 0.0
    return per_token, average


def main():
    if len(sys.argv) > 1:
        token_string = " ".join(sys.argv[1:])
    else:
        token_string = sys.stdin.read()
    token_string = token_string.strip()
    if not token_string:
        print("usage: python3 dependency_distance_from_tokens.py '<tag string>'", file=sys.stderr)
        sys.exit(1)

    try:
        per_token, average = dependency_distance(token_string)
    except GrammarError as e:
        print(f"Could not parse this tag string as a sentence: {e}", file=sys.stderr)
        sys.exit(1)

    print("pos  tag        head  distance")
    for i, tag, head, dist in per_token:
        head_str = "ROOT" if head is None else str(head)
        dist_str = "-" if dist is None else str(dist)
        print(f"{i:>3}  {tag:<10} {head_str:<5} {dist_str}")
    print()
    print(f"average dependency distance: {average:.4f}")


if __name__ == "__main__":
    main()
