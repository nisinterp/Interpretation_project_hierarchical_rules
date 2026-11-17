"""
branching_factor_from_tokens.py

Average branching factor of a construction, computed from the same tag
string format as dependency_distance_from_tokens.py (the grammar tags from
hier_gen/cfgs/tag_token_map.txt: det, n_s/n_p, v_trans/v_intrans,
aux_s/aux_p, rel, prep).

Definition
----------
Average number of dependents per head. A higher value means more
modification is packed onto each head.
  1. Parse the tag string into a dependency tree.
  2. For each head (a token with at least one dependent), count its
     dependents.
  3. Average branching factor = mean(dependents per head).

Tokens with no dependents (leaves such as det, rel, aux) are not heads, so
they are left out of the mean. If they were included, every tree would score
(n - 1) / n, because a tree with n tokens always has n - 1 dependency arcs,
and the metric would only reflect sentence length.

The parse and the head assignments are the same ones the dependency distance
scripts use: parse_sentence() and build_head_map() are imported from
dependency_distance_from_tokens.py, so the two metrics always describe the
same tree. As with dependency distance, only the declarative half of a
results_per_type.csv "type" field is used, and the actual words do not
matter, because the tree shape depends only on the tag sequence.

Usage
-----
    python3 branching_factor_from_tokens.py "det n_p rel aux_p v_intrans aux_p v_intrans"

    # a full results_per_type.csv "type" field also works:
    python3 branching_factor_from_tokens.py "det n_p rel aux_p v_intrans aux_p v_intrans quest aux_p det n_p rel aux_p v_intrans v_intrans"

    # or read the string from stdin:
    echo "det n_p aux_p v_intrans" | python3 branching_factor_from_tokens.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dependency_distance_from_tokens import (  # noqa: E402
    GrammarError,
    build_head_map,
    extract_declarative_tags,
    parse_sentence,
)


def branching_factor(token_string):
    """
    Returns (tags, dependents_per_head, average) for `token_string`:
      tags: the declarative-order tag list that was parsed
      dependents_per_head: dict head_position -> number of dependents, for
                           every token that has at least one dependent
      average: mean number of dependents per head
    """
    tags = extract_declarative_tags(token_string)
    subject_node, main_aux_idx, main_verb_idx, object_node = parse_sentence(tags)
    heads = build_head_map(subject_node, main_aux_idx, main_verb_idx, object_node)

    dependents_per_head = Counter(h for h in heads.values() if h is not None)
    average = (
        sum(dependents_per_head.values()) / len(dependents_per_head)
        if dependents_per_head
        else 0.0
    )
    return tags, dict(dependents_per_head), average


def average_branching_factor(token_string):
    """Just the number, e.g. for adding a column to results_per_type.csv."""
    return branching_factor(token_string)[2]


def main():
    if len(sys.argv) > 1:
        token_string = " ".join(sys.argv[1:])
    else:
        token_string = sys.stdin.read()
    token_string = token_string.strip()
    if not token_string:
        print("usage: python3 branching_factor_from_tokens.py '<tag string>'", file=sys.stderr)
        sys.exit(1)

    try:
        tags, dependents_per_head, average = branching_factor(token_string)
    except GrammarError as e:
        print(f"Could not parse this tag string as a sentence: {e}", file=sys.stderr)
        sys.exit(1)

    print("pos  tag        dependents")
    for i, tag in enumerate(tags):
        n = dependents_per_head.get(i, 0)
        print(f"{i:>3}  {tag:<10} {n if n else '-'}")
    print()
    print(f"heads: {len(dependents_per_head)}, arcs: {sum(dependents_per_head.values())}")
    print(f"average branching factor: {average:.4f}")


if __name__ == "__main__":
    main()
