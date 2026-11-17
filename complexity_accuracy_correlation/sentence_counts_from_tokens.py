"""
sentence_counts_from_tokens.py

Four count-based complexity metrics for a construction, computed from the
same tag string format as dependency_distance_from_tokens.py (the grammar
tags from hier_gen/cfgs/tag_token_map.txt: det, n_s/n_p, v_trans/v_intrans,
aux_s/aux_p, rel, prep):

  Clauses per Sentence  - main clause + every relative clause (subject- or
                          object-extracted) anywhere in the tree.
  Verbs per Sentence    - number of lexical verbs (v_trans / v_intrans).
                          Auxiliaries (aux_s / aux_p) are not counted: they
                          are function words that depend on a verb, not
                          verbs heading their own predicate.
  NPs per Sentence      - every noun phrase, including NPs nested inside
                          PPs and relative clauses. In this grammar each NP
                          is "det N (modifier)", so this equals the noun count.
  Verb/Noun Ratio       - verbs / nouns (nouns = n_s / n_p).

Each template is exactly one sentence, so "per sentence" is the count for
that template. As with the other metrics, only the declarative half of a
results_per_type.csv "type" field is used, and the actual words do not
matter, because the tree depends only on the tag sequence.

Clauses and NPs are counted from the parse tree (parse_sentence() imported
from dependency_distance_from_tokens.py), not by matching tags, so they
count the same structure the other metrics are computed on. Every clause in
this grammar has exactly one lexical verb, so Clauses per Sentence and Verbs
per Sentence always come out equal here. They are still computed
independently, so a mismatch would expose a parsing problem.

Usage
-----
    python3 sentence_counts_from_tokens.py "det n_p rel aux_p v_intrans aux_p v_intrans"

    # a full results_per_type.csv "type" field also works:
    python3 sentence_counts_from_tokens.py "det n_p rel aux_p v_intrans aux_p v_intrans quest aux_p det n_p rel aux_p v_intrans v_intrans"

    # or read the string from stdin:
    echo "det n_p aux_p v_intrans" | python3 sentence_counts_from_tokens.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dependency_distance_from_tokens import (  # noqa: E402
    GrammarError,
    extract_declarative_tags,
    parse_sentence,
)

VERB_TAGS = ("v_trans", "v_intrans")


def _walk_np(node, counts):
    """Count this NP and everything embedded in it."""
    counts["nps"] += 1
    mod = node["mod"]
    if mod is None:
        return
    if mod["type"] == "pp":
        _walk_np(mod["np"], counts)
    elif mod["type"] == "rc_subj":
        counts["clauses"] += 1
        if mod["obj"] is not None:
            _walk_np(mod["obj"], counts)
    elif mod["type"] == "rc_obj":
        counts["clauses"] += 1
        _walk_np(mod["embedded"], counts)
    else:
        raise GrammarError(f"unknown mod type {mod['type']}")


def sentence_counts(token_string):
    """
    Returns a dict with:
      clauses, verbs, nps, nouns    (integer counts)
      clauses_per_sentence, verbs_per_sentence, nps_per_sentence,
      verb_noun_ratio               (the four metrics)
    """
    tags = extract_declarative_tags(token_string)
    subject_node, _main_aux_idx, _main_verb_idx, object_node = parse_sentence(tags)

    counts = {"clauses": 1, "nps": 0}  # 1 = the main clause
    _walk_np(subject_node, counts)
    if object_node is not None:
        _walk_np(object_node, counts)

    verbs = sum(1 for t in tags if t in VERB_TAGS)
    nouns = sum(1 for t in tags if t.startswith("n_"))

    return {
        "clauses": counts["clauses"],
        "verbs": verbs,
        "nps": counts["nps"],
        "nouns": nouns,
        "clauses_per_sentence": counts["clauses"],
        "verbs_per_sentence": verbs,
        "nps_per_sentence": counts["nps"],
        "verb_noun_ratio": verbs / nouns if nouns else 0.0,
    }


def clauses_per_sentence(token_string):
    return sentence_counts(token_string)["clauses_per_sentence"]


def verbs_per_sentence(token_string):
    return sentence_counts(token_string)["verbs_per_sentence"]


def nps_per_sentence(token_string):
    return sentence_counts(token_string)["nps_per_sentence"]


def verb_noun_ratio(token_string):
    return sentence_counts(token_string)["verb_noun_ratio"]


def main():
    if len(sys.argv) > 1:
        token_string = " ".join(sys.argv[1:])
    else:
        token_string = sys.stdin.read()
    token_string = token_string.strip()
    if not token_string:
        print("usage: python3 sentence_counts_from_tokens.py '<tag string>'", file=sys.stderr)
        sys.exit(1)

    try:
        c = sentence_counts(token_string)
    except GrammarError as e:
        print(f"Could not parse this tag string as a sentence: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"clauses per sentence: {c['clauses_per_sentence']}")
    print(f"verbs per sentence:   {c['verbs_per_sentence']}")
    print(f"NPs per sentence:     {c['nps_per_sentence']}")
    print(f"verb/noun ratio:      {c['verb_noun_ratio']:.4f}  ({c['verbs']} verbs / {c['nouns']} nouns)")


if __name__ == "__main__":
    main()
