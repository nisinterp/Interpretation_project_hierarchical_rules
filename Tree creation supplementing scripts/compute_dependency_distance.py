"""
compute_dependency_distance.py

Loads the JSON dependency tree written by creating_json_tree.py
(example_json_tree.txt by default), and computes the sentence's
Mean Dependency Distance (MDD).

Dependency distance, definition
--------------------------------
For a dependency relation between a word at linear position `id` and its
governor at position `head`, the dependency distance of that relation is

    DD = |id - head|

(Liu, 2008, "Dependency Distance as a Metric of Language Comprehension
Difficulty"; this is the standard definition used throughout the
quantitative/dependency-syntax literature, including work on UD treebanks.)
The root word has no governor inside the sentence (head = 0 in the UD
scheme) and is excluded from the count, so a sentence with n words
contributes exactly n-1 distances (one per real dependency relation).

Mean Dependency Distance (MDD) is simply the average of those distances
over the whole sentence:

    MDD = (1 / (n - 1)) * sum(|id_i - head_i|)   over all non-root words i

Punctuation: some studies exclude punctuation tokens from MDD on the
grounds that they aren't processed like content/function words. There is
no single agreed convention, so this script reports MDD both ways
(all tokens, and punctuation excluded) rather than silently picking one.

Note on multiword tokens: the "multiword_tokens" entries in the JSON file
(e.g. "doesn't" as the surface form of "does" + "n't") are metadata only --
they don't have their own id/head and must NOT be included in the distance
calculation. This script only ever reads the "tokens" list, so they are
naturally excluded; this note just makes that explicit.

This script has no dependency on stanza/torch -- it only needs the
standard library (`json`), so it will run in any Python 3 environment,
including outside the venv used for creating_json_tree.py.

Usage
-----
    python3 compute_dependency_distance.py [path/to/tree.json]

With no argument, it defaults to "example_json_tree.txt" in this folder.
"""

import json
import os
import sys

DEFAULT_INPUT = "example_json_tree.txt"


def load_tree(path):
    with open(path) as f:
        return json.load(f)


def dependency_distances(tokens, include_punct=True):
    """Return the list of |id - head| distances, one per non-root word.
    Multiword-token entries are never in `tokens` to begin with (see
    module docstring), so nothing needs to be filtered out for those."""
    distances = []
    for tok in tokens:
        if tok["head"] == 0:
            continue  # root: no governor inside the sentence
        if not include_punct and tok["upos"] == "PUNCT":
            continue
        distances.append(abs(tok["id"] - tok["head"]))
    return distances


def mean_dependency_distance(distances):
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def main():
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    else:
        in_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_INPUT)

    data = load_tree(in_path)  # "turns [the file] into JSON": json.load parses
                                # the saved text back into a Python dict/list
                                # structure (the in-memory JSON object) to work with.
    tokens = data["tokens"]

    distances_all = dependency_distances(tokens, include_punct=True)
    distances_no_punct = dependency_distances(tokens, include_punct=False)

    mdd_all = mean_dependency_distance(distances_all)
    mdd_no_punct = mean_dependency_distance(distances_no_punct)

    print(f"Sentence: {data['text']}\n")
    print("Per-word dependency distance:")
    id_to_text = {tok["id"]: tok["text"] for tok in tokens}
    for tok in tokens:
        if tok["head"] == 0:
            print(f"  {tok['id']:>2}  {tok['text']:<12} head=ROOT      distance=  -")
            continue
        d = abs(tok["id"] - tok["head"])
        head_text = id_to_text.get(tok["head"], "?")
        print(f"  {tok['id']:>2}  {tok['text']:<12} head={tok['head']:>2} ({head_text:<10}) distance={d}")

    print()
    print(f"Words (incl. punct.): {len(tokens)}   Relations counted: {len(distances_all)}")
    print(f"Mean Dependency Distance (MDD), all tokens incl. punctuation: {mdd_all:.4f}")
    print(f"Mean Dependency Distance (MDD), punctuation excluded:        {mdd_no_punct:.4f}")

    # Also emit the result as JSON on stdout, in case this script's output
    # is meant to be piped into something else rather than just read.
    result = {
        "text": data["text"],
        "n_words": len(tokens),
        "n_relations": len(distances_all),
        "mdd_all_tokens": mdd_all,
        "mdd_excluding_punct": mdd_no_punct,
    }
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
