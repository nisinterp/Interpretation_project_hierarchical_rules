"""
First test.py

Parses one declarative sentence from the paper's question-formation dataset
(the first example in `hier_gen/data_utils/question_formation_data/question.train`)
into a Universal Dependencies (UD) tree, using Stanza's English pipeline.

Why Stanza and not spaCy
-------------------------
Stanza's pipelines are trained directly on UD treebank data and output
genuine, current UD v2 relations (e.g. `obj`, `nmod:poss`, `advmod` for
negation). spaCy's bundled `en_core_web_sm`, by contrast, uses an older
ClearNLP/OntoNotes-derived scheme that only *resembles* UD (it still uses
pre-UDv2 labels like `dobj`, `poss`, `neg`). When both were tried on this
exact sentence during development, spaCy also mis-parsed the non-negated
"do"-support construction ("your vultures do admire", not natural affirmative
English) by treating "admire" as a noun object of "do" rather than "do" as
an auxiliary of "admire" -- Stanza got this right. See "tree building
journal.txt" in this folder for the full comparison and other issues hit
along the way.

IMPORTANT CAVEAT for this project
----------------------------------
Stanza's English model is trained on real English treebanks (UD English-EWT
etc.), not on the paper's synthetic mini-language. The mini-language's
grammar uses unconditional periphrastic "do"-support ("do admire", not just
"doesn't admire"), which almost never occurs in natural affirmative English.
The parser handled *this particular* sentence correctly, but it is being run
out-of-distribution, so do not treat its output as ground truth without
spot-checking further examples -- see the journal for details.

Output
------
Writes "First_test_tree" (no extension) in this same folder, containing:
  1. The parse in standard CoNLL-U format (the format the UD project itself
     uses to distribute treebanks).
  2. A human-readable indented dependency tree for quick visual inspection.

Setup
-----
    python3 -m venv .venv
    source .venv/bin/activate          # on Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt
    python "First test.py"

On first run, Stanza automatically downloads its English UD pipeline
(tokenize, mwt, pos, lemma, depparse processors) to ~/stanza_resources/
(internet access required once; cached afterwards).
"""

import os
import stanza
from stanza.utils.conll import CoNLL


# The first example sentence extracted from question.train (see the earlier
# extraction step: it is the first line of the file, a "decl" -- identity --
# pair, so the sentence appears verbatim as both the input and the output).
SENTENCE = (
    "your peacock that your vultures do admire doesn't high_five "
    "my salamanders who do accept the peacocks ."
)

OUTPUT_FILENAME = "First_test_tree"


def load_pipeline():
    """
    Load Stanza's English UD pipeline, downloading it on first use.

    We deliberately do NOT pre-tokenize the sentence ourselves (e.g. by
    splitting on whitespace) and hand Stanza a list of tokens. Early testing
    showed that forcing pre-tokenization (`tokenize_pretokenized=True`)
    suppresses Stanza's built-in contraction-splitting step, which UD
    English treebanks rely on (e.g. "doesn't" -> two syntactic words "does"
    + "n't"). Letting Stanza run its own tokenizer/MWT-splitter on the raw
    string handles both that AND the underscore-joined toy verb
    ("high_five", an artifact of this paper's vocabulary) correctly at the
    same time -- see the journal for the failed alternative.
    """
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        verbose=False,
    )


def render_tree(sentence):
    """
    Build a simple, human-readable indented rendering of a Stanza
    `Sentence`'s dependency tree, e.g.:

        high_five [root]
        |-- peacock [nsubj]
        |   |-- your [nmod:poss]
        |   `-- admire [acl:relcl]
        ...
    """
    words = sentence.words
    children_of = {0: []}
    for w in words:
        children_of.setdefault(w.head, []).append(w)
    for kids in children_of.values():
        kids.sort(key=lambda w: w.id)

    lines = []

    def walk(word_id, prefix, is_last):
        word = words[word_id - 1]
        connector = "`-- " if is_last else "|-- "
        lines.append(f"{prefix}{connector}{word.text} [{word.deprel}]")
        next_prefix = prefix + ("    " if is_last else "|   ")
        kids = children_of.get(word.id, [])
        for i, kid in enumerate(kids):
            walk(kid.id, next_prefix, i == len(kids) - 1)

    roots = children_of.get(0, [])
    for i, root in enumerate(roots):
        lines.append(f"{root.text} [root]")
        kids = children_of.get(root.id, [])
        for j, kid in enumerate(kids):
            walk(kid.id, "", j == len(kids) - 1)

    return "\n".join(lines)


def main():
    nlp = load_pipeline()
    doc = nlp(SENTENCE)
    sentence = doc.sentences[0]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILENAME)

    # CoNLL.write_doc2conll produces a standards-compliant CoNLL-U file
    # (including the "# text = ..." / "# sent_id = ..." comment header,
    # morphological FEATS, and the multi-word-token range line for
    # "doesn't" -> "does" + "n't"), so we let Stanza write that part
    # directly rather than hand-assembling it.
    CoNLL.write_doc2conll(doc, out_path)

    with open(out_path, "a") as f:
        f.write("\n# source: hier_gen/data_utils/question_formation_data/question.train, line 1\n\n")
        f.write("# ==== Human-readable dependency tree ====\n")
        f.write(render_tree(sentence))
        f.write("\n")

    print(f"Parsed sentence:\n  {SENTENCE}\n")
    print("Human-readable tree:")
    print(render_tree(sentence))
    print(f"\nSaved CoNLL-U + tree to: {out_path}")


if __name__ == "__main__":
    main()
