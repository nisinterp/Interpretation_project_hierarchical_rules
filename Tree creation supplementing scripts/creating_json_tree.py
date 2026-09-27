"""
creating_json_tree.py

Rewrite of "First test.py" that serializes the parsed Universal Dependencies
tree as plain JSON instead of CoNLL-U + an ASCII tree, so that the result
can be loaded and processed programmatically -- specifically, by
compute_dependency_distance.py in this same folder, which computes the
sentence's Mean Dependency Distance (MDD) from it.

Same sentence, same Stanza pipeline and the same tokenization choices as
"First test.py" (do NOT pre-tokenize -- see "tree building journal.txt" for
why that broke contraction splitting on "doesn't"). The only thing that
changed is the output format.

JSON schema written to example_json_tree.txt
-----------------------------------------------
{
  "text": "<the full sentence>",
  "tokens": [                     // one entry per SYNTACTIC word (i.e. per
                                   // CoNLL-U word line, NOT per surface
                                   // token -- "doesn't" contributes two
                                   // entries here, "does" and "n't", each
                                   // with its own head/deprel, exactly as
                                   // UD treebanks are annotated)
    {
      "id": 1,                    // 1-based position in the sentence --
                                   // this is what dependency distance is
                                   // measured over
      "text": "your",
      "lemma": "your",
      "upos": "PRON",             // universal POS tag
      "xpos": "PRP$",             // language-specific (Penn Treebank) tag
      "feats": "Case=Gen|...",    // morphological features, or null
      "head": 2,                  // id of the governing word; 0 = root
      "deprel": "nmod:poss"
    },
    ...
  ],
  "multiword_tokens": [           // surface contractions that were split
                                   // into >1 syntactic word above, kept
                                   // here only for reference -- they are
                                   // NOT separate entries in "tokens" and
                                   // must stay out of any distance
                                   // calculation (they have no head/id of
                                   // their own in the UD scheme)
    {"id": "8-9", "text": "doesn't"}
  ]
}

Why this shape: dependency distance (Liu, 2008) is defined per dependency
relation as |position(word) - position(its head)|, using the words' linear
positions in the sentence. Flattening the tree to a list of
{id, head, ...} records -- rather than nesting children inside parents --
makes every relation directly readable as one (id, head) pair, which is
exactly what a distance calculation needs; nesting would just have to be
flattened again on the consumer side.

Setup: see requirements.txt in this folder (same environment as
"First test.py" -- Stanza + CPU-only torch).
"""

import json
import os

import stanza

SENTENCE = (
    "your peacock that your vultures do admire doesn't high_five "
    "my salamanders who do accept the peacocks ."
)

OUTPUT_FILENAME = "example_json_tree.txt"


def load_pipeline():
    """Same pipeline/tokenization choice as First test.py -- see its
    load_pipeline() docstring for why we let Stanza tokenize the raw
    string itself instead of pre-splitting it."""
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        verbose=False,
    )


def sentence_to_json(sentence):
    """Flatten a Stanza Sentence into the JSON-serializable dict described
    in the module docstring."""
    tokens = [
        {
            "id": w.id,
            "text": w.text,
            "lemma": w.lemma,
            "upos": w.upos,
            "xpos": w.xpos,
            "feats": w.feats,
            "head": w.head,
            "deprel": w.deprel,
        }
        for w in sentence.words
    ]

    multiword_tokens = [
        {"id": f"{tok.words[0].id}-{tok.words[-1].id}", "text": tok.text}
        for tok in sentence.tokens
        if len(tok.words) > 1
    ]

    return {
        "text": sentence.text,
        "tokens": tokens,
        "multiword_tokens": multiword_tokens,
    }


def main():
    nlp = load_pipeline()
    doc = nlp(SENTENCE)
    sentence = doc.sentences[0]
    data = sentence_to_json(sentence)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILENAME)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"Parsed sentence:\n  {SENTENCE}\n")
    print(f"Saved JSON dependency tree ({len(data['tokens'])} words, "
          f"{len(data['multiword_tokens'])} multiword token(s)) to: {out_path}")


if __name__ == "__main__":
    main()
