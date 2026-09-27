"""
grammar_ud_converter.py

A deterministic, rule-based Universal Dependencies converter for the
question-formation grammar used in Qin, Saphra & Alvarez-Melis (EMNLP 2025)
and shipped in hier_gen/data_utils/question_formation_data/question.train.

Why rule-based instead of Stanza
-----------------------------------
Building question_train_with_trees.tsv means parsing 200,000 sentences
(a declarative + a question form for all 100,000 rows). Benchmarked on this
machine (4 CPU cores, no GPU), Stanza parses ~4 sentences/second, which
would take ~14 hours for the full file -- confirmed by an actual timed run,
see "building_trees_journal.txt".

But there is a better option specific to this dataset: every sentence in
question.train is generated from a small, fully known, finite CFG (the same
grammar reverse-engineered for recursion.py and documented in
cfgs/tag_token_map.txt). That means we do not need a statistical parser to
*guess* the structure -- we can derive the exact, correct UD tree from the
grammar rules directly, deterministically, with no ambiguity and no
inference cost. Two concrete pieces of evidence this is not just faster but
more *correct* for this dataset (both found by spot-checking Stanza on
constructed examples before writing this module -- see the journal):
  - Stanza mis-parsed a bare "do sleep" (no negation) as "do" [VERB, root
    of the relative clause] + "sleep" [NOUN, object of "do"], because
    "sleep" is a common noun in real English and the construction "do V"
    without negation/emphasis barely occurs in real affirmative English.
    The grammar-based converter has no such ambiguity: it knows "sleep" is
    always v_intrans and "do" is always aux, by construction.
  - Stanza attached object-position PPs ("... entertain your zebras behind
    some unicorns") as an oblique argument of the main verb (`obl`) rather
    than as a modifier of the preceding object noun (`nmod`), even though
    the CFG's own object-NP-then-PP production intends the PP to modify
    the noun. Real-English attachment statistics favor the verb-oblique
    reading; the grammar tells us the intended reading directly, so this
    converter always emits `nmod` for these PPs, on both subject and
    object position, for consistency with the CFG structure that actually
    generated the sentence.

Grammar covered (validated against all 100,000 lines of question.train --
see the journal for the validation run and its 100% coverage result)
------------------------------------------------------------------------
    S    -> NP AUX V .              (V = v_intrans)
    S    -> NP AUX V NP .           (V = v_trans)
    NP   -> Det N
    NP   -> Det N RC-subj           relative clause, subject-extracted
    NP   -> Det N RC-obj            relative clause, object-extracted
    NP   -> Det N PP
    RC-subj -> rel AUX V            V = v_intrans
    RC-subj -> rel AUX V NP         V = v_trans
    RC-obj  -> rel NP AUX V         V always v_trans (gap = object)
    PP   -> prep NP

Question formation: front the AUX that immediately follows the fully-built
subject NP (this is the paper's "hierarchical" rule -- see the top-level
project's earlier work in recursion.py for the same logic applied
generatively rather than by parsing existing sentences).

UD labels used, and where each came from
-------------------------------------------
Cross-checked against Stanza's own output on hand-picked, unambiguous
example sentences from this grammar (documented in the journal) wherever
the construction wasn't itself ambiguous for a real-English parser:
  det (articles the/some) -> upos DET, xpos DT, deprel det
  det (possessives my/your/our/her) -> upos PRON, xpos PRP$, deprel
      nmod:poss
  noun -> upos NOUN, xpos NN/NNS, feats Number=Sing/Plur
  bare verb (v_trans/v_intrans) -> upos VERB, xpos VB, feats VerbForm=Inf
  aux do/does(+n't) -> upos AUX, xpos VBP/VBZ, deprel aux; "n't" split out
      as its own PART/RB token, deprel advmod, feats Polarity=Neg -- this
      MWT split mirrors the UD English convention for contractions (see
      "First test.py"/"creating_json_tree.py": Stanza does the same split)
  rel who -> upos PRON, xpos WP; rel that -> upos PRON, xpos WDT; deprel is
      nsubj if the RC is subject-extracted, obj if object-extracted
  prep -> upos ADP, xpos IN, deprel case (head = the NP's own head noun)
  relative clause verb -> deprel acl:relcl, head = the noun it modifies
  main verb -> deprel root, head = 0
  period/question mark -> upos PUNCT, xpos ".", deprel punct

Output shape matches creating_json_tree.py's schema exactly ({"text":...,
"tokens":[...], "multiword_tokens":[...]}), so the same
compute_dependency_distance.py can be pointed at trees produced here.
"""

import os
from collections import defaultdict

CFG_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "hier_gen", "cfgs"
)

# ---------------------------------------------------------------------------
# Lexical lookup tables. The CFG only distinguishes tokens by their coarse
# tag (det/n_s/n_p/v_trans/.../aux_s/aux_p/rel/prep); a genuine UD parse
# needs finer-grained facts about each *lexeme* (e.g. "my" is a possessive
# pronoun, not an article) that the CFG tag alone doesn't carry. These were
# fixed by inspecting Stanza's own analysis of these exact words in
# unambiguous contexts (see the journal) and are standard UD English
# annotation choices, not specific to this project.
# ---------------------------------------------------------------------------

DET_LEXEMES = {
    "the": dict(upos="DET", xpos="DT", feats="Definite=Def|PronType=Art", possessive=False),
    "some": dict(upos="DET", xpos="DT", feats="PronType=Ind", possessive=False),
    "my": dict(upos="PRON", xpos="PRP$", feats="Case=Gen|Number=Sing|Person=1|Poss=Yes|PronType=Prs", possessive=True),
    "your": dict(upos="PRON", xpos="PRP$", feats="Case=Gen|Person=2|Poss=Yes|PronType=Prs", possessive=True),
    "our": dict(upos="PRON", xpos="PRP$", feats="Case=Gen|Number=Plur|Person=1|Poss=Yes|PronType=Prs", possessive=True),
    "her": dict(upos="PRON", xpos="PRP$", feats="Case=Gen|Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs", possessive=True),
}

REL_LEXEMES = {
    "who": dict(upos="PRON", xpos="WP", feats="PronType=Rel"),
    "that": dict(upos="PRON", xpos="WDT", feats="PronType=Rel"),
}

# aux surface form -> (rendered aux text, is-negated, xpos, feats)
AUX_LEXEMES = {
    "do": dict(aux_text="do", neg=False, xpos="VBP", feats="Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin"),
    "does": dict(aux_text="does", neg=False, xpos="VBZ", feats="Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"),
    "don't": dict(aux_text="do", neg=True, xpos="VBP", feats="Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin"),
    "doesn't": dict(aux_text="does", neg=True, xpos="VBZ", feats="Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"),
}


def load_grammar(cfg_dir=CFG_DIR_DEFAULT):
    """Load tag<->token vocabulary and a plural->singular noun lemma map
    from the study's own cfgs/tag_token_map.txt (same file used by
    recursion.py and generate_qf_data_from_cfg.py)."""
    type_to_token_map = defaultdict(list)
    token2tag = {}
    with open(os.path.join(cfg_dir, "tag_token_map.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag, token = line.split("\t")
            type_to_token_map[tag].append(token)
            if token not in token2tag:
                token2tag[token] = tag

    # n_s and n_p are listed in matching order (see tree building journal /
    # tag_token_map.txt: each n_p entry is the plural of the n_s entry at
    # the same position), so zip them for a plural -> singular lemma map.
    plural_to_singular = dict(zip(type_to_token_map["n_p"], type_to_token_map["n_s"]))

    return token2tag, plural_to_singular


# ---------------------------------------------------------------------------
# Parsing: recursive descent over (token, tag) pairs, following exactly the
# grammar in the module docstring. Returns a tree of NP nodes referencing
# ORIGINAL (declarative-order) token positions; nothing here depends on
# surface word order beyond "subject NP, then AUX, then V, then object NP?".
# ---------------------------------------------------------------------------

class GrammarError(ValueError):
    pass


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
            # object-extracted RC: rel NP AUX V   (gap = object, no trailing NP)
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
    """Parse a full declarative-order token sequence (no trailing period).
    Returns (subject_node, main_aux_idx, main_verb_idx, object_node_or_None).
    Raises GrammarError if the sentence doesn't match the grammar, or if
    tokens are left over at the end."""
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
        raise GrammarError(f"leftover tokens after parsing: {idx} != {len(tags)}")
    return subject_node, main_aux_idx, main_verb_idx, object_node


# ---------------------------------------------------------------------------
# UD relation assignment: walk the parsed tree and record, for every
# original token index, (head_original_idx_or_None, deprel). None marks the
# sentence root. Everything here operates purely on original (declarative)
# indices -- linearization/renumbering happens later in render_ud, so this
# step is agnostic to whether the final surface form is declarative or
# question order.
# ---------------------------------------------------------------------------

def build_relations(tokens, subject_node, main_aux_idx, main_verb_idx, object_node):
    relations = {}

    def emit_np(node, parent_idx, deprel):
        noun_idx = node["noun_idx"]
        det_idx = node["det_idx"]
        relations[noun_idx] = (parent_idx, deprel)
        det_text = tokens[det_idx]
        det_deprel = "nmod:poss" if DET_LEXEMES[det_text]["possessive"] else "det"
        relations[det_idx] = (noun_idx, det_deprel)

        mod = node["mod"]
        if mod is None:
            return
        if mod["type"] == "pp":
            relations[mod["prep_idx"]] = (mod["np"]["noun_idx"], "case")
            emit_np(mod["np"], noun_idx, "nmod")
        elif mod["type"] == "rc_subj":
            relations[mod["rel_idx"]] = (mod["verb_idx"], "nsubj")
            relations[mod["aux_idx"]] = (mod["verb_idx"], "aux")
            relations[mod["verb_idx"]] = (noun_idx, "acl:relcl")
            if mod["obj"] is not None:
                emit_np(mod["obj"], mod["verb_idx"], "obj")
        elif mod["type"] == "rc_obj":
            relations[mod["rel_idx"]] = (mod["verb_idx"], "obj")
            emit_np(mod["embedded"], mod["verb_idx"], "nsubj")
            relations[mod["aux_idx"]] = (mod["verb_idx"], "aux")
            relations[mod["verb_idx"]] = (noun_idx, "acl:relcl")
        else:
            raise GrammarError(f"unknown mod type {mod['type']}")

    relations[main_verb_idx] = (None, "root")
    emit_np(subject_node, main_verb_idx, "nsubj")
    relations[main_aux_idx] = (main_verb_idx, "aux")
    if object_node is not None:
        emit_np(object_node, main_verb_idx, "obj")

    return relations


# ---------------------------------------------------------------------------
# Rendering: turn (tokens, tags, relations) into the final JSON tree, in a
# given surface order (declarative order, or question order with the main
# aux fronted). Handles the aux/n't multiword split here, since that's a
# surface-realization detail, not a structural one.
# ---------------------------------------------------------------------------

def _lexical_info(token, tag, plural_to_singular):
    if tag == "det":
        info = DET_LEXEMES[token]
        return dict(lemma=token, upos=info["upos"], xpos=info["xpos"], feats=info["feats"])
    if tag == "n_s":
        return dict(lemma=token, upos="NOUN", xpos="NN", feats="Number=Sing")
    if tag == "n_p":
        return dict(lemma=plural_to_singular[token], upos="NOUN", xpos="NNS", feats="Number=Plur")
    if tag in ("v_trans", "v_intrans"):
        return dict(lemma=token, upos="VERB", xpos="VB", feats="VerbForm=Inf")
    if tag == "rel":
        info = REL_LEXEMES[token]
        return dict(lemma=token, upos=info["upos"], xpos=info["xpos"], feats=info["feats"])
    if tag == "prep":
        return dict(lemma=token, upos="ADP", xpos="IN", feats=None)
    raise GrammarError(f"no lexical info rule for tag {tag!r} (token {token!r})")


def render_ud(tokens, tags, relations, main_verb_idx, order, final_punct, lexicon_fn):
    """
    order: list of original indices, one per token, in the desired surface
    order (declarative order = list(range(len(tokens))); question order =
    [main_aux_idx] + everything else in original relative order).
    final_punct: "." or "?"
    lexicon_fn: (token, tag) -> {"lemma","upos","xpos","feats"} for
        non-auxiliary tokens (nouns, dets, verbs, rel-pronouns, preps).
    """
    new_id_of = {}
    plan = []  # list of (kind, original_idx, new_id) where kind in {"plain","aux","neg"}

    next_id = 1
    for i in order:
        tag = tags[i]
        tok = tokens[i]
        if tag.startswith("aux_") and AUX_LEXEMES[tok]["neg"]:
            new_id_of[i] = next_id
            plan.append(("aux", i, next_id))
            next_id += 1
            plan.append(("neg", i, next_id))
            next_id += 1
        else:
            new_id_of[i] = next_id
            plan.append(("plain", i, next_id))
            next_id += 1

    punct_id = next_id

    out_tokens = []
    multiword_tokens = []
    surface_words = []

    for kind, i, new_id in plan:
        tok = tokens[i]
        tag = tags[i]
        head_i, deprel = relations[i]
        head_new_id = 0 if head_i is None else new_id_of[head_i]

        if kind == "aux":
            info = AUX_LEXEMES[tok]
            out_tokens.append(dict(
                id=new_id, text=info["aux_text"], lemma="do",
                upos="AUX", xpos=info["xpos"], feats=info["feats"],
                head=head_new_id, deprel=deprel,
            ))
            surface_words.append(tok)  # the contracted surface form, e.g. "doesn't"
        elif kind == "neg":
            info = AUX_LEXEMES[tok]
            out_tokens.append(dict(
                id=new_id, text="n't", lemma="not",
                upos="PART", xpos="RB", feats="Polarity=Neg",
                head=head_new_id, deprel="advmod",
            ))
            # no separate surface word: "n't" was already included with the
            # aux half above, as part of the undivided contraction "doesn't"
            aux_new_id = new_id - 1
            multiword_tokens.append({"id": f"{aux_new_id}-{new_id}", "text": tok})
        else:
            if tag.startswith("aux_"):
                info = AUX_LEXEMES[tok]
                lex = dict(lemma="do", upos="AUX", xpos=info["xpos"], feats=info["feats"])
            else:
                lex = lexicon_fn(tok, tag)
            out_tokens.append(dict(
                id=new_id, text=tok, lemma=lex["lemma"], upos=lex["upos"],
                xpos=lex["xpos"], feats=lex["feats"], head=head_new_id, deprel=deprel,
            ))
            surface_words.append(tok)

    out_tokens.append(dict(
        id=punct_id, text=final_punct, lemma=final_punct,
        upos="PUNCT", xpos=".", feats=None,
        head=new_id_of[main_verb_idx], deprel="punct",
    ))
    surface_words.append(final_punct)

    return {
        "text": " ".join(surface_words),
        "tokens": out_tokens,
        "multiword_tokens": multiword_tokens,
    }


def build_converter(cfg_dir=CFG_DIR_DEFAULT):
    """Factory: loads the grammar vocabulary once and returns a
    `sentence_to_trees(decl_tokens)` function bound to it."""
    token2tag, plural_to_singular = load_grammar(cfg_dir)
    lexicon_fn = lambda tok, tag: _lexical_info(tok, tag, plural_to_singular)

    def tags_of(toks):
        return [token2tag[t] for t in toks]

    def sentence_to_trees(decl_tokens):
        """
        decl_tokens: declarative-order token list, NOT including the
        trailing period.
        Returns (declarative_json, question_json) -- both in the same
        schema as creating_json_tree.py's output.
        """
        tags = tags_of(decl_tokens)
        subject_node, main_aux_idx, main_verb_idx, object_node = parse_sentence(tags)
        relations = build_relations(decl_tokens, subject_node, main_aux_idx, main_verb_idx, object_node)

        decl_order = list(range(len(decl_tokens)))
        decl_json = render_ud(decl_tokens, tags, relations, main_verb_idx, decl_order, ".", lexicon_fn)

        quest_order = [main_aux_idx] + [i for i in decl_order if i != main_aux_idx]
        quest_json = render_ud(decl_tokens, tags, relations, main_verb_idx, quest_order, "?", lexicon_fn)

        return decl_json, quest_json

    return sentence_to_trees
