"""
Validator for files produced by generate_qf_data_from_cfg_extended.py (--mode extended).

It does not use any code from the generator. It has its own lexicon (the
paper's vocabulary) and a recursive-descent parser for the grammar. For each
line it checks that:

  * every word is in the paper's vocabulary, and (with --train_file) occurs in
    the training data, so models trained on it have the word in their vocabulary;
  * the declarative parses as   NP_S  RC  aux  VP  "."   with agreement (RC aux
    agrees with its own subject, main aux agrees with the head of NP_S);
    NPs are  det n (prep det n)*  and at most one NP carries a PP chain,
    whose depth is <= --max_pp;
  * the question parses as      aux  NP_S  RC  VP_bare  "?"   with the fronted
    aux agreeing with the main subject;
  * the question is "main aux + declarative without the main aux" (the
    hierarchical rule);
  * the RC aux token differs from the main aux token, so the linear rule
    (front the first aux) gives a different FIRST WORD, which is what the
    evaluation scores;
  * in object RCs the RC subject has the number required by --orc_number;
  * the .type (and optional .meta.jsonl) lines match the sentence.

For the whole set it checks coverage (24 skeletons, every PP depth x site,
every type in the types csv at least once) and that the types csv is exactly
the set of types the grammar allows. It also checks that the paper's 56
templates are its {PP depth 0, or depth 1 on O} part, with the object-RC
templates' RC number flipped when --orc_number different.
"""

import argparse
import itertools
import json
import sys
from collections import Counter

import pandas as pd

LEX = {
    "det": "the some my your our her",
    "n_s": "newt orangutan peacock quail raven salamander tyrannosaurus unicorn vulture walrus "
           "xylophone yak zebra",
    "n_p": "newts orangutans peacocks quails ravens salamanders tyrannosauruses unicorns vultures "
           "walruses xylophones yaks zebras",
    "aux_s": "does doesn't",
    "aux_p": "do don't",
    "v_trans": "entertain amuse high_five applaud confuse admire accept remember comfort",
    "v_intrans": "giggle smile sleep swim wait move change read eat",
    "rel": "who that",
    "prep": "around near with upon by behind above below",
}
TOK2TAG = {w: tag for tag, words in LEX.items() for w in words.split()}
assert len(TOK2TAG) == sum(len(v.split()) for v in LEX.values()) == 64  # 69-entry model vocab = these 64 + . ? quest decl <pad>


class ParseError(Exception):
    pass


class Parser:
    """Recursive descent over a TAG sequence."""

    def __init__(self, tags):
        self.t = list(tags)
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, *allowed):
        tag = self.peek()
        if tag not in allowed:
            raise ParseError(f"pos {self.i}: expected {allowed}, got {tag!r} in {' '.join(self.t)}")
        self.i += 1
        return tag

    def np(self):
        """det n (prep det n)*"""
        self.eat("det")
        num = self.eat("n_s", "n_p")[-1]
        chain = []
        while self.peek() == "prep":
            self.eat("prep")
            self.eat("det")
            chain.append(self.eat("n_s", "n_p")[-1])
        return {"num": num, "chain": chain}

    def aux(self, num):
        a = self.eat("aux_s", "aux_p")
        if a[-1] != num:
            raise ParseError(f"pos {self.i - 1}: {a} does not agree with number {num}: {' '.join(self.t)}")
        return self.i - 1

    def subject_and_rc(self):
        subj = self.np()
        self.eat("rel")
        r = None
        if self.peek() in ("aux_s", "aux_p"):
            rc_aux = self.aux(subj["num"])
            if self.eat("v_intrans", "v_trans") == "v_trans":
                r, rc = self.np(), "src_trans"
            else:
                rc = "src_intrans"
        else:
            r = self.np()
            rc_aux = self.aux(r["num"])
            self.eat("v_trans")
            rc = "orc"
        return subj, rc, r, rc_aux

    def end(self):
        if self.i != len(self.t):
            raise ParseError(f"trailing material at {self.i}: {' '.join(self.t)}")


def parse_decl(tags):
    p = Parser(tags)
    subj, rc, r, rc_aux = p.subject_and_rc()
    main_aux = p.aux(subj["num"])
    o = p.np() if p.eat("v_intrans", "v_trans") == "v_trans" else None
    p.end()
    return {"S": subj, "R": r, "O": o, "rc": rc, "rc_aux": rc_aux, "main_aux": main_aux,
            "vp": "trans" if o else "intrans"}


def parse_question(tags):
    p = Parser(tags)
    fronted = p.eat("aux_s", "aux_p")
    subj, _, _, _ = p.subject_and_rc()
    if fronted[-1] != subj["num"]:
        raise ParseError(f"fronted {fronted} does not agree with the subject: {' '.join(tags)}")
    if p.eat("v_intrans", "v_trans") == "v_trans":
        p.np()
    p.end()


def describe(parse):
    slots = [s for s in "SRO" if parse[s] is not None]
    num = {s: parse[s]["num"] for s in slots}
    skel = f"S={num['S']}|rc={parse['rc']}" + (f":R={num['R']}" if "R" in num else "") + \
           f"|vp={parse['vp']}" + (f":O={num['O']}" if "O" in num else "")
    chains = {s: len(parse[s]["chain"]) for s in slots if parse[s]["chain"]}
    if len(chains) > 1:
        raise ParseError(f"PP chains on more than one NP: {chains}")
    site, depth = next(iter(chains.items())) if chains else ("", 0)
    return skel, depth, site


def expected_skeletons(orc_number):
    out = []
    for s in "sp":
        for rc, rs in (("src_intrans", [None]), ("src_trans", ["s", "p"]),
                       ("orc", [s if orc_number == "same" else {"s": "p", "p": "s"}[s]])):
            for r in rs:
                for vp, os_ in (("intrans", [None]), ("trans", ["s", "p"])):
                    for o in os_:
                        out.append(f"S={s}|rc={rc}" + (f":R={r}" if r else "") + f"|vp={vp}" +
                                   (f":O={o}" if o else ""))
    return out


def main(args):
    errors = []
    data = open(args.data).read().split("\n")
    types = open(args.type).read().split("\n")
    meta = [json.loads(l) for l in open(args.meta)] if args.meta else None
    if len(data) != len(types) or (meta is not None and len(meta) != len(data)):
        errors.append("data/type/meta files have different numbers of lines")
    train_words = None
    if args.train_file:
        train_words = set()
        for line in open(args.train_file):
            train_words.update(line.split())

    stats = Counter()
    info = []
    for ln, (line, tline) in enumerate(zip(data, types)):
        try:
            inp, out = line.split("\t")
            inp_toks, out_toks = inp.split(), out.split()
            marker = inp_toks[-1]
            assert marker in ("quest", "decl"), f"bad marker {marker}"
            assert inp_toks[-2] == "." and out_toks[-1] == ("?" if marker == "quest" else ".")
            decl, q = inp_toks[:-2], out_toks[:-1]
            unknown = sorted({w for w in decl + q if w not in TOK2TAG})
            assert not unknown, f"words outside the paper's vocabulary: {unknown}"
            if train_words is not None:
                unseen = sorted({w for w in inp_toks + out_toks if w not in train_words})
                assert not unseen, f"words not in the training file: {unseen}"
            tags = [TOK2TAG[w] for w in decl]
            t_in, t_out = tline.split("\t")
            assert t_in.split() == tags + [marker], "type line (input) does not match the tokens"

            parse = parse_decl(tags)
            skel, depth, site = describe(parse)
            if depth > args.max_pp:
                raise ParseError(f"PP depth {depth} > {args.max_pp}")
            if parse["rc"] == "orc":
                same = parse["R"]["num"] == parse["S"]["num"]
                if same != (args.orc_number == "same"):
                    raise ParseError(f"object RC subject number violates --orc_number {args.orc_number}")
            if decl[parse["rc_aux"]] == decl[parse["main_aux"]]:
                raise ParseError(f"RC aux and main aux are the same token ({decl[parse['main_aux']]})")
            stats["aux_number_differs"] += tags[parse["rc_aux"]] != tags[parse["main_aux"]]
            if marker == "quest":
                m = parse["main_aux"]
                assert q == [decl[m]] + decl[:m] + decl[m + 1:], "question is not main-aux fronting"
                parse_question([TOK2TAG[w] for w in q])
                assert t_out.split() == [TOK2TAG[w] for w in q], "type line (output) mismatch"
                first_aux = next(i for i, t in enumerate(tags) if t.startswith("aux"))
                stats["linear_first_word_differs"] += decl[first_aux] != q[0]
                stats["quest"] += 1
            else:
                assert q == decl
                stats["decl"] += 1
            if meta is not None:
                mm = meta[ln]
                got = (mm["type"], mm["skeleton"], mm["pp_depth"], mm["pp_site"],
                       mm["main_aux_index"], mm["rc_aux_index"])
                want = (" ".join(tags), skel, depth, site, parse["main_aux"], parse["rc_aux"])
                if got != want:
                    raise ParseError(f"meta {got} != parsed {want}")
            info.append({"type": " ".join(tags), "skeleton": skel, "depth": depth, "site": site,
                         "rc": parse["rc"], "len": len(tags), "quest": marker == "quest"})
        except (ParseError, AssertionError, ValueError, StopIteration) as e:
            errors.append(f"line {ln + 1}: {e}")

    print(f"lines: {len(data)}   parsed OK: {len(info)}   errors: {len(errors)}")
    for e in errors[:10]:
        print("  ERROR", e)
    if not info:
        print("SOME CHECKS FAILED")
        return 1
    print(f"quest: {stats['quest']}  decl: {stats['decl']}")
    print(f"linear rule (front the first aux) gives a different first word: "
          f"{stats['linear_first_word_differs']}/{stats['quest']}")
    print(f"RC aux differs from the main aux in number: {stats['aux_number_differs']}/{len(info)} "
          f"(object RCs: {sum(i['rc'] == 'orc' for i in info)}; subject RCs agree with the main "
          f"subject, so they differ only in polarity)")
    ok = not errors

    def report(name, seen, expected):
        nonlocal ok
        missing = sorted(set(expected) - set(seen), key=str)
        extra = sorted(set(seen) - set(expected), key=str)
        good = not missing and not extra
        ok &= good
        print(f"[{'OK' if good else 'FAIL'}] {name}: {len(set(seen) & set(expected))}/{len(set(expected))}"
              + (f" missing={missing[:6]}" if missing else "") + (f" unexpected={extra[:6]}" if extra else ""))

    report("skeletons", [i["skeleton"] for i in info], expected_skeletons(args.orc_number))
    report("PP (site, depth)", [(i["site"], i["depth"]) for i in info if i["depth"]],
           [(s, k) for s in "SRO" for k in range(1, args.max_pp + 1)])
    depth_counts = Counter(i["depth"] for i in info)
    print("       sentences per PP depth: " + ", ".join(f"{k}: {depth_counts[k]}" for k in sorted(depth_counts)))
    print(f"       length min/mean/max: {min(i['len'] for i in info)}/"
          f"{sum(i['len'] for i in info) / len(info):.1f}/{max(i['len'] for i in info)}")

    if args.types_csv:
        lst = pd.read_csv(args.types_csv, index_col=0, keep_default_na=False)
        listed = set(lst["type"])
        # the list must be exactly what the grammar allows: rebuild it from the parser's point of view
        bad = []
        for t in listed:
            try:
                p = parse_decl(t.split())
                _, d, _ = describe(p)
                if d > args.max_pp or (p["rc"] == "orc" and
                                       (p["R"]["num"] == p["S"]["num"]) != (args.orc_number == "same")):
                    bad.append(t)
            except ParseError:
                bad.append(t)
        n_expected = 0
        for sk in expected_skeletons(args.orc_number):
            n_slots = 1 + ("R=" in sk) + ("O=" in sk)
            n_expected += 1 + n_slots * sum(2 ** k for k in range(1, args.max_pp + 1))
        good = not bad and len(listed) == len(lst) == n_expected
        ok &= good
        print(f"[{'OK' if good else 'FAIL'}] types csv lists exactly the grammar's types: "
              f"{len(lst)} rows, expected {n_expected}, {len(bad)} invalid")
        quest_counts = Counter(i["type"] for i in info if i["quest"])
        good = all(quest_counts.get(t, 0) == n for t, n in zip(lst["type"], lst["number"])) and \
            set(quest_counts) <= listed
        ok &= good
        print(f"[{'OK' if good else 'FAIL'}] types csv 'number' column matches the data")
        report("every possible type generated at least once", list(quest_counts), listed)

        if args.orig_types:
            orig = pd.read_csv(args.orig_types)["type"].tolist()
            mapped = []
            for t in orig:
                tags = t.split()
                p = parse_decl(tags)
                if p["rc"] == "orc" and args.orc_number == "different":
                    # flip the RC subject (the noun right after "rel") and the RC aux
                    i_n = tags.index("rel") + 2
                    flip = {"n_s": "n_p", "n_p": "n_s", "aux_s": "aux_p", "aux_p": "aux_s"}
                    tags[i_n] = flip[tags[i_n]]
                    tags[p["rc_aux"]] = flip[tags[p["rc_aux"]]]
                mapped.append(" ".join(tags))
            subset = set(lst[(lst.pp_depth == 0) | ((lst.pp_depth == 1) & (lst.pp_site == "O"))]["type"])
            good = set(mapped) == subset and len(set(mapped)) == len(orig)
            ok &= good
            print(f"[{'OK' if good else 'FAIL'}] the paper's {len(orig)} templates"
                  + (" (object-RC number flipped)" if args.orc_number == "different" else "")
                  + f" = the list's {{PP depth 0, or depth 1 on O}} part ({len(subset)} types)")

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data file (<decl> . quest\\t<question> ?)")
    ap.add_argument("--type", required=True, help="matching .type file")
    ap.add_argument("--meta", default=None, help="optional .meta.jsonl from the generator")
    ap.add_argument("--types_csv", default=None, help="list of all possible types from the generator")
    ap.add_argument("--orig_types", default=None, help="the paper's quest_test_type.csv")
    ap.add_argument("--train_file", default=None,
                    help="optional training file (e.g. question.train); every word must occur in it")
    ap.add_argument("--max_pp", type=int, default=3)
    ap.add_argument("--orc_number", choices=["different", "same"], default="different")
    sys.exit(main(ap.parse_args()))
