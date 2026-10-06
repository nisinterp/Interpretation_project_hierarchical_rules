# Interpretation_project_hierarchical_rules

Extensions and reproducibility materials for **Qin, Saphra & Alvarez-Melis,
["Data Drives Unstable Hierarchical Generalization in LMs"](https://aclanthology.org/2025.emnlp-main.593/)
(EMNLP 2025)** — the paper that trains small transformer LMs on a synthetic
English *question-formation* task to study whether they learn a genuinely
**hierarchical** rule for auxiliary-fronting ("front the structurally main
auxiliary") or a **linear** shortcut ("front the first/last auxiliary"),
and how training-data complexity and diversity determine which one wins.

Official code/data for the paper: <https://github.com/sunnytqin/hier_gen>
(referred to below as `hier_gen/`). **This repository does not duplicate
that code** — it adds reproducibility scripts, extended test data, and a
dependency-distance analysis layer on top of it. You need a local clone of
`hier_gen/` to actually train or evaluate models; see Setup below.

## What this project adds, in one paragraph each

1. **Training/evaluation reproduction scripts** (`training_and_evaluating_models/`)
   that reproduce the paper's Section 2.3 question-formation training run
   and the Section 3.2 "Center Embed" training condition, plus evaluation
   scripts for both the paper's own val/test sets and arbitrary custom test
   files.
2. **Deeper center-embedding test data** (`qf_data/`): the paper's own
   grammar caps relative-clause embedding at depth 1 (one clause nested
   inside the matrix sentence). We generate genuine depth-2..5 *multiply*
   center-embedded test sentences — sentences no model in the paper was
   ever trained on — to check whether a model's apparent hierarchical
   competence actually generalizes to deeper recursion, or collapses.
3. **Universal Dependencies trees for the training data**
   (`Question data with trees/`, built by scripts in
   `Tree creation supplementing scripts/`): every declarative/question pair
   in `question.train`, each parsed into a UD dependency tree, for
   downstream syntactic-complexity analysis.
4. **Per-construction dependency-distance analysis**
   (`Construction dependency distance/`): given a model's accuracy broken
   down by sentence-construction type, compute each construction's mean
   dependency distance (Liu, 2008) directly and deterministically from its
   grammar template — a candidate explanatory variable for *why* some
   constructions are harder than others.

## Repository structure

```
Interpretation_project_hierarchical_rules/
├── training_and_evaluating_models/
│   ├── train_qf_original.sh            # reproduce paper Sec. 2.3 (original QF training)
│   ├── train_qf_center_embed.sh        # reproduce paper Sec. 3.2 "Center Embed" condition
│   ├── evaluate_qf_orig_data.py        # evaluate checkpoints on question.val / question.test
│   └── evaluate_qf_custom_data.py      # evaluate checkpoints on any custom test file (e.g. qf_data/test/*)
│
├── qf_data_generation/
│   └── recursion.py                    # generates qf_data/'s depth-2..5 test sentences (see below)
│
├── qf_data/
│   ├── test/                           # depth-2..5 center-embedded test sentences (declarative + quest pairs)
│   │   ├── question_recursion_depth2.test
│   │   ├── question_recursion_depth3.test
│   │   ├── question_recursion_depth4.test
│   │   └── question_recursion_depth5.test
│   └── test_type/                      # matching grammatical-tag-template files (one line per sentence)
│       └── question_recursion_depth{2,3,4,5}.test.type
│
├── Tree creation supplementing scripts/
│   ├── First test.py                   # first worked example: Stanza-based UD parse of one sentence
│   ├── creating_json_tree.py           # same, output reshaped to JSON for programmatic use
│   ├── compute_dependency_distance.py  # loads a JSON UD tree, computes Mean Dependency Distance
│   ├── grammar_ud_converter.py         # deterministic, grammar-based UD converter (see "Why rule-based" below)
│   ├── build_question_train_tsv.py     # runs grammar_ud_converter.py over all of question.train
│   ├── split_tsv.py                    # splits a large TSV into GitHub-pushable chunks
│   ├── requirements.txt                # Python deps for the Tree-creation scripts (Stanza + CPU torch)
│   ├── First_test_tree / example_json_tree.txt   # example outputs of the two scripts above
│   ├── tree building journal.txt       # narrated build log: spaCy vs. Stanza, the Stanza-vs-rule-based
│   │                                     decision, every bug hit and how it was found/fixed
│   └── building_trees_journal.txt      # narrated build log for build_question_train_tsv.py specifically,
│                                         including the failed GitHub push attempts and why
│
├── Question data with trees/
│   └── question_train_with_trees_part{1,2,3,4}.tsv   # id, declarative, declarative_tree (UD JSON),
│                                                        question, question_tree (UD JSON) -- all 100,000
│                                                        rows of question.train, split into 4 files to
│                                                        clear GitHub's 100MB per-file push limit
│
└── Construction dependency distance/
    ├── results_per_type.csv                      # source: one model checkpoint's accuracy per sentence
    │                                                 construction (from jail_lexicon_attitude_study)
    ├── results_per_type_seed228embonly.csv        # source: a second checkpoint's per-construction accuracy
    ├── result_per_type_add.csv                    # = results_per_type.csv + avg_dependency_distance column
    ├── result_per_type_seed228embonly_add.csv      # = results_per_type_seed228embonly.csv + that column
    ├── add_avg_dependency_distance.py             # computes that column for a results_per_type.csv-shaped file
    └── dependency_distance_from_tokens.py         # same calculation, standalone, for one tag string at a time
```

## Setup

**1. Get the paper's own code.** Clone the official repository and set up
its environment (local/forked `transformers==4.36.0.dev0`, etc. — see its
own README for exact instructions):

```bash
git clone https://github.com/sunnytqin/hier_gen.git
```

**2. Drop this repo's scripts into the `hier_gen/` checkout.**
`training_and_evaluating_models/*.sh` and `qf_data_generation/recursion.py`
are written to run from inside `hier_gen/` (they expect `train_transformers.py`,
`data_utils/`, `cfgs/`, etc. alongside them, and the shell scripts locate
`hier_gen/` automatically by searching for `train_transformers.py` next to
themselves or in a `code/` subfolder — see each script's own header comment,
or set `CODE_DIR` explicitly). The simplest layout:

```bash
cp training_and_evaluating_models/*.sh hier_gen/
cp training_and_evaluating_models/*.py hier_gen/
cp qf_data_generation/recursion.py hier_gen/
```

**3. Python environment for the UD-tree / dependency-distance scripts**
(a *separate*, lighter environment from the one used for training —
these don't need PyTorch at all except for `creating_json_tree.py` and
`First test.py`, which use Stanza):

```bash
cd "Tree creation supplementing scripts"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # Stanza + CPU-only torch; see the file for the exact pinned versions
```
`add_avg_dependency_distance.py` and `dependency_distance_from_tokens.py`
(in `Construction dependency distance/`) use only the Python standard
library — no environment needed for those.

## Reproducing the study, end to end

### A. Train and evaluate the paper's own setup (Section 2.3)

```bash
cd hier_gen
bash train_qf_original.sh          # GPU required; see the script header for SEEDS/GPUS/OUT_DIR etc.
python evaluate_qf_orig_data.py runs/qf_section2/checkpoints/*/checkpoint_300000.pth \
    --per_type --out results.csv   # writes results.csv and results_per_type.csv
```
`results_per_type.csv`'s shape (`checkpoint,type,acc,n`) is exactly what
`Construction dependency distance/add_avg_dependency_distance.py` expects.

### B. Train on the "Center Embed" condition (Section 3.2)

```bash
bash train_qf_center_embed.sh      # generates question_D_center_embed.train itself if missing
```

### C. Extend beyond the paper: test deeper center embedding

The paper's grammar allows at most **one** level of relative-clause
embedding (`aux_count <= 3`). `recursion.py` generates genuinely recursive,
multiply center-embedded sentences — e.g. depth 2:
`"the dog that the cat that the mouse chased bit barked ."` — structurally
analogous to the classic psycholinguistic center-embedding examples
(Chomsky & Miller, 1963), and never present anywhere in the paper's own
training data. This directly probes whether a model's apparent
hierarchical rule ("front the structurally main aux, not the first one")
is genuinely recursive, or a shallow heuristic that only happened to match
the hierarchical rule at the one depth it was trained on.

```bash
# one depth-2 test file, 20,000 examples (already generated and committed
# in qf_data/test/ for depths 2-5; regenerate with, e.g.:)
python recursion.py --depth 2 --num_samples 20000 --split test
# or sweep several depths in one call:
python recursion.py --depths 2,3,4,5 --num_samples 20000 --split test
```

Evaluate a trained checkpoint against these out-of-distribution depths:

```bash
python evaluate_qf_custom_data.py runs/.../checkpoint_300000.pth \
    --test_file qf_data/test/question_recursion_depth2.test \
    --test_file qf_data/test/question_recursion_depth3.test \
    --test_file qf_data/test/question_recursion_depth4.test \
    --test_file qf_data/test/question_recursion_depth5.test \
    --out depth_results.csv
```

> **Design note on `recursion.py`'s grammar**: its object NP is
> deliberately kept clause-free (`Det N (PP)?`, never a relative clause),
> so the matrix auxiliary is always the *last* auxiliary in the sentence.
> A model exploiting "front the last aux" would score perfectly on these
> files despite that being a linear, non-hierarchical rule — this matters
> if you're using `acc_unambiguous` as evidence of genuine recursion rather
> than a different shallow heuristic. The paper's own grammar avoids this
> by also allowing an RC on the *object* NP.

### D. Build Universal Dependencies trees for `question.train`

```bash
cd "Tree creation supplementing scripts"
python "First test.py"                 # worked example: one sentence, Stanza, CoNLL-U + ASCII tree
python creating_json_tree.py           # same sentence, JSON output instead
python build_question_train_tsv.py     # all 100,000 rows of question.train -> question_train_with_trees.tsv
python split_tsv.py --parts 4          # split into GitHub-pushable chunks (already done; see
                                        #   "Question data with trees/")
python compute_dependency_distance.py  # Mean Dependency Distance of a single parsed sentence
```

**Why a deterministic grammar-based converter (`grammar_ud_converter.py`)
instead of Stanza for the full 100,000-row pass:** benchmarked on 4 CPU
cores with no GPU, Stanza parses ~4 sentences/second — the full pass (200,000
parses: a declarative *and* a question form per row) would take ~14 hours.
But `question.train` is generated from a small, fully known, finite CFG
(`hier_gen/cfgs/tag_token_map.txt`), so its correct UD structure can be
derived deterministically rather than statistically inferred — about
200,000× faster, and, as it turns out, more reliable: Stanza itself
mis-parses this grammar's unconditional "do"-support (e.g. treats a bare,
non-negated `"do sleep"` as a noun object of "do", since real affirmative
English almost never uses bare periphrastic "do V"), and attaches
object-position PPs to the verb (`obl`) rather than the noun the grammar
actually intends (`nmod`). `grammar_ud_converter.py`'s rules were validated
against Stanza on unambiguous examples, then checked for exact structural
coverage and correct question-transformation reconstruction across **all
100,000 rows of `question.train`, with zero errors**. Full narrative —
every dead end, every bug, and how each was found — is in
`"Tree creation supplementing scripts/tree building journal.txt"` and
`"...building_trees_journal.txt"`.

### E. Per-construction dependency-distance analysis

Given a `checkpoint,type,acc,n` file from step A's `--per_type` evaluation
(`type` is a grammatical-tag *template* like
`det n_p rel aux_p v_intrans aux_p v_intrans quest aux_p det n_p rel aux_p v_intrans v_intrans`,
not an actual sentence — every concrete sentence sharing a template has an
*identical* dependency-distance structure, since distance depends only on
grammatical category, never on which word fills a slot):

```bash
cd "Construction dependency distance"
python add_avg_dependency_distance.py --input results_per_type.csv --output result_per_type_add.csv
# or, for a single construction string, no CSV needed:
python dependency_distance_from_tokens.py "det n_p rel aux_p v_intrans aux_p v_intrans"
```
This lets you correlate a construction's mean dependency distance against
the model's accuracy on it — a candidate syntactic-complexity explanation
for which constructions a model gets wrong.

## Key design decisions and caveats (see the journals for full detail)

- **UD label fidelity**: Stanza's English pipeline was used (not spaCy),
  since spaCy's bundled `en_core_web_sm` uses an older ClearNLP/OntoNotes
  scheme (`dobj`, `poss`, `neg`) rather than current UD v2 relations
  (`obj`, `nmod:poss`, `advmod`) — and visibly mis-parsed one of this
  grammar's two "do"-support constructions on the very first sentence
  tried. See `tree building journal.txt`, section 2.
- **Every off-the-shelf parser here is being run out-of-distribution.**
  This grammar's toy vocabulary and unconditional periphrastic "do"-support
  don't occur in real English; even Stanza, and even the grammar-validated
  `grammar_ud_converter.py`'s own design, were cross-checked against hand
  -picked unambiguous examples rather than assumed correct outright.
- **`recursion.py`'s center-embedding is on the subject side only**,
  matching the classic psycholinguistic center-embedding paradigm and the
  worked example in the script's own docstring — see the design note under
  step C above for what this does and doesn't let you conclude from
  `acc_unambiguous` on these files.
- **Large files**: `question_train_with_trees.tsv` (one file, ~367MB) is
  not in this repo — GitHub hard-rejects any file over 100MB on a normal
  push. `Question data with trees/` has it pre-split into 4 parts
  (~90-98MB each) instead; regenerate the unsplit version with
  `build_question_train_tsv.py` if you need it as one file.
