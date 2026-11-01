# G1 freeze record

G1's bounds were frozen once, before any held-out request existed:
- run: `python3 -I -B tools/conformance/decision_g1/compare.py freeze`;
- at: 2026-11-01T19:36:12Z, on commit `57cfc22`, with no other change;
- committed: in the same commit as this record.

`bounds.json` has sha256
`338767ebbc89dc9536b181b16896938c3652659d7b8aefbee3e1382293e97065`, and
records:

```json
"frozen_with": {
  "comparator": {
    "file": "tools/conformance/decision_g1/compare.py",
    "sha256": "d2df51acf3d3055d5e30c47088c23a49bbb7f13e1fc22613a2fad4b367967384"
  },
  "heldout_recipe": {
    "file": "tools/conformance/decision_g1/make_heldout.py",
    "sha256": "806ac31b8b63fb9533a86aa9e32c2d93b3ccc9d897cf1cdcda3389cf95935913",
    "seed": 407709422033
  }
}
```

The seed 407709422033 is `0x5EED6011D1`.

The conditions below are those of the review of the pre-freeze commits
`f59a21e..57cfc22` (Fable, "Freeze: GO"), relayed by the coordinator. They
replace the list in the same reviewer's Accept of the recipe amendment
`c7dbf77`.

## 1. The recipe lineage, the generating recipe and the frozen comparator

| `make_heldout.py` sha256 | Commit | Status |
|---|---|---|
| `08ad800750cdf59986a23f0cf6f981572b8ff53c7d80854bcba6d685545f92da` | `a530d1b` | committed before any representative F16 observation; superseded, **never generated** |
| `193487e75ded33996d59dbbf01f1603ed4559d13b6aa2de414b0f8f8077ef955` | `c7dbf77` | padding-mechanics amendment, accepted; superseded |
| `9cb8c4ff61b14d7d6067a9c867386c39742cc4cd15c97d433abc4a52a8439b6a` | `f59a21e` | module docstring only; superseded |
| `806ac31b8b63fb9533a86aa9e32c2d93b3ccc9d897cf1cdcda3389cf95935913` | `596a3e2` | recording only; **the generating recipe** |

- The seed is `0x5EED6011D1`, unchanged since `a530d1b`.
- The frozen comparator is `compare.py`
  `d2df51acf3d3055d5e30c47088c23a49bbb7f13e1fc22613a2fad4b367967384`
  (`57cfc22`).
- `bounds.json` stamps exactly these in `frozen_with`.
- `tests/test_make_heldout.py` at `596a3e2` and `57cfc22` is
  `6ae0b84b6850154dcd5d7d0c2390f0c8f3a24e08056a79731fe805adcdb6a46e`.

## 2. Content-neutral and recording-only

**The amendment `c7dbf77`**, per the Accept:
- `git diff a530d1b c7dbf77` touches `make_heldout.py` only in
  `with_padding`, the new `MAX_WORD_LEVELS = 40` and `pad`.
- Byte-identical:
  - `SEED`, and `SLOTS` (40 slots, with targets 63/64/65/600/1,500/4,064);
  - `TOPICS`, `FILLER_WORDS`, and the fixed ambiguous and borderline
    scenarios;
  - `assign_topics`, `build_request`, `check_domain`, `strict_load`,
    `OracleMeasure`, `generate`, `allowlist_for` and `main`.
- Nothing in the diff reads a logit, an answer or an observation.
  `OracleMeasure` still runs `decision-oracle run --vocab-only` and returns
  `len(token_ids)` per pass.

**The later commits**, compared at function level with every docstring
stripped:
- `c7dbf77` → `f59a21e` changes nothing: the whole-module ASTs are
  identical, and so are the bytes before and after the module docstring.
- `f59a21e` → `596a3e2` changes `main` and adds `measured_with`, nothing
  else. Byte-identical: `generate`, `pad`, `with_padding`, `build_request`,
  `check_domain`, `SEED`, `SLOTS`, `TOPICS` and the fixed scenarios.
- `596a3e2` → `57cfc22` changes nothing.

`make_heldout.py --recipe` prints the same slot table at all three.

`measured_with` runs `decision-oracle --identity` and hashes two files.
`main` calls it after `generate` and before writing anything.

**No generation.** `main()` and `generate()` have not run with `SEED` (P11's
statement). The branch holds no held-out file. The generating tests use
seeds `0x7E57` and `0x7E58`, and the five `measured_with` tests call only
`measured_with`, on a shell stand-in that answers `--identity` alone.

## 3. Disclosure

As disclosed before the freeze, and recorded with the amendment:

> A dry run of make_heldout.py with development seeds 1-6 (never
> 0x5EED6011D1; no held-out request generated) and upstream's
> vocabulary-only preparation as the token counter found the word-padding
> phase could stall: on a JSON note every filler word added two tokens at the
> closing quote, so seed 3's 1,500-token slot stopped at 1,499. With the real
> seed that would have failed generation after the freeze.

The dry runs:
- read token counts only (vocabulary-only preparation, nothing evaluated)
  and wrote no case file;
- left one leftover: an interrupted run (seeds 1–3) left the
  development-seed input it was measuring in a temporary directory, deleted
  on 2026-11-01 before this freeze.

The development-seed evidence of the Accept of `c7dbf77`:
- `a530d1b`'s recipe stalls with development seed 3 at 1,499 tokens on slot
  14 (`g1h-14-choice-json-object-1500-tokens`).
- `c7dbf77`'s recipe hits all six targets on development seeds 1–6.
- Seed 3 run twice gives the same 40-request digest (`386e67993c0a8ac6…`).

## 4. What the bounds were frozen from

The freeze was made at `57cfc22` by `compare.py freeze`, which first passed
`compare.py integrity`. `g1/heldout` was absent. The inputs were the official
representative collections only:

| Input | sha256 |
|---|---|
| `g1/oracle/d1-2b-f16/cpu-t4-fa-off/collection.json` | `49f58b3d399f50ac6e741fa8cb4d744e2b70187651a507cd70092cecc8efcca0` |
| `g1/oracle/d1-2b-f16/cpu-t4-fa-on/collection.json` | `71c000c5534c92105d534437b1bdb16d4398d6faf3ef2843ae2a2b304f483f6c` |
| `hf/provenance.json` (`provenance.py --check` binds every HF observation) | `5bfc3e4ac3ac50e78a4e75d35c0ef839ea161925421b2cbf134200f4d8500db8` |

The frozen `observed` values:

| Setting | Questions (one-group) | Cases | Max AGE | At | Bound | Headroom | Discrete changes |
|---|---|---|---|---|---|---|---|
| `cpu-t4-fa-off` | 32 (1) | 26 | `0.0034017562866210938` | `hf-choice-letter-keys/planet` | 0.011 | 3.23362 | 0 |
| `cpu-t4-fa-on` | 32 (1) | 26 | `0.008427619934082031` | `hf-state-special-token-key/injection` | 0.02 | 2.37315 | 0 |

The frozen identities:
- `representative/hf`: `a43c3ae84a336fed20e79c0c3b0999ea86bc509f191e716c8b2df2edfa636099`
- `representative/d1-2b-f16/cpu-t4-fa-off`: `b4931eaf2d971de8ca5f8fd72bdc8198a24b1f03a07b10a32353a5deb85ceba2`
- `representative/d1-2b-f16/cpu-t4-fa-on`: `6a746ba307d5c29c4987caed86c57ea095d38d897fb07f34a4f7036d4ad8d6b5`

`bounds.json` (`338767ebbc89dc9536b181b16896938c3652659d7b8aefbee3e1382293e97065`)
lists 188 input files by repository-relative path, so its bytes are
reproducible. The d1-2b-q4km collections pass the integrity checks but do not
enter the bounds.

## 5. Post-freeze amendment policy

Verbatim; it supersedes condition 5 of the Accept of `c7dbf77`.

> **Post-freeze amendment policy.** (1) Default: a changed `compare.py` or `make_heldout.py` refuses every command; nobody edits `bounds.json` by hand. (2) Exactly one exception, "successor adoption", for two classes only: a comparator change that is metric-neutral (`compare_question`, `certificate`, `summarize`, `clipped_bound` with its four constants, the exclusion kinds and inclusion rules, `DIFFERENCE_FIELDS`/`differences`, `check_structure`, `check_join`, `check_membership`, `verify_exclusion`, `native_identity`, `hf_identity`, `reproduce_answer` and the upstream arithmetic unchanged or strictly stricter, shown by a function-level AST comparison recorded with the review); and a recipe change that is padding-mechanics-only (seed, slots, targets, topics, fixed scenarios and domain rules byte-identical; token counts the only input), made only while no held-out request exists. Once held-out bytes exist the recipe is never changed; a generation failure is a result. (3) Procedure: a reviewer other than P11 records the classification; the successor must reproduce the frozen representative results exactly (`bounds.json`'s `observed` block and `bounds`, the committed `report.json` and `q4-drift.json`, apart from the digest fields) on the representative set and the synthetic tests only, with no held-out number read while deciding; if held-out reports already exist, a comparator successor must also reproduce them byte for byte apart from digest fields, or it is a new experiment. (4) The adoption is recorded append-only: `frozen_with` keeps the frozen digest and gains a `successors` entry (file, old sha256, new sha256, commit, reviewer, reason), written by a dedicated `compare.py adopt` command, which is itself the first amendment under this policy if it is ever needed; until it exists, the coordinator's ruling in `g1-freeze-record.md` and the reviewed commit are the record. The bounds never change. (5) Everything else (the metric, the bound rule, the cap, inclusion rules, any weakened identity or structure check, any seed, slot, target or topic change) is a new experiment under C16: a new freeze and a new held-out set.

## 6. At generation

`compare.py integrity` enforces all of these, and refuses unless they hold:
- **`generation.json`** must hold:
  - the frozen recipe and seed;
  - the measuring tool, by repository-relative path, with its sha256 and
    stamped identity; the identity's fingerprint must equal every held-out
    collection's build;
  - the pinned vocabulary, with its sha256;
  - the measurement count;
  - the request digests and `pass_tokens`, with every target hit.
- **The held-out HF run log** must carry its passing float32-parity record
  and the full numerics tuple, from the representative run's environment.
- **The order:** `compare.py integrity` must pass before any held-out number
  is read, and the gate's identity check must accept every `heldout/<side>`
  fingerprint.

## Considered and not taken: an explicit HF-side length check (m2)

`check_generation` could compare each HF pass length to the generation's
`pass_tokens` directly, so a mismatch would name the generation record rather
than "the two sides evaluated different things". It was not taken: the HF
lengths are already bound transitively. `integrity` runs the d1-2b-f16 vs HF
comparison checks on the held-out corpus. There, `check_structure` holds each
HF observation's usage to its evaluated tokens, and `differences` holds the
HF token ids to the oracle's, which `check_generation` holds to `pass_tokens`.
Taking it would also have changed the frozen comparator.
