# CCIR LRAT Retriever — Canonical M10 Reproduction

> This archive contains the public reproduction code and contracts only. It does not contain the submitted checkpoint, tokenizer, training pairs, trajectory archive, offline corpus, or B-leaderboard attachment.

This repository has one supported training target: **M10**. It rebuilds the
query-disjoint M10 training set from organizer-provided LRAT inputs and trains
`Qwen3-Embedding-0.6B` for one fixed epoch with learning rate `2e-6`.

The canonical entry point is:

```text
code/reproduce_m10.py
```

`profiles/` and alternate model-training entry points are intentionally absent.

## B-leaderboard submission identity

The platform submission for 2026-07-31 is:

```jsonl
{"date":"2026-07-31","hf_repo_id":"Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729","github_repo_url":"https://github.com/seraphic663/ccir-lrat-retriever"}
```

Submitted checkpoint:

```text
Hugging Face: Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729
HF commit: 9cdfdba51359ec65069a748cf8c00c55477016bb
model.safetensors SHA-256:
cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984
```

The separately published Hugging Face repository contains the full model in
its root. This archive intentionally does not mirror that model. It is not a
LoRA or adapter checkpoint.

## What “reproduction” means

M10 training is GPU-distributed bf16 training. A second run is expected to
reproduce the fixed data, command, method and evaluation performance, but is
not expected to produce byte-identical floating-point weights.

This repository therefore keeps two identities separate:

1. **Submitted-checkpoint identity** — the HF checkpoint above must retain its
   exact SHA-256.
2. **Retraining acceptance** — an independent run must complete the fixed
   11,764-step contract and reproduce the documented dev1500 metrics within
   absolute tolerance `0.002`.

The tolerance is this repository's reproducibility criterion, not an organizer
rule or a claim that the B-leaderboard Agent score has already been reproduced.

## Pinned official inputs

Only the following inputs are accepted. The canonical runner checks every
listed byte identity before training.

| Input | Revision / identity |
|---|---|
| LRAT pairs | `Yuqi-Zhou/LRAT-Train@238b874226706bdf36e1684433b1455318ada22d` |
| `LRAT-training-pairs.jsonl` | SHA-256 `dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9` |
| `trajectories.tar.gz` | SHA-256 `fb8ca29a7807e334fa0eab2d22fd3c3d52852c2f42f534969c4b1605578617a9` |
| Offline corpus | `Lk123/wiki-25-512@fc1e312568b14385c04f41bc09157d8fa4c20658` |
| `wiki-25-512.jsonl` | 18,496,937,987 bytes; SHA-256 `4d795938bb89cbd7e7467a8da4e772f7ae95e6b533181aeace2a5e3fd3de6393` |
| Base model | `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Base `model.safetensors` | SHA-256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| Base `tokenizer.json` | SHA-256 `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` |

Machine-readable source declarations are in `DATA_SOURCES.json`.

M10 does not generate new trajectories, call an external API, add external
training data, use a teacher model, or initialize from another submitted
retriever.

## Reference environment and resources

Reference environment:

```text
Python 3.10.16
PyTorch 2.7.0 + CUDA 12.6
Transformers 4.53.2
2 × NVIDIA A40
```

Create the environment from the fully resolved lock:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

`requirements.txt` lists the direct dependencies; `requirements-lock.txt`
freezes the resolved environment used for reproduction.

Measured requirements on 2 × A40:

| Stage | Measured time |
|---|---:|
| Provenance + 18.5GB corpus verification + split | about 5 minutes |
| Dry-run | about 16 seconds |
| Full M10 training | 10h 15m |
| Independent dev1500 evaluation | about 10 minutes |
| Core end-to-end total | about 10h 30m |

Allow at least **80GB free disk** for raw inputs, work files, cache and
recoverable training checkpoints. Network download time is not included.

## One-command canonical reproduction

Prepare these local paths:

```text
/data/LRAT-training-pairs.jsonl
/data/LRAT-trajectories.tar.gz
/data/offline_corpus.jsonl
/models/Qwen3-Embedding-0.6B/
```

First run the full identity/provenance/split gate and print the immutable
training command:

```bash
python code/reproduce_m10.py \
  --pairs /data/LRAT-training-pairs.jsonl \
  --trajectories /data/LRAT-trajectories.tar.gz \
  --corpus /data/offline_corpus.jsonl \
  --base-model /models/Qwen3-Embedding-0.6B \
  --output-root outputs/m10-dry-run \
  --cuda-visible-devices 0,1 \
  --dry-run
```

The dry-run must finish with:

```json
{"status": "dry_run_verified"}
```

Run the complete reproduction in a new output directory by removing only
`--dry-run`:

```bash
python code/reproduce_m10.py \
  --pairs /data/LRAT-training-pairs.jsonl \
  --trajectories /data/LRAT-trajectories.tar.gz \
  --corpus /data/offline_corpus.jsonl \
  --base-model /models/Qwen3-Embedding-0.6B \
  --output-root outputs/m10-full \
  --cuda-visible-devices 0,1
```

The runner:

1. verifies the exact pairs, trajectory archive, corpus, base weights and
   tokenizer;
2. rebuilds trajectory provenance;
3. verifies every referenced document against the complete offline corpus;
4. builds the deterministic query-disjoint split;
5. validates the exact `94,113`-row train JSONL;
6. launches the fixed two-process M10 command;
7. records all identities and the produced model SHA in
   `CANONICAL_REPRODUCTION.json`.

It refuses to overwrite an existing output root. GPU device ids may change,
but exactly two distinct devices are required. The process count, data,
configuration and training command cannot be overridden through this entry
point.

## Fixed M10 method

Normalized queries are formed with:

```text
strip → collapse consecutive whitespace → Unicode casefold
```

Groups are sorted by:

```text
SHA-256("ccir-early-stop-v1" + NUL + normalized_query)
```

The first 1,500 query groups form dev, the next 500 form locked test, and all
remaining groups form train. The resulting train identity is:

```text
rows: 94,113
bytes: 3,784,989,300
SHA-256:
158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9
normalized-query overlap with dev/test: 0
```

Training contract:

| Parameter | Value |
|---|---:|
| Base model | Qwen3-Embedding-0.6B |
| GPU processes | 2 |
| Epoch / optimizer steps | 1 / 11,764 |
| Per-device batch | 1 |
| Gradient accumulation | 4 |
| Effective query batch | 8 |
| Group size | 6 |
| Learning rate | `2e-6` |
| Warmup ratio | 0.1 |
| Temperature | 0.02 |
| Query / passage length | 128 / 512 |
| Cross-device negatives | enabled |
| Pooling / normalization | last token / L2 |
| Precision | bf16 |
| Seed / data seed | 20260716 |

The original submitted M10 run was interrupted after step 7,385 and resumed
from the complete step-6,000 checkpoint. The canonical reproduction uses one
uninterrupted run; `code/train.py --resume-from-checkpoint` is retained only
for fault recovery and is not used by `code/reproduce_m10.py`.

## Independent local reproduction result

On 2026-07-31, an isolated README-driven run on 2 × A40 rebuilt all inputs and
completed `11,764/11,764` steps. This run used the same M10 training contract
from public commit `d9671c2a211c4454ef91c268d91a6c9970e2cf07`;
the present hardening changes identity checks and documentation, not the
training algorithm.

| Metric | Submitted M10 reference | Independent run | Delta |
|---|---:|---:|---:|
| R@1 | 0.630000 | 0.628000 | -0.002000 |
| R@5 | 0.905333 | 0.906000 | +0.000667 |
| R@10 | 0.974667 | 0.974667 | 0 |
| MRR | 0.752016 | 0.750895 | -0.001121 |

The independent model SHA-256 was:

```text
02e8856ea597f39d27d41963d62bc204d6b057ebd2fc6e744e2bd273a6a56e37
```

It differs from the submitted checkpoint SHA, as expected for distributed
bf16 retraining. All four dev metrics meet the repository's absolute
`0.002` acceptance tolerance. This experiment did not rerun the A-leaderboard
Agent score and cannot claim a B-Agent score before the organizer publishes
that evaluator.

The submitted M10's historical official A-leaderboard result was:

| Total | Recall | Success | Avg Steps | Agent |
|---:|---:|---:|---:|---|
| 40.6 | 46.3 | 21.1 | 16.0 | Qwen3.5-4B |

## Inference

The default `competition` contract follows the published evaluator: official
query instruction prefix, query/document limits 8192/4096, EOS/last-token
pooling, L2 normalization and 1024-dimensional output.

```bash
python code/inference.py \
  --checkpoint /models/m10-checkpoint \
  --input queries.jsonl \
  --output query_embeddings.jsonl \
  --mode query \
  --contract competition
```

For document encoding, change `--mode query` to `--mode document`. Input JSONL
contains `id` and `text`; output JSONL contains `id` and `embedding`.

## Package contents

The post-deadline reproduction ZIP contains:

```text
submission_B_m10_final_20260731/
├── code/
│   ├── reproduce_m10.py
│   ├── build_provenance.py
│   ├── preprocess.py
│   ├── prepare_split.py
│   ├── train.py
│   ├── inference.py
│   ├── config.yaml
│   └── vendor/
├── checkpoint/
├── evidence/
├── requirements.txt
├── requirements-lock.txt
├── README.md
├── DATA_SOURCES.json
├── COMPLIANCE.md
├── MODEL_IDENTITY.json
├── BUILD_MANIFEST.json
└── validate_submission.py
```

The public GitHub repository does not duplicate the 2.38GB HF checkpoint or
competition data.

## Static checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q code tests
```

## Sources and licenses

- Official data and code: `Yuqi-Zhou/LRAT-Train`, `Yuqi-Zhou/LRAT`
- Base model: `Qwen/Qwen3-Embedding-0.6B`
- `code/vendor/FlagEmbedding/` is the minimal training runtime snapshot; its
  MIT license is included at `code/vendor/FlagEmbedding/LICENSE`.

Competition data, upstream models and vendored code remain subject to their
respective licenses.
