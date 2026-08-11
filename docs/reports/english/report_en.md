# Training an Agent-Trajectory Retriever

Final-Defense Proposal · v3.0

Team: DefaultGroup, CCIR Team of Renmin University of China; Team ID: 362259; Record ID: 998912; members: Liu Qi (captain) and Zhao Ziming (member); main contact: Liu Qi (+86 13798122637). The organizer's notice confirms that the team reached the top three of the B-leaderboard evaluation and advanced to the final defense.

## Abstract

In Agentic Search, an agent issues intermediate queries, examines snippets, chooses which documents to browse in full, and uses the resulting evidence in later reasoning. LRAT identifies Search-to-Browse transitions, unbrowsed candidates in the same search context, and post-Browse reasoning as useful signals of document utility. Our project did not regenerate agent trajectories, re-run the paper's judge, or introduce external data, teacher parameters, LoRA, or adapters. Instead, we built an auditable training and delivery path around the official Training pairs, Agent trajectories, offline Corpus, and the specified Qwen3-Embedding-0.6B. The final A-leaderboard result is Total 40.6, Recall 46.3, Success 21.1, and 16.0 average steps; the three-Agent B-leaderboard evaluation averages 56.18 Total.

## Motivation and Scope

Deep-research agents alternate among thinking, searching, browsing, and reasoning. Retrieval is therefore not only a relevance endpoint: it supplies evidence for the next reasoning step. Our implementation path is fixed inputs -> trajectory/pair provenance audit -> normalized-query isolation -> 1 positive + 5 negatives -> weighted contrastive training -> A/B evaluation -> independent loading, freezing, and reproduction read-back.

The LRAT paper treats browsed documents as candidate positives, unbrowsed documents from the same Search candidate set as a negative pool, and post-Browse reasoning as a way to filter weak positives and express relevance intensity. We use the published Training pairs as the training supervision. Trajectories and Corpus records are used for provenance and document-consistency checks; they are not used here to collect new trajectories, recompute labels or weights, or claim completion of the paper's Data Flywheel.

## Provenance and Query Isolation

For every pair record, we trace its query, positive document, negative document references, and `reweight_rate` back to trajectory events and verify the referenced text against the fixed Corpus. Among 96,504 Training pairs, 96,366 are stable mappings, 138 are ambiguous but field-consistent mappings, and 0 are mismatches. The audit covers 1,989,015 document references and 959,042 unique document IDs, with no missing documents or text mismatches.

We strip leading and trailing whitespace, collapse repeated whitespace, and apply Unicode case folding. All source rows sharing a normalized query form one group and are assigned together:

| Set | Query groups | Source rows | Role |
|---|---:|---:|---|
| Train | 78,890 | 94,113 | Full-parameter training |
| Dev | 1,500 | — | Development and selection |
| Locked test | 500 | — | Not used for training or selection |

The three sets have zero normalized-query overlap. The locked test was not used for model, method, or hyperparameter selection.

## Training Method

Each Training-pair row becomes one query, one positive passage, and five negatives randomly sampled from that row's `neg` list; the positive is fixed at candidate position 0. The resulting six-way candidate group is an input construction unit, not the optimizer batch size. Query and passage caps are 128 and 512 tokens.

Starting from the specified Qwen3-Embedding-0.6B, we update all parameters with shared encoder weights, last-token pooling, L2 normalization, and 1024-dimensional representations. With temperature `τ=0.02`, query loss `ℓ_j`, and the published pair weight `w_j`, the implemented objective is `L = Σ_j w_jℓ_j / Σ_j w_j`. The weight changes a query's contribution to the loss, not its positive/negative label.

Each of two A40 GPUs processes one candidate group. Same-order cross-device gathering makes the other query's candidates in-batch negatives and keeps each weight attached to its query. Gradient accumulation is 4, giving an effective query batch of 8 without merging the candidate pools of four micro-steps. The final run uses one epoch, 11,764 optimizer steps, per-device batch 1, group size 6, bf16, cross-device negatives, and learning rate `2e-6`.

## Configuration Selection and Results

We first fixed the learning rate at `1e-6`, trained on all 96,504 Training pairs for 1, 2, and 3 epochs, and submitted each model to the A leaderboard. The results were:

| Epochs | Total | Recall | Success | Avg. steps |
|---:|---:|---:|---:|---:|
| 1 | 40.1 | 44.8 | 21.0 | 15.4 |
| 2 | 38.2 | 42.7 | 18.3 | 15.5 |
| 3 | 38.7 | 43.4 | 19.0 | 15.8 |

We therefore selected one epoch. We then fixed the 94,113-row query-disjoint train set and compared four learning rates for 500 steps on dev1500:

| Learning rate | R@1 | MRR |
|---:|---:|---:|
| `3e-7` | 0.4913 | 0.6419 |
| `5e-7` | 0.5073 | 0.6571 |
| `1e-6` | 0.5433 | 0.6831 |
| **`2e-6`** | **0.5640** | **0.6993** |

The formal A-leaderboard comparison is:

| Candidate | Total ↑ | Recall ↑ | Success ↑ | Avg. steps ↓ |
|---|---:|---:|---:|---:|
| Qwen3-Embedding-0.6B (Base) | 28.5 | 28.5 | 10.6 | 17.8 |
| Official LRAT baseline | 34.0 | 36.1 | 15.5 | 16.8 |
| Query-disjoint (`1e-6`) | 39.5 | 45.0 | 19.8 | 15.9 |
| **Query-disjoint (`2e-6`)** | **40.6** | **46.3** | **21.1** | 16.0 |

The final candidate is 12.1 Total points above the base and 6.6 above the official LRAT baseline. Within the same query-disjoint recipe, changing the learning rate moves Total from 39.5 to 40.6. These are comparisons between reported leaderboard rows and are not a single-variable causal attribution.

The B-leaderboard Totals are 67.42 for DeepSeek-V4-Flash-0731, 60.96 for Qwen3.5-35B-A3B, and 40.16 for gpt-oss-120b, averaging 56.18; the range is 27.26. This is direct cross-Agent runnability evidence. Because matched per-Agent baselines are not available, we report absolute results and do not claim universal improvement across agents.

## Delivery and Conclusion

The final model passed fresh CPU loading, query/document inference smoke, freeze-manifest checks, anonymous Hugging Face revision read-back, and GitHub/reproduction-package validation. Detailed hashes, commits, and file-level evidence are kept in the reproduction appendix rather than the main narrative.

Our contribution is an auditable pipeline around fixed official supervision: provenance explains where the signals came from, query-disjoint grouping protects the development boundary, explicit candidate construction and weighted loss record the training semantics, staged experiments select the epoch and learning rate, and A/B evaluation reports both measured performance and its cross-Agent limits. The project does not present the competition training run as a complete re-execution of the LRAT paper's trajectory collection, judge, or Data Flywheel.

## References

1. Yuqi Zhou et al. “Learning to Retrieve from Agent Trajectories.” arXiv:2604.04949, 2026.
2. [BC-Plus A Leaderboard](https://huggingface.co/spaces/Yuqi-Zhou/BC-Plus-Leaderboard).
3. [CCIR Cup 2026 competition page](https://www.xir.cn/competition/1170).

## Appendix: Delivery Evidence and Scope

The Hugging Face evidence is anonymous API read-back of the public commit, file table, file size, and LFS SHA. It is not described as a fresh local re-download and hash recomputation of the 2.38 GB weight.

| Evidence layer | Recorded check | Result |
|---|---|---|
| Manifest | Base model, raw pairs, provenance, corpus, query-disjoint split, command, and model identity | Fixed and traceable through `reproduce_m10.py`, manifests, and the freeze record |
| Independent load | Fresh CPU process loads the frozen model; query/document inference smoke | `Qwen3Model`, 595,776,512 parameters, 1024-dimensional outputs |
| Canonical dry-run | Recheck provenance, corpus, train split, and the two-GPU command without training | `dry_run_verified`; 96,504 provenance rows, 11,215,099 corpus rows, 94,113 train rows |
| HF public read-back | Anonymous API read-back of public commit, root file table, file size, and LFS SHA | Public repo; `model.safetensors` 2,383,139,480 bytes; LFS SHA equals the frozen checkpoint SHA |
| GitHub and ZIP | Repository checks plus package manifest, CRC, and validators | `main@31949f6`; tests, compile, JSON/diff checks, 45-file ZIP, CRC and package validators pass |

The frozen identities are:

| Artifact | Identity |
|---|---|
| Base model | `Qwen3-Embedding-0.6B`; SHA-256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| Raw pairs | 96,504 rows; SHA-256 `dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9` |
| Query-disjoint train | 94,113 rows; SHA-256 `158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9` |
| Provenance | 96,366 stable / 138 ambiguous / 0 mismatch; SHA-256 `0dd510fcc41444bfcff99381ce077e5048375855add8498f9ab10afabea4b690` |
| Offline corpus | 11,215,099 rows; SHA-256 `4d795938bb89cbd7e7467a8da4e772f7ae95e6b533181aeace2a5e3fd3de6393` |
| Final checkpoint | `model.safetensors`; SHA-256 `cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984` |
| Freeze manifest | SHA-256 `d56ceafcc110ea37d480d1f137b7306a63422afc1b4e257e44119e27ada11b2a` |
| Hugging Face | `Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729`; commit `9cdfdba51359ec65069a748cf8c00c55477016bb` |
| GitHub | `seraphic663/ccir-lrat-retriever`; `main@31949f6f53722f06e91bd2ded6ec7f3a48037bba` |
| B-leaderboard JSONL | `versions/M10/submission_B_m10_20260731.jsonl`; SHA-256 `02c271173ad6bf5e438972bf9e617c66e1625bb13f5a7086c1b6bcf0167b3f49` |
| Reproduction ZIP | 45 files, 2,387,249,735 bytes; SHA-256 `c46f15d34a09dfe3108971ce56f837f1a23bbe39e35ee3a541bb609aea159984` |

These checks establish traceability and delivery-contract consistency. They do not guarantee a byte-identical checkpoint on every hardware and runtime combination, and API metadata read-back is not a local re-download and hash recomputation.
