# Data Compliance

## Allowed inputs

| Input | Purpose | Identity |
|---|---|---|
| `Yuqi-Zhou/LRAT-Train/LRAT-training-pairs.jsonl` | Sole source of training rows | 96,504 rows, SHA-256 `dd75a3f...afbe1b9` |
| `Yuqi-Zhou/LRAT-Train/trajectories.tar.gz` | Provenance validation only | SHA-256 `fb8ca29...78617a9` |
| Organizer-designated LRAT offline corpus | Document ID/text validation only | SHA-256 `4d79593...de6393` |
| `Qwen/Qwen3-Embedding-0.6B` | Sole initialization checkpoint | weight SHA-256 `0437e45...e23fd` |

## Explicitly absent

This repository contains no path that generates Agent trajectories, calls an external API, imports external training corpora, mines examples with another retriever, distills another model, or consumes another model's inference output.

M10 is the default profile. It performs only a deterministic partition of the organizer-released pair rows; the partitioner neither edits a training row nor creates supervision. M01 trains the released pair file unchanged and is retained only as a fallback.

## Reproduction gate

M10 requires all of the following before launch:

1. exact official pair-file hash, size, row count, schema, and reweight derivation;
2. every pair row traced to an official raw trajectory, with every negative document ID present;
3. every referenced positive and negative document matched exactly to the fixed offline corpus;
4. exact Qwen3-Embedding-0.6B base-weight hash;
5. fixed split salt, exact derived train hash, 94,113 rows, 1,500 dev query groups, 500 locked-test query groups, and zero normalized-query overlap.
