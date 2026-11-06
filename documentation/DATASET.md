# Benchmark Dataset and Knowledge Repository Documentation

## 1. Dataset Overview

The HalluciGuard-OKF evaluation framework incorporates two synchronized data repositories:
1. **Curated OKF Knowledge Repository (`knowledge/artifacts/curated_knowledge.json`):** Ground-truth knowledge store containing verified Open Knowledge Format artifacts against which claims are evaluated.
2. **Benchmark Evaluation Suite (`datasets/benchmark/benchmark_data.json`):** Gold-standard benchmark containing realistic generation scenarios, extracted claims, provenance tags, temporal states, and ground-truth verification labels.

---

## 2. Curated Knowledge Base Statistics

- **Total Verified Artifacts:** 31
- **Domain Distribution:**
  - Science & Physics: 9 artifacts (e.g., speed of light, gravitational acceleration, Planck's constant, water boiling point, DNA structure).
  - Geography & Geopolitics: 7 artifacts (e.g., Australia capital, Mount Everest elevation, Amazon River, France capital, Tokyo population).
  - Technology & Computer Science: 6 artifacts (e.g., World Wide Web invention, Python creation, Linux kernel, Bitcoin whitepaper).
  - Medicine & Biology: 5 artifacts (e.g., human chromosomes, penicillin discovery, hemoglobin, insulin).
  - History & Space Exploration: 4 artifacts (e.g., Apollo 11 moon landing, Berlin Wall fall, Magna Carta).
- **Format:** Conforms to strict `OKFArtifact` Pydantic v2 specification including cryptographic identity, provenance verification chain, primary source designation, and temporal validity intervals.

---

## 3. Benchmark Dataset Statistics

- **Total Scenarios:** 72 realistic question-answering and generation contexts.
- **Total Evaluated Claims:** 84 unique atomic claims.
- **Class Distribution:**
  - `SUPPORTED`: 33 claims (39.3%) — Claims fully entailed by verified knowledge base entries.
  - `CONTRADICTED`: 29 claims (34.5%) — Claims directly conflicting with verified knowledge base entries (factual hallucinations, attribute clashes, numerical distortions).
  - `UNVERIFIED`: 22 claims (26.2%) — Claims asserting facts outside the knowledge base (unsupported claims, fabricated citations, hypothetical entities).
- **Zero Data Leakage Guarantee:**
  - Stratified 70% / 15% / 15% train/val/test split.
  - Scaler parameters ($\mu, \sigma$) fitted exclusively on training claims ($N = 58$) and stored to `models/multisignal_head/feature_scaler.json`.
  - Zero test partition claims ($N = 13$) or validation partition claims ($N = 13$) were exposed during feature normalizer fitting or hyperparameter tuning.

---

## 4. Hallucination Typology Coverage

The benchmark includes explicit representation across 10 distinct hallucination categories:

1. **Factual Inversions:** Claims directly reversing empirical reality (e.g. Canberra vs Sydney).
2. **Numerical Hallucinations:** Fabricated constants, altered measurements, or swapped orders of magnitude.
3. **Temporal Inconsistencies:** Conflating historical milestones (e.g. World Wide Web in 1989 vs 1994).
4. **Entity Attribute Clashes:** Swapping inventors, authors, or attributes across real entities.
5. **Fabricated Citations:** References to fictitious authors, non-existent journals, or synthetic institutions.
6. **Temporal Staleness:** Stating outdated historical facts as current status.
7. **Negation Flipping:** Manipulating polarity or double negations.
8. **Unsupported Generalizations:** Extrapolations that cannot be substantiated from source evidence.
9. **Subtle Context Drift:** Statements that match keywords but shift underlying semantic relations.
10. **Multi-Claim Blends:** Sentences containing one supported and one contradicted proposition.

---

## 5. Current Research Pipeline and External Dataset Adapters (2026-11-05)

The primary `BenchmarkDataLoader` path now uses local retrieval-derived evidence features, grouped by scenario ID. Gold `evidence` fields are excluded unless the caller explicitly selects `evidence_mode="oracle_evidence"`; oracle results must be reported separately. Split group IDs, dataset/config hashes, execution modes, and per-example validation/test predictions are recorded by `train.py`.

Local-only adapters are available in `datasets/adapters.py`:

- `load_triviaplus(path)` reads a user-provided local Parquet file when optional pandas/Parquet dependencies are installed. It retains the official split and source sentence labels. `supplementary` stays outside the three-class claim taxonomy; `not-mentioned` maps to `INSUFFICIENT_EVIDENCE`. The code does not download or redistribute the dataset. TRIVIA+ is published as CC BY-NC-ND 4.0; verify current terms and do not distribute transformed benchmark records without permission. See the [official TRIVIA+ repository](https://github.com/amazon-science/hallucination-benchmark-trivialplus).
- `load_ragtruth(response.jsonl, source_info.jsonl)` joins records by source ID and preserves span labels/offsets. It explicitly does not convert span annotations into claim-level labels. See the [official RAGTruth repository](https://github.com/ParticleMedia/RAGTruth); underlying source-corpus rights may differ from the repository's MIT license.
- RAGuard is not integrated. Its optional challenge format is not required for the core workflow.

Neither external adapter has been run against an acquired benchmark in this environment. No external dataset was downloaded. The custom benchmark remains small and is not evidence of generalization.
