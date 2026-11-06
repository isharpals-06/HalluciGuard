# RESEARCH GAP AND NOVELTY ANALYSIS: HALLUCIGUARD-OKF

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Audit Target:** Scientific Novelty, Literature Positioning, and High-Impact Research Roadmap  
**Auditor Roles:** Senior AI/ML Research Scientist, LLM & Transformer Architect, NLP Research Engineer  
**Date:** November 3, 2026  
**Status:** COMPLETE  

---

## 1. Scientific Novelty Assessment

### 1.1 Is HalluciGuard-OKF a Genuine Research Contribution?
In its current implementation, **HalluciGuard-OKF does not constitute a novel scientific contribution (`CONFIRMED`)**. 

Rather, it is an early, highly constrained integration of well-known established techniques:
- **Claim Decomposition:** Decomposing texts into atomic propositions has been extensively researched since ClaimBuster (Hassan et al., 2017), FEVER (Thorne et al., 2018), and FActScore (Min et al., 2023). The implementation's regex-based decomposition is a primitive approximation of existing academic parsers.
- **Retrieval-Augmented Verification:** Retrieving evidence from a knowledge store to verify claims is the standard pipeline established in Fact-Checking (Guo et al., 2022) and RAG verification frameworks (RAGAS, TruLens).
- **Multi-Signal Feature Fusion:** Hand-crafting similarity and NLI features into an MLP classification head is a standard tabular ML technique that predates modern deep learning.
- **The "Open Knowledge Format (OKF)" Concept:** While the project markets "Google's Open Knowledge Format (OKF)", in reality, the schema simply implements standard JSON fields for source citation, timestamps, and verification status. This is functionally equivalent to classical semantic web ontologies (PROV-O, Dublin Core, Wikidata qualifiers, RDF reification) and temporal knowledge graphs.

### 1.2 The Gap Between Vision and Implementation
The vision outlined in the README—creating an auditable, temporal-aware, provenance-grounded verification system—is **scientifically valuable**. However, the current code relies on shortcut heuristics (hardcoded Australian capital city checks, regex year matching) that bypass the actual scientific problems of hallucination detection.

---

## 2. Five Rigorous, High-Impact Research Directions

To transform this project into a top-tier scientific contribution (suitable for publication at ACL, EMNLP, NeurIPS, or ICLR), the following five research directions are formulated.

---

### Research Direction 1: Uncertainty-Aware Atomic Claim Decomposition with Adaptive Granularity
1. **Research Problem:**
   Existing claim decomposition models (e.g. FActScore, Minicons) split text into either overly fine-grained tokens (losing coreferential context) or compound sentences (burying subtle hallucinations). They treat all extracted claims with uniform certainty.
2. **Existing Limitation:**
   Fixed heuristics or rigid LLM prompting fail on polysemous sentences, metaphors, and complex syntactic dependencies, frequently producing orphaned fragments whose truth value cannot be evaluated.
3. **Proposed Contribution:**
   Develop an **Adaptive Granularity Decomposition Network (AGD-Net)** that outputs hierarchical claim trees rather than flat lists. Each atomic claim is assigned an extraction epistemic uncertainty score $U_e \in [0, 1]$ conditioned on syntactic dependency parsing and sentence-level semantic preservation.
4. **Technical Implementation Requirements:**
   - Fine-tune a lightweight model (e.g. Llama-3.2-3B or Qwen-2.5-3B) with LoRA on syntactic dependency corpora annotated with claim boundaries.
   - Constrain decoding using Grammar-Guided Generation (GBNF / Outlines) to guarantee well-formed standalone proposition triples $(S, P, O, \text{Modifiers})$.
5. **Required Datasets:**
   - FActScore biography dataset (Min et al., 2023).
   - CoNLL-2012 Shared Task (OntoNotes 5.0) for coreference resolution.
   - Benchmark subset of 1,000 multi-sentence LLM responses with human-annotated atomic boundaries.
6. **Evaluation Metrics:**
   - Claim Atomicity Score, Context Independence (CI-Score), Coreference Accuracy, Human Alignment F1.
7. **Expected Difficulty:** Medium-High.
8. **Research Risks:** Over-fragmentation creating un-verifiable micro-claims; high computational cost of autoregressive decomposition.
9. **How Novelty is Demonstrated:**
   Demonstrate higher downstream verification F1 on FActScore benchmark compared to vanilla ChatGPT/GPT-4 extraction baselines.

---

### Research Direction 2: Temporal Conflict Resolution in Attributed Fact Verification
1. **Research Problem:**
   World facts evolve over time (e.g. CEOs, political office holders, corporate valuations, scientific consensus). LLMs frequently hallucinate by blending valid historical facts with current states.
2. **Existing Limitation:**
   Standard NLI models treat premises as atemporal. Given premise *"Rishi Sunak served as Prime Minister until July 2024"* and hypothesis *"Rishi Sunak is the current Prime Minister"*, standard NLI models often output Entailment or Neutral because they lack grounding in the reference query timestamp.
3. **Proposed Contribution:**
   Formulate a **Temporal-Constraint NLI Framework (T-NLI)** that explicitly binds temporal validity intervals $[t_{\text{start}}, t_{\text{end}}]$ to NLI premise-hypothesis pairs, mapping temporal consistency into 4D interval algebra (Allen's Interval Algebra).
4. **Technical Implementation Requirements:**
   - Temporal tagger (e.g. SUTime / HeidelTime) integrated into premise and hypothesis encoding.
   - Cross-attention cross-encoder modifying DeBERTa-v3 attention matrices with temporal distance bias.
5. **Required Datasets:**
   - TempLama (Dhingra et al., 2022).
   - TimeQA (Chen et al., 2021).
   - Curated temporal contradiction benchmark across political and sports domain changes.
6. **Evaluation Metrics:**
   - Temporal F1, Stale-State Contradiction Detection Recall, Macro-AUC on time-sensitive claims.
7. **Expected Difficulty:** High.
8. **Research Risks:** Ambiguity in undated web corpora; complex temporal scope expressions in informal text.
9. **How Novelty is Demonstrated:**
   Achieve state-of-the-art performance on TempLama and TimeQA hallucination detection where standard NLI models fail.

---

### Research Direction 3: Cross-Document Evidence Reconciliation and Contradiction Disambiguation
1. **Research Problem:**
   In real-world verification, retrieved documents frequently disagree (conflicting statistical estimates, partisan news sources, outdated vs. updated studies). Existing detectors naively take the top-1 retrieved document or concatenate text blindly.
2. **Existing Limitation:**
   HalluciGuard-OKF takes strictly `retrieved[0]` ([inference.py#L125](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L125)), ignoring all consensus or disagreement across sources.
3. **Proposed Contribution:**
   Design a **Consensus-Graph Evidence Verification Network (CGE-Net)**. When top-$K$ evidence documents are retrieved, the system constructs an evidence graph where edges represent mutual entailment or contradiction between sources, weighted by source institutional credibility priors.
4. **Technical Implementation Requirements:**
   - Graph Neural Network (GNN) / Graph Attention Network (GAT) layer operating on evidence embeddings.
   - Joint objective combining multi-document agreement with claim entailment.
5. **Required Datasets:**
   - Multi-News contradiction subsets.
   - FEVEROUS (Aly et al., 2021) multi-hop table and text evidence dataset.
   - SciFact (Wadden et al., 2020) biomedical claim verification with conflicting findings.
6. **Evaluation Metrics:**
   - Evidence Selection F1, Source Disagreement Disambiguation Accuracy, Claim-Level AUROC.
7. **Expected Difficulty:** High.
8. **Research Risks:** Graph construction adds $O(K^2)$ NLI overhead per claim.
9. **How Novelty is Demonstrated:**
   Prove that CGE-Net prevents false rejections when low-quality contradictory sources pollute the retrieval pool.

---

### Research Direction 4: Robust Adversarial Claim Attribution via Counterfactual Probing
1. **Research Problem:**
   Hallucination detectors are vulnerable to surface lexical adversarial attacks (keyword stuffing, negation insertion, misleading paraphrasing), as demonstrated in Phase 4.
2. **Existing Limitation:**
   Detectors over-rely on lexical similarity features, allowing adversaries to deceive verification engines into approving false claims simply by mirroring source vocabulary.
3. **Proposed Contribution:**
   Develop a **Counterfactual Attribution Probing Engine (CAPE)**. For each verified claim, the engine automatically generates minimal counterfactual entity/relation edits and tests whether the verification score drops sharply. If editing the core factual assertion does not diminish the verification score, the detector flags the claim as ungrounded.
4. **Technical Implementation Requirements:**
   - Masked language model or diffusion text editor to generate minimal counterfactual perturbations.
   - Sensitivity gradient analysis computing $\nabla_{\text{claim}} \text{Score}$.
5. **Required Datasets:**
   - Adversarial NLI (ANLI) (Nie et al., 2020).
   - PAWS (Paraphrase Adversaries from Word Scrambling) (Zhang et al., 2019).
   - Synthetic perturbation suite generated on HaluEval.
6. **Evaluation Metrics:**
   - Robustness under Perturbation ($\Delta \text{F1}$), Attack Success Rate (ASR) reduction, Certified Accuracy.
7. **Expected Difficulty:** Medium-High.
8. **Research Risks:** High inference latency due to multi-sample counterfactual generation.
9. **How Novelty is Demonstrated:**
   Demonstrate superior robustness against lexical adversarial attacks on ANLI and perturbed RAG datasets without sacrificing standard accuracy.

---

### Research Direction 5: Self-Calibrated Selective Abstention under Incomplete Knowledge
1. **Research Problem:**
   Hallucination detectors operating in production must know what they do not know. In open-domain environments, knowledge bases are inherently incomplete.
2. **Existing Limitation:**
   Current systems either guess or assign arbitrary static default confidence scores ($0.95$ for unverified in line 204 of `inference.py`), leading to poor calibration and severe overconfidence.
3. **Proposed Contribution:**
   A **Conformal Risk-Controlled Selective Verification Framework (CRCS-Verify)** based on Conformal Prediction theory. The system guarantees a user-specified bound on the False Positive Rate (e.g., $\text{FPR} \le 5\%$) by adaptively expanding prediction sets to $\{\text{SUPPORTED}, \text{UNVERIFIED}\}$ or abstaining from classification when epistemic uncertainty is elevated.
4. **Technical Implementation Requirements:**
   - Conformal prediction calibration on validation non-conformity scores.
   - Dual-head uncertainty estimation separating aleatoric (data) and epistemic (retrieval) uncertainty.
5. **Required Datasets:**
   - Open-domain FEVER + Real-world enterprise RAG generation traces (Finance, Healthcare).
6. **Evaluation Metrics:**
   - Empirical Coverage, Abstention Rate, Brier Score, Expected Calibration Error (ECE) at bounded risk.
7. **Expected Difficulty:** Medium.
8. **Research Risks:** Conservative abstention rates degrading overall utility.
9. **How Novelty is Demonstrated:**
   Provide mathematical finite-sample guarantees on hallucination detection precision across open-domain datasets.
