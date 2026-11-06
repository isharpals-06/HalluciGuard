# ARCHITECTURAL AND ENGINEERING DECISIONS: HALLUCIGUARD-OKF
**Document:** Architecture Decision Records (ADRs)  
**Author:** NLP & LLM Systems Architect, Senior Software Architect  
**Date:** November 3, 2026  
**Status:** ACTIVE  

---

## ADR-001: Environment and Dependency Management using `uv` and Python 3.11

- **Decision:** Use `uv` to manage a dedicated Python 3.11 virtual environment (`.venv`) with strict dependency pinning in `pyproject.toml`.
- **Context:** The host machine defaults to Python 3.14, for which PyTorch does not provide official prebuilt Windows wheels, causing runtime crashes (`ModuleNotFoundError: No module named 'torch'`).
- **Alternatives Considered:**
  1. Relying on default host Python 3.14 without PyTorch (disables neural models and breaks existing codebase).
  2. Installing full Conda / Docker stack (violates resource constraints and introduces excessive complexity).
  3. `uv` with Python 3.11 (`cpython-3.11.15-windows-x86_64-none` already present in `uv` registry).
- **Reason for Selection:** `uv` is extremely fast, reproducible, and provides native virtual environments that run cleanly on Windows without external daemon overhead.
- **Trade-offs:** Requires running commands with `.venv\Scripts\python.exe` or `uv run`.
- **Impact:** Solves the primary execution blocker; ensures 100% reproducible execution across local developer environments.

---

## ADR-002: Modular Atomic Claim Extraction with Syntactic Clause Decomposition

- **Decision:** Implement a modular claim extraction pipeline combining syntactic clause boundary splitting, pronoun coreference preservation, and structured proposition formation, with clear fallback handling.
- **Context:** The existing implementation in `ml/claim_extraction/extractor.py` relied on brittle, hardcoded regular expressions specifically matching *"Tesla was founded in 2008 by Elon Musk"*.
- **Alternatives Considered:**
  1. Full large-scale LLM extraction on every query (high latency, external API dependency, cost).
  2. Pure spaCy parser (requires separate model asset download).
  3. Hybrid syntactic clause decomposition with regular expression fallback and optional LLM provider enhancement.
- **Reason for Selection:** Ensures the system runs offline with zero external dependencies while decomposing compound sentences, relative clauses, and multi-sentence paragraphs robustly.
- **Trade-offs:** Pure syntactic clause splitting cannot resolve every colloquial metaphor without a model.
- **Impact:** Eliminates test-cheating regexes; generalizes claim extraction across arbitrary multi-claim paragraphs.

---

## ADR-003: Core Research Contribution — Uncertainty-Aware Verification with Selective Abstention

- **Decision:** Frame the core research novelty around **Selective Prediction and Abstention under Epistemic Uncertainty**, comparing against forced-label baselines.
- **Context:** Previous claims of a "novel deep neural multi-signal architecture" were invalidated because the MLP was trained on 19 samples and performed identically to a 10-line heuristic.
- **Alternatives Considered:**
  1. Claiming novelty in the OKF data format (invalid; OKF is an ontology schema, not an algorithmic discovery).
  2. Claiming novelty in BM25 retrieval (invalid; standard IR technique).
  3. Formulating a formal selective classification framework with risk-coverage analysis for factual verification.
- **Reason for Selection:** Directly answers RQ3 and RQ4; addresses the fundamental hallucination problem where models make confident incorrect claims when evidence is missing.
- **Trade-offs:** Requires calculating risk-coverage curves and coverage metrics on held-out splits.
- **Impact:** Provides a scientifically defensible, publishable research hypothesis.

---

## ADR-004: Safe Serialization and Elimination of Insecure Pickling

- **Decision:** Replace unverified `torch.load()` and `joblib.load()` with `safetensors` for model weights and JSON/NumPy `.npz` arrays for scaler parameters.
- **Context:** The security audit identified VULN-01 (CVSS 9.8 Critical), where loading `.pt` and `.pkl` files allows arbitrary remote code execution via pickle exploitation.
- **Alternatives Considered:**
  1. Adding `weights_only=True` to `torch.load()` (mitigates PyTorch, but does not fix `joblib.load()`).
  2. Migrating to `safetensors` and JSON-serialized parameter dictionaries.
- **Reason for Selection:** Completely eliminates arbitrary code execution vectors across both model and preprocessor artifacts.
- **Trade-offs:** Requires a one-time conversion script for legacy checkpoints.
- **Impact:** Resolves CVSS 9.8 critical vulnerability to achieve enterprise security compliance.

---

## ADR-005: Lightweight FastAPI Web Application Architecture

- **Decision:** Build a responsive, research-oriented web UI served directly by FastAPI and Uvicorn using clean semantic HTML5, CSS3, and modern vanilla JavaScript.
- **Context:** The README claimed a complex React/Node.js frontend that did not exist. Introducing a separate Node.js toolchain adds unnecessary build overhead.
- **Alternatives Considered:**
  1. Setting up a full React / Next.js npm project (heavyweight, requires Node.js, complex deployment).
  2. Streamlit / Gradio (introduces bulky additional dependencies).
  3. Single-service FastAPI with integrated static/HTML5 dashboard.
- **Reason for Selection:** Runs directly out-of-the-box on the existing Python runtime with zero external build tools, sub-millisecond load times, and clean UI components for claim-level inspection.
- **Trade-offs:** Custom UI components must be styled cleanly without npm packages.
- **Impact:** Delivers a functional web interface with zero extra toolchain requirements.
