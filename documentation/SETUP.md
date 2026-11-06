# Installation and Setup Guide

## 1. System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS (tested on Windows 11).
- **Python Version:** Python 3.10 or Python 3.11 (Python 3.11 recommended; avoid 3.14 due to missing PyTorch wheels).
- **Hardware Acceleration:** CPU fully supported. CUDA 11.8+ / 12.1+ supported if NVIDIA GPU is present.
- **Disk Space:** ~2 GB for virtual environment and model dependencies.

---

## 2. Virtual Environment Setup

We recommend `uv` for fast, reproducible dependency installation:

```bash
# Clone the repository
git clone https://github.com/your-org/HalluciGuard-OKF.git
cd HalluciGuard-OKF

# Create Python 3.11 virtual environment
uv venv --python 3.11 .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
source .venv/bin/activate
```

---

## 3. Dependency Installation

Install pinned core dependencies:

```bash
# Using uv (fast)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

### Core Packages:
- `torch>=2.0.0`
- `safetensors>=0.4.0`
- `transformers>=4.30.0`
- `pydantic>=2.0.0`
- `scikit-learn>=1.3.0`
- `fastapi>=0.100.0`
- `uvicorn>=0.20.0`
- `matplotlib>=3.7.0`
- `pyyaml>=6.0`
- `pytest>=7.0.0`

---

## 4. Verification of Installation

Verify your environment using the CLI diagnostic tool:

```bash
python cli.py info
```

Expected output:
```
=======================================================
      HALLUCIGUARD-OKF SYSTEM INFORMATION
=======================================================
Python Version:       3.11.x
PyTorch Version:      2.15.x
CUDA Available:       False (or True if GPU detected)
SafeTensors Support:  0.8.x
Knowledge Store:      31 Verified OKF Artifacts
Model Checkpoint:     EXISTS (models\multisignal_head\best_model.safetensors)
Feature Scaler:       EXISTS (models\multisignal_head\feature_scaler.json)
Supported Providers:  local, gemini, openai
=======================================================
```

Run test suite to confirm end-to-end functionality:
```bash
pytest tests
```
All 29 tests should pass in under 15 seconds.
