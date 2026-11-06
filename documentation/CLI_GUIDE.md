# HalluciGuard-OKF CLI User Guide

> Current status (2026-11-05): this guide describes an earlier CLI. Default abstention now uses the validation-selected threshold stored in model metadata; `--abstain-threshold` overrides it. Confidence calibration has not been established beyond a small custom validation set. See `IMPLEMENTATION_STATUS.md`.

The HalluciGuard-OKF research command-line interface provides high-level commands for verifying text, executing reproducible benchmark evaluations, running ablation suites, and inspecting system diagnostics.

## Entry Point
```bash
python cli.py [COMMAND] [OPTIONS]
```

---

## 1. Commands

### 1.1 `verify`
Verifies an input text string or file for factual hallucinations and unsupported assertions.

```bash
# Verify direct text string (Table display)
python cli.py verify --text "The capital city of Australia is Sydney."

# Output structured JSON
python cli.py verify --text "The speed of light in a vacuum is 299,792,458 m/s." --format json

# Verify a file with selective abstention
python cli.py verify --file sample_text.txt --abstain-threshold 0.44

# Export verification report to file
python cli.py verify --text "The moon landing occurred in 1969." --output report.json
```

**Options:**
- `--text`, `-t`: Direct string to analyze.
- `--file`, `-f`: Path to plain-text file to analyze.
- `--query`, `-q`: Optional user prompt or question context.
- `--abstain-threshold`: Optional explicit confidence threshold override. If omitted, use the model metadata threshold selected on validation. Use `--forced` only for forced classification.
- `--provider`: LLM provider (`local`, `gemini`, `openai`).
- `--format`: Display format (`table` or `json`).
- `--output`, `-o`: File path to save output JSON.

**Exit Codes:**
- `0`: Success (All claims verified / supported or clean).
- `1`: Processing or file read error.
- `2`: Hallucination detected (One or more claims contradicted).

---

### 1.2 `evaluate`
Evaluates the pipeline on a specified benchmark dataset and prints summary metric tables.

```bash
python cli.py evaluate --benchmark datasets/benchmark/benchmark_data.json
```

---

### 1.3 `benchmark`
Executes the full research benchmark suite including Baselines 1–3, the proposed selective prediction model, and all five ablation experiments. Generates Risk-Coverage and Calibration plots.

```bash
python cli.py benchmark --output-dir evaluation/results --plot-dir evaluation/plots
```

---

### 1.4 `info`
Displays diagnostic system telemetry including Python version, PyTorch device, SafeTensors version, verified OKF artifact count, and model checkpoint status.

```bash
python cli.py info
```
