# Configuration Guide

HalluciGuard-OKF supports flexible configuration through YAML files, environment variables, and CLI parameters.

---

## 1. Environment Variables (`.env`)

To configure cloud providers and logging, copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Supported Variables:
| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Optional | `""` | Google Gemini API key for external generation |
| `OPENAI_API_KEY` | Optional | `""` | OpenAI API key for external generation |
| `DEFAULT_PROVIDER` | No | `local` | Active provider (`local`, `gemini`, `openai`) |
| `HALLUCIGUARD_ENV` | No | `development` | Runtime environment (`development`, `production`) |
| `LOG_LEVEL` | No | `INFO` | Console logging verbosity (`DEBUG`, `INFO`, `WARNING`) |

---

## 2. Training Configuration (`configs/train.yaml`)

Training hyperparameters, data split ratios, and artifact paths are controlled via `configs/train.yaml`:

```yaml
experiment:
  name: "halluciguard_multisignal_v1"
  seed: 42
  description: "PyTorch Multi-Signal Classification Head with class weighting and calibration"

data:
  benchmark_path: "datasets/benchmark/benchmark_data.json"
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  stratify: true
  random_seed: 42

training:
  epochs: 80
  batch_size: 16
  learning_rate: 0.003
  weight_decay: 0.0001
  patience: 12
  min_delta: 0.001
  loss_type: "weighted_cross_entropy"
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"
  t_max: 80
  eta_min: 0.00005

model:
  input_dim: 10
  hidden_dims: [64, 32]
  dropout: 0.20
  num_classes: 3

artifacts:
  output_dir: "models/multisignal_head"
  checkpoint_name: "best_model.pt"
  metadata_name: "model_metadata.json"
  scaler_name: "feature_scaler.json"
```

---

## 3. Pipeline Runtime Defaults

> Values below are historical unless repeated in `configs/model_config.yaml`. Current model metadata may set the validation-selected threshold. Runtime modes and current feature schema are described in `IMPLEMENTATION_STATUS.md`.

The inference pipeline accepts configurable thresholds in `inference.py`:

- `min_relevance_threshold` (default: `0.35`): Minimum combined retrieval score required to consider candidate evidence valid.
- `abstain_threshold` (default: `0.0` in forced mode, `0.44` in selective mode): Calibrated confidence threshold below which claims are marked as `ABSTAINED`.
- `alpha` (default: `0.60`): Weight given to dense semantic retrieval vs sparse BM25 retrieval ($1 - \alpha$).
