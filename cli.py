"""
HalluciGuard-OKF Research Command-Line Interface (CLI).
Provides commands to verify text, evaluate datasets, benchmark baselines and ablations,
and inspect system status.

Usage:
  python cli.py verify --text "The capital of Australia is Sydney."
  python cli.py evaluate --benchmark datasets/benchmark/benchmark_data.json
  python cli.py benchmark
  python cli.py info
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import HalluciGuardPipeline
from knowledge.store import OKFKnowledgeStore


def format_table(headers: list, rows: list) -> str:
    """Formats a clean ASCII table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [sep, header_str, sep]
    for row in rows:
        row_str = "| " + " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row)) + " |"
        lines.append(row_str)
    lines.append(sep)
    return "\n".join(lines)


def cmd_verify(args) -> int:
    """Verifies a text string or file for hallucinations."""
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
            return 1
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("[ERROR] Either --text or --file must be provided.", file=sys.stderr)
        return 1

    pipeline = HalluciGuardPipeline(
        abstain_threshold=args.abstain_threshold,
        provider_name=args.provider,
    )

    result = pipeline.analyze(response_text=text, query=args.query or "")

    if args.format == "json":
        output_str = json.dumps(result, indent=2)
        print(output_str)
    else:
        # Table format
        print("\n" + "=" * 65)
        print("  HALLUCIGUARD-OKF CLAIM VERIFICATION REPORT")
        print("=" * 65)
        print(f"Overall Status:       {result['overall_status']}")
        print(f"Hallucination Score:  {result['hallucination_score']:.4f}")
        print(f"Total Claims:         {result['metrics']['total_claims']}")
        print(f"Supported:            {result['metrics']['supported_claims']}")
        print(f"Contradicted:         {result['metrics']['contradicted_claims']}")
        print(f"Unverified:           {result['metrics']['unverified_claims']}")
        print(f"Abstained:            {result['metrics']['abstained_claims']}")
        print(f"Average Confidence:   {result['metrics']['average_confidence']:.4f}")
        print(f"Mean Uncertainty:     {result['metrics']['mean_uncertainty']:.4f}")
        print("-" * 65)

        headers = ["ID", "Claim Text", "Status", "Conf", "Uncert", "Abst"]
        rows = []
        for c in result["claims"]:
            short_text = c["claim_text"] if len(c["claim_text"]) <= 35 else c["claim_text"][:32] + "..."
            rows.append([
                c["claim_id"],
                short_text,
                c["status"],
                f"{c['confidence']:.2f}",
                f"{c['uncertainty_score']:.2f}",
                "YES" if c["abstained"] else "NO",
            ])
        print(format_table(headers, rows))

        print("\nClaims Explanations & Evidence Grounding:")
        for c in result["claims"]:
            print(f"\n[{c['claim_id']}] ({c['status']}) {c['claim_text']}")
            print(f"  Explanation: {c['explanation']}")
            if c.get("evidence"):
                ev = c["evidence"][0]
                print(f"  Evidence:    [{ev['okf_object_id']}] {ev['text']}")
                print(f"  Source:      {c['provenance'].get('source_name', 'N/A')} (Verified: {c['provenance'].get('verified')})")
            if c["abstained"]:
                print(f"  Abstention:  {c['abstention_reason']}")
        print("=" * 65 + "\n")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Report saved to {out_p}")

    # Return exit code: 2 if hallucination detected, 0 if clean
    if result["overall_status"] == "HALLUCINATED":
        return 2
    return 0


def cmd_evaluate(args) -> int:
    """Evaluates the pipeline on a benchmark dataset."""
    from evaluation.experiment import run_experiments
    print(f"[INFO] Evaluating benchmark dataset: {args.benchmark}")
    results = run_experiments(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
    )
    headers = ["Model / Metric", "Accuracy", "Macro F1", "ECE", "Coverage"]
    rows = []
    for exp_name, m in results.items():
        if exp_name == "risk_coverage_curve":
            continue
        rows.append([
            exp_name,
            str(m.get("accuracy", "N/A")),
            str(m.get("macro_f1", "N/A")),
            str(m.get("expected_calibration_error", "N/A")),
            str(m.get("coverage", 1.0)),
        ])
    print("\n" + format_table(headers, rows) + "\n")
    return 0


def cmd_benchmark(args) -> int:
    """Runs complete baseline comparison and ablation suite."""
    from evaluation.experiment import run_experiments
    print("[INFO] Executing complete HalluciGuard-OKF Benchmark Suite...")
    results = run_experiments(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
    )
    headers = ["Configuration", "Accuracy", "Macro F1", "Macro Precision", "Macro Recall", "ECE", "Coverage"]
    rows = []
    for k, m in results.items():
        if k == "risk_coverage_curve":
            continue
        rows.append([
            k,
            str(m.get("accuracy", "N/A")),
            str(m.get("macro_f1", "N/A")),
            str(m.get("macro_precision", "N/A")),
            str(m.get("macro_recall", "N/A")),
            str(m.get("expected_calibration_error", "N/A")),
            str(m.get("coverage", 1.0)),
        ])
    print("\n" + format_table(headers, rows) + "\n")
    return 0


def cmd_info(args) -> int:
    """Displays system status, model versions, and knowledge stats."""
    import torch
    try:
        import safetensors
        sf_ver = safetensors.__version__
    except ImportError:
        sf_ver = "Not Installed"

    store = OKFKnowledgeStore("knowledge/artifacts/curated_knowledge.json")
    artifacts = store.get_all()

    model_ckpt = Path("models/multisignal_head/best_model.safetensors")
    scaler_json = Path("models/multisignal_head/feature_scaler.json")

    print("\n" + "=" * 55)
    print("      HALLUCIGUARD-OKF SYSTEM INFORMATION")
    print("=" * 55)
    print(f"Python Version:       {sys.version.split()[0]}")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"CUDA Available:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device:           {torch.cuda.get_device_name(0)}")
    print(f"SafeTensors Support:  {sf_ver}")
    print(f"Knowledge Store:      {len(artifacts)} Verified OKF Artifacts")
    print(f"Model Checkpoint:     {'EXISTS' if model_ckpt.exists() else 'MISSING'} ({model_ckpt})")
    print(f"Feature Scaler:       {'EXISTS' if scaler_json.exists() else 'MISSING'} ({scaler_json})")
    print(f"Supported Providers:  local, gemini (API key), openai (API key)")
    print("=" * 55 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="halluciguard",
        description="HalluciGuard-OKF: Uncertainty-Aware Claim-Level Hallucination Detection System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify text or file for factual hallucinations")
    p_verify.add_argument("--text", "-t", type=str, help="Raw text string to verify")
    p_verify.add_argument("--file", "-f", type=str, help="Path to text file to verify")
    p_verify.add_argument("--query", "-q", type=str, default="", help="Original user prompt or query context")
    p_verify.add_argument("--abstain-threshold", type=float, default=0.0, help="Confidence threshold for selective abstention (default: 0.0)")
    p_verify.add_argument("--provider", type=str, default="local", help="LLM provider: local, gemini, openai")
    p_verify.add_argument("--format", choices=["table", "json"], default="table", help="Output display format")
    p_verify.add_argument("--output", "-o", type=str, help="Optional path to save JSON results")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run evaluation on benchmark dataset")
    p_eval.add_argument("--benchmark", type=str, default="datasets/benchmark/benchmark_data.json", help="Path to benchmark JSON")
    p_eval.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory to save evaluation results")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full baseline and ablation experiments")
    p_bench.add_argument("--benchmark", type=str, default="datasets/benchmark/benchmark_data.json", help="Path to benchmark JSON")
    p_bench.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory to save results")
    p_bench.add_argument("--plot-dir", type=str, default="evaluation/plots", help="Directory to save plots")

    # info
    subparsers.add_parser("info", help="Display system configuration and model status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "verify":
        return cmd_verify(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "info":
        return cmd_info(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
