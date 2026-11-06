"""
HalluciGuard-OKF Research Web Application & API.
FastAPI backend providing interactive demonstration, claim verification,
knowledge browsing, and benchmark visualization.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import HalluciGuardPipeline
from knowledge.store import OKFKnowledgeStore


app = FastAPI(
    title="HalluciGuard-OKF Research Demo",
    description="Uncertainty-Aware Atomic Claim Hallucination Detection System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")), name="static")

# Global pipeline instance (cached for responsiveness)
PIPELINE = None


def get_pipeline(abstain_threshold: Optional[float] = None, provider_name: str = "local") -> HalluciGuardPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = HalluciGuardPipeline(
            abstain_threshold=abstain_threshold,
            provider_name=provider_name,
        )
    else:
        PIPELINE.abstain_threshold = abstain_threshold
        PIPELINE.scorer.abstain_threshold = abstain_threshold
    return PIPELINE


class VerifyRequest(BaseModel):
    text: str = Field(..., description="Text or LLM response to verify")
    query: Optional[str] = Field("", description="Original user prompt or query context")
    abstain_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override validation-selected abstention threshold")
    provider: Optional[str] = Field("local", description="LLM provider: local, gemini, openai")


@app.get("/api/info")
def get_info():
    import torch
    try:
        import safetensors
        sf_ver = safetensors.__version__
    except ImportError:
        sf_ver = "Not Installed"

    store = OKFKnowledgeStore(str(PROJECT_ROOT / "knowledge/artifacts/curated_knowledge.json"))
    artifacts = store.get_all()
    model_ckpt = PROJECT_ROOT / "models/multisignal_head/best_model.safetensors"
    config_path = PROJECT_ROOT / "configs/model_config.yaml"
    run_path = PROJECT_ROOT / "evaluation/results/current_training_run.json"
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        retrieval_config = config.get("retrieval", {})
        execution_modes = {
            "retrieval_mode": config.get("embedding", {}).get("mode", "unknown"),
            "nli_mode": config.get("nli", {}).get("mode", "unknown"),
            "evidence_mode": "retrieval",
            "top_k": retrieval_config.get("top_k"),
            "min_relevance_threshold": retrieval_config.get("min_relevance_threshold"),
        }
    except (OSError, ValueError, TypeError):
        execution_modes = {}
    if run_path.exists():
        try:
            with open(run_path, "r", encoding="utf-8") as run_file:
                saved_run = json.load(run_file)
                run_metadata = saved_run.get("metadata", {})
            execution_modes.update({
                "retrieval_mode": run_metadata.get("dense_retrieval_mode", execution_modes.get("retrieval_mode")),
                "nli_mode": run_metadata.get("nli_mode", execution_modes.get("nli_mode")),
                "evidence_mode": run_metadata.get("evidence_mode", execution_modes.get("evidence_mode")),
                "top_k": run_metadata.get("top_k", execution_modes.get("top_k")),
                "min_relevance_threshold": run_metadata.get("min_relevance_threshold", execution_modes.get("min_relevance_threshold")),
                "selected_abstention_threshold": run_metadata.get("selected_abstention_threshold"),
            })
        except (OSError, ValueError, TypeError):
            pass

    return {
        "status": "ready",
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "safetensors_support": sf_ver,
        "verified_artifacts_count": len(artifacts),
        "model_loaded": model_ckpt.exists(),
        "supported_providers": ["local", "gemini", "openai"],
        "execution_modes": execution_modes,
    }


@app.get("/api/knowledge")
def get_knowledge():
    store = OKFKnowledgeStore(str(PROJECT_ROOT / "knowledge/artifacts/curated_knowledge.json"))
    artifacts = store.get_all()
    return [
        {
            "id": art.identity.id,
            "statement": art.content.statement,
            "entity": art.content.entity,
            "attribute": art.content.attribute,
            "value": art.content.value,
            "category": art.content.category,
            "source_name": art.source.name,
            "source_type": art.source.source_type,
            "reliability": art.source.reliability_estimate,
            "verified": (art.verification.status == "verified"),
            "freshness": art.lifecycle.freshness_state,
        }
        for art in artifacts
    ]


@app.get("/api/benchmark")
def get_benchmark_results():
    current_path = PROJECT_ROOT / "evaluation/results/current_training_run.json"
    if current_path.exists():
        with open(current_path, "r", encoding="utf-8") as f:
            current = json.load(f)
        result = dict(current.get("baseline_comparison", {}))
        result["selective_policy"] = current.get("selective_metrics", {})
        result["run_metadata"] = current.get("metadata", {})
        return result

    results_path = PROJECT_ROOT / "evaluation/results/experiment_results.json"
    if not results_path.exists():
        from evaluation.experiment import run_experiments
        run_experiments()

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/evaluation")
def get_current_evaluation():
    current_path = PROJECT_ROOT / "evaluation/results/current_training_run.json"
    if not current_path.exists():
        raise HTTPException(status_code=404, detail="No saved training run is available.")
    with open(current_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/experiments")
def get_saved_experiments():
    experiment_path = PROJECT_ROOT / "evaluation/results/experiment_results.json"
    if not experiment_path.exists():
        raise HTTPException(status_code=404, detail="No separate experiment artifact is available.")
    with open(experiment_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/plots/{plot_name}")
def get_plot(plot_name: str):
    allowed_plots = {
        "risk_coverage": PROJECT_ROOT / "evaluation/plots/risk_coverage_curve.png",
        "calibration": PROJECT_ROOT / "evaluation/plots/calibration_curve.png",
    }
    if plot_name not in allowed_plots:
        raise HTTPException(status_code=404, detail="Plot not found")

    plot_file = allowed_plots[plot_name]
    if not plot_file.exists():
        from evaluation.experiment import run_experiments
        run_experiments()

    return FileResponse(plot_file, media_type="image/png")


@app.post("/api/verify")
def verify_text(req: VerifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    pipeline = get_pipeline(
        abstain_threshold=req.abstain_threshold,
        provider_name=req.provider or "local",
    )
    result = pipeline.analyze(response_text=req.text, query=req.query or "")
    return result


@app.get("/", response_class=HTMLResponse)
def get_index():
    index_file = Path(__file__).resolve().parent / "templates" / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>HalluciGuard-OKF API is running. UI template loading...</h1>"


def render_dashboard(screen: str, title: str, active: str) -> str:
    templates = Path(__file__).resolve().parent / "templates"
    shell = (templates / "dashboard_shell.html").read_text(encoding="utf-8")
    content = (templates / "screens" / f"{screen}.html").read_text(encoding="utf-8")
    values = {
        "{{TITLE}}": title,
        "{{PAGE}}": screen,
        "{{CONTENT}}": content,
        "{{ACTIVE_VERIFY}}": "active" if active == "verify" else "",
        "{{ACTIVE_EVALUATION}}": "active" if active == "evaluation" else "",
        "{{ACTIVE_KNOWLEDGE}}": "active" if active == "knowledge" else "",
        "{{ACTIVE_ARCHITECTURE}}": "active" if active == "architecture" else "",
    }
    for marker, value in values.items():
        shell = shell.replace(marker, value)
    return shell


@app.get("/dashboard", include_in_schema=False)
def dashboard_redirect():
    return RedirectResponse(url="/dashboard/verify", status_code=307)


@app.get("/dashboard/verify", response_class=HTMLResponse, include_in_schema=False)
def verification_screen():
    return render_dashboard("verify", "Live verification", "verify")


@app.get("/dashboard/evaluation", response_class=HTMLResponse, include_in_schema=False)
def evaluation_screen():
    return render_dashboard("evaluation", "Evaluation & ablation", "evaluation")


@app.get("/dashboard/knowledge", response_class=HTMLResponse, include_in_schema=False)
def knowledge_screen():
    return render_dashboard("knowledge", "Knowledge base", "knowledge")


@app.get("/dashboard/architecture", response_class=HTMLResponse, include_in_schema=False)
def architecture_screen():
    return render_dashboard("architecture", "Architecture", "architecture")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)
