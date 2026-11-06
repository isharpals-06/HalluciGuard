"""
Integration tests for HalluciGuard-OKF CLI and Web Application API.
"""

import subprocess
import sys
import pytest
from fastapi.testclient import TestClient
from web.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_cli_info_command():
    """Test `python cli.py info` runs cleanly with exit code 0."""
    res = subprocess.run(
        [sys.executable, "cli.py", "info"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "HALLUCIGUARD-OKF SYSTEM INFORMATION" in res.stdout
    assert "31 Verified OKF Artifacts" in res.stdout


def test_cli_verify_supported():
    """Test `python cli.py verify` with verified claim returns code 0."""
    res = subprocess.run(
        [sys.executable, "cli.py", "verify", "--text", "The speed of light in a vacuum is exactly 299,792,458 meters per second."],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "VERIFIED" in res.stdout
    assert "SUPPORTED" in res.stdout


def test_cli_verify_hallucination():
    """Test `python cli.py verify` with contradicted claim returns code 1 or 2."""
    res = subprocess.run(
        [sys.executable, "cli.py", "verify", "--text", "The capital city of Australia is Sydney."],
        capture_output=True,
        text=True,
    )
    # Pipeline exits with non-zero when hallucination is detected
    assert res.returncode in (1, 2)
    assert "HALLUCINATED" in res.stdout
    assert "Canberra" in res.stdout


def test_web_api_info_endpoint(client):
    """Test GET /api/info returns ready status and artifact counts."""
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["verified_artifacts_count"] >= 30
    assert data["model_loaded"] is True


def test_web_api_knowledge_endpoint(client):
    """Test GET /api/knowledge returns list of OKF artifacts."""
    res = client.get("/api/knowledge")
    assert res.status_code == 200
    artifacts = res.json()
    assert isinstance(artifacts, list)
    assert len(artifacts) >= 30
    assert "statement" in artifacts[0]


def test_web_api_benchmark_endpoint(client):
    """Test GET /api/benchmark returns evaluation results."""
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "heuristic_baseline" in data
    assert "proposed_model" in data
    assert "selective_policy" in data


def test_web_api_verify_endpoint(client):
    """Test POST /api/verify with supported and unverified inputs."""
    # 1. Supported
    res1 = client.post(
        "/api/verify",
        json={"text": "The speed of light in a vacuum is exactly 299,792,458 meters per second."},
    )
    assert res1.status_code == 200
    assert res1.json()["overall_status"] == "VERIFIED"

    # 2. Empty text rejection
    res2 = client.post("/api/verify", json={"text": "   "})
    assert res2.status_code == 400


def test_web_api_plots_endpoint(client):
    """Test GET /api/plots/risk_coverage returns image/png."""
    res = client.get("/api/plots/risk_coverage")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 10000


def test_frontend_routes_and_favicon(client):
    """The public landing page and dedicated research screens are served."""
    landing = client.get("/")
    assert landing.status_code == 200
    assert "Check factual claims against evidence" in landing.text
    assert "/static/favicon.svg" in landing.text

    for route, marker in (
        ("/dashboard/verify", "Inspect a response, claim by claim."),
        ("/dashboard/evaluation", "RESEARCH EVALUATION"),
        ("/dashboard/knowledge", "Knowledge base"),
        ("/dashboard/architecture", "Architecture"),
    ):
        response = client.get(route)
        assert response.status_code == 200
        assert marker in response.text
        assert "LOCAL RESEARCH DEMO" not in response.text

    favicon = client.get("/static/favicon.svg")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"] == "image/svg+xml"


def test_frontend_evaluation_data_endpoints(client):
    evaluation = client.get("/api/evaluation")
    assert evaluation.status_code == 200
    assert "test_metrics" in evaluation.json()

    experiments = client.get("/api/experiments")
    assert experiments.status_code == 200
    assert "Ablation_wo_Freshness" in experiments.json()
