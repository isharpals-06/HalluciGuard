"""
Optional Cloud Providers (Gemini, OpenAI, Claude) for Claim Extraction and Verification.
All providers strictly require environment variables, implement timeouts and fallbacks.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

from ml.providers.base import VerificationProvider, ProviderClaim, ProviderVerificationResult

logger = logging.getLogger("HalluciGuard.Providers")


class GeminiProvider(VerificationProvider):
    """Google Gemini API Provider via standard REST endpoint."""

    def __init__(self, model: str = "gemini-1.5-flash", timeout_seconds: float = 8.0):
        super().__init__(name="gemini", timeout_seconds=timeout_seconds)
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def extract_claims(self, text: str) -> List[ProviderClaim]:
        if not self.is_available():
            return []
        prompt = (
            "Decompose the following text into discrete, atomic factual claims. "
            "Output strictly valid JSON list of strings, with no markdown code fences: [\"claim 1\", \"claim 2\"]:\n\n"
            f"{text}"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 500},
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                cleaned = content.replace("```json", "").replace("```", "").strip()
                claims_raw = json.loads(cleaned)
                return [
                    ProviderClaim(
                        claim_id=f"C{i+1:03d}",
                        claim_text=c,
                        source_sentence=c,
                        extraction_confidence=0.95,
                    )
                    for i, c in enumerate(claims_raw)
                ]
        except Exception as e:
            logger.warning(f"Gemini claim extraction fallback triggered: {type(e).__name__}")
            return []

    def verify_claim(self, claim: str, evidence: str) -> ProviderVerificationResult:
        if not self.is_available():
            return ProviderVerificationResult(
                status="INSUFFICIENT_EVIDENCE", confidence=0.5, explanation="Provider unavailable"
            )
        prompt = (
            f"Premise Evidence: {evidence}\n"
            f"Hypothesis Claim: {claim}\n\n"
            "Evaluate if the premise entails or contradicts the hypothesis. "
            "Reply strictly with JSON: {\"status\": \"SUPPORTED\" | \"CONTRADICTED\" | \"INSUFFICIENT_EVIDENCE\", \"confidence\": float 0.0-1.0, \"explanation\": \"reason\"}"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200},
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                cleaned = content.replace("```json", "").replace("```", "").strip()
                res = json.loads(cleaned)
                return ProviderVerificationResult(
                    status=res.get("status", "INSUFFICIENT_EVIDENCE"),
                    confidence=float(res.get("confidence", 0.70)),
                    explanation=res.get("explanation", "Verified by Gemini"),
                )
        except Exception as e:
            logger.warning(f"Gemini verification fallback triggered: {type(e).__name__}")
            return ProviderVerificationResult(
                status="INSUFFICIENT_EVIDENCE", confidence=0.50, explanation=f"Remote call failed: {type(e).__name__}"
            )


class OpenAIProvider(VerificationProvider):
    """OpenAI API Provider."""

    def __init__(self, model: str = "gpt-4o-mini", timeout_seconds: float = 8.0):
        super().__init__(name="openai", timeout_seconds=timeout_seconds)
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def extract_claims(self, text: str) -> List[ProviderClaim]:
        if not self.is_available():
            return []
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Decompose into atomic claims. Return JSON list of strings.",
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()
                claims_raw = json.loads(content)
                return [
                    ProviderClaim(
                        claim_id=f"C{i+1:03d}",
                        claim_text=c,
                        source_sentence=c,
                        extraction_confidence=0.95,
                    )
                    for i, c in enumerate(claims_raw)
                ]
        except Exception as e:
            logger.warning(f"OpenAI extraction fallback triggered: {type(e).__name__}")
            return []

    def verify_claim(self, claim: str, evidence: str) -> ProviderVerificationResult:
        if not self.is_available():
            return ProviderVerificationResult(
                status="INSUFFICIENT_EVIDENCE", confidence=0.5, explanation="Provider unavailable"
            )
        url = "https://api.openai.com/v1/chat/completions"
        prompt = (
            f"Evidence: {evidence}\nClaim: {claim}\n"
            "Return JSON: {\"status\": \"SUPPORTED\"|\"CONTRADICTED\"|\"INSUFFICIENT_EVIDENCE\", \"confidence\": float, \"explanation\": str}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()
                res = json.loads(content)
                return ProviderVerificationResult(
                    status=res.get("status", "INSUFFICIENT_EVIDENCE"),
                    confidence=float(res.get("confidence", 0.7)),
                    explanation=res.get("explanation", "Verified by OpenAI"),
                )
        except Exception as e:
            logger.warning(f"OpenAI verification fallback triggered: {type(e).__name__}")
            return ProviderVerificationResult(
                status="INSUFFICIENT_EVIDENCE", confidence=0.5, explanation=f"Remote call failed: {type(e).__name__}"
            )
