"""
Provider Factory for HalluciGuard-OKF.
Provides seamless switching between Local, Gemini, and OpenAI providers.
"""

from typing import Optional
from ml.providers.base import VerificationProvider
from ml.providers.local_provider import LocalProvider
from ml.providers.cloud_providers import GeminiProvider, OpenAIProvider


def get_provider(provider_name: str = "local") -> VerificationProvider:
    """
    Returns the requested verification provider.
    Falls back gracefully to LocalProvider if a cloud provider is unavailable.
    """
    name_clean = provider_name.lower().strip() if provider_name else "local"

    if name_clean == "gemini":
        prov = GeminiProvider()
        if prov.is_available():
            return prov
        print("[Provider] GEMINI_API_KEY not found or invalid. Falling back to LocalProvider.")
        return LocalProvider()

    elif name_clean in ("openai", "gpt"):
        prov = OpenAIProvider()
        if prov.is_available():
            return prov
        print("[Provider] OPENAI_API_KEY not found or invalid. Falling back to LocalProvider.")
        return LocalProvider()

    return LocalProvider()
