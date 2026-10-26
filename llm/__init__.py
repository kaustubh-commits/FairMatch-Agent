"""Provider-agnostic LLM wrapper: Gemini (default), Ollama (fallback), or none (demo-safe)."""

from llm.client import DEMO_SAFE, LLMClient, LLMError, make_client

__all__ = ["DEMO_SAFE", "LLMClient", "LLMError", "make_client"]
