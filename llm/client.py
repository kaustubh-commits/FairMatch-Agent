"""
A tiny, provider-agnostic LLM wrapper.

    client = make_client("gemini")      # or "ollama" / "demo-safe"
    text = client.complete("Say hi", json_mode=False)

Every provider has the same `complete(prompt, json_mode)` method and raises
LLMError when anything goes wrong (no key, network error, bad response).
The agents catch LLMError and switch to the rule-based demo-safe mode, so a
failing API never breaks the demo.

Only the Python standard library is used (urllib), so there is nothing extra
to install and you can read exactly what is sent over the network.

Settings come from environment variables (or the .env file):
  LLM_PROVIDER    gemini | ollama | demo-safe   (default: gemini)
  GEMINI_API_KEY  your key
  GEMINI_MODEL    default: gemini-3.5-flash
  GEMINI_THINKING low | high | empty to use the model default (default: low, faster)
  OLLAMA_HOST     default: http://localhost:11434
  OLLAMA_MODEL    default: llama3.2
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

DEMO_SAFE = "demo-safe"
# Google retires models for new keys (2.5 already returns 404), so if this one
# stops working, set GEMINI_MODEL in .env to a newer flash model.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_OLLAMA_MODEL = "llama3.2"
TIMEOUT_SECONDS = 60   # thinking models can take a while

ROOT = Path(__file__).resolve().parent.parent


class LLMError(Exception):
    """Raised for any LLM failure. Callers fall back to demo-safe mode."""


def load_env(path: Path = ROOT / ".env") -> None:
    """
    Minimal .env reader: KEY=value lines, '#' comments ignored.
    Values already set in the real environment win.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# "Busy, try again" errors (rate limit, overloaded) get a couple of retries.
RETRY_STATUSES = {429, 500, 503}
RETRY_WAITS = [2, 5]   # seconds to wait before retry 1 and retry 2


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    """POST a JSON payload and return the decoded JSON reply (or raise LLMError)."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            if e.code in RETRY_STATUSES and attempt < len(RETRY_WAITS):
                time.sleep(RETRY_WAITS[attempt])
                continue
            # Show the provider's error message, never the request (it holds the key).
            detail = " ".join(e.read().decode(errors="replace").split())[:200]
            retried = f" (after {attempt} retries)" if attempt else ""
            raise LLMError(f"HTTP {e.code}{retried}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"network error: {e}") from None
        except json.JSONDecodeError:
            raise LLMError("reply was not JSON") from None


class LLMClient:
    """Base class. `name` is shown in the trace."""

    name = "base"

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Google Gemini via the REST generateContent endpoint."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL, thinking: str = "low"):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set (add it to .env)")
        self.api_key, self.model, self.thinking = api_key, model, thinking
        self.name = f"gemini:{model}"

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        config = {"temperature": 0}
        if json_mode:
            config["responseMimeType"] = "application/json"
        if self.thinking:  # our tasks are simple; less "thinking" = faster replies
            config["thinkingConfig"] = {"thinkingLevel": self.thinking}
        reply = _post_json(
            self.URL.format(model=self.model),
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
             "generationConfig": config},
            headers={"x-goog-api-key": self.api_key},
        )
        try:
            parts = reply["candidates"][0]["content"]["parts"]
            # Skip the model's internal "thought" parts; keep only the answer text.
            return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except (KeyError, IndexError):
            raise LLMError(f"unexpected Gemini reply: {str(reply)[:200]}") from None


class OllamaClient(LLMClient):
    """A local model served by Ollama (https://ollama.com)."""

    def __init__(self, host: str = "http://localhost:11434", model: str = DEFAULT_OLLAMA_MODEL):
        self.host, self.model = host.rstrip("/"), model
        self.name = f"ollama:{model}"

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        payload = {"model": self.model, "stream": False, "options": {"temperature": 0},
                   "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            payload["format"] = "json"
        reply = _post_json(f"{self.host}/api/chat", payload, headers={})
        try:
            return reply["message"]["content"].strip()
        except KeyError:
            raise LLMError(f"unexpected Ollama reply: {str(reply)[:200]}") from None


class ChainClient(LLMClient):
    """
    Try each client in order; the first that answers wins.
    Used for "Gemini, and if that fails, Ollama".
    """

    def __init__(self, clients: list[LLMClient]):
        self.clients = clients
        self.name = clients[0].name + "".join(f" (fallback {c.name})" for c in clients[1:])
        self.last_used: Optional[str] = None

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        errors = []
        for client in self.clients:
            try:
                text = client.complete(prompt, json_mode)
                self.last_used = client.name
                return text
            except LLMError as e:
                errors.append(f"{client.name}: {e}")
        raise LLMError("; ".join(errors))


def make_client(provider: Optional[str] = None) -> tuple[Optional[LLMClient], str]:
    """
    Build the client for `provider` (or LLM_PROVIDER from the environment).

    Returns (client, note). client is None in demo-safe mode, or when no
    provider could be set up; `note` says why, for the trace.
      gemini -> Gemini, falling back to Ollama on each failed call
      ollama -> Ollama only
    """
    load_env()
    provider = (provider or os.environ.get("LLM_PROVIDER") or "gemini").lower()
    ollama = OllamaClient(os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                          os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))

    if provider == DEMO_SAFE:
        return None, "demo-safe mode: rule-based planner and template explainer, no LLM"
    if provider == "ollama":
        return ollama, f"using {ollama.name}"
    if provider == "gemini":
        try:
            gemini = GeminiClient(os.environ.get("GEMINI_API_KEY", ""),
                                  os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
                                  os.environ.get("GEMINI_THINKING", "low"))
        except LLMError as e:
            return ollama, f"{e}; trying {ollama.name} instead"
        chain = ChainClient([gemini, ollama])
        return chain, f"using {chain.name}"
    raise ValueError(f"Unknown LLM provider '{provider}'. Use gemini, ollama or demo-safe.")
