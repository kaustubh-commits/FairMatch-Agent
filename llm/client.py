"""
A tiny, provider-agnostic LLM wrapper.

    client = make_client("gemini")      # or "ollama" / "demo-safe"
    text = client.complete("Say hi", json_mode=False)

Every provider has the same `complete(prompt, json_mode)` method and raises
LLMError when anything goes wrong (no key, quota, network error, bad reply).
Each LLMError has a `kind` (e.g. "quota") so the trace can show a friendly
line instead of a raw JSON error. The agents catch LLMError and switch to the
rule-based demo-safe mode, so a failing API never breaks the demo.

In "gemini" mode the clients are tried in order:
    GEMINI_MODEL -> GEMINI_FALLBACK_MODEL -> Ollama
Each Gemini model has its own free-tier quota, so when the main model's
daily quota is used up the lighter model usually still answers.

Only the Python standard library is used (urllib), so there is nothing extra
to install and you can read exactly what is sent over the network.

Settings come from environment variables (or the .env file):
  LLM_PROVIDER           gemini | ollama | demo-safe   (default: gemini)
  GEMINI_API_KEY         your key
  GEMINI_MODEL           default: gemini-3.5-flash
  GEMINI_FALLBACK_MODEL  default: gemini-3.5-flash-lite (empty = no fallback model)
  GEMINI_THINKING        low | high | empty for the model default (default: low, faster)
  OLLAMA_HOST            default: http://localhost:11434
  OLLAMA_MODEL           default: llama3.2
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
# Google retires models for new keys (2.5 already returns 404), so if these
# stop working, set GEMINI_MODEL / GEMINI_FALLBACK_MODEL in .env to newer ones.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_FALLBACK = "gemini-3.5-flash-lite"
DEFAULT_OLLAMA_MODEL = "llama3.2"
TIMEOUT_SECONDS = 60   # thinking models can take a while

ROOT = Path(__file__).resolve().parent.parent

# Error kinds -> short, human wording for the trace.
FRIENDLY = {
    "quota": "quota reached",
    "overloaded": "overloaded",
    "not_found": "model not available for this key",
    "auth": "key rejected",
    "no_key": "no API key set",
    "bad_request": "request rejected",
    "network": "not reachable",
    "bad_reply": "unexpected reply",
    "error": "failed",
}
# Kinds that won't fix themselves within one run: stop calling that client.
PERMANENT = {"quota", "not_found", "auth", "no_key"}


class LLMError(Exception):
    """
    Any LLM failure. `kind` is one of FRIENDLY's keys, `provider` the client
    name. A chain of clients raises one LLMError whose `errors` lists them all.
    """

    def __init__(self, message: str, kind: str = "error", provider: str = ""):
        super().__init__(message)
        self.kind, self.provider = kind, provider
        self.errors: list[LLMError] = [self]


def describe(error: LLMError) -> str:
    """One friendly line, e.g. 'gemini-3.5-flash: quota reached · ollama: not reachable'."""
    parts = []
    for e in error.errors:
        name = e.provider.split(":", 1)[-1] if e.provider.startswith("gemini") else e.provider.split(":")[0]
        parts.append(f"{name or 'LLM'}: {FRIENDLY.get(e.kind, 'failed')}")
    return " · ".join(dict.fromkeys(parts))   # dict keeps order and drops duplicates


def headline(error: LLMError) -> str:
    """The most useful single phrase, e.g. 'Gemini quota reached'."""
    first = error.errors[0]
    who = "Gemini" if first.provider.startswith("gemini") else (first.provider.split(":")[0] or "LLM").title()
    return f"{who} {FRIENDLY.get(first.kind, 'failed')}"


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


# "Busy, try again" errors (server overloaded) get a couple of retries.
RETRY_STATUSES = {500, 503}   # 429 = quota used up: retrying only burns more quota
RETRY_WAITS = [2, 5]   # seconds to wait before retry 1 and retry 2
HTTP_KINDS = {429: "quota", 500: "overloaded", 503: "overloaded", 404: "not_found",
              401: "auth", 403: "auth", 400: "bad_request"}


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
            # Keep the provider's error message, never the request (it holds the key).
            detail = " ".join(e.read().decode(errors="replace").split())[:600]
            retried = f" (after {attempt} retries)" if attempt else ""
            raise LLMError(f"HTTP {e.code}{retried}: {detail}", HTTP_KINDS.get(e.code, "error")) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"network error: {e}", "network") from None
        except json.JSONDecodeError:
            raise LLMError("reply was not JSON", "bad_reply") from None


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
            raise LLMError("GEMINI_API_KEY is not set (add it to .env)", "no_key", f"gemini:{model}")
        self.api_key, self.model, self.thinking = api_key, model, thinking
        self.name = f"gemini:{model}"

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        config = {"temperature": 0}
        if json_mode:
            config["responseMimeType"] = "application/json"
        if self.thinking:  # our tasks are simple; less "thinking" = faster replies
            config["thinkingConfig"] = {"thinkingLevel": self.thinking}
        try:
            reply = _post_json(
                self.URL.format(model=self.model),
                {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                 "generationConfig": config},
                headers={"x-goog-api-key": self.api_key},
            )
            # Skip the model's internal "thought" parts; keep only the answer text.
            parts = reply["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except LLMError as e:
            e.provider = self.name
            raise
        except (KeyError, IndexError):
            raise LLMError(f"unexpected Gemini reply: {str(reply)[:300]}", "bad_reply", self.name) from None


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
        try:
            reply = _post_json(f"{self.host}/api/chat", payload, headers={})
            return reply["message"]["content"].strip()
        except LLMError as e:
            e.provider = self.name
            raise
        except KeyError:
            raise LLMError(f"unexpected Ollama reply: {str(reply)[:300]}", "bad_reply", self.name) from None


class ChainClient(LLMClient):
    """
    Try each client in order; the first that answers wins.
    A client that fails permanently (quota, missing model, bad key) is not
    tried again for the rest of this run. `last_errors` holds the failures
    that happened before the last successful answer, for the trace.
    """

    def __init__(self, clients: list[LLMClient]):
        self.clients = clients
        self.dead: set[str] = set()
        others = ", ".join(c.name.split(":", 1)[-1] if c.name.startswith("gemini") else c.name.split(":")[0]
                           for c in clients[1:])
        self.name = clients[0].name + (f" (fallbacks: {others})" if others else "")
        self.last_used: Optional[str] = None
        self.last_errors: list[LLMError] = []

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        errors = []
        for client in self.clients:
            if client.name in self.dead:
                continue
            try:
                text = client.complete(prompt, json_mode)
                self.last_used, self.last_errors = client.name, errors
                return text
            except LLMError as e:
                errors.append(e)
                if e.kind in PERMANENT:
                    self.dead.add(client.name)
        if not errors:   # every client was already dead
            errors = [LLMError("all LLM providers failed earlier in this run", "quota", self.clients[0].name)]
        chained = LLMError("; ".join(f"{e.provider}: {e}" for e in errors), errors[0].kind, errors[0].provider)
        chained.errors = errors
        raise chained


def make_client(provider: Optional[str] = None) -> tuple[Optional[LLMClient], str]:
    """
    Build the client for `provider` (or LLM_PROVIDER from the environment).

    Returns (client, note). client is None in demo-safe mode; `note` says
    what is being used, for the trace.
      gemini -> Gemini main model, then the fallback model, then Ollama
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
        key = os.environ.get("GEMINI_API_KEY", "")
        thinking = os.environ.get("GEMINI_THINKING", "low")
        main = os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        backup = os.environ.get("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK)
        if not key:
            return ollama, "No Gemini key in .env; trying Ollama, then rule-based"
        clients = [GeminiClient(key, main, thinking)]
        if backup and backup != main:
            clients.append(GeminiClient(key, backup, thinking))
        chain = ChainClient(clients + [ollama])
        return chain, f"using {chain.name}"
    raise ValueError(f"Unknown LLM provider '{provider}'. Use gemini, ollama or demo-safe.")
