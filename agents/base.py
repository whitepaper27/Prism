"""
GeminiResearchAgent -- base class for PRISM research tooling agents.

Reuses patterns from:
  prism/crystal.py:37  -- genai.Client initialization
  prism/crystal.py:100 -- JSON fence stripping
  prism/ablation.py:649 -- exponential backoff retry

These agents are TOOLING for writing the paper, NOT part of the PRISM
architecture itself. They do not extend PRISMAgent.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")


class GeminiResearchAgent:
    """Base class for PRISM research agents using Gemini API."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def _call(self, prompt: str, system_prompt: str = "", max_retries: int = 8) -> str:
        """Call Gemini and return raw text response with retry logic."""
        contents = prompt
        if system_prompt:
            contents = f"{system_prompt}\n\n---\n\n{prompt}"

        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name, contents=contents
                )
                return response.text.strip()
            except Exception as e:
                err = str(e)
                retryable = any(s in err for s in [
                    "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL",
                    "503", "overloaded", "quota", "ServerError", "ClientError",
                ])
                if retryable and attempt < max_retries - 1:
                    wait = min(600, 60 * (2 ** min(attempt, 3)))
                    print(f"  [retry] {err[:80]}... waiting {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise

    def _call_json(self, prompt: str, system_prompt: str = "") -> dict:
        """Call Gemini and parse JSON response. Falls back to empty dict."""
        raw = self._call(prompt, system_prompt)
        # Strip markdown JSON fences (pattern from prism/crystal.py:100-101)
        clean = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
        clean = re.sub(r"\n?```$", "", clean, flags=re.IGNORECASE)
        try:
            return json.loads(clean.strip())
        except json.JSONDecodeError:
            print(f"  [warn] JSON parse failed, returning raw text")
            return {"raw_text": raw}
