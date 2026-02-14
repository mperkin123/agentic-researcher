import os
from typing import Any, Dict, List, Optional

import httpx


def openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return key


def strategy_models() -> tuple[str, str]:
    """Return (planner_model, worker_model)."""
    # You can override with env if you want
    planner = os.getenv("OPENAI_PLANNER_MODEL", "gpt-5")
    worker = os.getenv("OPENAI_WORKER_MODEL", "gpt-4.1-mini")
    return planner, worker


async def responses_create(
    *,
    model: str,
    input_text: str,
    json_schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Minimal OpenAI Responses API call (no SDK dependency)."""
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/responses")
    headers = {
        "Authorization": f"Bearer {openai_api_key()}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "input": input_text,
        "temperature": temperature,
    }

    # If schema is provided, request JSON output.
    if json_schema is not None:
        payload["text"] = {"format": {"type": "json_schema", "name": "output", "schema": json_schema}}

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


def extract_text(resp: Dict[str, Any]) -> str:
    # Responses API has multiple output types; handle common case.
    outs = resp.get("output", []) or []
    chunks: List[str] = []
    for o in outs:
        for c in (o.get("content", []) or []):
            if c.get("type") == "output_text":
                chunks.append(c.get("text", ""))
    return "\n".join(chunks).strip()


def extract_json(resp: Dict[str, Any]) -> Any:
    outs = resp.get("output", []) or []
    for o in outs:
        for c in (o.get("content", []) or []):
            if c.get("type") == "output_text":
                # if the API returned JSON in text
                import json
                try:
                    return json.loads(c.get("text", ""))
                except Exception:
                    continue
            if c.get("type") == "output_json":
                return c.get("json")
    return None
