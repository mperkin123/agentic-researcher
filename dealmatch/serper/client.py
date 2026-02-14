import os
from typing import Any, Dict

import httpx


def serper_api_key() -> str:
    key = os.getenv("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY is required")
    return key


def serper_endpoint() -> str:
    return os.getenv("SERPER_ENDPOINT", "https://google.serper.dev/search")


async def serper_search(query: str, num: int = 10) -> Dict[str, Any]:
    headers = {
        "X-API-KEY": serper_api_key(),
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": int(num)}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(serper_endpoint(), headers=headers, json=payload)
        r.raise_for_status()
        return r.json()
