from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Strategy:
    # Query templates the system will use to discover more domains
    query_templates: List[str]
    # Domain block substrings
    block_substr: List[str]
    # How many domains to propose per batch
    propose_batch: int = 50
    # Serper results per query
    serper_num: int = 10

    @staticmethod
    def default(seed_domains: List[str], seed_phrases: Dict[str, List[str]]) -> "Strategy":
        phrases = []
        for dom, ps in (seed_phrases or {}).items():
            phrases.extend(ps or [])
        phrases = [p.strip() for p in phrases if (p or "").strip()]
        phrases = list(dict.fromkeys(phrases))
        # Simple initial templates; LLM will adapt these.
        templates = [
            "aircraft parts distributor",
            "aviation parts supplier",
            "PMa parts supplier",
        ]
        if phrases:
            templates.append("\"{phrase}\" supplier")
            templates.append("\"{phrase}\" parts")
        return Strategy(query_templates=templates, block_substr=["github.", "linkedin.", "facebook.", "wikipedia."])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_templates": self.query_templates,
            "block_substr": self.block_substr,
            "propose_batch": self.propose_batch,
            "serper_num": self.serper_num,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Strategy":
        return Strategy(
            query_templates=list((d or {}).get("query_templates") or []),
            block_substr=list((d or {}).get("block_substr") or []),
            propose_batch=int((d or {}).get("propose_batch") or 50),
            serper_num=int((d or {}).get("serper_num") or 10),
        )
