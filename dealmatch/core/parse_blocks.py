from __future__ import annotations

import re
import urllib.parse
from typing import Dict, List, Tuple


def domain_only(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # accept URLs
    if "://" in s:
        host = urllib.parse.urlparse(s).hostname or ""
    else:
        host = s.split("/")[0]
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


_re_urlish = re.compile(r"(?i)^(https?://|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,})(/.*)?$")


def looks_like_urlish(line: str) -> bool:
    t = (line or "").strip()
    if not t:
        return False
    return bool(_re_urlish.match(t))


def _dedupe(out: Dict[str, List[str]]) -> Dict[str, List[str]]:
    for s, toks in list(out.items()):
        seen = set()
        deduped: List[str] = []
        for t in toks:
            k = (t or "").strip()
            if not k:
                continue
            lk = k.lower()
            if lk in seen:
                continue
            seen.add(lk)
            deduped.append(k)
        out[s] = deduped
    return out


def parse_seed_parts_blocks(text: str) -> Dict[str, List[str]]:
    """Parse textarea text into {seed_domain: [part_tokens...]}

    Supported input styles:
    1) Preferred (line-based blocks):
       <seed-url>
       <part token>
       <part token>

       <seed-url>
       ...

    2) Space-separated mega-line paste:
       seed1.com PN1 PN2 seed2.com PN3 ...
       (any URL-ish token starts a new seed block; everything else becomes a part token)
    """

    raw = (text or "").strip()
    if not raw:
        return {}

    # Heuristic: if the paste is mostly one huge line or very long lines, parse by whitespace tokens.
    lines = [ln.strip() for ln in raw.splitlines()]
    long_line = any(len(ln) > 260 for ln in lines)
    few_lines = len([ln for ln in lines if ln]) <= 3

    if long_line or few_lines:
        # Token mode
        toks = re.split(r"\s+", raw)
        seed = ""
        out: Dict[str, List[str]] = {}
        for tok in toks:
            tok = (tok or "").strip()
            if not tok:
                continue
            if looks_like_urlish(tok):
                seed = domain_only(tok)
                if seed:
                    out.setdefault(seed, [])
                continue
            if not seed:
                continue
            out[seed].append(tok)
        return _dedupe(out)

    # Line mode
    seed = ""
    out = {}
    for ln in lines:
        if not ln:
            continue
        if looks_like_urlish(ln):
            seed = domain_only(ln)
            if seed:
                out.setdefault(seed, [])
            continue
        if not seed:
            continue
        out[seed].append(ln)

    return _dedupe(out)
