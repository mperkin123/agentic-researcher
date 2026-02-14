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


_re_urlish = re.compile(r"(?i)^(https?://|www\.|[a-z0-9-]+\.[a-z]{2,})(/.*)?$")


def looks_like_urlish(line: str) -> bool:
    t = (line or "").strip()
    if not t:
        return False
    return bool(_re_urlish.match(t))


def parse_seed_parts_blocks(text: str) -> Dict[str, List[str]]:
    """Parse textarea text into {seed_domain: [part_tokens...]}

    Format: blocks separated by blank lines; first line of a block is a URL/domain.
    But we also handle continuous paste: any URL-ish line starts a new block.
    """

    lines = [ln.strip() for ln in (text or "").splitlines()]

    seed = ""
    out: Dict[str, List[str]] = {}

    for ln in lines:
        if not ln:
            continue
        if looks_like_urlish(ln):
            seed = domain_only(ln)
            if seed:
                out.setdefault(seed, [])
            continue
        if not seed:
            # ignore leading tokens until first seed line
            continue
        tok = ln.strip()
        if tok:
            out[seed].append(tok)

    # de-dupe within seed, preserve order
    for s, toks in list(out.items()):
        seen = set()
        deduped = []
        for t in toks:
            k = t.strip()
            if not k:
                continue
            if k.lower() in seen:
                continue
            seen.add(k.lower())
            deduped.append(k)
        out[s] = deduped

    return out
