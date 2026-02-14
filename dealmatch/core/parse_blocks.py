from __future__ import annotations

import re
import urllib.parse
from typing import Dict, List, Tuple


_ZW_CHARS = "\ufeff\u200b\u200c\u200d\u2060"

# Common Excel/Word punctuation/whitespace oddities
_NBSP = "\u00a0"  # non-breaking space
_NARROW_NBSP = "\u202f"
_THIN_SPACE = "\u2009"
_SMART_QUOTES = {
    "\u201c": '"',  # left double
    "\u201d": '"',  # right double
    "\u2018": "'",  # left single
    "\u2019": "'",  # right single
}


def _clean(s: str) -> str:
    s = s or ""
    # remove zero-width chars / BOM that often appear in copy-pastes
    for ch in _ZW_CHARS:
        s = s.replace(ch, "")
    # normalize weird spaces
    s = s.replace(_NBSP, " ").replace(_NARROW_NBSP, " ").replace(_THIN_SPACE, " ")
    # normalize smart quotes to ASCII quotes so shlex can parse them
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    return s


def domain_only(s: str) -> str:
    s = _clean((s or "").strip())
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
    t = _clean((line or "").strip())
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

    raw = _clean((text or "")).strip()
    if not raw:
        return {}

    lines = [ln.rstrip("\n") for ln in raw.splitlines()]

    # TSV mode (Excel-friendly): domain<TAB>item, domain may be blank to mean "same as above".
    if any("\t" in ln for ln in lines):
        seed = ""
        out: Dict[str, List[str]] = {}
        for ln in lines:
            if not ln.strip():
                continue
            cols = ln.split("\t")
            col0 = _clean((cols[0] or "")).strip() if len(cols) >= 1 else ""
            col1 = _clean((cols[1] or "")).strip() if len(cols) >= 2 else ""
            if col0:
                if looks_like_urlish(col0):
                    seed = domain_only(col0)
                    if seed:
                        out.setdefault(seed, [])
                else:
                    # if first col isn't urlish, treat it as item under current seed
                    col1 = (col0 + (" " + col1 if col1 else "")).strip()
            if seed and col1:
                out.setdefault(seed, []).append(col1)
        return _dedupe(out)

    # Heuristic: if the paste is mostly one huge line or very long lines, parse by whitespace tokens.
    stripped_lines = [ln.strip() for ln in lines]
    long_line = any(len(ln) > 260 for ln in stripped_lines)
    few_lines = len([ln for ln in stripped_lines if ln]) <= 3

    if long_line or few_lines:
        # Token mode. Prefer shell-style parsing so quotes work (Excel one-cell pastes).
        import shlex

        try:
            toks = shlex.split(raw)
        except Exception:
            toks = re.split(r"\s+", raw)

        toks = [_clean(t) for t in toks]

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
    out: Dict[str, List[str]] = {}
    for ln in stripped_lines:
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
