import asyncio
import os
import random
import time
from datetime import datetime

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dealmatch.core.models import Run
from dealmatch.db.session import SessionLocal
from dealmatch.llm.client import extract_json, responses_create, strategy_models
from dealmatch.serper.client import serper_search


# --- Demo models live in core.models (imported there) ---
from dealmatch.core.models import (
    DemoEdge,
    DemoEdgeStatus,
    DemoEvent,
    DemoEventType,
    DemoRunControl,
    DemoSerperQuery,
    DemoSerperQueryStatus,
    DemoSupplier,
    DemoSupplierStatus,
    DemoPart,
    DemoPartStatus,
)


# Demo pacing + budget
SERPER_BUDGET = int(os.getenv("DEMO_SERPER_BUDGET", "2000"))
SERPER_CONCURRENCY = int(os.getenv("DEMO_SERPER_CONCURRENCY", "5"))
SCRAPE_CONCURRENCY = int(os.getenv("DEMO_SCRAPE_CONCURRENCY", "20"))

# Global slow-motion factor. 1.0 = normal, 1.7 ~= 70% slower.
SLOW_FACTOR = float(os.getenv("DEMO_SLOW_FACTOR", "1.7"))

# Keep the dashboard "alive" by maintaining a backlog of pending queries.
QUERY_BACKLOG_TARGET = int(os.getenv("DEMO_QUERY_BACKLOG_TARGET", "80"))

# Limit concurrent unqualified suppliers so we qualify as we go.
MAX_UNQUALIFIED_SUPPLIERS = int(os.getenv("DEMO_MAX_UNQUALIFIED_SUPPLIERS", "10"))

# Slow down the visible SERP wall a bit (per lane, between queries).
SERPER_LANE_DELAY_S = float(os.getenv("DEMO_SERPER_LANE_DELAY_S", "0.35")) * SLOW_FACTOR

MAX_SUPPLIERS_PER_QUERY = int(os.getenv("DEMO_MAX_SUPPLIERS_PER_QUERY", "12"))
MAX_PAGES_PER_SUPPLIER = int(os.getenv("DEMO_MAX_PAGES_PER_SUPPLIER", "3"))

USER_AGENT = os.getenv(
    "DEMO_UA",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
)


def _now() -> datetime:
    return datetime.utcnow()


# Simple per-process throttles/caches to reduce SSE spam.
_LAST_RUN_STATE_EMIT: dict[int, float] = {}


def emit(db: Session, *, run_id: int, type: DemoEventType, data: dict):
    ev = DemoEvent(run_id=run_id, type=type, data=data)
    db.add(ev)
    db.commit()


def emit_run_state_throttled(db: Session, *, run_id: int, state: str, serper_calls_used: int, now: str, min_interval_s: float = 5.0):
    """Emit RUN_STATE at most once per min_interval_s per run.

    The UI doesn't need per-Serper-call precision and too many SSE messages can
    overwhelm the frontend.
    """
    t = time.time()
    last = float(_LAST_RUN_STATE_EMIT.get(int(run_id), 0.0))
    if (t - last) < float(min_interval_s):
        return
    _LAST_RUN_STATE_EMIT[int(run_id)] = t
    emit(
        db,
        run_id=run_id,
        type=DemoEventType.RUN_STATE,
        data={"state": state, "serper_calls_used": int(serper_calls_used), "now": now},
    )


def ensure_control(db: Session, run_id: int) -> DemoRunControl:
    c = db.execute(select(DemoRunControl).where(DemoRunControl.run_id == run_id)).scalar_one_or_none()
    if c:
        return c
    c = DemoRunControl(run_id=run_id, state="stopped", serper_calls_used=0, serper_budget=SERPER_BUDGET)
    db.add(c)
    db.commit()
    return c


def normalize_part(tok: str) -> str:
    t = (tok or "").strip()
    t = t.replace("\u00a0", " ")
    return t


def extract_part_from_query(q: str) -> str | None:
    """Best-effort extract a part token from the demo query string.

    Our queries are often like: "{PART}" aircraft parts supplier
    """
    import re

    s = (q or "").strip()
    m = re.search(r"\"([^\"]{2,80})\"", s)
    if not m:
        return None
    tok = m.group(1).strip()
    return tok or None


JUNK_DOMAIN_SUBSTR = [
    "archive.org",
    "amazonaws.com",
    "pdfcoffee.",
    "yumpu.",
    "federalregister.",
    "researchgate.",
    "slideshare.",
    "scribd.",
    "github.",
    "linkedin.",
    "facebook.",
    "wikipedia.",
]


def is_junk_domain(d: str) -> bool:
    """Heuristic junk filter.

    Can be disabled with DEMO_DISABLE_JUNK_FILTER=1.
    """
    if os.getenv("DEMO_DISABLE_JUNK_FILTER", "").strip() in ("1", "true", "yes", "on"):
        return False
    dd = (d or "").lower().strip()
    if not dd:
        return True
    return any(sub in dd for sub in JUNK_DOMAIN_SUBSTR)


def part_uncertain(tok: str) -> bool:
    t = (tok or "").strip()
    if not t:
        return True
    # If it's clearly a part-like token, don't LLM.
    import re

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-\/\.]{3,40}", t) and any(ch.isdigit() for ch in t):
        return False
    return True


async def llm_validate_part(tok: str) -> tuple[bool, float, str]:
    planner, worker = strategy_models()
    schema = {
        "type": "object",
        "properties": {
            "is_part": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["is_part", "confidence", "reason"],
        "additionalProperties": False,
    }
    prompt = (
        "You are classifying whether a token is a legitimate aircraft part number or part name token used in aviation parts commerce.\n"
        "Return JSON only.\n\n"
        f"TOKEN: {tok!r}\n"
        "Criteria: part numbers often include digits and dashes; part names are short but specific; reject generic words, sentences, and garbage."
    )
    resp = await asyncio.wait_for(
        responses_create(model=worker, input_text=prompt, json_schema=schema, temperature=0.1),
        timeout=float(os.getenv("DEMO_LLM_TIMEOUT_S", "8")),
    )
    j = extract_json(resp) or {}
    return bool(j.get("is_part")), float(j.get("confidence") or 0.5), (j.get("reason") or "").strip()


async def llm_qualify_supplier(domain: str, homepage_text: str) -> tuple[bool, float, str]:
    planner, worker = strategy_models()
    schema = {
        "type": "object",
        "properties": {
            "is_aircraft_parts_seller": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["is_aircraft_parts_seller", "confidence", "reason"],
        "additionalProperties": False,
    }
    text = (homepage_text or "").strip()
    text = text[:4000]
    prompt = (
        "You are qualifying whether a website domain sells or distributes aircraft/aviation parts (inventory, RFQ, aerospace components).\n"
        "Return JSON only.\n\n"
        f"DOMAIN: {domain}\n"
        f"HOMEPAGE TEXT SNIPPET:\n{text}\n\n"
        "Answer true only if it's clearly relevant to aviation parts."
    )
    resp = await asyncio.wait_for(
        responses_create(model=worker, input_text=prompt, json_schema=schema, temperature=0.1),
        timeout=float(os.getenv("DEMO_LLM_TIMEOUT_S", "8")),
    )
    j = extract_json(resp) or {}
    return bool(j.get("is_aircraft_parts_seller")), float(j.get("confidence") or 0.5), (j.get("reason") or "").strip()


async def fetch_text(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=25, follow_redirects=True, headers=headers) as client:
        r = await client.get(url)
        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()
        if "text" not in ct and "html" not in ct and "json" not in ct:
            return ""
        return r.text


def extract_visibleish_text(html: str) -> str:
    # Very lightweight: strip tags.
    import re

    s = html or ""
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def find_part_hits(text: str, parts: list[str]) -> list[tuple[str, int]]:
    t = (text or "").lower()
    hits = []
    for p in parts:
        pp = (p or "").strip()
        if not pp:
            continue
        idx = t.find(pp.lower())
        if idx >= 0:
            hits.append((pp, idx))
    return hits


def snippet_around(text: str, idx: int, window: int = 90) -> str:
    if idx < 0:
        return ""
    a = max(0, idx - window)
    b = min(len(text), idx + window)
    return text[a:b].strip()


async def ensure_parts(db: Session, run: Run):
    # Seed parts are assumed valid (demo focus); but we still store them.
    seed_parts = []
    for dom, toks in (run.seed_phrases or {}).items():
        for t in toks or []:
            seed_parts.append(normalize_part(t))
    # dedupe
    seen = set()
    seed_parts = [p for p in seed_parts if not (p.lower() in seen or seen.add(p.lower()))]
    for p in seed_parts[:5000]:
        exists = db.execute(select(DemoPart).where(DemoPart.run_id == run.id, DemoPart.token == p)).scalar_one_or_none()
        if exists:
            continue
        db.add(DemoPart(run_id=run.id, token=p, status=DemoPartStatus.validated, confidence=0.8, reason="seed"))
    db.commit()


async def generate_queries(db: Session, run: Run):
    """Maintain a backlog of pending Serper queries.

    Important demo behavior: we *allow re-running* previously-done queries if needed.
    We only avoid duplicating queries that are currently pending/running.
    """

    pending_n = int(
        db.execute(
            select(func.count())
            .select_from(DemoSerperQuery)
            .where(DemoSerperQuery.run_id == run.id)
            .where(DemoSerperQuery.status == DemoSerperQueryStatus.pending)
        ).scalar()
        or 0
    )
    if pending_n >= QUERY_BACKLOG_TARGET:
        return

    need = max(0, QUERY_BACKLOG_TARGET - pending_n)

    # sample parts to drive queries (shuffle each top-up so we don't get "stuck")
    parts = (
        db.execute(
            select(DemoPart.token)
            .where(DemoPart.run_id == run.id, DemoPart.status == DemoPartStatus.validated)
            .limit(800)
        )
        .scalars()
        .all()
    )
    random.shuffle(parts)
    toks = parts[: min(len(parts), max(20, need // 2))] if parts else []

    qs: list[str] = []
    for t in toks:
        tt = (t or "").strip()
        if not tt:
            continue

        # Broad queries first (higher chance of organic results)
        qs.append(f'"{tt}" aircraft parts')
        qs.append(f'"{tt}" aviation')
        qs.append(f'"{tt}" supplier')
        qs.append(f'"{tt}" distributor')

        # One commerce-intent query (but keep it optional/sparse)
        qs.append(f'"{tt}" "Request a Quote"')

    for sd in (run.seed_domains or [])[:3]:
        qs.append(f"{sd} competitors")
        qs.append(f"{sd} aircraft parts")

    # Insert up to `need`, avoiding only duplicates that are currently pending/running.
    added = 0
    for q in qs:
        if added >= need:
            break
        in_flight = db.execute(
            select(DemoSerperQuery)
            .where(DemoSerperQuery.run_id == run.id)
            .where(DemoSerperQuery.query == q)
            .where(DemoSerperQuery.status.in_([DemoSerperQueryStatus.pending, DemoSerperQueryStatus.running]))
            .limit(1)
        ).scalar_one_or_none()
        if in_flight:
            continue
        db.add(DemoSerperQuery(run_id=run.id, query=q, status=DemoSerperQueryStatus.pending))
        added += 1

    db.commit()

    if added:
        emit(
            db,
            run_id=run.id,
            type=DemoEventType.DECISION,
            data={
                "title": "Queued Serper queries",
                "meta": f"added={added} pending_target={QUERY_BACKLOG_TARGET}",
                "body": (qs[:5] or []).__repr__(),
            },
        )


async def serper_worker(run_id: int, lane: int, sem: asyncio.Semaphore):
    # Each lane continuously pulls one pending query at a time.
    while True:
        await asyncio.sleep(0)  # yield
        async with sem:
            db = SessionLocal()
            try:
                try:
                    run = db.get(Run, run_id)
                    if not run:
                        return
                    ctl = ensure_control(db, run_id)
                    if ctl.state != "running":
                        await asyncio.sleep(0.35)
                        continue
                    if ctl.serper_calls_used >= ctl.serper_budget:
                        ctl.state = "stopped"
                        db.commit()
                        emit(db, run_id=run_id, type=DemoEventType.RUN_STATE, data={
                            "state": ctl.state,
                            "serper_calls_used": ctl.serper_calls_used,
                            "now": "Budget reached",
                        })
                        await asyncio.sleep(0.9)
                        continue

                    # If we already have a backlog of unqualified suppliers, pause discovery.
                    unq = int(
                        db.execute(
                            select(func.count())
                            .select_from(DemoSupplier)
                            .where(DemoSupplier.run_id == run_id)
                            .where(DemoSupplier.status.in_([DemoSupplierStatus.discovered, DemoSupplierStatus.qualifying]))
                        ).scalar()
                        or 0
                    )
                    if unq >= MAX_UNQUALIFIED_SUPPLIERS:
                        await asyncio.sleep(0.8 * SLOW_FACTOR)
                        continue

                    # keep queries stocked
                    await generate_queries(db, run)

                    q = db.execute(
                        select(DemoSerperQuery)
                        .where(DemoSerperQuery.run_id == run_id)
                        .where(DemoSerperQuery.status == DemoSerperQueryStatus.pending)
                        .order_by(DemoSerperQuery.id.asc())
                        .limit(1)
                    ).scalar_one_or_none()
                    if not q:
                        # nothing pending; back off a bit
                        await asyncio.sleep(0.6)
                        continue

                    q.status = DemoSerperQueryStatus.running
                    q.lane = lane
                    q.started_at = _now()
                    db.commit()

                    emit(db, run_id=run_id, type=DemoEventType.SERPER_LANE, data={
                        "lane": lane,
                        "meta": "running",
                        "query": q.query,
                        "serper_calls_used": ctl.serper_calls_used,
                    })

                    t0 = time.time()
                    try:
                        res = await serper_search(q.query, num=10)
                        ok = True
                    except Exception as e:
                        res = {"error": repr(e)}
                        ok = False

                    ctl.serper_calls_used += 1
                    q.finished_at = _now()
                    q.status = DemoSerperQueryStatus.done if ok else DemoSerperQueryStatus.error
                    q.result = res
                    db.commit()

                    took_ms = int((time.time() - t0) * 1000)

                    emit(db, run_id=run_id, type=DemoEventType.SERPER_LANE, data={
                        "lane": lane,
                        "meta": f"done ({took_ms}ms)",
                        "query": q.query,
                        "serper_calls_used": ctl.serper_calls_used,
                    })
                    emit_run_state_throttled(
                        db,
                        run_id=run_id,
                        state=ctl.state,
                        serper_calls_used=ctl.serper_calls_used,
                        now=f"Serper lane {lane+1}: {q.query}",
                        min_interval_s=5.0,
                    )

                    # slow down a touch so the UI is readable
                    await asyncio.sleep(SERPER_LANE_DELAY_S)

                    # Extract supplier domains from serper results
                    org = (res.get("organic") or []) if isinstance(res, dict) else []
                    domains_all: list[str] = []
                    for r in org:
                        link = (r.get("link") or "")
                        d = domain_only(link)
                        if d:
                            domains_all.append(d)

                    # unique preserve order
                    seen = set(); domains_all = [d for d in domains_all if not (d in seen or seen.add(d))]

                    domains_kept = [d for d in domains_all if not is_junk_domain(d)]

                    # tie discoveries to the part token that drove this query (best-effort)
                    part_tok = extract_part_from_query(q.query)

                    # Debug visibility: show what Serper gave us and what we kept.
                    if not domains_all:
                        emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                            "title": "Serper results: no domains",
                            "meta": f"lane={lane} organic={len(org)}",
                            "body": q.query,
                        })
                    elif not domains_kept:
                        emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                            "title": "Serper results: all domains filtered",
                            "meta": f"lane={lane} total={len(domains_all)}",
                            "body": str(domains_all[:10]),
                        })
                    else:
                        emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                            "title": "Serper domains extracted",
                            "meta": f"lane={lane} organic={len(org)} total={len(domains_all)} kept={len(domains_kept)}",
                            "body": str(domains_kept[:10]),
                        })

                    added = 0
                    for d in domains_kept[:MAX_SUPPLIERS_PER_QUERY]:
                        ex = db.execute(
                            select(DemoSupplier).where(DemoSupplier.run_id == run_id, DemoSupplier.domain == d)
                        ).scalar_one_or_none()
                        if ex:
                            continue
                        db.add(DemoSupplier(run_id=run_id, domain=d, status=DemoSupplierStatus.discovered, source_query_id=q.id))
                        added += 1

                        # Create a *single* researching edge per supplier+part so the UI has immediate feedback.
                        if part_tok:
                            edge_exists = (
                                db.execute(
                                    select(DemoEdge.id)
                                    .where(DemoEdge.run_id == run_id)
                                    .where(DemoEdge.supplier_domain == d)
                                    .where(DemoEdge.part_token == part_tok)
                                    .limit(1)
                                ).scalar_one_or_none()
                                is not None
                            )
                            if not edge_exists:
                                db.add(
                                    DemoEdge(
                                        run_id=run_id,
                                        supplier_domain=d,
                                        part_token=part_tok,
                                        # evidence_url is non-nullable in the DB schema; use empty string for
                                        # "discovered" edges that don't have a specific evidence URL yet.
                                        evidence_url="",
                                        evidence_snippet="Discovered via search; qualifying supplier…",
                                        status=DemoEdgeStatus.researching,
                                    )
                                )
                                db.commit()
                                emit(db, run_id=run_id, type=DemoEventType.EDGE, data={
                                    "supplier": d,
                                    "part": part_tok,
                                    "url": "",
                                    "snippet": "Discovered via search; qualifying supplier…",
                                    "status": DemoEdgeStatus.researching.value,
                                })

                    db.commit()
                except Exception as e:
                    emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                        "title": "Serper lane crashed (auto-recovering)",
                        "meta": f"lane={lane}",
                        "body": repr(e),
                    })
                    await asyncio.sleep(0.6)

            finally:
                db.close()


def domain_only(url: str) -> str:
    import re, urllib.parse

    s = (url or "").strip()
    if not s:
        return ""
    if "://" in s:
        host = urllib.parse.urlparse(s).hostname or ""
    else:
        host = s.split("/")[0]
    host = host.lower().strip(".")
    host = re.sub(r"^www\.", "", host)
    return host


async def supplier_qualifier_loop(run_id: int, sem: asyncio.Semaphore):
    while True:
        async with sem:
            db = SessionLocal()
            try:
                ctl = ensure_control(db, run_id)
                if ctl.state != "running":
                    await asyncio.sleep(0.4)
                    continue

                s = db.execute(
                    select(DemoSupplier)
                    .where(DemoSupplier.run_id == run_id)
                    .where(DemoSupplier.status == DemoSupplierStatus.discovered)
                    .order_by(DemoSupplier.id.asc())
                    .limit(1)
                ).scalar_one_or_none()
                if not s:
                    await asyncio.sleep(0.25)
                    continue

                s.status = DemoSupplierStatus.qualifying
                db.commit()

                url = f"https://{s.domain}/"
                try:
                    html = await fetch_text(url)
                    text = extract_visibleish_text(html)
                except Exception as e:
                    s.status = DemoSupplierStatus.rejected
                    s.reason = f"fetch_failed: {e!r}"
                    s.confidence = 0.0
                    db.commit()
                    emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                        "title": "Supplier rejected",
                        "meta": s.domain,
                        "body": f"Could not fetch homepage. {e!r}",
                    })
                    continue

                try:
                    ok, conf, reason = await llm_qualify_supplier(s.domain, text)
                except Exception as e:
                    ok, conf, reason = False, 0.0, f"llm_error: {e!r}"

                s.confidence = conf
                s.reason = reason
                s.status = DemoSupplierStatus.qualified if ok else DemoSupplierStatus.rejected
                db.commit()

                emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                    "title": "Supplier qualified" if ok else "Supplier rejected",
                    "meta": f"{s.domain} (conf={conf:.2f})",
                    "body": reason,
                })

                # Flip any researching edges for this supplier to validated/rejected.
                new_edge_status = DemoEdgeStatus.validated if ok else DemoEdgeStatus.rejected
                rows = (
                    db.execute(
                        select(DemoEdge)
                        .where(DemoEdge.run_id == run_id)
                        .where(DemoEdge.supplier_domain == s.domain)
                        .where(DemoEdge.status == DemoEdgeStatus.researching)
                        .limit(200)
                    )
                    .scalars()
                    .all()
                )
                for e in rows:
                    e.status = new_edge_status
                    # Keep some evidence (domain-level qualification)
                    if not e.evidence_snippet:
                        e.evidence_snippet = reason[:280]
                db.commit()
                for e in rows:
                    emit(db, run_id=run_id, type=DemoEventType.EDGE, data={
                        "supplier": e.supplier_domain,
                        "part": e.part_token,
                        "url": e.evidence_url,
                        "snippet": (e.evidence_snippet or reason)[:280],
                        "status": new_edge_status.value,
                    })
            finally:
                db.close()
        await asyncio.sleep(0.05 * SLOW_FACTOR)


async def scraper_loop(run_id: int, sem: asyncio.Semaphore):
    while True:
        async with sem:
            db = SessionLocal()
            try:
                ctl = ensure_control(db, run_id)
                if ctl.state != "running":
                    await asyncio.sleep(0.4)
                    continue

                sup = db.execute(
                    select(DemoSupplier)
                    .where(DemoSupplier.run_id == run_id)
                    .where(DemoSupplier.status == DemoSupplierStatus.qualified)
                    .where(DemoSupplier.pages_fetched < MAX_PAGES_PER_SUPPLIER)
                    .order_by(DemoSupplier.last_scraped_at.asc().nullsfirst(), DemoSupplier.id.asc())
                    .limit(1)
                ).scalar_one_or_none()
                if not sup:
                    await asyncio.sleep(0.25)
                    continue

                # pick a URL to fetch
                base = f"https://{sup.domain}"
                candidates = [base + "/", base + "/inventory", base + "/products", base + "/search"]
                url = candidates[min(sup.pages_fetched, len(candidates)-1)]

                try:
                    html = await fetch_text(url)
                    text = extract_visibleish_text(html)
                except Exception:
                    sup.pages_fetched += 1
                    sup.last_scraped_at = _now()
                    db.commit()
                    continue

                sup.pages_fetched += 1
                sup.last_scraped_at = _now()
                db.commit()

                # part list sample (limit for perf)
                parts = db.execute(
                    select(DemoPart.token).where(DemoPart.run_id == run_id, DemoPart.status == DemoPartStatus.validated).limit(800)
                ).scalars().all()

                hits = find_part_hits(text, parts)
                random.shuffle(hits)
                hits = hits[:12]

                for part, idx in hits:
                    snip = snippet_around(text, idx)

                    # Deduplicate by (supplier, part) rather than (supplier, part, url).
                    # This keeps the demo UI fast and reduces SSE events dramatically.
                    already = (
                        db.execute(
                            select(DemoEdge.id)
                            .where(DemoEdge.run_id == run_id)
                            .where(DemoEdge.supplier_domain == sup.domain)
                            .where(DemoEdge.part_token == part)
                            .limit(1)
                        ).scalar_one_or_none()
                        is not None
                    )
                    if already:
                        continue

                    # Insert edge as validated (supplier already qualified; parts are validated)
                    st = DemoEdgeStatus.validated
                    db.add(
                        DemoEdge(
                            run_id=run_id,
                            supplier_domain=sup.domain,
                            part_token=part,
                            evidence_url=url,
                            evidence_snippet=snip[:280],
                            status=st,
                        )
                    )
                    db.commit()

                    emit(db, run_id=run_id, type=DemoEventType.EDGE, data={
                        "supplier": sup.domain,
                        "part": part,
                        "url": url,
                        "snippet": snip[:280],
                        "status": st.value,
                    })

            finally:
                db.close()
        await asyncio.sleep(0.03 * SLOW_FACTOR)


async def seed_researching_edges(run_id: int):
    """Emit a few 'researching' edges for showmanship.

    These come from discovered suppliers before qualification, and then flip later.
    """
    while True:
        await asyncio.sleep(1.0 * SLOW_FACTOR)
        db = SessionLocal()
        try:
            ctl = ensure_control(db, run_id)
            if ctl.state != "running":
                continue

            # pick a discovered supplier and a part, emit researching edge that may get rejected
            sup = db.execute(select(DemoSupplier).where(DemoSupplier.run_id == run_id, DemoSupplier.status.in_([DemoSupplierStatus.discovered, DemoSupplierStatus.qualifying])).order_by(DemoSupplier.id.desc()).limit(1)).scalar_one_or_none()
            part = db.execute(select(DemoPart).where(DemoPart.run_id == run_id).order_by(DemoPart.id.asc()).limit(1)).scalar_one_or_none()
            if not sup or not part:
                continue

            url = f"https://{sup.domain}/"
            status = DemoEdgeStatus.researching

            # Don't spam: only emit a researching edge if we don't already have
            # any edge for this supplier+part.
            exists_any = (
                db.execute(
                    select(DemoEdge.id)
                    .where(DemoEdge.run_id == run_id)
                    .where(DemoEdge.supplier_domain == sup.domain)
                    .where(DemoEdge.part_token == part.token)
                    .limit(1)
                ).scalar_one_or_none()
                is not None
            )
            if not exists_any:
                emit(db, run_id=run_id, type=DemoEventType.EDGE, data={
                    "supplier": sup.domain,
                    "part": part.token,
                    "url": url,
                    "snippet": "Researching this relationship…",
                    "status": status.value,
                })
        finally:
            db.close()


async def run_demo_loop():
    print("demo worker: starting", flush=True)

    async def heartbeat():
        while True:
            await asyncio.sleep(5.0)
            try:
                db = SessionLocal()
                try:
                    ctl = db.execute(
                        select(DemoRunControl).where(DemoRunControl.state == "running").order_by(DemoRunControl.run_id.desc()).limit(1)
                    ).scalar_one_or_none()
                    rid = int(ctl.run_id) if ctl else 0
                    if rid:
                        c = db.get(DemoRunControl, rid)
                        emit_run_state_throttled(
                            db,
                            run_id=rid,
                            state=(c.state if c else "unknown"),
                            serper_calls_used=int((c.serper_calls_used if c else 0)),
                            now="worker heartbeat",
                            min_interval_s=5.0,
                        )
                finally:
                    db.close()
            except Exception as e:
                print(f"demo worker: heartbeat error {e!r}", flush=True)

    hb_task = asyncio.create_task(heartbeat())

    # Single-run demo: prefer the newest run that is explicitly marked running.
    while True:
        try:
            db = SessionLocal()
            try:
                # Find newest running demo control.
                running_ctl = db.execute(
                    select(DemoRunControl)
                    .where(DemoRunControl.state == "running")
                    .order_by(DemoRunControl.run_id.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if running_ctl:
                    run = db.get(Run, running_ctl.run_id)
                else:
                    run = db.execute(select(Run).order_by(Run.id.desc()).limit(1)).scalar_one_or_none()
                if not run:
                    await asyncio.sleep(1.0 * SLOW_FACTOR)
                    continue
                ensure_control(db, run.id)
                await ensure_parts(db, run)
                run_id = int(run.id)
                print(f"demo worker: attached run_id={run_id}", flush=True)
                emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                    "title": "Worker attached",
                    "meta": f"run_id={run_id}",
                    "body": f"serper_concurrency={SERPER_CONCURRENCY} scrape_concurrency={SCRAPE_CONCURRENCY}",
                })

                # Prime some Serper queries immediately so the demo doesn't depend on lane scheduling.
                try:
                    await generate_queries(db, run)
                    pending_n = db.execute(
                        select(func.count())
                        .select_from(DemoSerperQuery)
                        .where(DemoSerperQuery.run_id == run_id)
                        .where(DemoSerperQuery.status == DemoSerperQueryStatus.pending)
                    ).scalar() or 0
                    emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                        "title": "Query primed",
                        "meta": f"pending={int(pending_n)}",
                        "body": "initial query queue created",
                    })
                    print(f"demo worker: primed queries pending={int(pending_n)}", flush=True)
                except Exception as e:
                    emit(db, run_id=run_id, type=DemoEventType.DECISION, data={
                        "title": "Query prime failed",
                        "meta": "attach",
                        "body": repr(e),
                    })
                    print(f"demo worker: query prime failed {e!r}", flush=True)
            finally:
                db.close()

            # Spawn loops for this run
            serp_sem = asyncio.Semaphore(SERPER_CONCURRENCY)
            qual_sem = asyncio.Semaphore(int(os.getenv("DEMO_SUPPLIER_QUAL_CONCURRENCY", "12")))
            scr_sem = asyncio.Semaphore(SCRAPE_CONCURRENCY)

            tasks = []
            for lane in range(SERPER_CONCURRENCY):
                tasks.append(asyncio.create_task(serper_worker(run_id, lane=lane, sem=serp_sem)))
            tasks.append(asyncio.create_task(supplier_qualifier_loop(run_id, sem=qual_sem)))
            for _ in range(SCRAPE_CONCURRENCY):
                tasks.append(asyncio.create_task(scraper_loop(run_id, sem=scr_sem)))
            tasks.append(asyncio.create_task(seed_researching_edges(run_id)))

            # Keep running; if a different run becomes active (newer running control), restart loops.
            last_run_id = run_id
            while True:
                await asyncio.sleep(1.0 * SLOW_FACTOR)
                db = SessionLocal()
                try:
                    running_ctl = db.execute(
                        select(DemoRunControl)
                        .where(DemoRunControl.state == "running")
                        .order_by(DemoRunControl.run_id.desc())
                        .limit(1)
                    ).scalar_one_or_none()
                    active = (
                        int(running_ctl.run_id)
                        if running_ctl
                        else int(db.execute(select(Run.id).order_by(Run.id.desc()).limit(1)).scalar_one_or_none() or 0)
                    )
                    if active and active != last_run_id:
                        print(f"demo worker: active run changed {last_run_id} -> {active}, restarting loops", flush=True)
                        break
                finally:
                    db.close()

            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(f"demo worker: FATAL loop error {e!r}\n{tb}", flush=True)

            # Also surface this in the demo UI (DB-backed) so we aren't blind on Render.
            try:
                db = SessionLocal()
                try:
                    ctl = db.execute(
                        select(DemoRunControl).where(DemoRunControl.state == "running").order_by(DemoRunControl.run_id.desc()).limit(1)
                    ).scalar_one_or_none()
                    rid = int(ctl.run_id) if ctl else 0
                    if rid:
                        emit(
                            db,
                            run_id=rid,
                            type=DemoEventType.DECISION,
                            data={
                                "title": "Worker fatal error",
                                "meta": "auto-retrying",
                                "body": (repr(e) + "\n" + tb)[:1800],
                            },
                        )
                finally:
                    db.close()
            except Exception:
                pass

            await asyncio.sleep(1.0 * SLOW_FACTOR)


if __name__ == "__main__":
    asyncio.run(run_demo_loop())
