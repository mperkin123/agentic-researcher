import asyncio
import json
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fastapi.middleware.cors import CORSMiddleware

from dealmatch.core.models import (
    FeedbackEvent,
    Run,
    RunStatus,
    Target,
    TargetStatus,
    Task,
    TaskType,
    DemoEvent,
    DemoEventType,
    DemoRunControl,
    DemoSerperQuery,
    DemoSerperQueryStatus,
    DemoSupplier,
    DemoSupplierStatus,
    DemoEdge,
    DemoEdgeStatus,
    DemoPart,
    DemoPartStatus,
)
from dealmatch.core.parse_blocks import parse_seed_parts_blocks
from dealmatch.core.strategy import Strategy
from dealmatch.db.session import db_session

app = FastAPI(title="Agentic Researcher")

# --- CORS (added by patch script) ---
app.add_middleware(
    CORSMiddleware,
    # Allow Lovable hosted apps and local dev
    allow_origin_regex=r"^(https://.*\.(lovable\.app|lovableproject\.com)|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?)$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- end CORS ---

templates = Jinja2Templates(directory="app/templates")


class RunCreate(BaseModel):
    seed_domains: list[str] = Field(min_length=3, max_length=3)
    seed_phrases: dict[str, list[str]] = Field(default_factory=dict)
    directions: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """<html><body>
    <h1>Agentic Researcher</h1>
    <p>API is running.</p>
    <ul>
      <li><a href="/runs/new">New run (hardcoded)</a></li>
      <li><a href="/runs/new/custom">New run (custom UI)</a></li>
      <li>POST /runs</li>
      <li>POST /runs/{id}/start</li>
      <li>GET /runs/{id}/review</li>
    </ul>
    </body></html>"""


@app.get("/runs/new")
def new_run_hardcoded(db: Session = Depends(db_session)):
    """Create+start a run from hardcoded seed blocks, then redirect to review.

    This exists purely to make browser testing easy (no curl/webhooks).
    """

    # Prefer an env var override; fall back to repo file.
    import os

    blocks = (os.environ.get("DEFAULT_RUN_BLOCKS") or "").strip()
    if not blocks:
        # repo-local default
        try:
            from pathlib import Path

            seeds_path = (Path(__file__).resolve().parent.parent / "seeds_marc.txt").resolve()
            with open(seeds_path, "r", encoding="utf-8", errors="replace") as f:
                blocks = (f.read() or "").strip()
        except FileNotFoundError:
            blocks = ""

    if not blocks:
        raise HTTPException(500, "no hardcoded seed blocks available")

    r = _create_run_from_blocks(blocks=blocks, directions="", db=db)
    return RedirectResponse(url=f"/runs/{r.id}/review", status_code=303)


@app.get("/runs/new/custom", response_class=HTMLResponse)
def new_run_custom(request: Request):
    return templates.TemplateResponse("new_run.html", {"request": request, "default_blocks": "", "directions": ""})


@app.post("/runs/preview", response_class=HTMLResponse)
async def preview_run(request: Request, blocks: str = Form(...), directions: str = Form(default="")):
    seed_parts = parse_seed_parts_blocks(blocks)
    seeds = list(seed_parts.keys())
    counts = {s: len(seed_parts.get(s) or []) for s in seeds}
    sample = {s: (seed_parts.get(s) or [])[:10] for s in seeds}

    # Basic validation: require 1+ seeds and 1+ tokens total
    total = sum(counts.values())
    err = None
    if len(seeds) < 1:
        err = "No seed URLs/domains detected. Start each block with a URL or domain."
    elif total < 1:
        err = "No part tokens detected under seeds. Add part numbers/names on lines after each seed."

    if err:
        # Debug help: show what the parser saw (first chars + codepoints)
        head = blocks[:200]
        dbg = {
            "seeds_detected": seeds,
            "counts": counts,
            "head_repr": repr(head),
            "head_codepoints": [hex(ord(c)) for c in head[:80]],
        }
        return templates.TemplateResponse(
            "new_run.html",
            {"request": request, "default_blocks": blocks, "directions": directions, "error": err, "debug": dbg},
            status_code=400,
        )

    # Escape for hidden inputs (keep as-is; Jinja auto-escapes)
    return templates.TemplateResponse(
        "preview_run.html",
        {
            "request": request,
            "seeds": seeds,
            "counts": counts,
            "sample": sample,
            "directions": directions or "Domains only. Build part→seller graph.",
            "blocks_escaped": blocks,
            "directions_escaped": directions,
        },
    )


def _create_run_from_blocks(*, blocks: str, directions: str, db: Session) -> Run:
    seed_parts = parse_seed_parts_blocks(blocks)
    seeds = list(seed_parts.keys())
    if len(seeds) < 1:
        raise HTTPException(400, "no seeds detected")

    # Store parts in seed_phrases for now (schema compatibility)
    seed_phrases = seed_parts

    strat = Strategy.default(seeds, seed_phrases).to_dict()
    r = Run(
        seed_domains=seeds,
        seed_phrases=seed_phrases,
        directions=directions or "Domains only. Build part→seller graph.",
        strategy=strat,
        status=RunStatus.created,
    )
    db.add(r)
    db.commit()
    db.refresh(r)

    # start immediately
    r.status = RunStatus.running
    db.add(Task(run_id=r.id, type=TaskType.propose_targets))
    db.commit()
    return r


@app.post("/runs/create_from_blocks")
def create_from_blocks(blocks: str = Form(...), directions: str = Form(default=""), db: Session = Depends(db_session)):
    r = _create_run_from_blocks(blocks=blocks, directions=directions, db=db)
    return RedirectResponse(url=f"/runs/{r.id}/review", status_code=303)


@app.post("/runs/create_from_text")
async def create_from_text(request: Request, db: Session = Depends(db_session)):
    """Programmatic escape hatch.

    Accepts either:
    - text/plain body: raw blocks text
    - application/json: {"blocks": "...", "directions": "..."}

    Returns JSON {id, review_url}.
    """
    ct = (request.headers.get("content-type") or "").lower()
    directions = ""
    blocks = ""
    if "application/json" in ct:
        j = await request.json()
        blocks = (j.get("blocks") or "").strip()
        directions = (j.get("directions") or "").strip()
    else:
        blocks = (await request.body()).decode("utf-8", errors="replace").strip()

    r = _create_run_from_blocks(blocks=blocks, directions=directions, db=db)
    return {"id": r.id, "review_url": f"/runs/{r.id}/review"}


@app.post("/runs")
def create_run(payload: RunCreate, db: Session = Depends(db_session)):
    strat = Strategy.default(payload.seed_domains, payload.seed_phrases).to_dict()
    r = Run(
        seed_domains=payload.seed_domains,
        seed_phrases=payload.seed_phrases,
        directions=payload.directions,
        strategy=strat,
        status=RunStatus.created,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return {"id": r.id, "status": r.status}


@app.post("/runs/{run_id}/start")
def start_run(run_id: int, db: Session = Depends(db_session)):
    r = db.get(Run, run_id)
    if not r:
        raise HTTPException(404, "run not found")
    if r.status not in (RunStatus.created, RunStatus.waiting_feedback):
        return {"id": r.id, "status": r.status}

    r.status = RunStatus.running
    t = Task(run_id=r.id, type=TaskType.propose_targets)
    db.add(t)
    db.commit()
    return {"id": r.id, "status": r.status, "task_enqueued": t.type}


@app.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(db_session)):
    r = db.get(Run, run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return {
        "id": r.id,
        "status": r.status,
        "seed_domains": r.seed_domains,
        "seed_phrases_keys": list((r.seed_phrases or {}).keys()),
    }


@app.get("/runs/{run_id}/tasks")
def list_tasks(run_id: int, db: Session = Depends(db_session)):
    rows = (
        db.execute(select(Task).where(Task.run_id == run_id).order_by(Task.id.asc()).limit(500))
        .scalars()
        .all()
    )
    return [
        {
            "id": t.id,
            "type": t.type,
            "status": t.status,
            "created_at": t.created_at,
            "started_at": t.started_at,
            "finished_at": t.finished_at,
            "error": t.error,
            "result": t.result,
        }
        for t in rows
    ]


# --- Demo UI ---


def _ensure_demo_control(db: Session, run_id: int) -> DemoRunControl:
    import os

    c = db.get(DemoRunControl, run_id)
    if c:
        return c
    budget = int(os.getenv("DEMO_SERPER_BUDGET", "2000"))
    c = DemoRunControl(run_id=run_id, state="stopped", serper_calls_used=0, serper_budget=budget)
    db.add(c)
    db.commit()
    return c


def _ensure_demo_parts_seeded(db: Session, run_id: int) -> int:
    """Ensure DemoPart rows exist for this run.

    This makes the demo robust even if the worker hasn't attached yet.
    Returns number of parts inserted.
    """
    from sqlalchemy import select

    existing = db.execute(select(func.count()).select_from(DemoPart).where(DemoPart.run_id == run_id)).scalar() or 0
    if int(existing) > 0:
        return 0

    r = db.get(Run, run_id)
    if not r:
        return 0

    toks: list[str] = []
    for _seed, parts in (r.seed_phrases or {}).items():
        for p in parts or []:
            t = (p or "").strip()
            if t:
                toks.append(t)

    # dedupe (case-insensitive)
    seen = set()
    deduped: list[str] = []
    for t in toks:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(t)

    added = 0
    for t in deduped[:5000]:
        db.add(DemoPart(run_id=run_id, token=t, status=DemoPartStatus.validated, confidence=0.8, reason="seed"))
        added += 1

    if added:
        db.commit()
    return added


def _demo_prime_queries(db: Session, run_id: int, backlog_target: int = 80) -> int:
    """Seed DemoSerperQuery backlog from validated DemoPart tokens.

    We do this in the web app so the demo UI can become lively even if the
    background demo worker hasn't attached yet (or its Serper lanes crashed).

    Returns number of queries inserted.
    """

    from sqlalchemy import select

    # How many pending already?
    pending_n = (
        db.execute(
            select(func.count())
            .select_from(DemoSerperQuery)
            .where(DemoSerperQuery.run_id == run_id)
            .where(DemoSerperQuery.status == DemoSerperQueryStatus.pending)
        ).scalar()
        or 0
    )
    if int(pending_n) >= int(backlog_target):
        return 0

    need = max(0, int(backlog_target) - int(pending_n))
    if need <= 0:
        return 0

    # sample validated parts
    parts = (
        db.execute(
            select(DemoPart.token)
            .where(DemoPart.run_id == run_id)
            .where(DemoPart.status == DemoPartStatus.validated)
            .limit(800)
        )
        .scalars()
        .all()
    )

    # generate broader queries (match worker/demo.py behavior)
    qs: list[str] = []
    sample = parts[: min(len(parts), max(20, need // 4))]
    for t in sample:
        tt = (t or "").strip()
        if not tt:
            continue
        qs.append(f'"{tt}" aircraft parts')
        qs.append(f'"{tt}" aviation')
        qs.append(f'"{tt}" supplier')
        qs.append(f'"{tt}" distributor')
        qs.append(f'"{tt}" "Request a Quote"')

    # include a few competitor-style queries from seeds
    r = db.get(Run, run_id)
    if r:
        for sd in (r.seed_domains or [])[:3]:
            if sd:
                qs.append(f"{sd} competitors")
                qs.append(f"{sd} aircraft parts")

    added = 0
    for q in qs:
        if added >= need:
            break
        # Avoid duplicates that are in-flight (pending/running)
        in_flight = (
            db.execute(
                select(DemoSerperQuery)
                .where(DemoSerperQuery.run_id == run_id)
                .where(DemoSerperQuery.query == q)
                .where(DemoSerperQuery.status.in_([DemoSerperQueryStatus.pending, DemoSerperQueryStatus.running]))
                .limit(1)
            )
            .scalar_one_or_none()
        )
        if in_flight:
            continue
        db.add(DemoSerperQuery(run_id=run_id, query=q, status=DemoSerperQueryStatus.pending))
        added += 1

    if added:
        db.commit()
    return added


@app.get("/demo", response_class=HTMLResponse)
def demo_home(db: Session = Depends(db_session)):
    # Single-run demo: redirect to newest running run if any, else latest run.
    ctl = db.execute(select(DemoRunControl).where(DemoRunControl.state == "running").order_by(DemoRunControl.run_id.desc()).limit(1)).scalar_one_or_none()
    r = db.get(Run, ctl.run_id) if ctl else db.execute(select(Run).order_by(Run.id.desc()).limit(1)).scalar_one_or_none()
    if not r:
        return RedirectResponse(url="/runs/new", status_code=303)
    _ensure_demo_control(db, r.id)
    return RedirectResponse(url=f"/runs/{r.id}/demo", status_code=303)


@app.get("/demo/run")
def demo_latest_run(db: Session = Depends(db_session)):
    """Return the active demo run id (newest running), else newest run id."""
    ctl = db.execute(select(DemoRunControl).where(DemoRunControl.state == "running").order_by(DemoRunControl.run_id.desc()).limit(1)).scalar_one_or_none()
    rid = int(ctl.run_id) if ctl else int(db.execute(select(Run.id).order_by(Run.id.desc()).limit(1)).scalar_one_or_none() or 0)
    return {"run_id": rid}


@app.get("/runs/{run_id}/demo", response_class=HTMLResponse)
def demo_run(request: Request, run_id: int, db: Session = Depends(db_session)):
    r = db.get(Run, run_id)
    if not r:
        raise HTTPException(404, "run not found")
    _ensure_demo_control(db, run_id)
    return templates.TemplateResponse("demo.html", {"request": request, "run_id": run_id})


@app.post("/runs/{run_id}/demo/start")
def demo_start(run_id: int, db: Session = Depends(db_session)):
    # Enforce single-active demo: stop every other demo run.
    # This prevents multiple runs burning Serper budget in parallel.
    others = (
        db.execute(select(DemoRunControl).where(DemoRunControl.run_id != run_id).where(DemoRunControl.state == "running"))
        .scalars()
        .all()
    )
    for o in others:
        o.state = "stopped"
    if others:
        db.commit()

    c = _ensure_demo_control(db, run_id)

    # If env budget changed, keep control row in sync for new demos.
    import os

    desired_budget = int(os.getenv("DEMO_SERPER_BUDGET", "2000"))
    if int(c.serper_budget or 0) != desired_budget:
        c.serper_budget = desired_budget
        db.commit()

    # Ensure parts exist for demo.
    seeded = _ensure_demo_parts_seeded(db, run_id)
    if seeded:
        db.add(
            DemoEvent(
                run_id=run_id,
                type=DemoEventType.DECISION,
                data={
                    "title": "Seeded parts",
                    "meta": f"added={seeded}",
                    "body": "Seeded DemoPart rows from run.seed_phrases",
                },
            )
        )
        db.commit()

    # Prime some pending queries so the UI has immediate motion, even if the
    # background demo worker isn't attached yet.
    backlog = int(os.getenv("DEMO_QUERY_BACKLOG_TARGET", "80"))
    primed = _demo_prime_queries(db, run_id, backlog_target=backlog)
    if primed:
        db.add(
            DemoEvent(
                run_id=run_id,
                type=DemoEventType.DECISION,
                data={
                    "title": "Primed Serper queries",
                    "meta": f"added={primed} pending_target={backlog}",
                    "body": "Inserted DemoSerperQuery backlog (pending)",
                },
            )
        )
        db.commit()

    c.state = "running"
    db.commit()
    db.add(
        DemoEvent(
            run_id=run_id,
            type=DemoEventType.RUN_STATE,
            data={"state": c.state, "serper_calls_used": c.serper_calls_used, "now": "Running"},
        )
    )
    db.commit()
    return {"ok": True, "state": c.state, "primed_queries": primed}


@app.get("/runs/{run_id}/demo/control")
def demo_control(run_id: int, db: Session = Depends(db_session)):
    """Debug: return demo control row so we can see if the worker should be running."""
    c = db.get(DemoRunControl, run_id)
    if not c:
        return {"run_id": run_id, "exists": False}
    return {
        "run_id": run_id,
        "exists": True,
        "state": c.state,
        "serper_calls_used": c.serper_calls_used,
        "serper_budget": c.serper_budget,
        "updated_at": getattr(c, "updated_at", None),
    }


@app.get("/runs/{run_id}/demo/stats")
def demo_stats(run_id: int, db: Session = Depends(db_session)):
    """Debug: snapshot of demo tables so we can see if the worker is making progress."""

    def grouped_counts(model, status_col, enum_cls=None):
        rows = db.execute(
            select(status_col, func.count()).where(getattr(model, "run_id") == run_id).group_by(status_col)
        ).all()
        out = {}
        for st, cnt in rows:
            k = st.value if hasattr(st, "value") else str(st)
            out[k] = int(cnt)
        return out

    parts_total = int(db.execute(select(func.count()).select_from(DemoPart).where(DemoPart.run_id == run_id)).scalar() or 0)
    edges_total = int(db.execute(select(func.count()).select_from(DemoEdge).where(DemoEdge.run_id == run_id)).scalar() or 0)
    suppliers_total = int(db.execute(select(func.count()).select_from(DemoSupplier).where(DemoSupplier.run_id == run_id)).scalar() or 0)
    queries_total = int(db.execute(select(func.count()).select_from(DemoSerperQuery).where(DemoSerperQuery.run_id == run_id)).scalar() or 0)

    return {
        "run_id": run_id,
        "parts_total": parts_total,
        "parts_by_status": grouped_counts(DemoPart, DemoPart.status, DemoPartStatus),
        "suppliers_total": suppliers_total,
        "suppliers_by_status": grouped_counts(DemoSupplier, DemoSupplier.status, DemoSupplierStatus),
        "queries_total": queries_total,
        "queries_by_status": grouped_counts(DemoSerperQuery, DemoSerperQuery.status, DemoSerperQueryStatus),
        "edges_total": edges_total,
        "edges_by_status": grouped_counts(DemoEdge, DemoEdge.status, DemoEdgeStatus),
    }


@app.post("/runs/{run_id}/demo/stop")
def demo_stop(run_id: int, db: Session = Depends(db_session)):
    c = _ensure_demo_control(db, run_id)
    c.state = "stopped"
    db.commit()
    db.add(DemoEvent(run_id=run_id, type=DemoEventType.RUN_STATE, data={"state": c.state, "serper_calls_used": c.serper_calls_used, "now": "Stopped"}))
    db.commit()
    return {"ok": True, "state": c.state}


@app.post("/runs/{run_id}/demo/reset")
def demo_reset(run_id: int, db: Session = Depends(db_session)):
    # Start a brand-new run using the hardcoded blocks flow.
    return RedirectResponse(url="/runs/new", status_code=303)


@app.get("/runs/{run_id}/events")
async def demo_events(run_id: int, after_id: int = 0, db: Session = Depends(db_session)):
    # SSE stream of demo events. We poll the DB for new rows.
    _ensure_demo_control(db, run_id)

    async def gen():
        nonlocal after_id
        from dealmatch.db.session import SessionLocal

        # On connect, send an initial state.
        db2 = SessionLocal()
        try:
            c = db2.get(DemoRunControl, run_id)
            if c:
                init = {"id": 0, "type": "RUN_STATE", "state": c.state, "serper_calls_used": c.serper_calls_used, "now": ""}
                yield f"data: {json.dumps(init)}\n\n"
        finally:
            db2.close()

        while True:
            dbi = SessionLocal()
            try:
                rows = (
                    dbi.execute(
                        select(DemoEvent)
                        .where(DemoEvent.run_id == run_id)
                        .where(DemoEvent.id > after_id)
                        .order_by(DemoEvent.id.asc())
                        .limit(200)
                    )
                    .scalars()
                    .all()
                )
                if rows:
                    for ev in rows:
                        after_id = max(after_id, ev.id)
                        payload = {"id": ev.id, "type": ev.type.value}
                        payload.update(ev.data or {})
                        yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.35)
            finally:
                dbi.close()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/runs/{run_id}/targets")
def list_targets(run_id: int, status: str | None = None, db: Session = Depends(db_session)):
    q = select(Target).where(Target.run_id == run_id)
    if status:
        q = q.where(Target.status == TargetStatus(status))
    rows = db.execute(q.order_by(Target.id.desc()).limit(500)).scalars().all()
    return [
        {
            "id": t.id,
            "domain": t.domain,
            "score": t.score,
            "status": t.status,
            "evidence": t.evidence,
        }
        for t in rows
    ]


@app.get("/runs/{run_id}/review", response_class=HTMLResponse)
def review(request: Request, run_id: int, db: Session = Depends(db_session)):
    r = db.get(Run, run_id)
    if not r:
        raise HTTPException(404, "run not found")
    targets = (
        db.execute(
            select(Target)
            .where(Target.run_id == run_id)
            .where(Target.status == TargetStatus.proposed)
            .order_by(Target.score.desc(), Target.id.desc())
            .limit(200)
        )
        .scalars()
        .all()
    )
    return templates.TemplateResponse("review.html", {"request": request, "run": r, "targets": targets})


@app.post("/runs/{run_id}/feedback")
async def submit_feedback(run_id: int, request: Request, db: Session = Depends(db_session)):
    r = db.get(Run, run_id)
    if not r:
        raise HTTPException(404, "run not found")

    form = await request.form()
    accepted: list[str] = []
    rejected: list[str] = []

    for k, v in form.items():
        if not k.startswith("d:"):
            continue
        dom = k.split(":", 1)[1]
        if v == "accept":
            accepted.append(dom)
        elif v == "reject":
            rejected.append(dom)

    notes = (form.get("notes") or "").strip()

    # persist feedback
    fb = FeedbackEvent(run_id=r.id, accepted_domains=accepted, rejected_domains=rejected, notes=notes or None)
    db.add(fb)

    # update target statuses
    for dom in accepted:
        t = db.execute(select(Target).where(Target.run_id == r.id, Target.domain == dom)).scalar_one_or_none()
        if t:
            t.status = TargetStatus.accepted
    for dom in rejected:
        t = db.execute(select(Target).where(Target.run_id == r.id, Target.domain == dom)).scalar_one_or_none()
        if t:
            t.status = TargetStatus.rejected

    # enqueue adapt strategy + propose next batch
    db.add(Task(run_id=r.id, type=TaskType.adapt_strategy, payload={"accepted": accepted, "rejected": rejected, "notes": notes}))
    db.add(Task(run_id=r.id, type=TaskType.propose_targets))

    r.status = RunStatus.running
    db.commit()

    return RedirectResponse(url=f"/runs/{r.id}/review", status_code=303)
