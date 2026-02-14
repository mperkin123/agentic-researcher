import asyncio
import os
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from dealmatch.core.models import Run, RunStatus, Task, TaskStatus, TaskType
from dealmatch.core.strategy import Strategy
from dealmatch.db.session import SessionLocal
from dealmatch.llm.client import extract_json, responses_create, strategy_models
from dealmatch.serper.client import serper_search


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


async def propose_targets(db: Session, run: Run):
    strat = Strategy.from_dict(run.strategy or {})

    planner_model, worker_model = strategy_models()

    # Use LLM to generate a small set of serper queries from seeds/phrases/directions.
    schema = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 25}
        },
        "required": ["queries"],
        "additionalProperties": False,
    }
    prompt = (
        "You are proposing search queries to discover aircraft parts supplier DOMAINS. "
        "Return queries only; do not include URLs.\n\n"
        f"Seeds: {run.seed_domains}\n"
        f"Seed phrases by seed: {run.seed_phrases}\n\n"
        f"Directions: {run.directions}\n\n"
        f"Current query templates: {strat.query_templates}\n"
        "Output JSON with key 'queries'."
    )
    resp = await responses_create(model=planner_model, input_text=prompt, json_schema=schema, temperature=0.2)
    j = extract_json(resp) or {}
    queries = [q.strip() for q in (j.get("queries") or []) if (q or "").strip()]

    # If LLM failed, fall back to templates.
    if not queries:
        queries = []
        for t in strat.query_templates:
            if "{phrase}" in t:
                # expand with first few phrases
                for dom, ps in (run.seed_phrases or {}).items():
                    for p in (ps or [])[:5]:
                        queries.append(t.replace("{phrase}", p))
            else:
                queries.append(t)
        queries = queries[:10]

    # Serper search each query, collect domains from results
    domains = []
    for q in queries[:10]:
        s = await serper_search(q, num=strat.serper_num)
        for r in (s.get("organic") or []):
            link = (r.get("link") or "")
            d = domain_only(link)
            if not d:
                continue
            if any(b in d for b in strat.block_substr):
                continue
            domains.append(d)
    # unique preserve order
    seen = set()
    domains = [d for d in domains if not (d in seen or seen.add(d))]

    # insert as proposed targets
    from dealmatch.core.models import Target

    added = 0
    for d in domains[: str(strat.propose_batch)]:
        exists = db.execute(select(Target).where(Target.run_id == run.id, Target.domain == d)).scalar_one_or_none()
        if exists:
            continue
        db.add(Target(run_id=run.id, domain=d, score=0.5, evidence={"source": "serper", "queries": queries[:10]}))
        added += 1
    run.status = RunStatus.waiting_feedback
    db.commit()
    return {"queries": queries[:10], "proposed": added, "total_domains": len(domains)}


async def adapt_strategy(db: Session, run: Run, payload: dict):
    strat = Strategy.from_dict(run.strategy or {})
    planner_model, worker_model = strategy_models()

    schema = {
        "type": "object",
        "properties": {
            "query_templates": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 40},
            "block_substr": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 80},
            "propose_batch": {"type": "integer", "minimum": 10, "maximum": 300},
            "serper_num": {"type": "integer", "minimum": 3, "maximum": 20},
            "notes": {"type": "string"},
        },
        "required": ["query_templates", "block_substr", "propose_batch", "serper_num"],
        "additionalProperties": False,
    }

    accepted = payload.get("accepted") or []
    rejected = payload.get("rejected") or []
    notes = payload.get("notes") or ""

    prompt = (
        "You are updating a search strategy to discover TARGET DOMAINS for an aviation parts supplier campaign.\n"
        "We will use Serper (google.serper.dev/search).\n\n"
        "Constraints:\n"
        "- Output ONLY JSON matching the schema.\n"
        "- query_templates should be reusable query strings; you may include {phrase} placeholder.\n"
        "- block_substr should include domain substrings to exclude (e.g. social sites, business databases).\n\n"
        f"Seeds: {run.seed_domains}\n"
        f"Seed phrases: {run.seed_phrases}\n\n"
        f"Directions: {run.directions}\n\n"
        f"Current strategy: {strat.to_dict()}\n\n"
        f"Feedback accepted domains: {accepted}\n"
        f"Feedback rejected domains: {rejected}\n"
        f"Additional notes: {notes}\n"
    )

    resp = await responses_create(model=planner_model, input_text=prompt, json_schema=schema, temperature=0.2)
    j = extract_json(resp)
    if not isinstance(j, dict):
        return {"status": "no_change", "reason": "llm_returned_non_object"}

    # sanitize
    new_strat = Strategy.from_dict({
        "query_templates": j.get("query_templates") or strat.query_templates,
        "block_substr": j.get("block_substr") or strat.block_substr,
        "propose_batch": j.get("propose_batch") or strat.propose_batch,
        "serper_num": j.get("serper_num") or strat.serper_num,
    })
    run.strategy = new_strat.to_dict()
    db.commit()
    return {"status": "updated", "strategy": run.strategy, "llm_notes": j.get("notes", "")}


async def worker_loop(poll_s: float = 1.0):
    while True:
        db = SessionLocal()
        try:
            task = db.execute(
                select(Task).where(Task.status == TaskStatus.pending).order_by(Task.id.asc()).limit(1)
            ).scalar_one_or_none()
            if not task:
                db.close()
                await asyncio.sleep(poll_s)
                continue

            task.status = TaskStatus.running
            task.started_at = datetime.utcnow()
            db.commit()

            run = db.get(Run, task.run_id)
            if not run:
                task.status = TaskStatus.error
                task.error = "run not found"
                task.finished_at = datetime.utcnow()
                db.commit()
                continue

            try:
                if task.type == TaskType.propose_targets:
                    res = await propose_targets(db, run)
                elif task.type == TaskType.adapt_strategy:
                    res = await adapt_strategy(db, run, task.payload or {})
                else:
                    res = {"status": "noop"}
                task.status = TaskStatus.done
                task.result = res
            except Exception as e:
                task.status = TaskStatus.error
                task.error = repr(e)
            task.finished_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()


if __name__ == "__main__":
    poll = float(os.getenv("WORKER_POLL_S", "1"))
    asyncio.run(worker_loop(poll_s=poll))
