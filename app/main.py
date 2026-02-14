from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from dealmatch.core.models import FeedbackEvent, Run, RunStatus, Target, TargetStatus, Task, TaskType
from dealmatch.core.strategy import Strategy
from dealmatch.db.session import db_session

app = FastAPI(title="Agentic Researcher")
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
      <li>POST /runs</li>
      <li>POST /runs/{id}/start</li>
      <li>GET  /runs/{id}/review</li>
    </ul>
    </body></html>"""


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
