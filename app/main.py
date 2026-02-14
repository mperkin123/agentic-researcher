from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from dealmatch.core.models import Run, RunStatus, Task, TaskType
from dealmatch.core.strategy import Strategy
from dealmatch.db.session import db_session

app = FastAPI(title="dealmatch")


class RunCreate(BaseModel):
    seed_domains: list[str] = Field(min_length=3, max_length=3)
    seed_phrases: dict[str, list[str]] = Field(default_factory=dict)
    directions: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """<html><body>
    <h1>dealmatch</h1>
    <p>API is running.</p>
    <ul>
      <li>POST /runs</li>
      <li>POST /runs/{id}/start</li>
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
    # enqueue initial task: propose targets
    t = Task(run_id=r.id, type=TaskType.propose_targets)
    db.add(t)
    db.commit()
    return {"id": r.id, "status": r.status, "task_enqueued": t.type}
