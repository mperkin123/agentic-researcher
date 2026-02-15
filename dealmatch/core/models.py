import enum
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dealmatch.db.base import Base

# Demo-only models (graph + events)
from dealmatch.core.demo_models import (  # noqa: F401
    DemoEdge,
    DemoEdgeStatus,
    DemoEvent,
    DemoEventType,
    DemoPart,
    DemoPartStatus,
    DemoRunControl,
    DemoSerperQuery,
    DemoSerperQueryStatus,
    DemoSupplier,
    DemoSupplierStatus,
)


class RunStatus(str, enum.Enum):
    created = "created"
    running = "running"
    waiting_feedback = "waiting_feedback"
    done = "done"
    error = "error"


class TargetStatus(str, enum.Enum):
    proposed = "proposed"
    accepted = "accepted"
    rejected = "rejected"
    verified_yes = "verified_yes"
    verified_no = "verified_no"


class TaskStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class TaskType(str, enum.Enum):
    propose_targets = "propose_targets"
    verify_targets = "verify_targets"
    adapt_strategy = "adapt_strategy"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.created)

    seed_domains: Mapped[list] = mapped_column(JSON)  # list[str]
    seed_phrases: Mapped[dict] = mapped_column(JSON)  # {domain: [phrases]}
    directions: Mapped[str] = mapped_column(Text)

    strategy: Mapped[dict] = mapped_column(JSON, default=dict)  # mutable strategy blob

    targets: Mapped[list["Target"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    tasks: Mapped[list["Task"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Target(Base):
    __tablename__ = "targets"
    __table_args__ = (UniqueConstraint("run_id", "domain", name="uq_run_domain"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    domain: Mapped[str] = mapped_column(String(255), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[TargetStatus] = mapped_column(Enum(TargetStatus), default=TargetStatus.proposed)

    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship(back_populates="targets")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    type: Mapped[TaskType] = mapped_column(Enum(TaskType))
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.pending)

    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship(back_populates="tasks")


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    accepted_domains: Mapped[list] = mapped_column(JSON, default=list)
    rejected_domains: Mapped[list] = mapped_column(JSON, default=list)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
