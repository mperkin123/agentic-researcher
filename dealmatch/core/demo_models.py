import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from dealmatch.db.base import Base


class DemoEventType(str, enum.Enum):
    RUN_STATE = "RUN_STATE"
    SERPER_LANE = "SERPER_LANE"
    EDGE = "EDGE"
    DECISION = "DECISION"


class DemoEvent(Base):
    __tablename__ = "demo_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    type: Mapped[DemoEventType] = mapped_column(Enum(DemoEventType))
    data: Mapped[dict] = mapped_column(JSON, default=dict)


class DemoRunControl(Base):
    __tablename__ = "demo_run_control"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    state: Mapped[str] = mapped_column(String(32), default="stopped")  # running|stopped

    serper_calls_used: Mapped[int] = mapped_column(Integer, default=0)
    serper_budget: Mapped[int] = mapped_column(Integer, default=5000)

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DemoSerperQueryStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class DemoSerperQuery(Base):
    __tablename__ = "demo_serper_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)

    query: Mapped[str] = mapped_column(Text)
    status: Mapped[DemoSerperQueryStatus] = mapped_column(Enum(DemoSerperQueryStatus), default=DemoSerperQueryStatus.pending)

    lane: Mapped[int | None] = mapped_column(Integer, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    result: Mapped[dict] = mapped_column(JSON, default=dict)


class DemoSupplierStatus(str, enum.Enum):
    discovered = "discovered"
    qualifying = "qualifying"
    qualified = "qualified"
    rejected = "rejected"


class DemoSupplier(Base):
    __tablename__ = "demo_suppliers"
    __table_args__ = (UniqueConstraint("run_id", "domain", name="uq_demo_supplier"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    domain: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[DemoSupplierStatus] = mapped_column(Enum(DemoSupplierStatus), default=DemoSupplierStatus.discovered)

    source_query_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    pages_fetched: Mapped[int] = mapped_column(Integer, default=0)
    last_scraped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DemoPartStatus(str, enum.Enum):
    seed = "seed"
    validated = "validated"
    rejected = "rejected"


class DemoPart(Base):
    __tablename__ = "demo_parts"
    __table_args__ = (UniqueConstraint("run_id", "token", name="uq_demo_part"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    token: Mapped[str] = mapped_column(Text)
    status: Mapped[DemoPartStatus] = mapped_column(Enum(DemoPartStatus), default=DemoPartStatus.seed)

    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class DemoEdgeStatus(str, enum.Enum):
    researching = "researching"
    validated = "validated"
    rejected = "rejected"


class DemoEdge(Base):
    __tablename__ = "demo_edges"
    __table_args__ = (
        UniqueConstraint("run_id", "supplier_domain", "part_token", "evidence_url", name="uq_demo_edge"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    supplier_domain: Mapped[str] = mapped_column(String(255), index=True)
    part_token: Mapped[str] = mapped_column(Text)

    evidence_url: Mapped[str] = mapped_column(Text)
    evidence_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[DemoEdgeStatus] = mapped_column(Enum(DemoEdgeStatus), default=DemoEdgeStatus.researching)

    validator_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    validator_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
