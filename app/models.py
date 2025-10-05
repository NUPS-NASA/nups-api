from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import CHAR, TypeDecorator

# Column type that maps to BIGINT in PostgreSQL while remaining INTEGER for SQLite
BIGINT_PK = Integer().with_variant(BigInteger(), "postgresql")

from .database import Base


dataset_data_association = Table(
    "dataset_data",
    Base.metadata,
    Column(
        "dataset_id",
        ForeignKey("dataset.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "data_id",
        ForeignKey("data.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    UniqueConstraint("dataset_id", "data_id"),
)


class GUID(TypeDecorator):
    """Database-agnostic UUID storage."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(str(value)))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class User(Base):
    """Registered account with minimal metadata."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    profile: Mapped["UserProfile"] = relationship(
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    followers: Mapped[list["Follow"]] = relationship(
        back_populates="following",
        foreign_keys="Follow.following_id",
        passive_deletes=True,
    )
    following: Mapped[list["Follow"]] = relationship(
        back_populates="follower",
        foreign_keys="Follow.follower_id",
        passive_deletes=True,
    )
    repositories: Mapped[list["Repository"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )
    project_memberships: Mapped[list["ProjectUser"]] = relationship(
        back_populates="user",
    )
    uploaded_project_repositories: Mapped[list["ProjectRepository"]] = relationship(
        back_populates="uploader",
    )
    starred_repositories: Mapped[list["StarredRepository"]] = relationship(
        back_populates="user",
    )
    pinned_projects: Mapped[list["PinnedProject"]] = relationship(
        back_populates="user",
    )


class UserProfile(Base):
    """Extended metadata for a user."""

    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    bio: Mapped[str | None] = mapped_column(Text, default=None)
    avatar_url: Mapped[str | None] = mapped_column(String(255), default=None)

    user: Mapped[User] = relationship(back_populates="profile")


class Follow(Base):
    """Directed follow relationship between two users."""

    __tablename__ = "follows"

    follower_id: Mapped[int] = mapped_column(
        BIGINT_PK,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    following_id: Mapped[int] = mapped_column(
        BIGINT_PK,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    follower: Mapped[User] = relationship(
        "User",
        foreign_keys=[follower_id],
        back_populates="following",
    )
    following: Mapped[User] = relationship(
        "User",
        foreign_keys=[following_id],
        back_populates="followers",
    )


class Repository(Base):
    """Data container owned by a user."""

    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("user_id", "name"),)

    owner: Mapped[User] = relationship(back_populates="repositories")
    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list["Session"]] = relationship(back_populates="repository")
    project_links: Mapped[list["ProjectRepository"]] = relationship(
        back_populates="repository",
    )
    starred_by: Mapped[list["StarredRepository"]] = relationship(
        back_populates="repository",
    )


class Project(Base):
    """Collaboration workspace containing multiple repositories."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    repository_links: Mapped[list["ProjectRepository"]] = relationship(
        back_populates="project",
    )
    memberships: Mapped[list["ProjectUser"]] = relationship(
        back_populates="project",
    )
    pinned_by: Mapped[list["PinnedProject"]] = relationship(
        back_populates="project",
    )


class ProjectRepository(Base):
    """Association between projects and repositories with upload metadata."""

    __tablename__ = "project_repositories"

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        primary_key=True,
    )
    uploaded_by: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="repository_links")
    repository: Mapped[Repository] = relationship(back_populates="project_links")
    uploader: Mapped[User] = relationship(
        back_populates="uploaded_project_repositories"
    )


class ProjectUser(Base):
    """Membership link between users and projects with role metadata."""

    __tablename__ = "project_users"

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="memberships")
    user: Mapped[User] = relationship(back_populates="project_memberships")


class StarredRepository(Base):
    """User-starred repositories for quick access."""

    __tablename__ = "starred_repositories"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        primary_key=True,
    )
    starred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[User] = relationship(back_populates="starred_repositories")
    repository: Mapped[Repository] = relationship(back_populates="starred_by")


class PinnedProject(Base):
    """User-specific project pinning with optional ordering."""

    __tablename__ = "pinned_projects"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    pinned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    position: Mapped[int | None] = mapped_column(Integer, default=None)

    user: Mapped[User] = relationship(back_populates="pinned_projects")
    project: Mapped[Project] = relationship(back_populates="pinned_by")


class Dataset(Base):
    """Versioned dataset within a repository."""

    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("repository_id", "version"),)

    repository: Mapped[Repository] = relationship(back_populates="datasets")
    data_items: Mapped[list["Data"]] = relationship(
        "Data",
        secondary=dataset_data_association,
        back_populates="datasets",
    )
    preprocess_items: Mapped[list["DatasetPreprocessData"]] = relationship(
        "DatasetPreprocessData",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list["Session"]] = relationship(back_populates="dataset")


class Data(Base):
    """Individual FITS artifact that can belong to multiple datasets."""

    __tablename__ = "data"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    fits_original_path: Mapped[str] = mapped_column(Text, nullable=False)
    fits_image_path: Mapped[str | None] = mapped_column(Text, default=None)
    fits_data_json: Mapped[dict | None] = mapped_column(JSON)
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    datasets: Mapped[list[Dataset]] = relationship(
        "Dataset",
        secondary=dataset_data_association,
        back_populates="data_items",
    )
    preprocess_links: Mapped[list["DatasetPreprocessData"]] = relationship(
        "DatasetPreprocessData",
        back_populates="data",
        cascade="all, delete-orphan",
    )


class DatasetPreprocessData(Base):
    """Association between datasets and preprocessing data grouped by category."""

    __tablename__ = "dataset_preprocess_data"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data_id: Mapped[int] = mapped_column(
        ForeignKey("data.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (UniqueConstraint("dataset_id", "data_id"),)

    dataset: Mapped[Dataset] = relationship(back_populates="preprocess_items")
    data: Mapped[Data] = relationship(back_populates="preprocess_links")


class Session(Base):
    """Processing run for a committed dataset version."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    run_id: Mapped[uuid.UUID] = mapped_column(GUID(), unique=True, nullable=False)
    repository_id: Mapped[int] = mapped_column(
        BIGINT_PK,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data_version: Mapped[int] = mapped_column(Integer, nullable=False)
    current_step: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    progress: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    repository: Mapped[Repository] = relationship(back_populates="sessions")
    dataset: Mapped[Dataset] = relationship(back_populates="sessions")
    pipeline_steps: Mapped[list["PipelineStep"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    lightcurve_result: Mapped["Lightcurve | None"] = relationship(
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )
    denoise_result: Mapped["Denoise | None"] = relationship(
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )
    candidate_result: Mapped["Candidate | None"] = relationship(
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )


class PipelineStep(Base):
    """Lifecycle record for a pipeline step within a session."""

    __tablename__ = "pipeline_steps"

    step_id: Mapped[int] = mapped_column(primary_key=True, index=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("sessions.run_id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    progress: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    data: Mapped[dict | None] = mapped_column(JSON)
    log: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[Session] = relationship(back_populates="pipeline_steps")


class Lightcurve(Base):
    """Lightcurve analysis output for a session."""

    __tablename__ = "lightcurve"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship(back_populates="lightcurve_result")


class Denoise(Base):
    """Denoise analysis output for a session."""

    __tablename__ = "denoise"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship(back_populates="denoise_result")


class Candidate(Base):
    """Candidate detection output for a session."""

    __tablename__ = "candidate"

    id: Mapped[int] = mapped_column(
        BIGINT_PK, primary_key=True, index=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data: Mapped[dict | None] = mapped_column(JSON)
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        server_default=text("false"),
        default=False,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship(back_populates="candidate_result")
