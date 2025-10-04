from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

class User(Base):
    """Registered account with authentication metadata."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
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


class UserProfile(Base):
    """Extended metadata for a user."""

    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    name: Mapped[str | None] = mapped_column(String(100), default=None)
    bio: Mapped[str | None] = mapped_column(Text, default=None)
    avatar_url: Mapped[str | None] = mapped_column(String(255), default=None)

    user: Mapped[User] = relationship(back_populates="profile")


class Follow(Base):
    """Directed follow relationship between two users."""

    __tablename__ = "follows"

    follower_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    following_id: Mapped[int] = mapped_column(
        BigInteger,
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
