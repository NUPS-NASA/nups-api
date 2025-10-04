from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr

class UserProfileBase(BaseModel):
    name: str | None = None
    bio: str | None = None
    avatar_url: str | None = None


class UserProfileCreate(UserProfileBase):
    pass


class UserProfileUpdate(UserProfileBase):
    """Fields that can be patched on an existing profile."""

    model_config = ConfigDict(extra="forbid")


class UserProfileRead(UserProfileBase):
    user_id: int

    model_config = ConfigDict(from_attributes=True)


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password_hash: str
    profile: UserProfileCreate | None = None


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    password_hash: str | None = None
    profile: UserProfileUpdate | None = None

    model_config = ConfigDict(extra="forbid")


class UserRead(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime
    profile: UserProfileRead | None = None

    model_config = ConfigDict(from_attributes=True)


class FollowCreate(BaseModel):
    """Request payload for following another user."""

    following_id: int

    model_config = ConfigDict(extra="forbid")


class FollowerCreate(BaseModel):
    """Request payload for adding a follower to a user."""

    follower_id: int

    model_config = ConfigDict(extra="forbid")


class FollowRead(BaseModel):
    follower_id: int
    following_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
