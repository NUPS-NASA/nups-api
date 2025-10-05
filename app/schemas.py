from __future__ import annotations

from datetime import date as Date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# -----------------------
# User & profile schemas
# -----------------------


class UserProfileBase(BaseModel):
    bio: str | None = Field(
        default=None,
        description="Short biography shown on user profiles.",
        examples=["Researcher focusing on near-earth object detection."],
    )
    avatar_url: str | None = Field(
        default=None,
        description="Absolute URL pointing to the user's avatar image.",
        examples=["https://cdn.example.com/avatars/alice.png"],
    )


class UserProfileCreate(UserProfileBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "bio": "Researcher focusing on near-earth object detection.",
                "avatar_url": "https://cdn.example.com/avatars/alice.png",
            }
        },
    )


class UserProfileUpdate(UserProfileBase):
    """Fields that can be patched on an existing profile."""

    model_config = ConfigDict(extra="forbid")


class UserProfileRead(UserProfileBase):
    user_id: int

    model_config = ConfigDict(from_attributes=True)


class UserBase(BaseModel):
    email: EmailStr = Field(
        description="Unique email address used for communication and login.",
        examples=["astro@example.com"],
    )


class UserCreate(UserBase):
    password: str = Field(
        description="Plaintext password that will be hashed before persistence.",
        min_length=8,
        examples=["astr0n0my!"],
    )
    profile: UserProfileCreate | None = Field(
        default=None,
        description="Optional profile details to create alongside the user.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "email": "astro@example.com",
                "password": "astr0n0my!",
                "profile": {
                    "bio": "Researcher focusing on near-earth object detection.",
                    "avatar_url": "https://cdn.example.com/avatars/alice.png",
                },
            }
        },
    )


class UserLogin(BaseModel):
    """Credentials payload for logging in an existing user."""

    email: EmailStr
    password: str = Field(min_length=8)

    model_config = ConfigDict(extra="forbid")


class UserUpdate(BaseModel):
    email: EmailStr | None = Field(
        default=None,
        description="Updated email address. Must remain unique across users.",
    )
    profile: UserProfileUpdate | None = Field(
        default=None,
        description="Profile fields to upsert for the user.",
    )

    model_config = ConfigDict(extra="forbid")


class UserRead(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime
    profile: UserProfileRead | None = Field(
        default=None,
        description="Associated profile information when available.",
    )

    model_config = ConfigDict(from_attributes=True)


class AuthTokens(BaseModel):
    access_token: str = Field(description="JWT access token used for authenticated requests.")
    refresh_token: str = Field(description="JWT refresh token used to obtain new access tokens.")
    token_type: str = Field(default="bearer", description="Token type hint for clients.")

    model_config = ConfigDict(extra="forbid")


class AuthLoginResponse(AuthTokens):
    user: UserRead = Field(description="Authenticated user resource.")


class AuthTokenRefreshRequest(BaseModel):
    refresh_token: str = Field(description="Refresh token previously issued during login.")

    model_config = ConfigDict(extra="forbid")


class AuthTokenRefreshResponse(AuthTokens):
    user: UserRead = Field(description="Authenticated user resource associated with the tokens.")


class FollowCreate(BaseModel):
    """Request payload for following another user."""

    following_id: int = Field(
        description="Identifier of the user that the caller will follow.",
        examples=[42],
    )

    model_config = ConfigDict(extra="forbid")


class FollowerCreate(BaseModel):
    """Request payload for adding a follower to a user."""

    follower_id: int = Field(
        description="Identifier of the user that will follow the path user.",
        examples=[7],
    )

    model_config = ConfigDict(extra="forbid")


class FollowRead(BaseModel):
    follower_id: int = Field(description="Follower user identifier.")
    following_id: int = Field(description="User being followed.")
    created_at: datetime = Field(description="Timestamp when the relationship was created.")

    model_config = ConfigDict(from_attributes=True)


# -----------------------
# Repository schemas
# -----------------------


class RepositoryBase(BaseModel):
    name: str = Field(
        description="Human-friendly repository name displayed in the UI.",
        examples=["hip-65211-lightcurves"],
    )
    description: str | None = Field(
        default=None,
        description="Optional longer-form description of the repository contents.",
    )


class RepositoryCreate(RepositoryBase):
    user_id: int = Field(
        description="Owner user identifier.",
        examples=[1],
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_id": 1,
                "name": "hip-65211-lightcurves",
                "description": "Time-series photometry for HIP 65211.",
            }
        },
    )


class RepositoryUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        description="New repository name. Must remain unique per user.",
    )
    description: str | None = Field(
        default=None,
        description="Updated description for the repository.",
    )

    model_config = ConfigDict(extra="forbid")


class SessionSummary(BaseModel):
    id: int | None = Field(default=None, description="Session identifier.")
    run_id: UUID | None = Field(default=None, description="Stable run identifier for the session.")
    status: str | None = Field(default=None, description="Current lifecycle status (e.g. running, finished).")
    current_step: str | None = Field(default=None, description="Name of the pipeline step currently executing.")
    progress: int | None = Field(
        default=None,
        description="Overall completion percentage for the session.",
    )
    started_at: datetime | None = Field(default=None, description="Timestamp when the session started.")
    finished_at: datetime | None = Field(default=None, description="Timestamp when the session finished, if applicable.")

    model_config = ConfigDict(from_attributes=True)


class RepositoryRead(RepositoryBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime | None = Field(default=None, description="Last update timestamp for the repository.")
    starred: bool = Field(
        default=False,
        description="Whether the requesting user has starred this repository.",
    )
    session: SessionSummary | None = Field(
        default=None,
        description="Latest session summary when requested.",
    )

    model_config = ConfigDict(from_attributes=True)


class StarRead(BaseModel):
    user_id: int = Field(description="User who starred the repository.")
    repository_id: int = Field(description="Starred repository identifier.")
    starred_at: datetime = Field(description="Timestamp of the starring action.")

    model_config = ConfigDict(from_attributes=True)


# -----------------------
# Project schemas
# -----------------------


class ProjectBase(BaseModel):
    name: str = Field(
        description="Project title displayed to collaborators.",
        examples=["TESS Transit Deep Dive"],
    )
    description: str | None = Field(
        default=None,
        description="Optional summary describing the project's focus.",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Datetime when the project became active.",
    )


class ProjectCreate(ProjectBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "name": "TESS Transit Deep Dive",
                "description": "Cross-mission validation for new exoplanet candidates.",
                "start_date": "2024-02-01T00:00:00Z",
            }
        },
    )


class ProjectUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        description="Updated project name.",
    )
    description: str | None = Field(
        default=None,
        description="Updated project description.",
    )
    start_date: datetime | None = Field(
        default=None,
        description="New project start datetime.",
    )

    model_config = ConfigDict(extra="forbid")


class ProjectRead(ProjectBase):
    id: int
    created_at: datetime
    updated_at: datetime
    members_count: int = Field(
        default=0,
        description="Computed count of users participating in the project.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="List of labels associated with the project (placeholder until storage is available).",
    )

    model_config = ConfigDict(from_attributes=True)


class ProjectMemberRead(BaseModel):
    project_id: int = Field(description="Project identifier the member belongs to.")
    user_id: int = Field(description="Member user identifier.")
    role: str = Field(description="Role assigned to the member (e.g. pm, member).")
    joined_at: datetime = Field(description="Timestamp when the user joined the project.")
    user: UserRead | None = Field(
        default=None,
        description="User metadata populated when eager-loaded.",
    )

    model_config = ConfigDict(from_attributes=True)


class ProjectMemberCreate(BaseModel):
    user_id: int = Field(description="Identifier of the user joining the project.")
    role: str = Field(description="Role assigned to the user within the project.")

    model_config = ConfigDict(extra="forbid")


class ProjectMemberUpdate(BaseModel):
    role: str = Field(description="New role to apply to the project member.")

    model_config = ConfigDict(extra="forbid")


class ProjectRepositoryLinkCreate(BaseModel):
    repository_id: int = Field(description="Repository linked to the project.")
    uploaded_by: int = Field(description="User that performed the upload and link.")

    model_config = ConfigDict(extra="forbid")


class PinRead(BaseModel):
    user_id: int = Field(description="User that pinned the project.")
    project_id: int = Field(description="Pinned project identifier.")
    position: int | None = Field(
        default=None,
        description="Optional explicit ordering for pinned projects.",
    )
    pinned_at: datetime = Field(description="Timestamp when the project was pinned.")
    project: ProjectRead | None = Field(
        default=None,
        description="Project details when eager-loaded.",
    )

    model_config = ConfigDict(from_attributes=True)


class PinCreate(BaseModel):
    project_id: int = Field(description="Project identifier to pin.")
    position: int | None = Field(
        default=None,
        description="Optional slot to insert the project at.",
    )

    model_config = ConfigDict(extra="forbid")


class PinReorder(BaseModel):
    project_ids: list[int] = Field(
        description="Ordered list of project identifiers representing the desired pin order.",
        examples=[[10, 5, 8]],
    )

    model_config = ConfigDict(extra="forbid")


# -----------------------
# Dataset & data schemas
# -----------------------


class DatasetCreate(BaseModel):
    repository_id: int = Field(description="Repository that owns the dataset version.")
    version: int = Field(
        description="Monotonic dataset version within the repository.",
        examples=[2],
    )
    captured_at: datetime | None = Field(
        default=None,
        description="Optional capture time associated with the dataset contents.",
    )

    model_config = ConfigDict(extra="forbid")


class DatasetRead(BaseModel):
    id: int
    repository_id: int
    version: int
    captured_at: datetime | None = Field(default=None, description="Capture timestamp for the dataset version.")
    created_at: datetime = Field(description="Timestamp when the dataset version was created.")

    model_config = ConfigDict(from_attributes=True)


class DataCreate(BaseModel):
    dataset_id: int = Field(description="Dataset identifier that the item belongs to.")
    hash: str = Field(
        description="Content hash used to deduplicate items.",
        examples=["0x8aab5c9f"],
    )
    fits_original_path: str = Field(
        description="Path to the persisted FITS file.",
        examples=["/data/uploads/HIP65211/2024-02-01/raw.fits"],
    )
    fits_image_path: str | None = Field(
        default=None,
        description="Optional derived image path associated with the FITS source.",
    )
    fits_data_json: dict | None = Field(
        default=None,
        description="Structured FITS metadata extracted from the file.",
    )
    metadata_json: dict | None = Field(
        default=None,
        description="Arbitrary metadata describing the data item.",
    )

    model_config = ConfigDict(extra="forbid")


class DataRead(BaseModel):
    id: int
    dataset_id: int
    hash: str
    fits_original_path: str
    fits_image_path: str | None = Field(
        default=None,
        description="Optional derived image path associated with the FITS source.",
    )
    fits_data_json: dict | None = Field(
        default=None,
        description="Structured FITS metadata extracted from the file.",
    )
    metadata_json: dict | None = Field(
        default=None,
        description="Arbitrary metadata describing the data item.",
    )
    created_at: datetime = Field(description="Record creation timestamp.")
    updated_at: datetime | None = Field(
        default=None,
        description="Timestamp of the most recent update, if any.",
    )

    model_config = ConfigDict(from_attributes=True)


# -----------------------
# Upload workflow schemas
# -----------------------


from pydantic import BaseModel, Field, ConfigDict

PreprocessCategory = Literal["dark", "bias", "flat"]


class TempUploadItem(BaseModel):
    temp_id: str = Field(
        description="Unique identifier for the staged upload item."
    )
    filename: str = Field(
        description="Original filename supplied by the client."
    )
    size_bytes: int = Field(
        description="Size of the uploaded file in bytes."
    )
    content_type: str | None = Field(
        default=None,
        description="Content type provided by the client during upload."
    )

    tmp_fits: str = Field(
        description="Temporary filesystem path where the staged FITS file is stored."
    )
    tmp_png: str | None = Field(
        default=None,
        description="Temporary filesystem path of the generated preview PNG image."
    )

    fits_header: dict | None = Field(
        default=None,
        description="Extracted FITS header information captured during staging."
    )

    metadata_json: dict | None = Field(
        default=None,
        description="Additional metadata describing the upload item (non-FITS data)."
    )

    model_config = ConfigDict(extra="forbid")


class TempPreprocessItem(BaseModel):
    temp_id: str = Field(description="Identifier for the staged preprocessing artifact.")
    category: PreprocessCategory = Field(description="Preprocessing category represented by the file.")
    filename: str = Field(description="Original filename captured during staging.")
    size_bytes: int = Field(description="Size of the uploaded file in bytes.")
    temp_path: str = Field(description="Temporary filesystem path for the staged preprocessing file.")
    tmp_png: str | None = Field(
        default=None,
        description="Optional preview image path for the staged preprocessing file.",
    )
    metadata_json: dict | None = Field(
        default=None,
        description="Additional metadata recorded for the preprocessing file.",
    )

    model_config = ConfigDict(extra="forbid")


class StageUploadsResponse(BaseModel):
    items: list[TempUploadItem] = Field(description="Staged science exposure files.")
    preprocess: dict[PreprocessCategory, list[TempPreprocessItem]] = Field(
        default_factory=dict,
        description="Staged preprocessing files grouped by category.",
    )

    model_config = ConfigDict(extra="forbid")


class UploadPreprocessCommitItem(BaseModel):
    temp_id: str = Field(description="Identifier of the staged preprocessing item.")
    category: PreprocessCategory = Field(description="Preprocessing category for the staged file.")
    temp_path: str = Field(description="Temporary path of the staged preprocessing file.")
    original_name: str | None = Field(
        default=None,
        description="Original filename to persist for the preprocessing file.",
    )
    metadata_json: dict | None = Field(
        default=None,
        description="Metadata supplied for the preprocessing file during commit.",
    )

    model_config = ConfigDict(extra="forbid")


class UploadCommitItem(BaseModel):
    temp_id: str = Field(description="Identifier of the staged item being committed.")
    fits_temp_path: str = Field(description="Temporary path of the staged FITS file.")
    image_temp_path: str | None = Field(
        default=None,
        description="Temporary path of the generated preview image.",
    )
    fits_data_json: dict | None = Field(
        default=None,
        description="FITS metadata supplied during commit.",
    )
    metadata_json: dict | None = Field(
        default=None,
        description="Arbitrary metadata supplied during commit.",
    )

    model_config = ConfigDict(extra="forbid")


class UploadCommitRequest(BaseModel):
    user_id: int = Field(description="Identifier of the user owning the new repository.")
    repository_name: str = Field(description="Name of the repository that will be created.")
    repository_description: str | None = Field(
        default=None,
        description="Optional description for the repository.",
    )
    captured_at: datetime | None = Field(
        default=None,
        description="Optional capture timestamp for the resulting dataset.",
    )
    items: list[UploadCommitItem] = Field(description="Staged upload items to commit.")
    preprocess_items: list[UploadPreprocessCommitItem] | None = Field(
        default=None,
        description="Staged preprocessing files to persist alongside the dataset.",
    )

    model_config = ConfigDict(extra="forbid")


# -----------------------
# Session & pipeline schemas
# -----------------------


class SessionRead(SessionSummary):
    repository_id: int | None = Field(
        default=None,
        description="Repository associated with the session.",
    )
    dataset_id: int | None = Field(
        default=None,
        description="Dataset associated with the session.",
    )
    data_id: int | None = Field(
        default=None,
        description="Data item processed by the session.",
    )
    data_version: int | None = Field(
        default=None,
        description="Version value tracked for the data item when the session ran.",
    )

    model_config = ConfigDict(from_attributes=True)


class PipelineStepRead(BaseModel):
    step_id: int = Field(description="Unique step identifier within the run.")
    run_id: UUID = Field(description="Run identifier that owns the pipeline step.")
    step_name: str = Field(description="Name of the pipeline step.")
    status: str = Field(description="Current status for the step (e.g. queued, running, done).")
    progress: int = Field(description="Completion percentage for the step.")
    data: dict | None = Field(
        default=None,
        description="Optional structured payload attached to the step.",
    )
    log: str | None = Field(
        default=None,
        description="Human-readable log output produced by the step.",
    )
    started_at: datetime | None = Field(
        default=None,
        description="Timestamp when the step started.",
    )
    finished_at: datetime | None = Field(
        default=None,
        description="Timestamp when the step finished.",
    )

    model_config = ConfigDict(from_attributes=True)


class CandidateRead(BaseModel):
    id: int = Field(description="Candidate identifier.")
    session_id: int = Field(description="Session that produced the candidate.")
    data: dict | None = Field(
        default=None,
        description="Structured detection payload captured during processing.",
    )
    is_verified: bool = Field(description="Manual verification flag for the candidate.")
    created_at: datetime = Field(description="Timestamp when the candidate was recorded.")

    model_config = ConfigDict(from_attributes=True)


class CandidateVerifyUpdate(BaseModel):
    is_verified: bool = Field(
        description="Whether the candidate has been verified by a reviewer.",
        examples=[True],
    )

    model_config = ConfigDict(extra="forbid")


class UploadCommitResponse(BaseModel):
    repository: RepositoryRead = Field(description="Repository created during commit.")
    dataset: DatasetRead = Field(description="Dataset that stores the committed data items.")
    data: list[DataRead] = Field(description="Committed science exposure items persisted to storage.")
    preprocess_data: list[DataRead] = Field(
        default_factory=list,
        description="Committed preprocessing files persisted to storage.",
    )
    sessions: list[SessionRead] = Field(description="Sessions spawned for each committed data item.")

    model_config = ConfigDict(extra="forbid")


# -----------------------
# Stats schemas
# -----------------------


class UserStatsRead(BaseModel):
    projects: int = Field(description="Number of projects the user participates in.")
    uploads: int = Field(description="Number of repositories owned by the user.")
    followers: int = Field(description="How many users follow this user.")
    following: int = Field(description="How many users the user follows.")


class ContributionBucket(BaseModel):
    date: Date = Field(description="Calendar date for the bucket entry.")
    count: int = Field(description="Number of contributions recorded for the day.")


class ContributionSkyPoint(BaseModel):
    ra: float = Field(description="Right ascension in degrees.")
    dec: float = Field(description="Declination in degrees.")
    repository_id: int = Field(description="Repository associated with the sky point.")


class ContributionRead(BaseModel):
    buckets: list[ContributionBucket] = Field(
        description="Time-series contribution buckets.",
        examples=[[{"date": "2024-01-01", "count": 3}]],
    )
    sky_points: list[ContributionSkyPoint] | None = Field(
        default=None,
        description="Optional set of sky coordinates representing contributions.",
    )


__all__ = [
    "CandidateRead",
    "CandidateVerifyUpdate",
    "ContributionBucket",
    "ContributionRead",
    "ContributionSkyPoint",
    "DataCreate",
    "DataRead",
    "DatasetCreate",
    "DatasetRead",
    "FollowCreate",
    "FollowRead",
    "FollowerCreate",
    "PinCreate",
    "PinRead",
    "PinReorder",
    "PipelineStepRead",
    "ProjectCreate",
    "ProjectMemberCreate",
    "ProjectMemberRead",
    "ProjectMemberUpdate",
    "ProjectRead",
    "ProjectRepositoryLinkCreate",
    "ProjectUpdate",
    "RepositoryCreate",
    "RepositoryRead",
    "RepositoryUpdate",
    "SessionRead",
    "SessionSummary",
    "StarRead",
    "StageUploadsResponse",
    "TempUploadItem",
    "TempPreprocessItem",
    "UploadCommitItem",
    "UploadCommitRequest",
    "UploadCommitResponse",
    "UploadPreprocessCommitItem",
    "UserCreate",
    "UserLogin",
    "UserProfileCreate",
    "UserProfileRead",
    "UserProfileUpdate",
    "UserRead",
    "UserStatsRead",
    "UserUpdate",
]
