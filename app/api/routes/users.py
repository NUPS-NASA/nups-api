"""User resource endpoints."""

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import Select, select
from sqlalchemy.orm import selectinload

from ... import models, schemas
from ..dependencies import DBSession

router = APIRouter(prefix="/users", tags=["users"])


def _user_with_profile_query() -> Select:
    """Base SELECT statement that eagerly loads a user's profile."""

    return select(models.User).options(selectinload(models.User.profile))


async def _get_user_or_404(user_id: int, db: DBSession) -> models.User:
    """Fetch a user by identifier or raise a 404 error."""

    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.post(
    "",
    response_model=schemas.UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user",
    response_description="The newly created user including profile data, when present.",
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "Email already exists.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Validation error when the payload is malformed.",
        },
    },
)
async def create_user(payload: schemas.UserCreate, db: DBSession) -> schemas.UserRead:
    """Persist a new user with optional profile data."""

    existing = await db.scalar(
        select(models.User).where(models.User.email == payload.email)
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = models.User(email=payload.email, password_hash=payload.password_hash)
    if payload.profile:
        user.profile = models.UserProfile(**payload.profile.model_dump())

    db.add(user)
    await db.commit()

    created_user = await db.scalar(
        _user_with_profile_query().where(models.User.id == user.id)
    )
    if created_user is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User creation failed")
    return created_user


@router.get(
    "",
    response_model=list[schemas.UserRead],
    summary="List users",
    response_description="All users ordered by identifier.",
)
async def list_users(db: DBSession) -> list[schemas.UserRead]:
    """Return every registered user with eager-loaded profile data."""

    result = await db.scalars(
        _user_with_profile_query().order_by(models.User.id)
    )
    return list(result)


@router.get(
    "/{user_id}",
    response_model=schemas.UserRead,
    summary="Retrieve a user",
    response_description="The user resource with profile data when available.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "User not found.",
        }
    },
)
async def get_user(user_id: int, db: DBSession) -> schemas.UserRead:
    """Retrieve a user by identifier."""

    user = await db.scalar(
        _user_with_profile_query().where(models.User.id == user_id)
    )
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.put(
    "/{user_id}",
    response_model=schemas.UserRead,
    summary="Replace a user's attributes",
    response_description="The updated user resource.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "User not found.",
        }
    },
)
async def update_user(user_id: int, payload: schemas.UserUpdate, db: DBSession) -> schemas.UserRead:
    """Update core user fields and optionally their profile."""

    user = await db.scalar(
        _user_with_profile_query().where(models.User.id == user_id)
    )
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if payload.email is not None:
        user.email = payload.email
    if payload.password_hash is not None:
        user.password_hash = payload.password_hash

    if payload.profile is not None:
        profile_updates = payload.profile.model_dump(exclude_unset=True)
        if user.profile is None:
            if profile_updates:
                user.profile = models.UserProfile(user_id=user_id, **profile_updates)
        else:
            for key, value in profile_updates.items():
                setattr(user.profile, key, value)

    await db.commit()
    await db.refresh(user)
    return user


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a user",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "User not found.",
        }
    },
)
async def delete_user(user_id: int, db: DBSession) -> Response:
    """Remove a user and their profile."""

    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.delete(user)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{user_id}/profile",
    response_model=schemas.UserProfileRead,
    summary="Retrieve a user's profile",
    response_description="The profile resource for the requested user.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "User or profile missing.",
        }
    },
)
async def get_user_profile(user_id: int, db: DBSession) -> schemas.UserProfileRead:
    """Return a user's profile, ensuring both user and profile exist."""

    user = await db.scalar(
        _user_with_profile_query().where(models.User.id == user_id)
    )
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return user.profile


@router.put(
    "/{user_id}/profile",
    response_model=schemas.UserProfileRead,
    summary="Upsert a user's profile",
    response_description="The updated profile resource.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "User not found.",
        }
    },
)
async def update_user_profile(
    user_id: int, payload: schemas.UserProfileUpdate, db: DBSession
) -> schemas.UserProfileRead:
    """Create or update a user's profile and return the current representation."""

    user = await db.scalar(
        _user_with_profile_query().where(models.User.id == user_id)
    )
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    profile_updates = payload.model_dump(exclude_unset=True)

    if user.profile is None:
        user.profile = models.UserProfile(user_id=user_id, **profile_updates)
    else:
        for key, value in profile_updates.items():
            setattr(user.profile, key, value)

    await db.commit()
    await db.refresh(user, attribute_names=["profile"])
    if user.profile is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Profile update failed")
    return user.profile


@router.post(
    "/{user_id}/following",
    response_model=schemas.FollowRead,
    status_code=status.HTTP_201_CREATED,
    summary="Follow another user",
    response_description="The follow relationship that was created.",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Users cannot follow themselves.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "User or target user not found.",
        },
        status.HTTP_409_CONFLICT: {
            "description": "Follow relationship already exists.",
        },
    },
)
async def follow_user(
    user_id: int, payload: schemas.FollowCreate, db: DBSession
) -> schemas.FollowRead:
    """Create a follow relationship where the path user follows another user."""

    if user_id == payload.following_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Users cannot follow themselves")

    await _get_user_or_404(user_id, db)
    target_user = await db.get(models.User, payload.following_id)
    if target_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    existing = await db.scalar(
        select(models.Follow).where(
            models.Follow.follower_id == user_id,
            models.Follow.following_id == payload.following_id,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Follow relationship already exists")

    follow = models.Follow(
        follower_id=user_id,
        following_id=payload.following_id,
    )
    db.add(follow)
    await db.commit()
    await db.refresh(follow)
    return follow


@router.delete(
    "/{user_id}/following/{following_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unfollow a user",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Follow relationship not found.",
        }
    },
)
async def unfollow_user(user_id: int, following_id: int, db: DBSession) -> Response:
    """Delete the follow relationship between two users."""

    follow = await db.scalar(
        select(models.Follow).where(
            models.Follow.follower_id == user_id,
            models.Follow.following_id == following_id,
        )
    )
    if follow is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Follow relationship not found")

    await db.delete(follow)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{user_id}/followers",
    response_model=schemas.FollowRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a follower",
    response_description="The follow relationship that was created.",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Users cannot follow themselves.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "User or follower not found.",
        },
        status.HTTP_409_CONFLICT: {
            "description": "Follow relationship already exists.",
        },
    },
)
async def add_follower(
    user_id: int, payload: schemas.FollowerCreate, db: DBSession
) -> schemas.FollowRead:
    """Create a follow relationship where another user follows the path user."""

    if user_id == payload.follower_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Users cannot follow themselves")

    await _get_user_or_404(user_id, db)
    follower_user = await db.get(models.User, payload.follower_id)
    if follower_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Follower user not found")

    existing = await db.scalar(
        select(models.Follow).where(
            models.Follow.follower_id == payload.follower_id,
            models.Follow.following_id == user_id,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Follow relationship already exists")

    follow = models.Follow(
        follower_id=payload.follower_id,
        following_id=user_id,
    )
    db.add(follow)
    await db.commit()
    await db.refresh(follow)
    return follow


@router.delete(
    "/{user_id}/followers/{follower_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a follower",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Follow relationship not found.",
        }
    },
)
async def remove_follower(user_id: int, follower_id: int, db: DBSession) -> Response:
    """Delete the follow relationship where the path user is followed by another user."""

    follow = await db.scalar(
        select(models.Follow).where(
            models.Follow.follower_id == follower_id,
            models.Follow.following_id == user_id,
        )
    )
    if follow is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Follow relationship not found")

    await db.delete(follow)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{user_id}/followers",
    response_model=list[schemas.UserRead],
    summary="List followers",
    response_description="Users that follow the requested account.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "User not found.",
        }
    },
)
async def list_followers(user_id: int, db: DBSession) -> list[schemas.UserRead]:
    """Return users that follow the specified account."""

    await _get_user_or_404(user_id, db)

    query = (
        _user_with_profile_query()
        .join(models.Follow, models.User.id == models.Follow.follower_id)
        .where(models.Follow.following_id == user_id)
        .order_by(models.User.id)
    )
    result = await db.scalars(query)
    return list(result)


@router.get(
    "/{user_id}/following",
    response_model=list[schemas.UserRead],
    summary="List following",
    response_description="Users that the requested account is following.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "User not found.",
        }
    },
)
async def list_following(user_id: int, db: DBSession) -> list[schemas.UserRead]:
    """Return users that the specified account is following."""

    await _get_user_or_404(user_id, db)

    query = (
        _user_with_profile_query()
        .join(models.Follow, models.User.id == models.Follow.following_id)
        .where(models.Follow.follower_id == user_id)
        .order_by(models.User.id)
    )
    result = await db.scalars(query)
    return list(result)


__all__ = ["router"]
