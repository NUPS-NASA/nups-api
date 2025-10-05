"""Community discussion endpoints."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import Select, func, select
from sqlalchemy.orm import selectinload

from ... import models, schemas
from ..dependencies import CurrentUser, DBSession

router = APIRouter(tags=["community"])


async def _attach_project_metadata(projects: Sequence[models.Project], db: DBSession) -> None:
    if not projects:
        return

    project_ids = [project.id for project in projects]
    counts_result = await db.execute(
        select(
            models.ProjectUser.project_id,
            func.count(models.ProjectUser.user_id).label("member_count"),
        )
        .where(models.ProjectUser.project_id.in_(project_ids))
        .group_by(models.ProjectUser.project_id)
    )
    counts_map = {row.project_id: row.member_count for row in counts_result}

    for project in projects:
        setattr(project, "members_count", counts_map.get(project.id, 0))
        setattr(project, "tags", [])


async def _attach_post_metadata(
    posts: Sequence[models.CommunityPost],
    current_user: models.User,
    db: DBSession,
) -> None:
    if not posts:
        return

    post_ids = [post.id for post in posts]

    likes_result = await db.execute(
        select(
            models.CommunityPostLike.post_id,
            func.count(models.CommunityPostLike.user_id).label("likes"),
        )
        .where(models.CommunityPostLike.post_id.in_(post_ids))
        .group_by(models.CommunityPostLike.post_id)
    )
    likes_map = {row.post_id: row.likes for row in likes_result}

    liked_result = await db.scalars(
        select(models.CommunityPostLike.post_id).where(
            models.CommunityPostLike.post_id.in_(post_ids),
            models.CommunityPostLike.user_id == current_user.id,
        )
    )
    liked_ids = set(liked_result.all())

    for post in posts:
        setattr(post, "likes_count", likes_map.get(post.id, 0))
        setattr(post, "liked", post.id in liked_ids)
        setattr(post, "can_delete", post.author_id == current_user.id)

    projects = [post.linked_project for post in posts if post.linked_project is not None]
    await _attach_project_metadata(projects, db)


def _base_post_query() -> Select[tuple[models.CommunityPost]]:
    return (
        select(models.CommunityPost)
        .options(
            selectinload(models.CommunityPost.author).selectinload(models.User.profile),
            selectinload(models.CommunityPost.comments)
            .selectinload(models.CommunityComment.author)
            .selectinload(models.User.profile),
            selectinload(models.CommunityPost.linked_project),
        )
        .order_by(models.CommunityPost.created_at.desc())
    )


async def _get_post_or_404(post_id: int, db: DBSession) -> models.CommunityPost:
    result = await db.scalars(_base_post_query().where(models.CommunityPost.id == post_id))
    post = result.unique().one_or_none()
    if post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    return post


@router.get(
    "/community/posts",
    response_model=list[schemas.CommunityPostRead],
    summary="List community posts",
)
async def list_community_posts(
    db: DBSession,
    current_user: CurrentUser,
    category: str | None = Query(
        None,
        description="Restrict results to a category (announcements, project-showcase, astrophoto-gallery, upload-hall-of-fame).",
    ),
) -> list[schemas.CommunityPostRead]:
    normalized_category = None if category in {None, "all"} else category
    if normalized_category and normalized_category not in schemas.COMMUNITY_POST_CATEGORIES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid category")

    query = _base_post_query()
    if normalized_category:
        query = query.where(models.CommunityPost.category == normalized_category)

    result = await db.scalars(query)
    posts = result.unique().all()

    await _attach_post_metadata(posts, current_user, db)
    return posts


@router.post(
    "/community/posts",
    response_model=schemas.CommunityPostRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a community post",
)
async def create_community_post(
    payload: schemas.CommunityPostCreate,
    db: DBSession,
    current_user: CurrentUser,
) -> schemas.CommunityPostRead:
    if payload.category not in schemas.COMMUNITY_POST_CATEGORIES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid category")

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Title cannot be empty")

    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Content cannot be empty")

    linked_project = None
    if payload.linked_project_id is not None:
        linked_project = await db.get(models.Project, payload.linked_project_id)
        if linked_project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Linked project not found")

    post = models.CommunityPost(
        author_id=current_user.id,
        title=title,
        content=content,
        category=payload.category,
        linked_project_id=payload.linked_project_id,
    )
    db.add(post)
    await db.commit()
    await db.refresh(post)

    await db.refresh(current_user, attribute_names=["profile"])
    post.author = current_user

    if linked_project is not None:
        await db.refresh(post, attribute_names=["linked_project"])

    await _attach_post_metadata([post], current_user, db)
    return post


@router.post(
    "/community/posts/{post_id}/comments",
    response_model=schemas.CommunityCommentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a comment to a community post",
)
async def create_community_comment(
    post_id: int,
    payload: schemas.CommunityCommentCreate,
    db: DBSession,
    current_user: CurrentUser,
) -> schemas.CommunityCommentRead:
    _ = await _get_post_or_404(post_id, db)

    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Comment cannot be empty")

    comment = models.CommunityComment(
        post_id=post_id,
        author_id=current_user.id,
        content=content,
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)
    await db.refresh(current_user, attribute_names=["profile"])
    comment.author = current_user

    return comment


async def _like_status_response(
    post: models.CommunityPost,
    current_user: models.User,
    db: DBSession,
) -> schemas.CommunityPostLikeStatus:
    await _attach_post_metadata([post], current_user, db)
    return schemas.CommunityPostLikeStatus(
        post_id=post.id,
        liked=getattr(post, "liked", False),
        likes_count=getattr(post, "likes_count", 0),
    )


@router.post(
    "/community/posts/{post_id}/likes",
    response_model=schemas.CommunityPostLikeStatus,
    status_code=status.HTTP_201_CREATED,
    summary="Like a community post",
)
async def like_community_post(
    post_id: int,
    db: DBSession,
    current_user: CurrentUser,
) -> schemas.CommunityPostLikeStatus:
    post = await _get_post_or_404(post_id, db)

    existing = await db.scalar(
        select(models.CommunityPostLike).where(
            models.CommunityPostLike.post_id == post_id,
            models.CommunityPostLike.user_id == current_user.id,
        )
    )
    if existing is None:
        db.add(models.CommunityPostLike(post_id=post_id, user_id=current_user.id))
        await db.commit()
        post = await _get_post_or_404(post_id, db)

    return await _like_status_response(post, current_user, db)


@router.delete(
    "/community/posts/{post_id}/likes",
    response_model=schemas.CommunityPostLikeStatus,
    summary="Remove a like from a community post",
)
async def unlike_community_post(
    post_id: int,
    db: DBSession,
    current_user: CurrentUser,
) -> schemas.CommunityPostLikeStatus:
    post = await _get_post_or_404(post_id, db)

    existing = await db.scalar(
        select(models.CommunityPostLike).where(
            models.CommunityPostLike.post_id == post_id,
            models.CommunityPostLike.user_id == current_user.id,
        )
    )
    if existing is None:
        return await _like_status_response(post, current_user, db)

    await db.delete(existing)
    await db.commit()
    post = await _get_post_or_404(post_id, db)
    return await _like_status_response(post, current_user, db)


@router.delete(
    "/community/posts/{post_id}/comments/{comment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a community comment",
)
async def delete_community_comment(
    post_id: int,
    comment_id: int,
    db: DBSession,
    current_user: CurrentUser,
) -> Response:
    _ = await _get_post_or_404(post_id, db)

    comment = await db.scalar(
        select(models.CommunityComment).where(
            models.CommunityComment.id == comment_id,
            models.CommunityComment.post_id == post_id,
        )
    )
    if comment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found")

    if comment.author_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not permitted to delete this comment")

    await db.delete(comment)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/community/posts/{post_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a community post",
)
async def delete_community_post(
    post_id: int,
    db: DBSession,
    current_user: CurrentUser,
) -> Response:
    post = await _get_post_or_404(post_id, db)

    if post.author_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not permitted to delete this post")

    await db.delete(post)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
