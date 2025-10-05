"""Session, pipeline step, and candidate endpoints."""

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select

from ... import models, schemas
from ..dependencies import DBSession

router = APIRouter(tags=["sessions"])


async def _get_repository_or_404(repository_id: int, db: DBSession) -> models.Repository:
    repository = await db.get(models.Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
    return repository


async def _get_session_or_404(session_id: int, db: DBSession) -> models.Session:
    session = await db.get(models.Session, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


@router.get(
    "/repositories/{repository_id}/sessions",
    response_model=list[schemas.SessionRead],
    summary="List sessions for a repository",
)
async def list_repo_sessions(repository_id: int, db: DBSession) -> list[schemas.SessionRead]:
    await _get_repository_or_404(repository_id, db)

    result = await db.scalars(
        select(models.Session)
        .where(models.Session.repository_id == repository_id)
        .order_by(models.Session.id.desc())
    )
    return result.all()


@router.get(
    "/repositories/{repository_id}/sessions/latest",
    response_model=schemas.SessionRead,
    summary="Get latest session for a repository",
)
async def get_latest_repo_session(repository_id: int, db: DBSession) -> schemas.SessionRead:
    await _get_repository_or_404(repository_id, db)

    session = await db.scalar(
        select(models.Session)
        .where(models.Session.repository_id == repository_id)
        .order_by(models.Session.id.desc())
        .limit(1)
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No session found")
    return session


@router.get(
    "/sessions/{session_id}",
    response_model=schemas.SessionRead,
    summary="Get session",
)
async def get_session(session_id: int, db: DBSession) -> schemas.SessionRead:
    return await _get_session_or_404(session_id, db)


@router.get(
    "/sessions/{session_id}/pipeline-steps",
    response_model=list[schemas.PipelineStepRead],
    summary="List pipeline steps of a session",
)
async def list_pipeline_steps(session_id: int, db: DBSession) -> list[schemas.PipelineStepRead]:
    session = await _get_session_or_404(session_id, db)

    result = await db.scalars(
        select(models.PipelineStep)
        .where(models.PipelineStep.run_id == session.run_id)
        .order_by(models.PipelineStep.step_id)
    )
    return result.all()


@router.get(
    "/sessions/{session_id}/candidates",
    response_model=list[schemas.CandidateRead],
    tags=["candidates"],
    summary="List candidates for a session",
)
async def list_candidates(session_id: int, db: DBSession) -> list[schemas.CandidateRead]:
    await _get_session_or_404(session_id, db)

    result = await db.scalars(
        select(models.Candidate)
        .where(models.Candidate.session_id == session_id)
        .order_by(models.Candidate.created_at.desc())
    )
    return result.all()


@router.patch(
    "/candidates/{candidate_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["candidates"],
    summary="Verify/unverify a candidate",
)
async def verify_candidate(
    candidate_id: int, payload: schemas.CandidateVerifyUpdate, db: DBSession
) -> Response:
    candidate = await db.get(models.Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")

    candidate.is_verified = payload.is_verified
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
