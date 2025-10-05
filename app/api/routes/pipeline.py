"""Read-only endpoints exposing the canonical analysis pipeline definition."""

from fastapi import APIRouter

from ... import schemas
from ...core import get_default_pipeline

router = APIRouter(tags=["pipeline"])


@router.get(
    "/pipeline",
    response_model=schemas.PipelineDefinitionRead,
    summary="Get canonical pipeline definition",
)
async def get_pipeline_definition() -> schemas.PipelineDefinitionRead:
    """Return the shared pipeline metadata including ordered steps."""

    steps = [
        schemas.PipelineDefinitionStep(
            order=index,
            name=step.name,
            title=step.title,
            description=step.description,
        )
        for index, step in enumerate(get_default_pipeline(), start=1)
    ]
    return schemas.PipelineDefinitionRead(steps=steps, total_steps=len(steps))


__all__ = ["router"]
