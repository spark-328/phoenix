"""Prompt version tags through which LLM evaluators record the version they run."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from strawberry.relay import GlobalID

from phoenix.db import models
from phoenix.db.helpers import llm_evaluators_pinned_by_prompt_version_tag
from phoenix.db.types.identifier import Identifier
from phoenix.server.api.exceptions import Conflict, NotFound
from phoenix.server.api.helpers.evaluators import (
    incompatible_dataset_override_ids,
    validate_consistent_llm_evaluator_and_prompt_version,
)


async def validate_prompt_version_tag_move(
    session: AsyncSession,
    tag: models.PromptVersionTag,
    prompt_version_id: int,
) -> None:
    """Let a tag that an LLM evaluator runs through move only to a version the evaluator can run.

    Moving the tag from the prompt side changes what the evaluator runs, so it gets the same
    checks as changing the version through the evaluator itself: the version's tool schema must
    match the evaluator's outputs, and every dataset binding override must still fit. Raises
    Conflict naming the evaluator. Tags no evaluator uses move freely.
    """
    evaluators = await llm_evaluators_pinned_by_prompt_version_tag(session, tag.id)
    if not evaluators:
        return
    prompt_version = await session.get(models.PromptVersion, prompt_version_id)
    if prompt_version is None:
        raise NotFound(f"Prompt version not found: {prompt_version_id}")
    now = datetime.now(timezone.utc)
    for evaluator in evaluators:
        evaluator_id = GlobalID("LLMEvaluator", str(evaluator.id))
        try:
            validate_consistent_llm_evaluator_and_prompt_version(prompt_version, evaluator)
        except ValueError as error:
            raise Conflict(
                f"Tag '{tag.name.root}' records the prompt version of evaluator {evaluator_id}, "
                f"which cannot run the target version: {error}"
            ) from error
        if incompatible := await incompatible_dataset_override_ids(
            session, evaluator, prompt_version
        ):
            raise Conflict(
                f"Tag '{tag.name.root}' records the prompt version of evaluator {evaluator_id}; "
                "dataset evaluator bindings override outputs that the target version does not "
                f"support: {', '.join(incompatible)}"
            )
        evaluator.updated_at = now


async def upsert_prompt_version_tag(
    session: AsyncSession,
    prompt_id: int,
    prompt_version_id: int,
    name: Identifier,
    description: Optional[str] = None,
    user_id: Optional[int] = None,
) -> models.PromptVersionTag:
    """Create or retarget a tag within the caller's transaction.

    A tag that an LLM evaluator runs through only moves to a version the evaluator can run;
    otherwise Conflict is raised and nothing changes.
    """
    existing_tag = await session.scalar(
        select(models.PromptVersionTag).where(
            models.PromptVersionTag.prompt_id == prompt_id,
            models.PromptVersionTag.name == name,
        )
    )

    if existing_tag:
        if existing_tag.prompt_version_id != prompt_version_id:
            await validate_prompt_version_tag_move(session, existing_tag, prompt_version_id)
        existing_tag.prompt_version_id = prompt_version_id
        if description is not None:
            existing_tag.description = description
        return existing_tag
    new_tag = models.PromptVersionTag(
        name=name,
        description=description,
        prompt_id=prompt_id,
        prompt_version_id=prompt_version_id,
        user_id=user_id,
    )
    session.add(new_tag)
    return new_tag
