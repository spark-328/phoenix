"""Moving an LLM evaluator's prompt version tag from the prompt side gets the evaluator's checks."""

from datetime import datetime
from secrets import token_hex
from typing import Any, AsyncIterator, Optional, Sequence
from urllib.parse import quote_plus

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy import select
from strawberry.relay import GlobalID

from phoenix.db import models
from phoenix.db.types.annotation_configs import (
    CategoricalAnnotationValue,
    CategoricalOutputConfig,
    OptimizationDirection,
)
from phoenix.db.types.evaluators import InputMapping
from phoenix.db.types.identifier import Identifier
from phoenix.db.types.model_provider import ModelProvider
from phoenix.db.types.prompts import (
    PromptChatTemplate,
    PromptMessage,
    PromptOpenAIInvocationParameters,
    PromptOpenAIInvocationParametersContent,
    PromptTemplateFormat,
    PromptTemplateType,
    PromptToolChoiceOneOrMore,
    PromptToolFunction,
    PromptToolFunctionDefinition,
    PromptTools,
    TextContentPart,
)
from phoenix.server.types import DbSessionFactory
from tests.unit.graphql import AsyncGraphQLClient

_EVALUATOR_LABELS = ("correct", "incorrect")


def _prompt_version(text: str, labels: Sequence[str] = _EVALUATOR_LABELS) -> models.PromptVersion:
    return models.PromptVersion(
        template_type=PromptTemplateType.CHAT,
        template_format=PromptTemplateFormat.MUSTACHE,
        template=PromptChatTemplate(
            type="chat",
            messages=[
                PromptMessage(role="user", content=[TextContentPart(type="text", text=text)])
            ],
        ),
        invocation_parameters=PromptOpenAIInvocationParameters(
            type="openai", openai=PromptOpenAIInvocationParametersContent()
        ),
        tools=PromptTools(
            type="tools",
            tools=[
                PromptToolFunction(
                    type="function",
                    function=PromptToolFunctionDefinition(
                        name="correctness",
                        description="correctness",
                        parameters={
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "enum": list(labels),
                                    "description": "correctness",
                                }
                            },
                            "required": ["label"],
                        },
                    ),
                )
            ],
            tool_choice=PromptToolChoiceOneOrMore(type="one_or_more"),
        ),
        response_format=None,
        model_provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        metadata_={},
    )


def _output_config(labels: Sequence[str]) -> CategoricalOutputConfig:
    return CategoricalOutputConfig(
        type="CATEGORICAL",
        name="correctness",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        values=[
            CategoricalAnnotationValue(label=label, score=float(i == 0))
            for i, label in enumerate(labels)
        ],
    )


class _Fixture:
    def __init__(self, evaluator: models.LLMEvaluator, prompt: models.Prompt) -> None:
        self.evaluator = evaluator
        self.prompt = prompt
        self.pinned_version, self.compatible_version, self.incompatible_version = (
            prompt.prompt_versions
        )
        tag = evaluator.prompt_version_tag
        assert tag is not None
        self.tag = tag
        self.tag_name = tag.name.root
        self.evaluator_gid = str(GlobalID("LLMEvaluator", str(evaluator.id)))
        self.updated_at = evaluator.updated_at


@pytest.fixture
async def pinned(db: DbSessionFactory) -> AsyncIterator[_Fixture]:
    """An evaluator whose tag pins the first of three versions.

    The second version keeps the evaluator's label set; the third uses different labels, so
    the evaluator cannot run it.
    """
    prompt = models.Prompt(
        name=Identifier.model_validate(f"tag-move-prompt-{token_hex(4)}"),
        description="tag move",
        prompt_versions=[
            _prompt_version("Judge {{output}}"),
            _prompt_version("Grade {{output}}"),
            _prompt_version("Rate {{output}}", labels=("good", "bad")),
        ],
    )
    evaluator = models.LLMEvaluator(
        name=Identifier.model_validate(f"tag-move-evaluator-{token_hex(4)}"),
        description="correctness",
        kind="LLM",
        output_configs=[_output_config(_EVALUATOR_LABELS)],
        prompt=prompt,
    )
    async with db() as session:
        session.add(evaluator)
        await session.flush()
        evaluator.prompt_version_tag = models.PromptVersionTag(
            name=Identifier.model_validate(f"{evaluator.name.root}-evaluator-{token_hex(4)}"),
            prompt_id=prompt.id,
            prompt_version_id=prompt.prompt_versions[0].id,
        )
        session.add(evaluator)
        await session.flush()
        await session.refresh(evaluator, attribute_names=["updated_at"])
    yield _Fixture(evaluator, prompt)
    async with db() as session:
        await session.execute(
            sa.delete(models.LLMEvaluator).where(models.LLMEvaluator.id == evaluator.id)
        )
        await session.execute(sa.delete(models.Prompt).where(models.Prompt.id == prompt.id))


@pytest.fixture
async def dataset_override(db: DbSessionFactory, pinned: _Fixture) -> AsyncIterator[str]:
    """A dataset binding whose output override adds a label the evaluator's versions lack."""
    binding = models.DatasetEvaluators(
        dataset=models.Dataset(name=f"tag-move-dataset-{token_hex(4)}", metadata_={}),
        evaluator_id=pinned.evaluator.id,
        name=pinned.evaluator.name,
        input_mapping=InputMapping(literal_mapping={}, path_mapping={"output": "$.output"}),
        output_configs=[_output_config(("correct", "incorrect", "unsure"))],
        project=models.Project(name=f"tag-move-project-{token_hex(4)}"),
    )
    async with db() as session:
        session.add(binding)
    yield str(GlobalID("DatasetEvaluator", str(binding.id)))
    async with db() as session:
        await session.execute(
            sa.delete(models.DatasetEvaluators).where(models.DatasetEvaluators.id == binding.id)
        )
        await session.execute(
            sa.delete(models.Dataset).where(models.Dataset.id == binding.dataset_id)
        )
        await session.execute(
            sa.delete(models.Project).where(models.Project.id == binding.project_id)
        )


class _EvaluatorState:
    def __init__(self, tag_id: Optional[int], tag_target: Optional[int], updated_at: datetime):
        self.tag_id = tag_id
        self.tag_target = tag_target
        self.updated_at = updated_at


async def _state(db: DbSessionFactory, evaluator_id: int) -> _EvaluatorState:
    async with db() as session:
        evaluator = await session.get(models.LLMEvaluator, evaluator_id)
        assert evaluator is not None
        target = (
            await session.scalar(
                select(models.PromptVersionTag.prompt_version_id).where(
                    models.PromptVersionTag.id == evaluator.prompt_version_tag_id
                )
            )
            if evaluator.prompt_version_tag_id is not None
            else None
        )
    return _EvaluatorState(evaluator.prompt_version_tag_id, target, evaluator.updated_at)


async def _version_count(db: DbSessionFactory, prompt_id: int) -> int:
    async with db() as session:
        count = await session.scalar(
            select(sa.func.count(models.PromptVersion.id)).where(
                models.PromptVersion.prompt_id == prompt_id
            )
        )
    assert count is not None
    return count


def _gid(version: models.PromptVersion) -> str:
    return str(GlobalID("PromptVersion", str(version.id)))


def _path(version: models.PromptVersion) -> str:
    return quote_plus(_gid(version))


class TestRestRoutes:
    async def test_tag_moves_to_a_version_the_evaluator_can_run(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory, pinned: _Fixture
    ) -> None:
        response = await httpx_client.post(
            f"v1/prompt_versions/{_path(pinned.compatible_version)}/tags",
            json={"name": pinned.tag_name},
        )
        assert response.status_code == 204, response.text
        state = await _state(db, pinned.evaluator.id)
        assert state.tag_target == pinned.compatible_version.id
        assert state.updated_at > pinned.updated_at

    async def test_tag_does_not_move_to_a_version_the_evaluator_cannot_run(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory, pinned: _Fixture
    ) -> None:
        response = await httpx_client.post(
            f"v1/prompt_versions/{_path(pinned.incompatible_version)}/tags",
            json={"name": pinned.tag_name},
        )
        assert response.status_code == 409, response.text
        assert pinned.evaluator_gid in response.text
        state = await _state(db, pinned.evaluator.id)
        assert state.tag_target == pinned.pinned_version.id
        assert state.updated_at == pinned.updated_at

    async def test_tag_does_not_move_when_a_dataset_override_cannot_follow(
        self,
        httpx_client: httpx.AsyncClient,
        db: DbSessionFactory,
        pinned: _Fixture,
        dataset_override: str,
    ) -> None:
        response = await httpx_client.post(
            f"v1/prompt_versions/{_path(pinned.compatible_version)}/tags",
            json={"name": pinned.tag_name},
        )
        assert response.status_code == 409, response.text
        assert dataset_override in response.text
        assert (await _state(db, pinned.evaluator.id)).tag_target == pinned.pinned_version.id

    async def test_other_tags_move_freely(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory, pinned: _Fixture
    ) -> None:
        response = await httpx_client.post(
            f"v1/prompt_versions/{_path(pinned.incompatible_version)}/tags",
            json={"name": "staging"},
        )
        assert response.status_code == 204, response.text
        assert (await _state(db, pinned.evaluator.id)).tag_target == pinned.pinned_version.id

    async def test_deleting_the_tag_unpins_the_evaluator(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory, pinned: _Fixture
    ) -> None:
        response = await httpx_client.delete(
            f"v1/prompt_versions/{_path(pinned.pinned_version)}/tags/{quote_plus(pinned.tag_name)}"
        )
        assert response.status_code == 204, response.text
        assert (await _state(db, pinned.evaluator.id)).tag_id is None

    async def test_creating_a_version_carries_the_tag_only_if_the_evaluator_can_run_it(
        self, httpx_client: httpx.AsyncClient, db: DbSessionFactory, pinned: _Fixture
    ) -> None:
        content: dict[str, Any] = (
            await httpx_client.get(f"v1/prompt_versions/{_path(pinned.pinned_version)}")
        ).json()["data"]
        content.pop("id")
        prompt_path = quote_plus(str(GlobalID("Prompt", str(pinned.prompt.id))))
        tags = [{"name": pinned.tag_name}]

        response = await httpx_client.post(
            f"v1/prompts/{prompt_path}/versions", json={"version": content, "tags": tags}
        )
        assert response.status_code == 201, response.text
        created_id = GlobalID.from_id(response.json()["data"]["id"]).node_id
        assert (await _state(db, pinned.evaluator.id)).tag_target == int(created_id)

        content["tools"]["tools"][0]["function"]["parameters"]["properties"]["label"]["enum"] = [
            "good",
            "bad",
        ]
        response = await httpx_client.post(
            f"v1/prompts/{prompt_path}/versions", json={"version": content, "tags": tags}
        )
        assert response.status_code == 409, response.text
        assert pinned.evaluator_gid in response.text
        assert await _version_count(db, pinned.prompt.id) == 4
        assert (await _state(db, pinned.evaluator.id)).tag_target == int(created_id)


class TestGraphQLMutations:
    _SET_TAG = """
        mutation($input: SetPromptVersionTagInput!) {
          setPromptVersionTag(input: $input) { promptVersionTag { id } }
        }
    """

    async def test_set_tag_moves_only_to_a_version_the_evaluator_can_run(
        self, gql_client: AsyncGraphQLClient, db: DbSessionFactory, pinned: _Fixture
    ) -> None:
        result = await gql_client.execute(
            self._SET_TAG,
            {
                "input": {
                    "promptVersionId": _gid(pinned.incompatible_version),
                    "name": pinned.tag_name,
                }
            },
        )
        assert result.errors and "cannot run the target version" in result.errors[0].message
        assert pinned.evaluator_gid in result.errors[0].message
        assert (await _state(db, pinned.evaluator.id)).tag_target == pinned.pinned_version.id

        result = await gql_client.execute(
            self._SET_TAG,
            {
                "input": {
                    "promptVersionId": _gid(pinned.compatible_version),
                    "name": pinned.tag_name,
                }
            },
        )
        assert not result.errors, result.errors
        assert (await _state(db, pinned.evaluator.id)).tag_target == pinned.compatible_version.id

    async def test_delete_tag_unpins_the_evaluator(
        self, gql_client: AsyncGraphQLClient, db: DbSessionFactory, pinned: _Fixture
    ) -> None:
        result = await gql_client.execute(
            """
            mutation($input: DeletePromptVersionTagInput!) {
              deletePromptVersionTag(input: $input) { prompt { id } }
            }
            """,
            {
                "input": {
                    "promptVersionTagId": str(GlobalID("PromptVersionTag", str(pinned.tag.id)))
                }
            },
        )
        assert not result.errors, result.errors
        assert (await _state(db, pinned.evaluator.id)).tag_id is None

    async def test_creating_a_version_the_evaluator_cannot_run_does_not_carry_the_tag(
        self, gql_client: AsyncGraphQLClient, db: DbSessionFactory, pinned: _Fixture
    ) -> None:
        result = await gql_client.execute(
            """
            mutation($input: CreateChatPromptVersionInput!) {
              createChatPromptVersion(input: $input) { id }
            }
            """,
            {
                "input": {
                    "promptId": str(GlobalID("Prompt", str(pinned.prompt.id))),
                    "tags": [{"name": pinned.tag_name}],
                    "promptVersion": {
                        "templateFormat": "MUSTACHE",
                        "template": {
                            "messages": [
                                {"role": "USER", "content": [{"text": {"text": "Rate {{output}}"}}]}
                            ]
                        },
                        "modelProvider": "OPENAI",
                        "modelName": "gpt-4o-mini",
                        "invocationParameters": {"openai": {}},
                    },
                }
            },
        )
        assert result.errors and "cannot run the target version" in result.errors[0].message
        assert await _version_count(db, pinned.prompt.id) == 3
        assert (await _state(db, pinned.evaluator.id)).tag_target == pinned.pinned_version.id
