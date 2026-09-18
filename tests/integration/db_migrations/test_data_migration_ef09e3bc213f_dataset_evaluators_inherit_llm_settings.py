import json
from datetime import datetime, timezone
from typing import Any, Literal, NamedTuple, Optional

from alembic.config import Config
from sqlalchemy import Connection, text
from sqlalchemy.ext.asyncio import AsyncEngine

from phoenix.db.types.annotation_configs import OutputConfig

from . import _down, _get_table_schema_info, _run_async, _TableSchemaInfo, _up

_DOWN = "a7f1c3e9d2b4"
_UP = "ef09e3bc213f"

_INPUT_MAPPING: dict[str, dict[str, str]] = {"literal_mapping": {}, "path_mapping": {}}


def _categorical(name: str, *labels: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "CATEGORICAL",
            "name": name,
            "optimization_direction": "MAXIMIZE",
            "values": [{"label": label, "score": float(i == 0)} for i, label in enumerate(labels)],
        }
    ]


# The shared definition of each LLM evaluator.
_LLM_CONFIGS = _categorical("correctness", "correct", "incorrect")
_STALE_LLM_CONFIGS = _categorical("hallucination", "factual", "hallucinated", "unsure")
# What the stale binding still holds: its evaluator's outputs before a later edit.
_STALE_COPY = _categorical("hallucination", "factual", "hallucinated")
_CODE_OVERRIDE = [
    {
        "type": "CONTINUOUS",
        "name": "latency",
        "optimization_direction": "MINIMIZE",
        "lower_bound": 0.0,
        "upper_bound": None,
        "description": None,
    }
]
_BUILTIN_OVERRIDE = _categorical("contains", "yes", "no")
# Written after the upgrade, standing in for an override set through the dataset evaluator API.
_NEW_OVERRIDE = _categorical("correctness", "right", "wrong")


class _Seed(NamedTuple):
    llm: int
    stale_llm: int
    code: int
    builtin_inherits: int
    builtin_override: int
    llm_evaluator: int
    stale_llm_evaluator: int


class _Binding(NamedTuple):
    description: Any
    output_configs: Any
    is_sql_null: bool
    is_json_null: bool


def _json(value: Any) -> str:
    return json.dumps(value)


def _insert(conn: Connection, sql: str, **params: Any) -> int:
    rowid = conn.execute(text(sql + " RETURNING id"), params).scalar()
    assert isinstance(rowid, int)
    return rowid


def _seed(conn: Connection) -> _Seed:
    now = datetime.now(timezone.utc)
    project = _insert(conn, "INSERT INTO projects (name) VALUES ('evaluators')")
    dataset = _insert(conn, "INSERT INTO datasets (name, metadata) VALUES ('qa', :m)", m=_json({}))
    version = _insert(
        conn,
        "INSERT INTO dataset_versions (dataset_id, metadata) VALUES (:d, :m)",
        d=dataset,
        m=_json({}),
    )
    example = _insert(conn, "INSERT INTO dataset_examples (dataset_id) VALUES (:d)", d=dataset)
    prompt = _insert(
        conn, "INSERT INTO prompts (name, metadata) VALUES ('correctness', :m)", m=_json({})
    )

    def evaluator(name: str, kind: str, description: Optional[str]) -> int:
        return _insert(
            conn,
            "INSERT INTO evaluators (name, description, metadata, kind) "
            "VALUES (:name, :description, :m, :kind)",
            name=name,
            description=description,
            m=_json({}),
            kind=kind,
        )

    def llm_evaluator(name: str, description: Optional[str], configs: list[Any]) -> int:
        evaluator_id = evaluator(name, "LLM", description)
        conn.execute(
            text(
                "INSERT INTO llm_evaluators (id, prompt_id, output_configs) "
                "VALUES (:id, :prompt, :configs)"
            ),
            {"id": evaluator_id, "prompt": prompt, "configs": _json(configs)},
        )
        return evaluator_id

    llm = llm_evaluator("correctness", "shared description", _LLM_CONFIGS)
    stale_llm = llm_evaluator("hallucination", None, _STALE_LLM_CONFIGS)
    code = evaluator("latency", "CODE", None)
    builtin = evaluator("contains", "BUILTIN", None)

    def binding(name: str, evaluator_id: int, description: Optional[str], configs: Any) -> int:
        return _insert(
            conn,
            "INSERT INTO dataset_evaluators "
            "(dataset_id, evaluator_id, name, description, output_configs, input_mapping, "
            "project_id) VALUES (:d, :e, :name, :description, :configs, :mapping, :project)",
            d=dataset,
            e=evaluator_id,
            name=name,
            description=description,
            configs=_json(configs),
            mapping=_json(_INPUT_MAPPING),
            project=project,
        )

    seed = _Seed(
        llm=binding("correctness", llm, "shared description", _LLM_CONFIGS),
        stale_llm=binding("hallucination", stale_llm, "old description", _STALE_COPY),
        code=binding("latency", code, "code override", _CODE_OVERRIDE),
        builtin_inherits=binding("contains", builtin, None, None),
        builtin_override=binding("contains_custom", builtin, None, _BUILTIN_OVERRIDE),
        llm_evaluator=llm,
        stale_llm_evaluator=stale_llm,
    )

    # Rows that cascade from dataset_evaluators, which a SQLite table rebuild must preserve.
    experiment = _insert(
        conn,
        "INSERT INTO experiments (dataset_id, dataset_version_id, name, repetitions, metadata) "
        "VALUES (:d, :v, 'run', 1, :m)",
        d=dataset,
        v=version,
        m=_json({}),
    )
    conn.execute(
        text("INSERT INTO experiment_jobs (id, type) VALUES (:id, 'PROMPT')"), {"id": experiment}
    )
    conn.execute(
        text(
            "INSERT INTO experiment_dataset_evaluators (experiment_id, dataset_evaluator_id) "
            "VALUES (:x, :b)"
        ),
        {"x": experiment, "b": seed.llm},
    )
    run = _insert(
        conn,
        "INSERT INTO experiment_runs "
        "(experiment_id, dataset_example_id, repetition_number, output, start_time, end_time) "
        "VALUES (:x, :e, 1, :o, :now, :now)",
        x=experiment,
        e=example,
        o=_json({}),
        now=now,
    )
    log = _insert(
        conn,
        "INSERT INTO experiment_logs (experiment_id, category, level, message) "
        "VALUES (:x, 'EVAL', 'ERROR', 'failed')",
        x=experiment,
    )
    conn.execute(
        text(
            "INSERT INTO experiment_eval_logs (id, experiment_run_id, dataset_evaluator_id) "
            "VALUES (:id, :run, :b)"
        ),
        {"id": log, "run": run, "b": seed.llm},
    )
    conn.commit()
    return seed


def _is_json_null(db_backend: Literal["sqlite", "postgresql"]) -> str:
    if db_backend == "postgresql":
        return "jsonb_typeof(output_configs) = 'null'"
    return "json_type(output_configs) = 'null'"


def _decode(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _bindings(conn: Connection, db_backend: Literal["sqlite", "postgresql"]) -> dict[int, _Binding]:
    rows = conn.execute(
        text(
            "SELECT id, description, output_configs, output_configs IS NULL, "
            f"COALESCE({_is_json_null(db_backend)}, FALSE) FROM dataset_evaluators"
        )
    ).all()
    return {
        row[0]: _Binding(
            description=row[1],
            output_configs=_decode(row[2]),
            is_sql_null=bool(row[3]),
            is_json_null=bool(row[4]),
        )
        for row in rows
    }


def _child_row_counts(conn: Connection) -> tuple[int, int]:
    return (
        conn.execute(text("SELECT COUNT(*) FROM experiment_dataset_evaluators")).scalar_one(),
        conn.execute(text("SELECT COUNT(*) FROM experiment_eval_logs")).scalar_one(),
    )


def _schema_info(
    conn: Connection, db_backend: Literal["sqlite", "postgresql"], schema: str
) -> _TableSchemaInfo:
    info = _get_table_schema_info(conn, "dataset_evaluators", db_backend, schema)
    assert info is not None
    return info


async def test_dataset_evaluators_inherit_llm_settings(
    _engine: AsyncEngine,
    _alembic_config: Config,
    _db_backend: Literal["sqlite", "postgresql"],
    _schema: str,
) -> None:
    await _up(_engine, _alembic_config, _DOWN, _schema)
    seed = await _run_async(_engine, _seed)

    def _snapshot(conn: Connection) -> tuple[_TableSchemaInfo, dict[int, _Binding]]:
        return _schema_info(conn, _db_backend, _schema), _bindings(conn, _db_backend)

    schema_before, before = await _run_async(_engine, _snapshot)
    assert "output_configs" not in schema_before["nullable_column_names"]
    assert before[seed.builtin_inherits].is_json_null

    await _up(_engine, _alembic_config, _UP, _schema)

    def _verify_upgraded(conn: Connection) -> None:
        schema_after = _schema_info(conn, _db_backend, _schema)
        assert schema_after["column_names"] == schema_before["column_names"]
        assert schema_after["index_names"] == schema_before["index_names"]
        assert schema_after["constraint_names"] == schema_before["constraint_names"]
        assert schema_after["nullable_column_names"] == (
            schema_before["nullable_column_names"] | {"output_configs"}
        )
        after = _bindings(conn, _db_backend)
        # LLM bindings inherit, including one whose copy had gone stale.
        for binding_id in (seed.llm, seed.stale_llm):
            assert after[binding_id].is_sql_null
            assert after[binding_id].description is None
        # A binding that already inherited stores SQL NULL instead of JSON null.
        assert after[seed.builtin_inherits].is_sql_null
        # Overrides on other evaluator kinds are untouched.
        for binding_id in (seed.code, seed.builtin_override):
            assert after[binding_id] == before[binding_id]
        assert _child_row_counts(conn) == (1, 1)
        if _db_backend == "sqlite":
            assert conn.execute(text("PRAGMA foreign_key_check")).all() == []

    await _run_async(_engine, _verify_upgraded)

    def _override_after_upgrade(conn: Connection) -> None:
        conn.execute(
            text("UPDATE dataset_evaluators SET output_configs = :configs WHERE id = :id"),
            {"configs": _json(_NEW_OVERRIDE), "id": seed.llm},
        )
        conn.execute(
            text("UPDATE dataset_evaluators SET description = 'dataset specific' WHERE id = :id"),
            {"id": seed.stale_llm},
        )
        conn.commit()

    await _run_async(_engine, _override_after_upgrade)
    await _down(_engine, _alembic_config, _DOWN, _schema)

    def _verify_downgraded(conn: Connection) -> None:
        assert _schema_info(conn, _db_backend, _schema) == schema_before
        after = _bindings(conn, _db_backend)
        # Overrides written after the upgrade survive the downgrade.
        assert after[seed.llm].output_configs == _NEW_OVERRIDE
        assert after[seed.stale_llm].description == "dataset specific"
        # Inheriting LLM bindings get their evaluator's current settings back.
        assert after[seed.llm].description == "shared description"
        assert after[seed.stale_llm].output_configs == _STALE_LLM_CONFIGS
        # The copied-back LLM configs parse the way the binding's column type reads them.
        restored = [
            OutputConfig.model_validate(config).root
            for config in after[seed.stale_llm].output_configs
        ]
        assert [config.name for config in restored] == ["hallucination"]
        # Other inheriting bindings store JSON null again.
        assert after[seed.builtin_inherits].is_json_null
        for binding_id in (seed.code, seed.builtin_override):
            assert after[binding_id] == before[binding_id]
        assert _child_row_counts(conn) == (1, 1)
        if _db_backend == "sqlite":
            assert conn.execute(text("PRAGMA foreign_key_check")).all() == []

    await _run_async(_engine, _verify_downgraded)
