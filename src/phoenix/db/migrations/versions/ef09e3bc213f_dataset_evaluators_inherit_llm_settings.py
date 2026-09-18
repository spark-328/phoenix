"""dataset evaluators inherit LLM evaluator settings

Revision ID: ef09e3bc213f
Revises: a7f1c3e9d2b4
Create Date: 2026-09-18 00:00:00.000000

A dataset evaluator overrides its evaluator's description and output configs
only where it stores them; NULL means the binding inherits the evaluator's
value.

- dataset_evaluators.output_configs becomes nullable, and a binding that
  inherits stores SQL NULL instead of the JSON value null, so `IS NULL`
  identifies inheriting bindings on both dialects.
- Bindings of LLM evaluators are reset to inherit. Before this revision an LLM
  evaluator had exactly one binding, created with it, and every mutation that
  wrote the evaluator's description and output configs wrote the same input
  onto that binding, so the values these bindings hold are copies of their
  evaluator's settings, never overrides a user chose.

The downgrade copies each LLM evaluator's settings back onto its inheriting
bindings, stores JSON null for the remaining inheriting bindings, and restores
NOT NULL.
"""

from typing import Any, Sequence, Union

from alembic import op
from sqlalchemy import JSON
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.compiler import compiles


class JSONB(JSON):
    __visit_name__ = "JSONB"


@compiles(JSONB, "sqlite")
def _(*args: Any, **kwargs: Any) -> str:
    return "JSONB"


JSON_ = JSON().with_variant(postgresql.JSONB(), "postgresql").with_variant(JSONB(), "sqlite")

# revision identifiers, used by Alembic.
revision: str = "ef09e3bc213f"
down_revision: Union[str, None] = "a7f1c3e9d2b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_LLM_BINDINGS = "evaluator_id IN (SELECT id FROM evaluators WHERE kind = 'LLM')"


def _is_json_null(column: str) -> str:
    if op.get_bind().dialect.name == "postgresql":
        return f"jsonb_typeof({column}) = 'null'"
    return f"json_type({column}) = 'null'"


def upgrade() -> None:
    with op.batch_alter_table("dataset_evaluators") as batch_op:
        batch_op.alter_column("output_configs", existing_type=JSON_, nullable=True)
    op.execute(
        "UPDATE dataset_evaluators SET output_configs = NULL "
        f"WHERE {_is_json_null('output_configs')}"
    )
    op.execute(
        f"UPDATE dataset_evaluators SET output_configs = NULL, description = NULL "
        f"WHERE {_LLM_BINDINGS}"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE dataset_evaluators SET output_configs = ("
        "SELECT llm_evaluators.output_configs FROM llm_evaluators "
        "WHERE llm_evaluators.id = dataset_evaluators.evaluator_id"
        f") WHERE output_configs IS NULL AND {_LLM_BINDINGS}"
    )
    op.execute(
        "UPDATE dataset_evaluators SET description = ("
        "SELECT evaluators.description FROM evaluators "
        "WHERE evaluators.id = dataset_evaluators.evaluator_id"
        f") WHERE description IS NULL AND {_LLM_BINDINGS}"
    )
    op.execute("UPDATE dataset_evaluators SET output_configs = 'null' WHERE output_configs IS NULL")
    with op.batch_alter_table("dataset_evaluators") as batch_op:
        batch_op.alter_column("output_configs", existing_type=JSON_, nullable=False)
