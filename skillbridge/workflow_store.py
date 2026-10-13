"""SQLite persistence and idempotency for completed SkillBridge workflows."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from skillbridge.orchestration import CandidateWorkflowResult


@dataclass(frozen=True)
class WorkflowClaim:
    state: str
    result: CandidateWorkflowResult | None = None


class SQLiteWorkflowStore:
    """Persist traceable workflow output; never store credentials or raw request data."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS skillbridge_workflows (
                    request_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('in_progress', 'completed')),
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    result_json TEXT
                )
                """
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def claim(self, request_id: str, request_hash: str, started_at: str) -> WorkflowClaim:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT request_hash, state, result_json FROM skillbridge_workflows WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO skillbridge_workflows "
                    "(request_id, request_hash, state, started_at) VALUES (?, ?, 'in_progress', ?)",
                    (request_id, request_hash, started_at),
                )
                return WorkflowClaim("claimed")
            if row["request_hash"] != request_hash:
                return WorkflowClaim("conflict")
            if row["state"] == "in_progress":
                return WorkflowClaim("in_progress")
            result = CandidateWorkflowResult.model_validate_json(row["result_json"])
            return WorkflowClaim("completed", result)

    def complete(self, result: CandidateWorkflowResult) -> None:
        result_json = result.model_dump_json()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE skillbridge_workflows
                SET state = 'completed', completed_at = ?, result_json = ?
                WHERE request_id = ? AND state = 'in_progress'
                """,
                (result.completed_at.isoformat(), result_json, result.request_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("workflow completion did not match an in-progress request")
