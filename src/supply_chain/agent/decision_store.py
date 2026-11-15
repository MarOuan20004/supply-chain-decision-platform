"""Persistance SQLite transactionnelle des decisions agentiques et du workflow HITL."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from scripts.migrate_decisions_table import apply_decisions_migration
from supply_chain.agent.schemas import (
    DecisionRecord,
    DecisionReport,
    ExecutionStatus,
)


class DecisionNotFoundError(KeyError):
    """Exception levee lorsque l'identifiant de decision est introuvable en base."""


class DecisionConflictError(RuntimeError):
    """Exception levee lors d'une double approbation/rejet sur une decision deja traitee."""


class SQLDecisionStore:
    """Depot SQLite paramétré pour la table 'decisions'."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        apply_decisions_migration(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def save_decision(
        self,
        trace_id: str,
        report: DecisionReport,
        decision_id: str | None = None,
    ) -> DecisionRecord:
        """Persiste un DecisionReport en base SQL et retourne le DecisionRecord."""
        rec_id = decision_id or f"dec-{uuid4().hex[:12]}"
        now_iso = datetime.now(UTC).isoformat()
        report_str = report.model_dump_json()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (
                    id, sku_id, trace_id, status, report_json,
                    created_at, decided_at, operator_id, operator_justification
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (rec_id, report.sku_id, trace_id, report.status.value, report_str, now_iso),
            )
            conn.commit()

        return DecisionRecord(
            id=rec_id,
            sku_id=report.sku_id,
            trace_id=trace_id,
            status=report.status,
            report_json=report,
            created_at=now_iso,
        )

    def list_pending(self) -> list[DecisionRecord]:
        """Retourne uniquement les decisions en statut PENDING_HUMAN_APPROVAL."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, sku_id, trace_id, status, report_json,
                       created_at, decided_at, operator_id, operator_justification
                FROM decisions
                WHERE status = ?
                ORDER BY created_at ASC
                """,
                (ExecutionStatus.PENDING_HUMAN_APPROVAL.value,),
            ).fetchall()

        return [self._row_to_record(r) for r in rows]

    def approve_or_reject(
        self,
        decision_id: str,
        operator_id: str,
        decision: Literal["approve", "reject"],
        justification: str,
    ) -> tuple[DecisionRecord, ExecutionStatus]:
        """Met a jour le statut d'une decision PENDING_HUMAN_APPROVAL de facon idempotente."""
        target_status = (
            ExecutionStatus.APPROVED_HUMAN
            if decision == "approve"
            else ExecutionStatus.REJECTED_HUMAN
        )
        now_iso = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()

            if row is None:
                raise DecisionNotFoundError(f"Decision '{decision_id}' introuvable.")

            current_status = ExecutionStatus(row["status"])
            if current_status != ExecutionStatus.PENDING_HUMAN_APPROVAL:
                raise DecisionConflictError(
                    f"La decision '{decision_id}' a deja ete traitee (statut actuel: "
                    f"{current_status.value}). Double operation interdite."
                )

            cursor = conn.execute(
                """
                UPDATE decisions
                SET status = ?, decided_at = ?, operator_id = ?, operator_justification = ?
                WHERE id = ? AND status = ?
                """,
                (
                    target_status.value,
                    now_iso,
                    operator_id,
                    justification,
                    decision_id,
                    ExecutionStatus.PENDING_HUMAN_APPROVAL.value,
                ),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise DecisionConflictError(
                    f"Conflit concurrentiel sur la decision '{decision_id}'."
                )

            updated_row = conn.execute(
                "SELECT * FROM decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()

        return self._row_to_record(updated_row), current_status

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(
            id=str(row["id"]),
            sku_id=str(row["sku_id"]),
            trace_id=str(row["trace_id"]),
            status=ExecutionStatus(row["status"]),
            report_json=DecisionReport.model_validate_json(row["report_json"]),
            created_at=str(row["created_at"]),
            decided_at=str(row["decided_at"]) if row["decided_at"] is not None else None,
            operator_id=str(row["operator_id"]) if row["operator_id"] is not None else None,
            operator_justification=(
                str(row["operator_justification"])
                if row["operator_justification"] is not None
                else None
            ),
        )


SqliteDecisionStore = SQLDecisionStore
