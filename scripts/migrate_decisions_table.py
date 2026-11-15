#!/usr/bin/env python3
"""Migration SQL idempotente ajoutant la table HITL 'decisions' au schema Supply Chain S4."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from supply_chain.config import settings

DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    sku_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    status TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    operator_id TEXT,
    operator_justification TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions (status);
CREATE INDEX IF NOT EXISTS idx_decisions_sku_id ON decisions (sku_id);
"""


def apply_decisions_migration(db_path: Path) -> None:
    """Applique le DDL de la table decisions sur la base SQLite cible."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(DECISIONS_DDL)
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migration SQL de la table HITL decisions.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=settings.database_path,
        help="Chemin vers la base SQLite (defaut: data/supply_chain.db).",
    )
    args = parser.parse_args()
    apply_decisions_migration(args.db_path)
    print(f"Migration 'decisions' appliquee avec succes sur {args.db_path}")


if __name__ == "__main__":
    main()
