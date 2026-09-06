from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine
from util.db import UsEquitiesTable
from util.db.connection import get_engine as _get_engine


def get_engine() -> Engine:
    """Re-exported from util.db.connection (reads DB_CONNECTION_CONFIG_PATH)."""
    return _get_engine()


def load_us_equities(
    engine: Engine | None = None,
    symbols: Sequence[str] | None = None,
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
    schema: str | None = None,
) -> pd.DataFrame:
    """Read-only pull of daily OHLCV rows from MarketLab.us_equities.

    alpha_engine is read-only: this is the only way it touches the DB
    populated by eodhd_client — no write path is exposed here.
    """
    engine = engine or get_engine()
    table = UsEquitiesTable(engine, schema=schema).table

    stmt = select(table)
    if symbols is not None:
        stmt = stmt.where(table.c.symbol.in_(symbols))
    if start is not None:
        stmt = stmt.where(table.c.date >= start)
    if end is not None:
        stmt = stmt.where(table.c.date <= end)

    return pd.read_sql(stmt, engine)
