from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import CursorResult, Engine

from alpha_engine.data import get_engine


@dataclass(frozen=True, slots=True)
class ArchiveTransferResult:
    requested_tickers: int
    inserted_rows: int
    deleted_rows: int
    updated_tickers: int


def archive_us_equities_tickers(
    tickers: pd.DataFrame | Sequence[tuple[str, str]] | Sequence[Mapping[str, object]],
    *,
    engine: Engine | None = None,
    schema: str = "dbo",
    source_table_name: str = "us_equities",
    archive_table_name: str = "us_equities_archived",
    tickers_table_name: str = "us_tickers",
) -> ArchiveTransferResult:
    """Archive all equity rows for the given ticker subset in one transaction.

    The operation is a real move, not a copy:
    1. insert missing rows into `schema.archive_table_name`
    2. delete matching rows from `schema.source_table_name`
    3. mark matching rows in `schema.tickers_table_name` as archived

    `tickers` may be the notebook's `tickers_subset_df`, a sequence of
    `(symbol, exchange)` tuples, or a sequence of mapping objects that
    contain `symbol` and `exchange` keys. Callers are expected to provide
    a unique target list.
    """
    if isinstance(tickers, pd.DataFrame):
        missing_columns = {"symbol", "exchange"}.difference(tickers.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"tickers DataFrame is missing required columns: {missing}")
        archive_targets = tickers.loc[:, ["symbol", "exchange"]].to_dict(orient="records")
    else:
        archive_targets = []
        for ticker in tickers:
            if isinstance(ticker, Mapping):
                if "symbol" not in ticker or "exchange" not in ticker:
                    raise ValueError("Each ticker mapping must contain symbol and exchange keys.")
                archive_targets.append({"symbol": ticker["symbol"], "exchange": ticker["exchange"]})
                continue

            if len(ticker) != 2:
                raise ValueError("Each ticker tuple must contain exactly (symbol, exchange).")
            archive_targets.append({"symbol": ticker[0], "exchange": ticker[1]})

    if not archive_targets:
        raise ValueError("tickers must contain at least one (symbol, exchange) pair")

    engine = engine or get_engine()
    _ensure_archive_table(engine, schema=schema, table_name=archive_table_name)

    qualified_source = _qualified_name(schema, source_table_name)
    qualified_archive = _qualified_name(schema, archive_table_name)
    qualified_tickers = _qualified_name(schema, tickers_table_name)

    create_temp_sql = text(
        """
        CREATE TABLE #archive_targets (
            [symbol] VARCHAR(32) NOT NULL,
            [exchange] VARCHAR(32) NOT NULL,
            PRIMARY KEY ([symbol], [exchange])
        );
        """
    )
    insert_temp_sql = text(
        """
        INSERT INTO #archive_targets ([symbol], [exchange])
        VALUES (:symbol, :exchange);
        """
    )
    insert_archive_sql = text(
        f"""
        INSERT INTO {qualified_archive} (
            [symbol],
            [exchange],
            [date],
            [open],
            [high],
            [low],
            [close],
            [adjusted_close],
            [volume]
        )
        SELECT
            source.[symbol],
            source.[exchange],
            source.[date],
            source.[open],
            source.[high],
            source.[low],
            source.[close],
            source.[adjusted_close],
            source.[volume]
        FROM {qualified_source} AS source
        INNER JOIN #archive_targets AS target
            ON target.[symbol] = source.[symbol]
           AND target.[exchange] = source.[exchange]
        LEFT JOIN {qualified_archive} AS archived
            ON archived.[symbol] = source.[symbol]
           AND archived.[exchange] = source.[exchange]
           AND archived.[date] = source.[date]
        WHERE archived.[symbol] IS NULL;
        """
    )
    delete_source_sql = text(
        f"""
        DELETE source
        FROM {qualified_source} AS source
        INNER JOIN #archive_targets AS target
            ON target.[symbol] = source.[symbol]
           AND target.[exchange] = source.[exchange];
        """
    )
    update_tickers_sql = text(
        f"""
        UPDATE tickers
        SET tickers.[archived] = 1
        FROM {qualified_tickers} AS tickers
        INNER JOIN #archive_targets AS target
            ON target.[symbol] = tickers.[symbol]
           AND target.[exchange] = tickers.[exchange]
        WHERE tickers.[archived] = 0;
        """
    )
    drop_temp_sql = text("DROP TABLE IF EXISTS #archive_targets;")

    with engine.begin() as connection:
        try:
            connection.execute(create_temp_sql)
            connection.execute(insert_temp_sql, archive_targets)
            inserted_rows = _result_rowcount(connection.execute(insert_archive_sql))
            deleted_rows = _result_rowcount(connection.execute(delete_source_sql))
            updated_tickers = _result_rowcount(connection.execute(update_tickers_sql))
        finally:
            connection.execute(drop_temp_sql)

    return ArchiveTransferResult(
        requested_tickers=len(archive_targets),
        inserted_rows=inserted_rows,
        deleted_rows=deleted_rows,
        updated_tickers=updated_tickers,
    )


def _qualified_name(schema: str, table_name: str) -> str:
    return f"[{schema}].[{table_name}]"


def _ensure_archive_table(engine: Engine, *, schema: str, table_name: str) -> None:
    qualified_archive = _qualified_name(schema, table_name)
    create_table_sql = text(
        f"""
        IF OBJECT_ID(N'{schema}.{table_name}', N'U') IS NULL
        BEGIN
            CREATE TABLE {qualified_archive} (
                [symbol] VARCHAR(32) NOT NULL,
                [exchange] VARCHAR(32) NOT NULL,
                [date] DATE NOT NULL,
                [open] NUMERIC(12, 4) NOT NULL,
                [high] NUMERIC(12, 4) NOT NULL,
                [low] NUMERIC(12, 4) NOT NULL,
                [close] NUMERIC(12, 4) NOT NULL,
                [adjusted_close] NUMERIC(18, 6) NOT NULL,
                [volume] BIGINT NOT NULL,
                CONSTRAINT [PK_{table_name}] PRIMARY KEY ([symbol], [exchange], [date])
            );
        END;
        """
    )
    with engine.begin() as connection:
        connection.execute(create_table_sql)


def _result_rowcount(result: CursorResult[Any]) -> int:
    return max(result.rowcount or 0, 0)
