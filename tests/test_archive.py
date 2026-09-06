from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from alpha_engine.maintenance.archive import archive_us_equities_tickers


def test_archive_us_equities_tickers_runs_single_transaction_for_explicit_targets() -> None:
    connection = MagicMock()
    connection.execute.side_effect = [
        MagicMock(rowcount=0),
        MagicMock(rowcount=0),
        MagicMock(rowcount=2),
        MagicMock(rowcount=8),
        MagicMock(rowcount=8),
        MagicMock(rowcount=2),
        MagicMock(rowcount=0),
    ]

    transaction = MagicMock()
    transaction.__enter__.return_value = connection
    transaction.__exit__.return_value = None

    engine = MagicMock()
    engine.begin.return_value = transaction

    tickers = pd.DataFrame(
        [
            {"symbol": "AAPL", "exchange": "NASDAQ"},
            {"symbol": "AAPL", "exchange": "NYSE"},
            {"symbol": "MSFT", "exchange": "NASDAQ"},
        ]
    )

    result = archive_us_equities_tickers(tickers, engine=engine)

    assert result.requested_tickers == 3
    assert result.inserted_rows == 8
    assert result.deleted_rows == 8
    assert result.updated_tickers == 2

    temp_insert_call = connection.execute.call_args_list[2]
    assert temp_insert_call.args[1] == [
        {"symbol": "AAPL", "exchange": "NASDAQ"},
        {"symbol": "AAPL", "exchange": "NYSE"},
        {"symbol": "MSFT", "exchange": "NASDAQ"},
    ]

    statements = [str(call.args[0]) for call in connection.execute.call_args_list]
    assert "IF OBJECT_ID(N'dbo.us_equities_archived', N'U') IS NULL" in statements[0]
    assert "CREATE TABLE #archive_targets" in statements[1]
    assert "INSERT INTO [dbo].[us_equities_archived]" in statements[3]
    assert "[audit_created_at]" not in statements[3]
    assert "INNER JOIN #archive_targets AS target" in statements[3]
    assert "DELETE source" in statements[4]
    assert "UPDATE tickers" in statements[5]
    assert "audit_updated_at" not in statements[5]
    assert "DROP TABLE IF EXISTS #archive_targets" in statements[6]


def test_archive_us_equities_tickers_requires_symbol_and_exchange_columns() -> None:
    tickers = pd.DataFrame([{"exchange": "NASDAQ"}])

    with pytest.raises(ValueError, match="missing required columns: symbol"):
        archive_us_equities_tickers(tickers, engine=MagicMock())

    tickers = pd.DataFrame([{"symbol": "AAPL"}])

    with pytest.raises(ValueError, match="missing required columns: exchange"):
        archive_us_equities_tickers(tickers, engine=MagicMock())
