# alpha_engine

Research repo for exploring EOD equity data populated by `eodhd_client`. Most of the package is read-only; the only write-path helper is the explicit archival maintenance utility in `alpha_engine.maintenance`.

Package management is handled by [uv](https://github.com/astral-sh/uv) (Astral). The project uses a shared virtual environment at `%USERPROFILE%\.venvs\py313`, not a per-project venv:

```
source ~/.venvs/py313/Scripts/activate
```

## Package layout

- `src/alpha_engine/data.py`: `get_engine()` (re-exported from `util.db.connection`) + `load_us_equities(...)` — a thin read-only `pandas.DataFrame` pull from `MarketLab.us_equities`, with optional `symbols`/`start`/`end` filters.
- `src/alpha_engine/maintenance/archive.py`: `archive_us_equities_tickers(...)` — moves all `dbo.us_equities` rows for a ticker subset into `dbo.us_equities_archived`, deletes them from the live table, and marks `dbo.us_tickers.archived = 1` in the same transaction.
- `notebooks/`: exploratory research notebooks, run on the shared `py313 (shared)` Jupyter kernel.

## Example

```python
from alpha_engine.data import load_us_equities

df = load_us_equities(symbols=["AAPL", "MSFT"], start="2026-01-01")
```

```python
from alpha_engine.maintenance import archive_us_equities_tickers

archive_us_equities_tickers(tickers_subset_df)
```
