# alpha_engine

Research repo for exploring EOD equity data populated by `eodhd_client`. Most of the package is read-only; the only write-path helper is the explicit archival maintenance utility in `alpha_engine.maintenance`.

Package management is handled by [uv](https://github.com/astral-sh/uv) (Astral). The project uses a shared virtual environment at `%USERPROFILE%\.venvs\py313`, not a per-project venv:

```
source ~/.venvs/py313/Scripts/activate
```

## Package layout

- `src/alpha_engine/data.py`: `Data` — frozen metadata wrapper for a strategy panel and its selection inputs.
- `src/alpha_engine/universe.py`: `get_engine()`, `load_us_equities(...)`, `load_strategy_adjusted_close_panel(...)`, and `build_strategy_universe(...)` — read-only database and universe loaders.
- `src/alpha_engine/maintenance/archive.py`: `archive_us_equities_tickers(...)` — moves all `dbo.us_equities` rows for a ticker subset into `dbo.us_equities_archived`, deletes them from the live table, and marks `dbo.us_tickers.archived = 1` in the same transaction.
- `notebooks/`: exploratory research notebooks, run on the shared `py313 (shared)` Jupyter kernel.

## Example

```python
from alpha_engine.universe import load_us_equities

df = load_us_equities(symbols=["AAPL", "MSFT"], start="2026-01-01")
```

```python
from alpha_engine.maintenance import archive_us_equities_tickers

archive_us_equities_tickers(tickers_subset_df)
```
