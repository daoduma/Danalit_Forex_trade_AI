"""Streamlit dashboard — reads ONLY SQLite/Parquet, never the gateway.

Run: python scripts/run_dashboard.py   (localhost)

All data loaders are pure pandas functions, testable without streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from danalit.config import load_config
from danalit.db import connect


def load_table(table: str, db_path: Optional[Path] = None, limit: int = 0) -> pd.DataFrame:
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        sql = f"SELECT * FROM {table} ORDER BY id DESC" + (f" LIMIT {limit}" if limit else "")
        return pd.DataFrame([dict(r) for r in con.execute(sql).fetchall()])
    finally:
        con.close()


def equity_frame(db_path: Optional[Path] = None) -> pd.DataFrame:
    df = load_table("equity_snapshots", db_path)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    return df.sort_values("ts_utc")


def recent_decisions(db_path: Optional[Path] = None, n: int = 50) -> pd.DataFrame:
    df = load_table("decisions", db_path, limit=n)
    if df.empty:
        return df
    return df[["ts_utc", "instrument", "action", "confidence", "veto_reason",
               "explanation", "mode"]]


def veto_counts(db_path: Optional[Path] = None) -> pd.DataFrame:
    df = load_table("decisions", db_path)
    if df.empty:
        return df
    return (df["veto_reason"].fillna("(traded)").value_counts()
            .rename_axis("reason").reset_index(name="count"))


def main() -> None:  # pragma: no cover — UI shell over the tested loaders
    import streamlit as st

    from danalit.data.collector_daemon import heartbeat_age_seconds

    st.set_page_config(page_title="Danalit", layout="wide")
    st.title("Danalit — trading dashboard")

    col_age = heartbeat_age_seconds()
    c1, c2, c3 = st.columns(3)
    eq = equity_frame()
    if not eq.empty:
        last = eq.iloc[-1]
        c1.metric("Equity", f"{last['equity']:.2f}", f"{last['mode']}")
        c2.metric("Open risk", f"{last['open_risk']:.2f}")
    c3.metric("Collector heartbeat",
              f"{col_age:.0f}s ago" if col_age is not None else "MISSING")

    if not eq.empty:
        st.subheader("Equity curve")
        st.line_chart(eq.set_index("ts_utc")[["equity", "balance"]])

    st.subheader("Last 50 decisions (with explanations and vetoes)")
    st.dataframe(recent_decisions(), use_container_width=True)

    st.subheader("Veto / outcome frequencies")
    vc = veto_counts()
    if not vc.empty:
        st.bar_chart(vc.set_index("reason")["count"])

    st.subheader("System events")
    st.dataframe(load_table("system_events", limit=30), use_container_width=True)
    st.caption("Reads SQLite/Parquet only; auto-refresh with R. "
               "Forward-test cost panel: run scripts/journal_report.py")


if __name__ == "__main__":
    main()
