"""
Streamlit dashboard for quant-factor-mining.

Run:  streamlit run dashboard/app.py
"""

import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_DB_PATH       = _ROOT / "outputs" / "factor_monitor.db"
_WEIGHT_PATH   = _ROOT / "outputs" / "weight_history.parquet"
_REGIME_PATH   = _ROOT / "outputs" / "regime_history.parquet"
_BACKTEST_DIR  = _ROOT / "outputs" / "backtests"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Factor Monitor",
    page_icon="📈",
    layout="wide",
)


# ── Data loaders (read-only, graceful on missing files) ────────────────────

@st.cache_data(ttl=30)
def load_sqlite_metrics():
    if not _DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(_DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT date, factor_name, icir, ic_mean, hhi, skip_count "
            "FROM factor_metrics ORDER BY recorded_at",
            conn,
        )


@st.cache_data(ttl=30)
def load_weight_history():
    if not _WEIGHT_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(_WEIGHT_PATH)


@st.cache_data(ttl=30)
def load_regime_history():
    if not _REGIME_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(_REGIME_PATH)


@st.cache_data(ttl=30)
def load_backtest():
    if not _BACKTEST_DIR.exists():
        return pd.DataFrame()
    files = sorted(_BACKTEST_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])  # most recent file


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    refresh_interval = st.slider(
        "Auto-refresh interval (s)", min_value=10, max_value=120, value=30, step=10
    )
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    last_updated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Last updated: {last_updated}")
    st.caption(f"DB: `{_DB_PATH.name}`")
    if _DB_PATH.exists():
        st.success("SQLite DB connected")
    else:
        st.warning("SQLite DB not found — run the factor monitor first")


# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Factor Health",
    "⚖️ Weight Evolution",
    "🌐 Regime State",
    "📈 Backtest",
])


# ── Tab 1: Factor Health ───────────────────────────────────────────────────
with tab1:
    st.header("Factor Health")
    metrics = load_sqlite_metrics()

    if metrics.empty:
        st.info("No data yet. Run FactorMonitor with a db_path to populate this tab.")
    else:
        latest = metrics.groupby("factor_name").last().reset_index()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ICIR per Factor")
            fig = px.bar(
                latest, x="factor_name", y="icir",
                color="icir", color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                labels={"icir": "ICIR", "factor_name": "Factor"},
                title="Rolling ICIR (latest snapshot)",
            )
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                          annotation_text="Threshold 0.3")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Weight Concentration (HHI)")
            hhi_val = latest["hhi"].iloc[0] if not latest.empty else float("nan")
            if not pd.isna(hhi_val):
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=hhi_val,
                    title={"text": "HHI"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "steelblue"},
                        "steps": [
                            {"range": [0, 0.33], "color": "#c8e6c9"},
                            {"range": [0.33, 0.5], "color": "#fff9c4"},
                            {"range": [0.5, 1.0], "color": "#ffcdd2"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3}, "value": 0.5},
                    },
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.metric("HHI", "N/A")

        st.subheader("IC Skip Count (staleness)")
        skip_table = latest[["factor_name", "skip_count"]].copy()
        skip_table["status"] = skip_table["skip_count"].apply(
            lambda x: "🔴 STALE" if x >= 21 else ("🟡 Warn" if x >= 5 else "🟢 OK")
        )
        st.dataframe(skip_table, use_container_width=True)


# ── Tab 2: Weight Evolution ────────────────────────────────────────────────
with tab2:
    st.header("Weight Evolution")
    weights = load_weight_history()

    if weights.empty:
        st.info(
            "No weight_history.parquet found. Save weight history with:\n"
            "```python\naf.get_weight_history().to_parquet('outputs/weight_history.parquet')\n```"
        )
    else:
        weights.index = pd.to_datetime(weights.index)
        fig = px.line(
            weights.reset_index().melt(id_vars="date", var_name="Factor", value_name="Weight"),
            x="date", y="Weight", color="Factor",
            title="Factor Weight Evolution Over Time",
            labels={"date": "Date", "Weight": "Weight"},
        )
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latest Weights")
        st.dataframe(
            weights.iloc[-1:].T.rename(columns={weights.index[-1]: "Weight"}),
            use_container_width=True,
        )


# ── Tab 3: Regime State ────────────────────────────────────────────────────
with tab3:
    st.header("Regime State")
    regimes = load_regime_history()

    if regimes.empty:
        st.info("No regime_history.parquet found. Save regime history from RegimeDetector output.")
    else:
        regimes.index = pd.to_datetime(regimes.index)
        latest_regime = (
            regimes.iloc[-1].get("regime", "unknown")
            if "regime" in regimes.columns
            else "unknown"
        )

        REGIME_COLORS = {
            "trending":       "#2196F3",
            "mean_reverting": "#4CAF50",
            "high_vol":       "#F44336",
            "neutral":        "#9E9E9E",
        }
        badge_color = REGIME_COLORS.get(latest_regime, "#9E9E9E")
        st.markdown(
            f"**Current regime:** "
            f"<span style='background:{badge_color};color:white;padding:4px 12px;"
            f"border-radius:12px;font-weight:bold'>{latest_regime.upper()}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        if "regime" in regimes.columns:
            code_map = {"trending": 1, "mean_reverting": 2, "high_vol": 3, "neutral": 0}
            regimes["regime_code"] = regimes["regime"].map(code_map).fillna(0)

            fig = px.area(
                regimes.reset_index().rename(columns={"index": "date"}),
                x="date", y="regime_code",
                color="regime",
                title="Regime Timeline",
                labels={"date": "Date", "regime_code": "Regime", "regime": "Regime"},
                color_discrete_map=REGIME_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True)

            freq = regimes["regime"].value_counts().reset_index()
            freq.columns = ["regime", "count"]
            fig_pie = px.pie(
                freq, names="regime", values="count",
                title="Regime Frequency",
                color="regime", color_discrete_map=REGIME_COLORS,
            )
            st.plotly_chart(fig_pie, use_container_width=True)


# ── Tab 4: Backtest ────────────────────────────────────────────────────────
with tab4:
    st.header("Backtest Results")
    bt = load_backtest()

    if bt.empty:
        st.info(
            "No backtest parquet files found in `outputs/backtests/`.\n"
            "Run a backtest and save equity curve to:\n"
            "`outputs/backtests/equity_curve_YYYYMMDD.parquet`"
        )
    else:
        bt.index = pd.to_datetime(bt.index)

        if "portfolio_value" in bt.columns:
            fig = px.line(
                bt.reset_index(),
                x=bt.index.name or "index",
                y="portfolio_value",
                title="Equity Curve",
                labels={"portfolio_value": "Portfolio Value ($)"},
            )
            fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        if "daily_return" in bt.columns:
            monthly_ret = bt["daily_return"].resample("ME").apply(
                lambda x: (1 + x).prod() - 1
            )
            pivot = pd.DataFrame({
                "year": monthly_ret.index.year,
                "month": monthly_ret.index.month_name().str[:3],
                "ret": monthly_ret.values,
            }).pivot(index="year", columns="month", values="ret")

            fig_hm = px.imshow(
                pivot * 100,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                title="Monthly Returns Heatmap (%)",
                labels={"color": "Return (%)"},
                text_auto=".1f",
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        if "portfolio_value" in bt.columns:
            pv = bt["portfolio_value"]
            total_ret = pv.iloc[-1] / pv.iloc[0] - 1
            ann_factor = 252 / max(len(pv), 1)
            ann_ret = (1 + total_ret) ** ann_factor - 1
            rolling_max = pv.cummax()
            max_dd = ((pv - rolling_max) / rolling_max).min()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Return", f"{total_ret:.1%}")
            m2.metric("Annualized Return", f"{ann_ret:.1%}")
            m3.metric("Max Drawdown", f"{max_dd:.1%}")


# ── Auto-refresh ───────────────────────────────────────────────────────────
time.sleep(refresh_interval)
st.cache_data.clear()
st.rerun()
