"""
LeakFinder - Datathon 2026, Track 4 (The Money Leak)
Executive Streamlit dashboard. Run:  streamlit run app.py

Everything is computed locally with streamlit, pandas, numpy and plotly. No external APIs.

Backend: if ml_scratch.py and pipeline.py sit next to this file, the ML Lab uses them (from-scratch NumPy
models). If they are missing, app.py falls back to compact built-in NumPy models, so the app always runs.
Data: CSVs are read from ./data (or next to this file). All cleaning happens in load_data().

Optional: for native dark widgets/tables, add .streamlit/config.toml with
    [theme]
    base = "dark"
    primaryColor = "#38bdf8"
    backgroundColor = "#0b1426"
    secondaryBackgroundColor = "#0f172a"
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="LeakFinder",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    :root {
        --lf-bg: #0b1426;
        --lf-panel: #0f172a;
        --lf-panel-raised: #17243a;
        --lf-line: rgba(177, 198, 224, 0.14);
        --lf-ink: #f1f5f9;
        --lf-muted: #94a3b8;
        --lf-card-padding: 20px;
        --lf-green: #10b981;
        --lf-red: #ef4444;
        --lf-amber: #f59e0b;
        --lf-sky: #38bdf8;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}

    .stApp {
        background:
            radial-gradient(ellipse at 78% 0%, rgba(16, 185, 129, 0.09), transparent 32%),
            radial-gradient(ellipse at 0% 22%, rgba(56, 189, 248, 0.045), transparent 28%),
            var(--lf-bg);
        color: var(--lf-ink);
        font-family: Inter, system-ui, -apple-system, sans-serif;
    }

    [data-testid="stAppViewContainer"] { background: transparent; }
    .block-container { max-width: 1480px; padding-top: 1.2rem; padding-bottom: 2rem; }
    h1, h2, h3, h4, h5 { color: var(--lf-ink) !important; letter-spacing: 0; }
    [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color: #b4c2d3 !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101d31 0%, #0c1729 100%);
        border-right: 1px solid var(--lf-line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color: var(--lf-ink); }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: #1b2b3e !important;
        border: 1px solid rgba(177, 198, 224, 0.18);
        color: #cbd5e1 !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] * { color: #cbd5e1 !important; }
    input, textarea, [data-baseweb="select"] > div { border-color: var(--lf-line) !important; }

    .metric-card {
        box-sizing: border-box;
        height: 100%;
        background: linear-gradient(145deg, rgba(23, 42, 65, 0.86), rgba(15, 23, 42, 0.94));
        border: 1px solid var(--lf-line);
        border-radius: 10px;
        padding: var(--lf-card-padding);
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.16);
        backdrop-filter: blur(16px);
    }
    .metric-title { color: #a8b8cc; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em; }
    .metric-value { color: #f8fafc; font-size: clamp(1.35rem, 1.8vw, 1.8rem); font-weight: 750; margin: 9px 0 5px; font-variant-numeric: tabular-nums; }
    .metric-sub { color: var(--lf-muted); font-size: 0.75rem; font-weight: 500; line-height: 1.45; }
    .text-alert { color: #ff817c; }
    .text-success { color: #55ddb0; }
    .text-warning { color: #f8c568; }

    .alert-card {
        background: rgba(239, 68, 68, 0.075);
        border: 1px solid rgba(177, 198, 224, 0.12);
        border-left: 3px solid var(--lf-red);
        border-radius: 8px;
        padding: 15px 17px;
        margin-bottom: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        display: grid !important;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 5px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid var(--lf-line);
        padding: 6px;
        border-radius: 9px;
        overflow: visible !important;
    }
    .stTabs [data-baseweb="tab"] {
        min-width: 0;
        min-height: 42px;
        justify-content: center;
        white-space: normal;
        text-align: center;
        color: #a6b5c8;
        border-radius: 6px;
        padding: 9px 8px;
        font-size: 0.86rem;
        font-weight: 650;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.14) !important;
        color: #6ce1b6 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--lf-green); }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid var(--lf-line);
        border-radius: 7px;
        background: #14243a;
        color: var(--lf-ink);
        font-weight: 650;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(16, 185, 129, 0.65);
        color: #8ce8c5;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--lf-line);
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stPlotlyChart"] { border: 1px solid rgba(177, 198, 224, 0.08); border-radius: 9px; }
    .site-nav { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:4px 0 26px; border-bottom:1px solid var(--lf-line); }
    .brand-lockup { display:flex; align-items:center; gap:11px; color:#f5f8fc; font-size:0.93rem; font-weight:800; }
    .brand-mark { width:30px; height:30px; display:grid; place-items:center; border-radius:8px; background:rgba(16,185,129,.15); border:1px solid rgba(16,185,129,.35); color:#63dfb2; font-size:.75rem; }
    .brand-edition { display:block; margin-top:2px; color:#9aadc3; font-size:.72rem; font-weight:500; }
    .nav-links { display:flex; align-items:center; gap:25px; }
    .nav-links a, .hero-actions a { color:#aebdd0; text-decoration:none; font-size:.82rem; font-weight:650; }
    .nav-links a:hover { color:#f1f5f9; }
    .nav-links .nav-cta, .hero-actions .button-primary { color:#06231c; background:var(--lf-green); border:1px solid var(--lf-green); border-radius:7px; padding:10px 14px; }
    .stApp [data-testid="stHorizontalBlock"]:has(.hero-title) { align-items:center; gap:clamp(28px,5vw,72px); padding:42px 0 48px; }
    .eyebrow, .section-kicker { color:#6ce1b6; font:700 .66rem/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.12em; text-transform:uppercase; }
    .eyebrow { display:flex; align-items:center; gap:9px; }
    .eyebrow-dot { width:7px; height:7px; border-radius:50%; background:var(--lf-green); box-shadow:0 0 14px rgba(16,185,129,.7); }
    .hero-title { max-width:720px; margin:18px 0 14px; color:#f4f7fb; font-size:clamp(2.35rem,4.4vw,4.35rem); font-weight:760; line-height:1.04; letter-spacing:0; }
    .hero-title span { color:#6ce1b6; }
    .hero-copy { max-width:590px; margin:0; color:#a5b7cc; font-size:1.04rem; line-height:1.75; }
    .hero-actions { display:flex; flex-wrap:wrap; align-items:center; gap:18px; margin-top:25px; }
    .hero-actions .button-primary { color:#06231c; padding:12px 16px; }
    .hero-actions .button-secondary { display:inline-flex; align-items:center; gap:8px; }
    .proof-line { margin-top:25px; color:#7f93aa; font: .7rem/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .preview-panel, .exposure-panel, .science-panel { background:linear-gradient(145deg,rgba(23,42,65,.9),rgba(13,27,47,.96)); border:1px solid rgba(177,198,224,.17); border-radius:12px; box-shadow:0 24px 70px rgba(0,0,0,.24); }
    .preview-panel { padding:21px; }
    .preview-top, .preview-foot { display:flex; justify-content:space-between; align-items:center; gap:12px; color:#8398b1; font: .66rem/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.07em; text-transform:uppercase; }
    .preview-live { color:#6ce1b6; white-space:nowrap; }
    .preview-title { margin:20px 0 3px; color:#f1f5f9; font-size:1.04rem; font-weight:750; }
    .preview-caption { color:#8498b0; font: .64rem ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.07em; }
    .preview-metrics { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; margin:16px 0; }
    .preview-metric { min-width:0; padding:12px; background:rgba(7,20,38,.48); border:1px solid rgba(177,198,224,.11); border-radius:7px; }
    .preview-metric-label { color:#a5b5c8; font-size:.69rem; font-weight:650; }
    .preview-metric-value { margin-top:5px; color:#f3f6fa; font:700 1.08rem ui-monospace, SFMono-Regular, Consolas, monospace; font-variant-numeric:tabular-nums; }
    .preview-metric-value.loss { color:#ff817c; }
    .preview-signals-title { margin:15px 0 7px; color:#d7e0ea; font-size:.68rem; font-weight:750; }
    .preview-signal { display:flex; justify-content:space-between; align-items:baseline; gap:12px; border-top:1px solid rgba(177,198,224,.1); padding:9px 0; color:#aab9cb; font-size:.7rem; }
    .preview-signal strong { color:#f1f5f9; font:650 .7rem ui-monospace, SFMono-Regular, Consolas, monospace; text-align:right; }
    .preview-foot { border-top:1px solid rgba(177,198,224,.12); margin-top:7px; padding-top:12px; text-transform:none; letter-spacing:0; }
    .section-wrap { padding:28px 0 45px; border-top:1px solid var(--lf-line); scroll-margin-top:20px; }
    .section-heading { margin:8px 0 7px; color:#f1f5f9; font-size:1.8rem; line-height:1.2; font-weight:740; }
    .section-intro { max-width:720px; margin:0; color:#9aadc3; font-size:.92rem; line-height:1.7; }
    .exposure-layout { display:grid; grid-template-columns:minmax(0,1.25fr) minmax(240px,.75fr); gap:14px; margin-top:17px; }
    .exposure-panel { padding:var(--lf-card-padding); }
    .exposure-panel h3 { margin:0 0 13px; font-size:.88rem; }
    .exposure-big { color:#f3f6fa; font-size:1.08rem; line-height:1.65; }
    .exposure-big strong { color:#6ce1b6; font-variant-numeric:tabular-nums; }
    .revenue-track { display:flex; height:10px; margin:18px 0 9px; background:rgba(148,163,184,.15); border-radius:99px; overflow:hidden; }
    .revenue-realized { display:block; height:100%; background:var(--lf-green); }
    .revenue-unrealized { display:block; height:100%; background:var(--lf-red); }
    .revenue-legend { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 16px; color:#aab9cb; font:.65rem/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .revenue-legend span { display:flex; align-items:center; gap:7px; min-width:0; }
    .revenue-swatch { flex:0 0 8px; width:8px; height:8px; border-radius:2px; }
    .revenue-swatch.realized { background:var(--lf-green); }
    .revenue-swatch.unrealized { background:var(--lf-red); }
    .exposure-side { display:grid; gap:12px; }
    .exposure-side .metric-card { padding:var(--lf-card-padding); }
    .exposure-side .metric-title { color:#a8b8cc; text-transform:none; letter-spacing:0; font-size:.82rem; }
    .exposure-side .metric-sub { color:var(--lf-muted) !important; }
    .module-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:19px; }
    .module-card { min-width:0; min-height:125px; padding:15px; background:linear-gradient(145deg,rgba(22,40,62,.88),rgba(14,25,43,.9)); border:1px solid var(--lf-line); border-radius:8px; }
    .module-number { color:#63d9b0; font: .63rem ui-monospace, SFMono-Regular, Consolas, monospace; }
    .module-title { margin:10px 0 5px; color:#eaf0f6; font-size:.84rem; font-weight:750; }
    .module-copy { color:#91a4ba; font-size:.73rem; line-height:1.55; }
    .science-panel { margin:8px 0 34px; padding:22px 24px; border-color:rgba(16,185,129,.25); background:linear-gradient(110deg,rgba(16,75,65,.28),rgba(17,37,60,.86) 55%,rgba(20,41,68,.75)); }
    .science-label { color:#65ddb4; font:700 .62rem ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.08em; text-transform:uppercase; }
    .science-copy { margin:11px 0 0; color:#e1e9f1; font-size:.97rem; line-height:1.7; }
    .science-copy strong { color:#74e3bd; }
    .science-foot { margin-top:12px; color:#91a5bb; font: .62rem ui-monospace, SFMono-Regular, Consolas, monospace; }
    .site-footer { display:flex; justify-content:space-between; gap:18px; padding:20px 0 8px; border-top:1px solid var(--lf-line); color:#9aadc3; font-size:.75rem; }
    .site-footer strong { color:#b8c6d5; }
    .stApp [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4)):has(.metric-card) {
        display:grid !important;
        grid-template-columns:repeat(4,minmax(0,1fr)) !important;
        gap:16px !important;
    }
    .stApp [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4)):has(.metric-card) > [data-testid="stColumn"] {
        width:auto !important;
        min-width:0 !important;
        flex:initial !important;
    }
    .stApp [data-testid="stHorizontalBlock"]:has(.exposure-panel) {
        display:grid !important;
        grid-template-columns:repeat(4,minmax(0,1fr)) !important;
        column-gap:16px !important;
        align-items:stretch;
        margin-top:16px !important;
    }
    .stApp [data-testid="stHorizontalBlock"]:has(.exposure-panel) > [data-testid="stColumn"] {
        width:auto !important;
        min-width:0 !important;
        flex:initial !important;
    }
    .stApp [data-testid="stHorizontalBlock"]:has(.exposure-panel) > [data-testid="stColumn"]:first-child { grid-column:1 / span 3; }
    .stApp [data-testid="stHorizontalBlock"]:has(.exposure-panel) > [data-testid="stColumn"]:last-child { grid-column:4; }
    @media (max-width: 1100px) {
        .stApp [data-testid="stHorizontalBlock"]:has(.hero-title),
        .stApp [data-testid="stHorizontalBlock"]:has(.exposure-panel) {
            display:flex !important;
            flex-direction:column !important;
            align-items:stretch !important;
            gap:18px !important;
        }
        .stApp [data-testid="stHorizontalBlock"]:has(.hero-title) > [data-testid="stColumn"],
        .stApp [data-testid="stHorizontalBlock"]:has(.exposure-panel) > [data-testid="stColumn"] {
            width:100% !important;
            min-width:0 !important;
            flex:1 1 auto !important;
            grid-column:auto !important;
        }
        .stApp [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4)):has(.metric-card) {
            display:grid !important;
            grid-template-columns:repeat(2,minmax(0,1fr)) !important;
            column-gap:10px !important;
            row-gap:26px !important;
        }
        .stApp [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4)):has(.metric-card) > [data-testid="stColumn"] {
            width:auto !important;
            min-width:0 !important;
            flex:initial !important;
        }
        .stApp [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4)) .metric-card {
            height:auto;
        }
        .nav-links { gap:12px; }
        .nav-links a:not(.nav-cta) { display:none; }
        .nav-links .nav-cta { white-space:nowrap; }
        .preview-panel { max-width:620px; }
        .module-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    @media (max-width: 600px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .stTabs [data-baseweb="tab-list"] { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .site-nav { align-items:flex-start; }
        .nav-links { gap:12px; }
        .nav-links a:not(.nav-cta) { display:none; }
        .brand-lockup { min-width:0; font-size:.8rem; }
        .brand-edition { font-size:.64rem; }
        .hero-title { font-size:2.4rem; }
        .preview-panel { padding:15px; }
        .exposure-layout { grid-template-columns:minmax(0,1fr); }
        .module-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
        .module-card { padding:12px; }
        .site-footer { flex-direction:column; }
    }
</style>
""", unsafe_allow_html=True)

# LeakFinder reference styling for the data-backed dashboard components.
st.markdown("""
<style>
    .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; white-space: nowrap; }
    .badge-high { background: rgba(239,68,68,0.14); color: #ffaaa4; border: 1px solid rgba(239,68,68,0.4); }
    .badge-medium { background: rgba(245,158,11,0.13); color: #f6cd7b; border: 1px solid rgba(245,158,11,0.4); }
    .badge-low { background: rgba(148,163,184,0.1); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.28); }
    .alert-head { display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; }
    .alert-title { font-weight: 700; font-size: .95rem; color: #f1f5f9; }
    .alert-rank { color: #8295aa; font-weight: 700; margin-right: 8px; }
    .alert-amt { font: 750 1.2rem ui-monospace, SFMono-Regular, Consolas, monospace; color: #ffffff; }
    .alert-basis { color: #91a3b8; font-size: 0.7rem; font-weight: 500; margin-left: 6px; }
    .alert-body { color: #b4c2d2; font-size: 0.79rem; margin-top: 7px; line-height: 1.55; }
    .banner { background: rgba(16,185,129,0.07); border: 1px solid rgba(16,185,129,0.25); border-radius: 9px; padding: 15px 17px; color: #dbe6ef; line-height: 1.6; margin: 6px 0 14px 0; }
    .banner-warn { background: rgba(245,158,11,0.07); border-color: rgba(245,158,11,0.3); }
    .banner-alert { background: rgba(239,68,68,0.07); border-color: rgba(239,68,68,0.3); }
    .banner b { color: #f7fafc; }
    .note { color: #91a3b8; font-size: 0.76rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------
# Optional backend modules (safe fallbacks if they are missing)
# ---------------------------------------------------------------------------------------------
try:
    import ml_scratch as ms  # from-scratch NumPy models
    import pipeline as pl    # Preprocessor (one-hot, median impute, scaling)
    BACKEND_OK = True
except Exception:  # pragma: no cover - depends on the user's folder
    ms, pl, BACKEND_OK = None, None, False

SKY, GREEN, RED, AMBER = "#38bdf8", "#10b981", "#ef4444", "#f59e0b"
SLATE, VIOLET = "#94a3b8", "#a78bfa"
STATUS_COLORS = {"Completed": GREEN, "Cancelled": RED, "Returned": AMBER, "Pending": SKY}
REQUIRED = ["customers", "products", "transactions", "inventory", "campaigns"]
STALE_DAYS = 30


# ---------------------------------------------------------------------------------------------
# Formatting + UI helpers
# ---------------------------------------------------------------------------------------------
def money(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s, a = ("-" if x < 0 else ""), abs(float(x))
    if a >= 1e6:
        return f"{s}${a / 1e6:.{d}f}M"
    if a >= 1e5:
        return f"{s}${a / 1e3:.0f}K"
    if a >= 1e3:
        return f"{s}${a / 1e3:.1f}K"
    return f"{s}${a:,.0f}"


def metric_card(title: str, value: str, sub: str = "", tone: str = "text-success") -> str:
    return (f'<div class="metric-card"><div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-sub {tone}">{sub}</div></div>')


def badge(conf: str) -> str:
    return f'<span class="badge badge-{conf.lower()}">{conf.upper()} CONFIDENCE</span>'


def banner(html: str, kind: str = "") -> None:
    st.markdown(f'<div class="banner {kind}">{html}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, title: str | None = None, height: int = 400) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter, system-ui, sans-serif", size=12),
        colorway=[SKY, GREEN, RED, AMBER, VIOLET, SLATE],
        margin=dict(l=10, r=10, t=56 if title else 24, b=10), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f1f5f9"),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color="#f1f5f9"), x=0.0))
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)", linecolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)", linecolor="rgba(148,163,184,0.25)")
    return fig


def plot(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, height: int | None = None) -> None:
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------------------------
# Data layer: load, clean, join
# ---------------------------------------------------------------------------------------------
def find_data_dir() -> Path | None:
    here = Path(__file__).resolve().parent
    for c in [here / "data", here, Path.cwd() / "data", Path.cwd(), here / "Datathon-MoneyLeak" / "data"]:
        if all((c / f"{n}.csv").exists() for n in REQUIRED):
            return c
    return None


@st.cache_data(show_spinner="Loading and cleaning the 5 data tables...")
def load_data(folder: str) -> dict:
    base = Path(folder)
    raw = {n: pd.read_csv(base / f"{n}.csv") for n in REQUIRED}
    for df in raw.values():
        for c in df.columns:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].str.strip()
    cu, pr, tx, inv, cp = (raw[n] for n in REQUIRED)
    log: list[tuple[str, int]] = []

    cu["signup_date"] = pd.to_datetime(cu["signup_date"], errors="coerce")
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    cp["date"] = pd.to_datetime(cp["date"], errors="coerce")
    for df, cols in [(tx, ["quantity", "unit_price", "discount_pct"]), (pr, ["unit_cost", "list_price", "supplier_lead_days"]),
                     (inv, ["stock_on_hand", "reorder_threshold", "stockout_flag", "supplier_delay_days"]),
                     (cp, ["spend", "impressions", "clicks", "conversions"])]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(cu); cu = cu.drop_duplicates("customer_id"); log.append(("Duplicate customers removed", n0 - len(cu)))
    n0 = len(pr); pr = pr.drop_duplicates("product_id"); log.append(("Duplicate products removed", n0 - len(pr)))
    n0 = len(tx); tx = tx.drop_duplicates("transaction_id"); log.append(("Duplicate transaction IDs removed", n0 - len(tx)))
    bad = tx.timestamp.isna() | tx.quantity.isna() | tx.unit_price.isna() | (tx.quantity <= 0) | (tx.unit_price < 0)
    log.append(("Transactions dropped (bad date, quantity or price)", int(bad.sum()))); tx = tx[~bad].copy()
    tx["status"] = tx["status"].str.title()

    t = tx.merge(pr, on="product_id", how="inner")
    log.append(("Transactions dropped (unknown product)", len(tx) - len(t)))
    t = t.merge(cu[["customer_id", "region", "customer_segment", "signup_date", "preferred_channel"]], on="customer_id", how="left")
    for c in ["region", "customer_segment", "preferred_channel"]:
        t[c] = t[c].fillna("Unknown")

    as_of = t["timestamp"].max()
    t["revenue"] = t.quantity * t.unit_price
    t["cost"] = t.quantity * t.unit_cost
    t["margin"] = t.revenue - t.cost
    t["margin_pct"] = np.where(t.revenue > 0, t.margin / t.revenue.replace(0, np.nan) * 100, 0.0)
    t["is_completed"] = (t.status == "Completed")
    t["is_cancelled"] = (t.status == "Cancelled")
    t["is_returned"] = (t.status == "Returned")
    t["failed"] = (t.is_cancelled | t.is_returned).astype(int)
    t["real_rev"] = np.where(t.is_completed, t.revenue, 0.0)
    t["real_margin"] = np.where(t.is_completed, t.margin, 0.0)
    t["list_below_cost"] = t.list_price < t.unit_cost
    t["date"] = t.timestamp.dt.normalize()
    t["month"] = t.timestamp.dt.to_period("M").astype(str)
    t["hour"] = t.timestamp.dt.hour
    t["dow"] = t.timestamp.dt.dayofweek
    t["tenure_days"] = (t.timestamp - t.signup_date).dt.days.fillna(0)
    t["age_days"] = (as_of - t.timestamp).dt.days
    t["stale"] = (t.status == "Pending") & (t.age_days > STALE_DAYS)

    cp = cp.copy()
    cp["valid"] = ((cp.clicks <= cp.impressions) & (cp.conversions <= cp.clicks) & (cp.conversions >= 0)).fillna(False)
    return dict(fact=t, products=pr, customers=cu, inventory=inv, campaigns=cp, as_of=as_of, log=log, folder=str(base))


# ---------------------------------------------------------------------------------------------
# Business calculations
# ---------------------------------------------------------------------------------------------
def compute_kpis(F: pd.DataFrame) -> dict:
    comp = F[F.is_completed]
    k = dict(orders=len(F), booked=F.revenue.sum(), realized=comp.revenue.sum(),
             cancelled=F[F.is_cancelled].revenue.sum(), returned=F[F.is_returned].revenue.sum(),
             pending=F[F.status == "Pending"].revenue.sum(), realized_margin=comp.margin.sum())
    k["net_leak"] = k["booked"] - k["realized"]
    k["failed_rev"] = k["cancelled"] + k["returned"]
    k["failed_orders"] = int(F.failed.sum())
    denom = k["realized"] + k["failed_rev"]
    k["completion_rate"] = k["realized"] / denom if denom > 0 else 0.0
    st_ = F[F.stale]
    k["stuck_n"], k["stuck_value"] = len(st_), st_.revenue.sum()
    k["stuck_recoverable"] = k["stuck_value"] * k["completion_rate"]
    k["price_loss"] = -comp[comp.list_below_cost].margin.sum()
    k["below_cost_lines"] = int((comp.margin < 0).sum())
    k["margin_pct"] = k["realized_margin"] / k["realized"] if k["realized"] > 0 else 0.0
    return k


def product_table(P: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    comp = F[F.is_completed]
    g = comp.groupby("product_id").agg(units=("quantity", "sum"), revenue=("revenue", "sum"), margin=("margin", "sum"))
    p = P.merge(g, on="product_id", how="left").fillna({"units": 0, "revenue": 0, "margin": 0})
    p["list_margin_pct"] = (p.list_price - p.unit_cost) / p.list_price * 100
    p["below_cost"] = p.list_price < p.unit_cost
    return p


def reprice_whatif(F: pd.DataFrame, target: float, vloss: float) -> tuple[float, pd.DataFrame]:
    """Lift every completed line priced under the floor cost/(1-target) up to the floor, then apply a volume haircut.
    Gain = new margin - old margin on the affected lines."""
    comp = F[F.is_completed]
    floor = comp.unit_cost.values / (1 - target)
    new_price = np.maximum(comp.unit_price.values, floor)
    aff = new_price > comp.unit_price.values + 1e-9
    new_margin = (new_price - comp.unit_cost.values) * comp.quantity.values * (1 - vloss)
    gain = np.where(aff, new_margin - comp.margin.values, 0.0)
    by = comp.assign(gain=gain, affected=aff).groupby("product_id").agg(gain=("gain", "sum"), affected_lines=("affected", "sum")).reset_index()
    return float(gain.sum()), by


def _pval(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def segment_table(G: pd.DataFrame, dim: str) -> pd.DataFrame:
    base = G.failed.mean()
    g = G.groupby(dim).agg(orders=("transaction_id", "size"), gross=("revenue", "sum"), cancelled=("is_cancelled", "sum"),
                           returned=("is_returned", "sum"), failed=("failed", "sum"), real_rev=("real_rev", "sum"),
                           net_margin=("real_margin", "sum")).reset_index()
    g["cancel_pct"] = g.cancelled / g.orders * 100
    g["return_pct"] = g.returned / g.orders * 100
    g["net_margin_pct"] = np.where(g.real_rev > 0, g.net_margin / g.real_rev.replace(0, np.nan) * 100, np.nan)
    se = np.sqrt(max(base * (1 - base), 1e-12) / g.orders)
    g["z"] = (g.failed / g.orders - base) / se
    g["p"] = [_pval(v) for v in g.z]
    g["verdict"] = np.where(g.p < 0.05 / max(1, len(g)), "Real signal", "Normal noise")
    return g


def driver_scan(G: pd.DataFrame, dims: dict[str, str]) -> pd.DataFrame:
    rows = []
    for label, col in dims.items():
        s = segment_table(G, col)
        top = s.sort_values("p").iloc[0]
        thr = 0.05 / (len(dims) * max(1, len(s)))
        rows.append(dict(Driver=label, **{"Most unusual group": str(top[col]), "Orders": int(top.orders),
                                         "Failure rate %": round(top.failed / top.orders * 100, 1), "z-score": round(top.z, 2),
                                         "p-value": round(top.p, 4), "Verdict": "Real signal" if top.p < thr else "Normal noise"}))
    return pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)


def build_stock(inv: pd.DataFrame, P: pd.DataFrame, fact: pd.DataFrame, as_of: pd.Timestamp, overstock_days: int) -> pd.DataFrame:
    sort_cols = ["date"] + (["inventory_record_id"] if "inventory_record_id" in inv.columns else [])
    last = inv.sort_values(sort_cols, kind="stable").groupby("product_id").tail(1).merge(P, on="product_id", how="left")
    recent = fact[(fact.timestamp > as_of - pd.Timedelta(days=90)) & (~fact.is_cancelled)]
    rate = (recent.groupby("product_id").quantity.sum() / 90).rename("daily_demand")
    last = last.merge(rate, on="product_id", how="left")
    last["daily_demand"] = last.daily_demand.fillna(0.0)
    last["days_remaining"] = last.stock_on_hand / last.daily_demand.clip(lower=0.01)
    last["restock_days"] = last.supplier_lead_days + last.supplier_delay_days
    below = last.stock_on_hand <= last.reorder_threshold
    short = last.days_remaining < last.restock_days
    last["risk"] = np.where(below & short, "High", np.where(below | short, "Medium", "Low"))
    last["revenue_at_risk"] = (last.restock_days - last.days_remaining).clip(lower=0) * last.daily_demand * last.list_price
    last["excess_units"] = (last.stock_on_hand - last.daily_demand * overstock_days).clip(lower=0)
    last["overstock_capital"] = last.excess_units * last.unit_cost
    last["why_flagged"] = [
        "; ".join(([f"stock {int(s)} <= reorder point {int(r)}"] if b else []) + ([f"covers {d:.0f}d but restock takes {l:.0f}d"] if sh else [])) or "healthy"
        for s, r, b, d, l, sh in zip(last.stock_on_hand, last.reorder_threshold, below, last.days_remaining, last.restock_days, short)]
    return last


def campaign_stats(C: pd.DataFrame) -> dict:
    v = C[C.valid]
    conv = v.conversions.sum()
    bench = v.spend.sum() / conv if conv > 0 else float("nan")
    excess = float(np.maximum(0, v.spend - v.conversions * bench).sum()) if conv > 0 else 0.0
    r = float(v.spend.corr(v.conversions)) if len(v) > 2 else float("nan")
    return dict(r=r, total_spend=float(C.spend.sum()), invalid_rows=int((~C.valid).sum()), invalid_spend=float(C[~C.valid].spend.sum()),
                benchmark_cpa=bench, excess=excess, valid_rows=len(v), total_conv=float(C.conversions.sum()))


def channel_economics(C: pd.DataFrame, customers: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    v = C[C.valid]
    ch = v.groupby("channel").agg(spend=("spend", "sum"), conversions=("conversions", "sum")).reset_index()
    ch["cac"] = ch.spend / ch.conversions.replace(0, np.nan)
    lm = fact[fact.is_completed].groupby("customer_id").margin.sum()
    cu = customers[["customer_id", "preferred_channel"]].copy()
    cu["lifetime_margin"] = cu.customer_id.map(lm).fillna(0.0)
    lmc = cu.groupby("preferred_channel").agg(customers=("customer_id", "size"), lifetime_margin=("lifetime_margin", "mean")).reset_index()
    out = ch.merge(lmc, left_on="channel", right_on="preferred_channel", how="left").drop(columns="preferred_channel")
    out["margin_to_cac"] = out.lifetime_margin / out.cac
    return out.sort_values("cac").reset_index(drop=True)


def daily_series(F: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "Realized revenue":
        s = F[F.is_completed].groupby("date").revenue.sum()
    else:
        s = F.groupby("date").size().astype(float)
    return s.reindex(pd.date_range(F.date.min(), F.date.max(), freq="D"), fill_value=0.0)


def project_series(s: pd.Series, horizon: int = 30, holdout: int = 30) -> tuple[pd.DataFrame, str, dict]:
    y = s.values.astype(float); n = len(y)
    flat = lambda yy: float(np.mean(yy[-28:]))
    stats = dict(mae_trend=float("nan"), mae_flat=float("nan"))
    use_trend = False
    if n >= holdout + 60:
        tr, te = y[:-holdout], y[-holdout:]
        coef = np.polyfit(np.arange(len(tr)), tr, 1)
        stats["mae_trend"] = float(np.mean(np.abs(te - np.polyval(coef, np.arange(len(tr), n)))))
        stats["mae_flat"] = float(np.mean(np.abs(te - flat(tr))))
        use_trend = stats["mae_trend"] < stats["mae_flat"]
    if use_trend:
        coef = np.polyfit(np.arange(n), y, 1)
        center = np.polyval(coef, np.arange(n, n + horizon)); sd = float(np.std(y - np.polyval(coef, np.arange(n))))
        method = "Linear trend (it beat the flat baseline on the last 30 unseen days)"
    else:
        center = np.full(horizon, flat(y)); sd = float(np.std(y[-90:]))
        method = "Flat baseline: mean of the last 28 days (a trend line did not beat it on unseen days)"
    dates = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fut = pd.DataFrame(dict(date=dates, forecast=np.clip(center, 0, None), low=np.clip(center - 1.28 * sd, 0, None), high=center + 1.28 * sd))
    return fut, method, stats


# ---------------------------------------------------------------------------------------------
# ML Lab (time-aware split: Jan-Sep train, Oct-Dec test). Uses ml_scratch/pipeline if present.
# ---------------------------------------------------------------------------------------------
class _Pre:
    """Fallback preprocessor: median impute, one-hot (<=15 levels), standardise."""
    def fit(self, df):
        self.num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.cat = [c for c in df.columns if c not in self.num and df[c].nunique() <= 15]
        self.med = {c: df[c].median() for c in self.num}
        self.levels = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.cat}
        raw = self._raw(df); self.mu, self.sd = raw.mean(0), raw.std(0) + 1e-9
        return self

    def _raw(self, df):
        parts = [df[c].fillna(self.med[c]).astype(float).values[:, None] for c in self.num]
        for c in self.cat:
            v = df[c].astype(str).values
            parts += [(v == lv).astype(float)[:, None] for lv in self.levels[c]]
        return np.hstack(parts)

    def transform(self, df):
        return (self._raw(df) - self.mu) / self.sd


class _Tree:
    def __init__(self, depth=4, min_leaf=25, n_thr=10, max_feats=None, seed=0):
        self.depth, self.min_leaf, self.n_thr, self.max_feats, self.rng = depth, min_leaf, n_thr, max_feats, np.random.default_rng(seed)

    def fit(self, X, y):
        self.root = self._grow(X, y, 0); return self

    def _grow(self, X, y, d):
        if d >= self.depth or len(y) < 2 * self.min_leaf or y.std() < 1e-9:
            return float(y.mean())
        n, p = X.shape
        feats = self.rng.choice(p, self.max_feats, replace=False) if self.max_feats and self.max_feats < p else range(p)
        base = ((y - y.mean()) ** 2).sum(); best = (0.0, None, None)
        for j in feats:
            col = X[:, j]
            for th in np.unique(np.quantile(col, np.linspace(0.1, 0.9, self.n_thr))):
                left = col <= th; nl = int(left.sum())
                if nl < self.min_leaf or n - nl < self.min_leaf:
                    continue
                yl, yr = y[left], y[~left]
                gain = base - ((yl - yl.mean()) ** 2).sum() - ((yr - yr.mean()) ** 2).sum()
                if gain > best[0]:
                    best = (gain, j, th)
        if best[1] is None:
            return float(y.mean())
        _, j, th = best; left = X[:, j] <= th
        return (j, th, self._grow(X[left], y[left], d + 1), self._grow(X[~left], y[~left], d + 1))

    def predict(self, X):
        out = np.empty(len(X))
        def walk(node, idx):
            if not isinstance(node, tuple):
                out[idx] = node; return
            j, th, l, r = node; m = X[idx, j] <= th
            walk(l, idx[m]); walk(r, idx[~m])
        walk(self.root, np.arange(len(X))); return out


class _LogReg:
    def fit(self, X, y, lr=0.1, epochs=80, l2=1e-3):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _ in range(epochs):
            g = 1 / (1 + np.exp(-np.clip(X @ self.w + self.b, -30, 30))) - y
            self.w -= lr * (X.T @ g / len(y) + l2 * self.w); self.b -= lr * g.mean()
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.clip(X @ self.w + self.b, -30, 30))); return np.c_[1 - p, p]


class _Forest:
    def __init__(self, n=15, depth=5, seed=0):
        self.n, self.depth, self.seed = n, depth, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); mf = max(2, int(np.sqrt(X.shape[1])))
        self.trees = []
        for i in range(self.n):
            idx = rng.integers(0, len(y), len(y)); self.trees.append(_Tree(self.depth, max_feats=mf, seed=i).fit(X[idx], y[idx].astype(float)))
        return self

    def predict_proba(self, X):
        p = np.mean([t.predict(X) for t in self.trees], axis=0); return np.c_[1 - p, p]


class _Boost:
    def __init__(self, n=30, lr=0.3, depth=3):
        self.n, self.lr, self.depth = n, lr, depth

    def fit(self, X, y):
        m = float(np.clip(y.mean(), 1e-4, 1 - 1e-4)); self.f0 = math.log(m / (1 - m)); F = np.full(len(y), self.f0); self.trees = []
        for _ in range(self.n):
            p = 1 / (1 + np.exp(-F)); t = _Tree(self.depth, max_feats=max(2, int(np.sqrt(X.shape[1])))).fit(X, y - p)
            F += self.lr * t.predict(X); self.trees.append(t)
        return self

    def predict_proba(self, X):
        F = np.full(len(X), self.f0)
        for t in self.trees:
            F += self.lr * t.predict(X)
        p = 1 / (1 + np.exp(-F)); return np.c_[1 - p, p]


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int); pos = int(y.sum()); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = pd.Series(s).rank().values
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def roc_points(y: np.ndarray, s: np.ndarray, max_pts: int = 300) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int); o = np.argsort(-s, kind="mergesort"); ys = y[o]
    tpr = np.r_[0, np.cumsum(ys) / max(1, ys.sum())]; fpr = np.r_[0, np.cumsum(1 - ys) / max(1, (1 - ys).sum())]
    if len(fpr) > max_pts:
        keep = np.linspace(0, len(fpr) - 1, max_pts).astype(int); fpr, tpr = fpr[keep], tpr[keep]
    return fpr, tpr


def top_decile_lift(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y); k = max(1, len(y) // 10); top = y[np.argsort(-s)[:k]]
    base = y.mean(); return float(top.mean() / base) if base > 0 else float("nan")


@st.cache_data(show_spinner="Training logistic regression, random forest and gradient boosting on Jan-Sep orders...")
def run_ml_lab(folder: str) -> dict:
    fact = load_data(folder)["fact"]
    feat = ["quantity", "unit_price", "discount_pct", "list_below_cost", "category", "region", "customer_segment", "channel",
            "payment_method", "tenure_days", "hour", "dow", "supplier_lead_days"]
    year = int(fact.timestamp.max().year)
    df = fact[feat + ["month", "failed"]].copy(); df["list_below_cost"] = df.list_below_cost.astype(int)
    tr, te = df[df.month <= f"{year}-09"], df[df.month > f"{year}-09"]
    pre = (pl.Preprocessor() if BACKEND_OK else _Pre()).fit(tr[feat])
    Xtr, Xte = pre.transform(tr[feat]), pre.transform(te[feat]); ytr, yte = tr.failed.values.astype(int), te.failed.values.astype(int)
    sub = np.random.default_rng(0).choice(len(Xtr), min(len(Xtr), 6000), replace=False)
    if BACKEND_OK:
        models = {"Logistic Regression": (ms.LogisticRegression(lr=0.1, epochs=60), False),
                  "Random Forest": (ms.RandomForest("classification", 25, 6), True),
                  "Gradient Boosting": (ms.GradientBoosting("classification", 40, 0.1, 3), True)}
    else:
        models = {"Logistic Regression": (_LogReg(), False), "Random Forest": (_Forest(), True), "Gradient Boosting": (_Boost(), True)}
    base_acc = float(max(yte.mean(), 1 - yte.mean())); rows, curves = [], {}
    for name, (mdl, use_sub) in models.items():
        t0 = time.time()
        mdl.fit(Xtr[sub] if use_sub else Xtr, ytr[sub] if use_sub else ytr)
        secs = time.time() - t0
        score = np.asarray(mdl.predict_proba(Xte))[:, 1]
        rows.append({"Model": name, "Test AUC": round(auc_score(yte, score), 3), "Accuracy": round(float(np.mean((score > 0.5).astype(int) == yte)), 3),
                     "Majority-class baseline": round(base_acc, 3), "Top-decile lift": round(top_decile_lift(yte, score), 2),
                     "Train seconds": round(secs, 1)})
        curves[name] = roc_points(yte, score)
    return dict(matrix=pd.DataFrame(rows), curves=curves, train_n=len(tr), test_n=len(te), fail_rate=float(yte.mean()),
                engine="from-scratch ml_scratch.py + pipeline.py" if BACKEND_OK else "built-in NumPy fallback models")


# =============================================================================================
# APP
# =============================================================================================
DIMS = {"Category": "category", "Region": "region", "Customer segment": "customer_segment", "Channel": "channel",
        "Payment method": "payment_method", "Discount %": "discount_pct", "Quantity": "quantity", "Month": "month",
        "Hour of day": "hour", "Weekday (0 = Mon)": "dow"}
BREAKDOWN = {"Category": "category", "Region": "region", "Customer segment": "customer_segment",
             "Payment method": "payment_method", "Order channel": "channel", "Month": "month"}


def leak_card(rank: int, name: str, amt: float, basis: str, conf: str, why: str) -> str:
    edge = {"HIGH": RED, "MEDIUM": AMBER, "LOW": SLATE}[conf]
    bg = {"HIGH": "rgba(239,68,68,0.10)", "MEDIUM": "rgba(245,158,11,0.09)", "LOW": "rgba(148,163,184,0.08)"}[conf]
    return (f'<div class="alert-card" style="border-left-color:{edge};background:{bg};">'
            f'<div class="alert-head"><div><span class="alert-rank">#{rank}</span><span class="alert-title">{name}</span></div>'
            f'<div><span class="alert-amt">{money(amt)}</span><span class="alert-basis">{basis}</span></div></div>'
            f'<div class="alert-body">{badge(conf)}&nbsp; {why}</div></div>')


data_dir = find_data_dir()
if data_dir is None:
    st.error("Could not find customers.csv, products.csv, transactions.csv, inventory.csv and campaigns.csv. "
             "Put them in a `data/` folder next to app.py.")
    st.stop()

D = load_data(str(data_dir))
FACT, PRODUCTS, CUSTOMERS, INV, CAMP, AS_OF = D["fact"], D["products"], D["customers"], D["inventory"], D["campaigns"], D["as_of"]

# ---------------------------------------------------------------- sidebar filters
with st.sidebar:
    st.markdown("### 🚨 LeakFinder")
    st.caption("Filters apply to Command Center, Investigate, Pricing, Forecast and Action Plan downloads.")
    dmin, dmax = FACT.date.min().date(), FACT.date.max().date()
    dr = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(dr, (list, tuple)):
        d0, d1 = dr[0], (dr[1] if len(dr) > 1 else dr[0])
    else:
        d0 = d1 = dr
    all_regions = sorted(FACT.region.unique())
    regions = st.multiselect("Region", all_regions, default=all_regions)
    st.caption(f"{len(regions)} regions selected")
    m_opts = [-3000, -100, -50, -25, 0, 10, 25, 50, 100]
    mlo, mhi = st.select_slider("Order-line margin % range", options=m_opts, value=(m_opts[0], m_opts[-1]),
                                format_func=lambda v: f"{v}%",
                                help="Keep only order lines whose margin % falls inside this range. The full range keeps every order.")
    st.caption("Region and margin filters do not apply to campaigns and inventory (they have no region or order margin). "
               "The date filter also applies to campaigns.")

mask = ((FACT.date >= pd.Timestamp(d0)) & (FACT.date <= pd.Timestamp(d1)) & FACT.region.isin(regions)
        & FACT.margin_pct.between(mlo, mhi))
F = FACT[mask]
CF = CAMP[(CAMP.date >= pd.Timestamp(d0)) & (CAMP.date <= pd.Timestamp(d1))]
with st.sidebar:
    st.markdown(f"**{len(F):,}** of {len(FACT):,} orders in view")
if F.empty:
    st.warning("No orders match the current filters. Widen the date range, regions or margin range in the sidebar.")
    st.stop()
FILTERED = len(F) != len(FACT)

# ---------------------------------------------------------------- shared calculations
K = compute_kpis(F)
PT = product_table(PRODUCTS, F)
N_BELOW = int((PRODUCTS.list_price < PRODUCTS.unit_cost).sum())
S = build_stock(INV, PRODUCTS, FACT, AS_OF, int(st.session_state.get("overstock_days", 180)))
HIGH = S[S.risk == "High"].sort_values("revenue_at_risk", ascending=False)
CAMP_STATS = campaign_stats(CF)
SCAN = driver_scan(F, DIMS)
N_REAL = int((SCAN.Verdict == "Real signal").sum())
TM = int(st.session_state.get("tm_pct", 20)); VL = int(st.session_state.get("vl_pct", 15))
GAIN_NOW, BY_PROD = reprice_whatif(F, TM / 100, VL / 100)

failure_pct = K["failed_orders"] / max(K["orders"], 1) * 100
lost_pct = max(0, min(100, (K["booked"] - K["realized"]) / max(K["booked"], 1) * 100))
realized_pct = 100 - lost_pct
preview_metrics = [
    ("Booked revenue", money(K["booked"]), ""),
    ("Realized revenue", money(K["realized"]), ""),
    ("Unrecovered leaks", money(K["net_leak"]), "loss"),
    ("Below-cost pricing loss", f"-{money(K['price_loss'])}", "loss"),
]
preview_metrics_html = "".join(
    f'<div class="preview-metric"><div class="preview-metric-label">{label}</div>'
    f'<div class="preview-metric-value {tone}">{value}</div></div>'
    for label, value, tone in preview_metrics
)

st.markdown(
    '<div class="site-nav">'
    '<div class="brand-lockup"><span class="brand-mark">LF</span><span>LeakFinder <span class="brand-edition">/ W V P S SRIRAJ (23BCE8414)</span></span></div>'
    '<nav class="nav-links" aria-label="Main navigation">'
    '<a href="#exposure">The exposure</a><a href="#capabilities">Capabilities</a>'
    '<a class="nav-cta" href="#command-center">Launch Command Center &#8599;</a></nav></div>',
    unsafe_allow_html=True,
)

hero_copy, preview_col = st.columns([1.08, 0.92], gap="large", vertical_alignment="center")
with hero_copy:
    st.markdown(
        '<div class="eyebrow"><span class="eyebrow-dot"></span>Datathon 2026 &middot; Track 4: The Money Leak</div>'
        '<h1 class="hero-title">Find Where Your Business Loses Money. <span>Prove It.</span> Fix It First.</h1>'
        '<p class="hero-copy">LeakFinder analyzes multi-table enterprise data to uncover negative-margin pricing, stuck pending orders and ad spend leakage&mdash;then gives leaders a clear, evidence-led path to act.</p>'
        '<div class="hero-actions"><a class="button-primary" href="#command-center">Launch Command Center &#8594;</a>'
        '<a class="button-secondary" href="#exposure">Explore the exposure &#8595;</a></div>'
        '<div class="proof-line">LOCAL DATA &nbsp;&middot;&nbsp; INSPECTABLE EVIDENCE &nbsp;&middot;&nbsp; NO EXTERNAL API</div>',
        unsafe_allow_html=True,
    )
with preview_col:
    st.markdown(
        '<section class="preview-panel"><div class="preview-top"><span>LeakFinder / Command Center</span>'
        '<span class="preview-live">&#8226; Live from supplied data</span></div>'
        '<div class="preview-title">Leak exposure overview</div>'
        '<div class="preview-caption">CURRENT FILTERS &middot; COMPUTED LOCALLY</div>'
        f'<div class="preview-metrics">{preview_metrics_html}</div>'
        '<div class="preview-signals-title">Prioritized leak signals</div>'
        f'<div class="preview-signal"><span>Below unit-cost pricing</span><strong>{N_BELOW} / {len(PRODUCTS)} products</strong></div>'
        f'<div class="preview-signal"><span>Orders pending over {STALE_DAYS} days</span><strong>{money(K["stuck_value"])}</strong></div>'
        f'<div class="preview-signal"><span>Cancelled &amp; returned orders</span><strong>{failure_pct:.1f}%</strong></div>'
        '<div class="preview-foot"><span>Live dataset &bull; filters apply</span><span>Evidence over estimates</span></div></section>',
        unsafe_allow_html=True,
    )
if FILTERED:
    st.markdown('<div class="note">Filters are active, so figures can differ from the full-dataset totals.</div>', unsafe_allow_html=True)

st.markdown(
    '<section class="section-wrap" id="exposure"><div class="section-kicker">01 / The exposure</div>'
    '<h2 class="section-heading">Revenue booked is not revenue kept.</h2>'
    '<p class="section-intro">The gap is visible. LeakFinder makes the underlying operational signals easier to investigate and prioritize.</p></section>',
    unsafe_allow_html=True,
)
exposure_metrics = st.columns(4, gap="small")
exposure_metrics[0].markdown(metric_card("Booked revenue", money(K["booked"]), f"{K['orders']:,} orders", "text-warning"), unsafe_allow_html=True)
exposure_metrics[1].markdown(metric_card("Realized revenue", money(K["realized"]), "Completed orders", "text-success"), unsafe_allow_html=True)
exposure_metrics[2].markdown(metric_card("Unrecovered leaks", money(K["net_leak"]), "Cancelled + returned + pending", "text-alert"), unsafe_allow_html=True)
exposure_metrics[3].markdown(metric_card("Below-cost pricing loss", f"-{money(K['price_loss'])}", "Completed order margin lost", "text-alert"), unsafe_allow_html=True)

exposure_left, exposure_right = st.columns([3, 1], gap="small")
with exposure_left:
    st.markdown(
        '<div class="exposure-panel"><div class="metric-title">Revenue bridge</div>'
        f'<h3>{money(K["booked"])} booked. {money(K["realized"])} realized.</h3>'
        '<div class="exposure-big">The distance between them deserves a closer look.</div>'
        f'<div class="revenue-track" role="img" aria-label="Revenue composition: {money(K["realized"])} realized of {money(K["booked"])} booked; {money(K["net_leak"])} not realized">'
        f'<span class="revenue-realized" style="width:{realized_pct:.1f}%" aria-hidden="true"></span>'
        f'<span class="revenue-unrealized" style="width:{lost_pct:.1f}%" aria-hidden="true"></span></div>'
        f'<div class="revenue-legend"><span><i class="revenue-swatch realized"></i>Realized {money(K["realized"])} ({realized_pct:.1f}%)</span>'
        f'<span><i class="revenue-swatch unrealized"></i>Not realized {money(K["net_leak"])} ({lost_pct:.1f}%)</span></div></div>',
        unsafe_allow_html=True,
    )
with exposure_right:
    st.markdown(
        '<div class="exposure-side">'
        + metric_card("Below-cost catalog", f"{N_BELOW} / {len(PRODUCTS)}", "Products priced below unit cost", "")
        + metric_card("Cash sitting in limbo", money(K["stuck_value"]), f"Pending orders over {STALE_DAYS} days", "")
        + '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<section class="section-wrap" id="capabilities"><div class="section-kicker">02 / From signal to action</div>'
    '<h2 class="section-heading">Every leak gets an investigation path.</h2>'
    '<p class="section-intro">A disciplined workflow for finding, validating and operationalizing financial leakage&mdash;not another score to explain away.</p></section>',
    unsafe_allow_html=True,
)
capabilities = [
    ("01", "Command Center", "Ranked leak stream with current exposure and revenue outcomes."),
    ("02", "Multi-Dim Investigate", "Slice cancellation and return patterns across business dimensions."),
    ("03", "Pricing Leak Engine", f"What-if repricing; modeled recovery {money(GAIN_NOW)} at current settings."),
    ("04", "True Stock Risk", f"Stockout exposure checked against reorder points and restock time; {len(HIGH)} critical now."),
    ("05", "Campaign Audit", f"Spend vs. conversions, correlation r = {CAMP_STATS['r']:.2f}."),
    ("06", "Forecast & Anomaly Detection", "Time-aware forecasts with spike and drop alerts."),
    ("07", "Honest ML Lab", "Time-aware evaluation with baseline metrics shown alongside model scores."),
    ("08", "Action Plan", "Prioritized operational queues and CSV exports."),
]
capability_cards = "".join(
    f'<article class="module-card"><div class="module-number">{number}</div>'
    f'<div class="module-title">{title}</div><div class="module-copy">{description}</div></article>'
    for number, title, description in capabilities
)
st.markdown(f'<div class="module-grid">{capability_cards}</div>', unsafe_allow_html=True)

st.markdown(
    '<section class="science-panel"><div class="science-label">Built on transparent data science &nbsp;|&nbsp; Datathon 2026 &middot; Track 4</div>'
    f'<p class="science-copy">LeakFinder tests <strong>{len(DIMS)} possible leakage drivers</strong> and separates statistically meaningful signals from ordinary variation. '
    f'The current scan found <strong>{N_REAL} validated driver signal(s)</strong>; model evaluation is presented with time-aware test results and baseline comparisons.</p>'
    '<div class="science-foot">Inspectable evidence. Honest uncertainty. Actionable findings.</div></section>',
    unsafe_allow_html=True,
)

st.markdown('<div id="command-center"></div>', unsafe_allow_html=True)

tabs = st.tabs(["🚨 Command Center", "🔍 Investigate", "💸 Pricing Leak", "📦 Stock Risk",
                "📢 Campaigns", "📈 Forecast & Alerts", "🧪 ML Lab", "📋 Action Plan"])

# =============================================================================================
# TAB 1: COMMAND CENTER
# =============================================================================================
leaks = [
    dict(name="Cancelled & returned orders", amt=K["failed_rev"], basis="revenue not realized", conf="MEDIUM",
         why=f"{K['failed_orders']:,} orders failed. The driver scan tested {len(SCAN)} factors and found {N_REAL} real signal(s), so there is no safe segment to target yet. Record cancel and return reasons."),
    dict(name="Pricing leak: products sold below cost", amt=K["price_loss"], basis="margin lost on completed orders", conf="HIGH",
         why=f"{N_BELOW} of {len(PRODUCTS)} products list below unit cost, so every unit loses money before any discount. This is arithmetic, not a model."),
    dict(name="Stuck pending orders", amt=K["stuck_value"], basis=f"locked over {STALE_DAYS} days", conf="MEDIUM",
         why=f"{K['stuck_n']:,} orders have been pending for more than {STALE_DAYS} days. About {money(K['stuck_recoverable'])} is recoverable at the {K['completion_rate']:.0%} historical completion rate."),
    dict(name="Marketing spend above benchmark", amt=CAMP_STATS["excess"], basis="upper bound, not guaranteed savings", conf="LOW",
         why=f"Ad spend and conversions are uncorrelated (r = {CAMP_STATS['r']:.2f}) and {CAMP_STATS['invalid_rows']} campaign rows are impossible. Audit tracking before moving budget."),
    dict(name="Stockout exposure", amt=float(HIGH.revenue_at_risk.sum()), basis="revenue at risk", conf="MEDIUM",
         why=f"{len(HIGH)} critical products are below their reorder point and will run out before restock arrives."),
]
leaks = sorted([x for x in leaks if x["amt"] > 0], key=lambda x: x["amt"], reverse=True)

with tabs[0]:
    c = st.columns(4)
    c[0].markdown(metric_card("Booked revenue", money(K["booked"]), f"{K['orders']:,} orders", "text-warning"), unsafe_allow_html=True)
    c[1].markdown(metric_card("Realized revenue", money(K["realized"]), f"{K['realized'] / K['booked']:.1%} of booked", "text-success"), unsafe_allow_html=True)
    c[2].markdown(metric_card("Net leaked cash", money(K["net_leak"]), "cancelled + returned + pending", "text-alert"), unsafe_allow_html=True)
    top_name = leaks[0]["name"] if leaks else "none"
    c[3].markdown(metric_card("Active leaks", str(len(leaks)), f"largest: {top_name}", "text-alert"), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### Priority leak stream")
        st.caption("Ranked by dollars at stake. Pricing is margin lost; the others are revenue or spend, so do not add them up.")
        for i, lk in enumerate(leaks, 1):
            st.markdown(leak_card(i, lk["name"], lk["amt"], lk["basis"], lk["conf"], lk["why"]), unsafe_allow_html=True)
    with right:
        st.markdown("#### From booked to realized revenue")
        wf = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Booked revenue", "Cancellations", "Returns", "Pending (unrealized)", "Realized revenue"],
            y=[K["booked"], -K["cancelled"], -K["returned"], -K["pending"], 0],
            text=[money(K["booked"]), "-" + money(K["cancelled"]), "-" + money(K["returned"]), "-" + money(K["pending"]), money(K["realized"])],
            textposition="outside", connector=dict(line=dict(color="rgba(148,163,184,0.4)")),
            increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)), totals=dict(marker=dict(color=SKY))))
        plot(style_fig(wf, height=430))
        st.caption("Pending orders are included so the bridge adds up: booked - cancelled - returned - pending = realized.")

    st.markdown("#### Revenue by outcome, per month")
    pv = F.pivot_table(index="month", columns="status", values="revenue", aggfunc="sum", fill_value=0)
    fig = go.Figure()
    for s_ in ["Completed", "Pending", "Returned", "Cancelled"]:
        if s_ in pv.columns:
            fig.add_trace(go.Bar(x=pv.index, y=pv[s_], name=s_, marker_color=STATUS_COLORS[s_]))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    plot(style_fig(fig, height=340))

# =============================================================================================
# TAB 2: INVESTIGATE
# =============================================================================================
with tabs[1]:
    st.markdown("#### Where do orders fail, and is it real?")
    cats_all, segs_all, pays_all = sorted(FACT.category.unique()), sorted(FACT.customer_segment.unique()), sorted(FACT.payment_method.unique())
    f1, f2, f3, f4 = st.columns(4)
    sel_reg = f1.multiselect("Region", all_regions, default=all_regions, key="inv_region")
    sel_cat = f2.multiselect("Category", cats_all, default=cats_all, key="inv_cat")
    sel_seg = f3.multiselect("Customer segment", segs_all, default=segs_all, key="inv_seg")
    sel_pay = f4.multiselect("Payment method", pays_all, default=pays_all, key="inv_pay")
    G = F[F.region.isin(sel_reg) & F.category.isin(sel_cat) & F.customer_segment.isin(sel_seg) & F.payment_method.isin(sel_pay)]
    if G.empty:
        st.warning("No orders match these filters.")
    else:
        dim_label = st.selectbox("Break down by", list(BREAKDOWN), index=0)
        dim = BREAKDOWN[dim_label]
        seg = segment_table(G, dim)
        base_fail = G.failed.mean()
        view = seg.assign(**{dim: seg[dim].astype(str)}).sort_values(dim)
        n_sig = int((seg.verdict == "Real signal").sum())
        banner(f"<b>{len(G):,} orders</b> in view. Overall failure rate (cancelled + returned): <b>{base_fail:.1%}</b>. "
               f"Groups flagged as a real signal in this view: <b>{n_sig}</b>. "
               + ("Differences between groups are what chance alone would produce." if n_sig == 0 else "See the Verdict column."),
               "banner-warn" if n_sig else "")
        bc1, bc2 = st.columns([1.1, 1])
        with bc1:
            out = view.rename(columns={dim: dim_label, "orders": "Orders", "gross": "Gross volume ($)", "cancel_pct": "Cancellation %",
                                       "return_pct": "Return %", "net_margin": "Net margin ($)", "net_margin_pct": "Net margin %", "verdict": "Verdict"})
            out = out[[dim_label, "Orders", "Gross volume ($)", "Cancellation %", "Return %", "Net margin ($)", "Net margin %", "Verdict"]]
            for col in ["Gross volume ($)", "Net margin ($)"]:
                out[col] = out[col].round(0)
            for col in ["Cancellation %", "Return %", "Net margin %"]:
                out[col] = out[col].round(1)
            table(out, height=360)
        with bc2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=view[dim], y=view.cancel_pct, name="Cancellation %", marker_color=RED))
            fig.add_trace(go.Bar(x=view[dim], y=view.return_pct, name="Return %", marker_color=AMBER))
            fig.update_layout(barmode="stack")
            fig.update_yaxes(ticksuffix="%")
            plot(style_fig(fig, f"Failure rate by {dim_label.lower()}", height=360))

        st.markdown("#### Order value vs lead time, by order status")
        samp = G.sample(min(len(G), 4000), random_state=1)
        fig = go.Figure()
        for s_ in ["Completed", "Pending", "Returned", "Cancelled"]:
            d_ = samp[samp.status == s_]
            if len(d_):
                fig.add_trace(go.Scattergl(x=d_.supplier_lead_days, y=d_.revenue, mode="markers", name=s_,
                                           marker=dict(size=5, color=STATUS_COLORS[s_], opacity=0.55),
                                           hovertemplate="Lead time %{x} d<br>Order value $%{y:,.0f}<extra>" + s_ + "</extra>"))
        fig.update_xaxes(title="Supplier lead time, days (the data has no shipping field, so this is the closest proxy)")
        fig.update_yaxes(title="Order value ($)", tickprefix="$", tickformat=",.0f")
        plot(style_fig(fig, height=400))
        st.caption(f"Random sample of {len(samp):,} orders for speed. No band of lead time or order value is dominated by one status.")

        with st.expander("Driver scan: does any factor really explain cancellations and returns?"):
            scan_g = driver_scan(G, DIMS)
            table(scan_g)
            st.caption("For each factor, the most unusual group is tested against the overall failure rate. A Bonferroni correction "
                       "(stricter threshold when many groups are tested) separates real signals from normal noise.")

# =============================================================================================
# TAB 3: PRICING LEAK
# =============================================================================================
with tabs[2]:
    st.markdown("#### Products priced below unit cost")
    below = PT[PT.below_cost].copy()
    ref_gain, _ = reprice_whatif(F, 0.20, 0.15)
    c = st.columns(4)
    c[0].markdown(metric_card("Products below cost", f"{len(below)} of {len(PT)}", "list price < unit cost", "text-alert"), unsafe_allow_html=True)
    c[1].markdown(metric_card("Margin lost", money(K["price_loss"]), "completed orders, list < cost", "text-alert"), unsafe_allow_html=True)
    c[2].markdown(metric_card("Lines sold at a loss", f"{K['below_cost_lines']:,}", "completed order lines", "text-warning"), unsafe_allow_html=True)
    c[3].markdown(metric_card("Reference recovery", money(ref_gain), "20% floor, 15% volume loss", "text-success"), unsafe_allow_html=True)
    st.write("")

    p1, p2 = st.columns([2, 1])
    cat_pick = p1.multiselect("Category", sorted(below.category.unique()), default=sorted(below.category.unique()), key="pr_cat")
    pid_q = p2.text_input("Search product ID", "", key="pr_search")
    st.markdown("##### What-if repricing simulator")
    s1, s2 = st.columns(2)
    tm = s1.slider("Target minimum margin %", 0, 30, 20, 1, key="tm_pct")
    vl = s2.slider("Elasticity volume loss %", 0, 20, 15, 1, key="vl_pct")
    gain, by = reprice_whatif(F, tm / 100, vl / 100)
    n_aff = int((by.affected_lines > 0).sum())
    sim_a, sim_b = st.columns([1, 2])
    sim_a.markdown(metric_card("Estimated margin recovery", money(gain), f"{n_aff} products repriced at {tm}% floor, {vl}% volume loss", "text-success"),
                   unsafe_allow_html=True)
    with sim_b:
        banner(f"Every completed order line priced under <b>cost / (1 - {tm}%)</b> is lifted to that floor, then <b>{vl}%</b> of those units are assumed lost. "
               f"The {n_aff} repriced products include the {len(below)} below-cost products plus products where discounts push the sold price under the floor. "
               "Assumes cost data is right; finance should confirm before repricing.")

    tbl = below[below.category.isin(cat_pick)].merge(by, on="product_id", how="left").fillna({"gain": 0, "affected_lines": 0})
    tbl["recommended_price"] = tbl.unit_cost / (1 - tm / 100)
    if pid_q.strip():
        tbl = tbl[tbl.product_id.str.contains(pid_q.strip(), case=False, na=False)]
    tbl = tbl.sort_values("margin")
    show = tbl[["product_id", "category", "unit_cost", "list_price", "recommended_price", "list_margin_pct", "units", "margin", "gain"]].rename(columns={
        "product_id": "Product", "category": "Category", "unit_cost": "Unit cost", "list_price": "Current price", "recommended_price": "Recommended price",
        "list_margin_pct": "List margin %", "units": "Units sold", "margin": "Realized margin ($)", "gain": "Simulated recovery ($)"}).round(2)
    st.markdown(f"##### {len(show)} products below cost (worst realized margin first)")
    table(show, height=330)

    top = tbl.head(15)
    if len(top):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top.product_id, y=top.list_price, name="Current price", marker_color=RED))
        fig.add_trace(go.Bar(x=top.product_id, y=top.unit_cost, name="Unit cost", marker_color=SLATE))
        fig.add_trace(go.Bar(x=top.product_id, y=top.recommended_price, name=f"Recommended ({tm}% margin)", marker_color=GREEN))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="$ per unit", tickprefix="$")
        plot(style_fig(fig, "15 worst products: current price vs unit cost vs recommended price", height=400))
    st.caption(f"Discounts are not the cause: unit_price already includes the discount, and discounts alone add only "
               f"{money(-F[(F.is_completed) & (~F.list_below_cost) & (F.margin < 0)].margin.sum())} of extra loss. The list price is the problem.")

# =============================================================================================
# TAB 4: STOCK RISK
# =============================================================================================
with tabs[3]:
    st.markdown("#### True inventory audit")
    od = st.slider("Overstock threshold: days of cover above which stock counts as excess", 60, 365, 180, 15, key="overstock_days")
    S_ = build_stock(INV, PRODUCTS, FACT, AS_OF, od)
    crit = S_[S_.risk == "High"].sort_values("revenue_at_risk", ascending=False)
    zero_ev = INV[INV.stock_on_hand == 0]
    over_cap = float(S_.overstock_capital.sum())
    c = st.columns(4)
    c[0].markdown(metric_card("Critical stockout products", str(len(crit)), "below reorder point and out before restock", "text-alert"), unsafe_allow_html=True)
    c[1].markdown(metric_card("Revenue at risk", money(crit.revenue_at_risk.sum()), "demand x days short x price", "text-warning"), unsafe_allow_html=True)
    c[2].markdown(metric_card("Overstock capital locked", money(over_cap), f"stock beyond {od} days of cover, at cost", "text-warning"), unsafe_allow_html=True)
    c[3].markdown(metric_card("Zero-stock events logged", str(len(zero_ev)), f"{zero_ev.product_id.nunique()} products, none flagged", "text-alert"), unsafe_allow_html=True)
    st.write("")
    if len(crit):
        lo_, hi_ = int(crit.stock_on_hand.min()), int(crit.stock_on_hand.max())
        banner(f"<b>Critical</b> means the latest on-hand stock is at or below the reorder point <b>and</b> days of inventory remaining are fewer than "
               f"supplier lead time plus delay. These {len(crit)} products hold {lo_} to {hi_} units now; none is at exactly zero in the latest snapshot. "
               f"The supplied stockout_flag is ignored: {int(((INV.stockout_flag == 1) & (INV.stock_on_hand > 0)).sum())} of "
               f"{int((INV.stockout_flag == 1).sum())} flagged records still have stock, and all {len(zero_ev)} real zero-stock records are unflagged.", "banner-alert")
        tb = crit[["product_id", "category", "stock_on_hand", "reorder_threshold", "daily_demand", "days_remaining", "restock_days", "revenue_at_risk", "why_flagged"]].rename(columns={
            "product_id": "Product", "category": "Category", "stock_on_hand": "Stock on hand", "reorder_threshold": "Reorder point",
            "daily_demand": "Units / day (90d)", "days_remaining": "Days of inventory remaining", "restock_days": "Restock time (days)",
            "revenue_at_risk": "Revenue at risk ($)", "why_flagged": "Why flagged"})
        table(tb.round(2))
    else:
        st.success("No product is both below its reorder point and set to run out before restock arrives.")

    fig = go.Figure(go.Bar(x=["Overstock capital lockup", "Immediate stockout risk"], y=[over_cap, float(crit.revenue_at_risk.sum())],
                           marker_color=[AMBER, RED], text=[money(over_cap), money(crit.revenue_at_risk.sum())], textposition="outside"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    a1, a2 = st.columns([1, 1])
    with a1:
        plot(style_fig(fig, "Cash tied up in excess stock vs revenue at risk from stockouts", height=380))
    with a2:
        sc = go.Figure()
        for r_, col in [("Low", GREEN), ("Medium", AMBER), ("High", RED)]:
            d_ = S_[S_.risk == r_]
            sc.add_trace(go.Scatter(x=d_.days_remaining, y=d_.restock_days, mode="markers", name=r_, text=d_.product_id,
                                    marker=dict(size=7, color=col, opacity=0.75), hovertemplate="%{text}<br>Cover %{x:.0f} d<br>Restock %{y:.0f} d<extra></extra>"))
        mx = float(min(S_.days_remaining.max(), 120))
        sc.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Cover = restock time", line=dict(color=SLATE, dash="dash")))
        sc.update_xaxes(title="Days of inventory remaining", range=[0, mx])
        sc.update_yaxes(title="Restock time (days)")
        plot(style_fig(sc, "Products above the dashed line run out before restock", height=380))
    st.caption("The overstock bar dwarfs the stockout bar: the bigger inventory problem is cash locked in slow movers, not lost sales.")
    with st.expander("Zero-stock events in the inventory log"):
        table(zero_ev[["inventory_record_id", "product_id", "date", "reorder_threshold", "stockout_flag"]].assign(date=lambda x: x.date.dt.date))

# =============================================================================================
# TAB 5: CAMPAIGNS
# =============================================================================================
with tabs[4]:
    st.markdown("#### Is ad spend buying conversions?")
    cs = CAMP_STATS
    c = st.columns(4)
    c[0].markdown(metric_card("Spend vs conversions", f"r = {cs['r']:.2f}", "no relationship", "text-alert"), unsafe_allow_html=True)
    c[1].markdown(metric_card("Total ad spend", money(cs["total_spend"]), f"{len(CF):,} campaigns", "text-warning"), unsafe_allow_html=True)
    c[2].markdown(metric_card("Impossible rows", str(cs["invalid_rows"]), f"{money(cs['invalid_spend'])} spend, excluded from r", "text-alert"), unsafe_allow_html=True)
    c[3].markdown(metric_card("Spend above benchmark", money(cs["excess"]), f"benchmark ${cs['benchmark_cpa']:.2f} per conversion", "text-warning"), unsafe_allow_html=True)
    st.write("")
    v = CF[CF.valid]
    fig = go.Figure()
    chan_cols = {"App": SKY, "Email": GREEN, "Social": AMBER, "Store": VIOLET, "Web": RED}
    for ch_, d_ in v.groupby("channel"):
        fig.add_trace(go.Scattergl(x=d_.spend, y=d_.conversions, mode="markers", name=ch_,
                                   marker=dict(size=6, opacity=0.55, color=chan_cols.get(ch_, SLATE))))
    if len(v) > 2:
        b1, b0 = np.polyfit(v.spend, v.conversions, 1)
        xs = np.array([v.spend.min(), v.spend.max()])
        fig.add_trace(go.Scatter(x=xs, y=b0 + b1 * xs, mode="lines", name="Best-fit line", line=dict(color="#f1f5f9", dash="dash")))
    fig.update_xaxes(title="Ad spend per campaign ($)", tickprefix="$")
    fig.update_yaxes(title="Conversions")
    plot(style_fig(fig, f"Ad spend vs conversions on {len(v):,} valid campaigns (r = {cs['r']:.2f})", height=420))
    st.caption("A flat best-fit line means more budget did not buy more conversions. "
               f"Rows with clicks above impressions or conversions above clicks ({cs['invalid_rows']}) are excluded from r and from realized conversions.")

    st.markdown("#### Channel economics: acquisition cost vs customer lifetime margin")
    ce = channel_economics(CF, CUSTOMERS, FACT)
    sp = make_subplots(rows=1, cols=2, subplot_titles=("CAC: ad spend per conversion", "Customer lifetime margin (avg per customer)"))
    sp.add_trace(go.Bar(x=ce.channel, y=ce.cac, marker_color=RED, name="CAC", text=[f"${x:,.2f}" for x in ce.cac], textposition="outside"), row=1, col=1)
    lm_ = ce.lifetime_margin.fillna(0)
    sp.add_trace(go.Bar(x=ce.channel, y=lm_, marker_color=GREEN, name="Lifetime margin", text=["n/a" if pd.isna(x) else f"${x:,.0f}" for x in ce.lifetime_margin], textposition="outside"), row=1, col=2)
    sp.update_layout(showlegend=False)
    sp.update_yaxes(tickprefix="$")
    plot(style_fig(sp, height=380))
    ce_show = ce.rename(columns={"channel": "Channel", "spend": "Valid spend ($)", "conversions": "Conversions", "cac": "CAC ($)",
                                 "customers": "Customers", "lifetime_margin": "Lifetime margin / customer ($)", "margin_to_cac": "Margin / CAC"}).round(2)
    table(ce_show)
    banner("<b>Read with care.</b> CAC here is ad spend divided by recorded conversions. Campaign conversions (about "
           f"{cs['total_conv']:,.0f}) are far larger than the customer base ({len(CUSTOMERS):,}) or order count ({len(FACT):,}), so tracking cannot be reconciled with sales. "
           "Lifetime margin is realized margin per customer by preferred channel; Social and Email have no matching sales channel, so it shows n/a. "
           "Fix conversion tracking before shifting budget.", "banner-warn")

# =============================================================================================
# TAB 6: FORECAST & ALERTS
# =============================================================================================
with tabs[5]:
    st.markdown("#### Daily anomalies and 30-day outlook")
    q1, q2 = st.columns([1, 2])
    metric = q1.radio("Metric", ["Realized revenue", "Order count"], horizontal=True)
    zthr = q2.slider("Z-score alert threshold (standard deviations from the daily mean)", 2.0, 4.0, 2.5, 0.1)
    ser = daily_series(F, metric)
    money_mode = metric == "Realized revenue"
    fmt = money if money_mode else (lambda v_: f"{v_:,.0f}")
    if len(ser) < 14:
        st.warning("Select at least 14 days to run anomaly detection and a projection.")
    else:
        sd = float(ser.std(ddof=0)); mu = float(ser.mean())
        z = (ser - mu) / sd if sd > 0 else ser * 0.0
        flag = z.abs() > zthr
        an = pd.DataFrame(dict(Date=ser.index[flag].date, Value=ser[flag].values, **{"Z-score": z[flag].values}))
        if len(an):
            an["Type"] = np.where(an["Z-score"] > 0, "Spike", "Drop")
            an["vs daily mean %"] = (an.Value / mu - 1) * 100
            an = an.reindex(an["Z-score"].abs().sort_values(ascending=False).index)
        k1, k2, k3 = st.columns(3)
        k1.markdown(metric_card("Days analysed", f"{len(ser):,}", f"daily mean {fmt(mu)}", "text-success"), unsafe_allow_html=True)
        k2.markdown(metric_card("Anomalous days", str(int(flag.sum())), f"|z| above {zthr:.1f}", "text-alert" if flag.sum() else "text-success"), unsafe_allow_html=True)
        k3.markdown(metric_card("Largest |z|", f"{float(z.abs().max()):.2f}", "single biggest daily deviation", "text-warning"), unsafe_allow_html=True)
        st.write("")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ser.index, y=ser.values, mode="lines", name="Daily", line=dict(color=SKY, width=1.2)))
        fig.add_trace(go.Scatter(x=ser.index, y=ser.rolling(7, min_periods=1).mean(), mode="lines", name="7-day average", line=dict(color=GREEN, width=2.2)))
        fig.add_trace(go.Scatter(x=ser.index[flag], y=ser[flag].values, mode="markers", name="Anomaly",
                                 marker=dict(size=10, color=RED, line=dict(color="#ffffff", width=1))))
        fig.add_hline(y=mu, line_dash="dash", line_color=SLATE)
        if money_mode:
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        plot(style_fig(fig, f"Daily {metric.lower()} with z-score alerts", height=380))
        if len(an):
            table(an.round(2))
        else:
            st.success(f"No day crosses |z| > {zthr:.1f}. Daily {metric.lower()} stays inside normal variation.")

        fut, method, st_stats = project_series(ser, 30)
        hist = ser.iloc[-120:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(fut.date) + list(fut.date[::-1]), y=list(fut.high) + list(fut.low[::-1]), fill="toself",
                                 fillcolor="rgba(56,189,248,0.14)", line=dict(width=0), name="80% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Actual", line=dict(color=SKY, width=1.2)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.rolling(7, min_periods=1).mean(), mode="lines", name="7-day average", line=dict(color=GREEN, width=2.2)))
        fig.add_trace(go.Scatter(x=fut.date, y=fut.forecast, mode="lines", name="30-day projection", line=dict(color=AMBER, width=2.6, dash="dash")))
        if money_mode:
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        plot(style_fig(fig, "30-day trend and projection", height=380))
        last30 = float(ser.iloc[-30:].sum()); proj30 = float(fut.forecast.sum())
        m1, m2 = st.columns(2)
        m1.markdown(metric_card("Last 30 days (actual)", fmt(last30), "most recent 30 days in view", "text-success"), unsafe_allow_html=True)
        m2.markdown(metric_card("Next 30 days (projection)", fmt(proj30), f"{(proj30 / last30 - 1) * 100:+.1f}% vs last 30 days" if last30 else "", "text-warning"), unsafe_allow_html=True)
        st.caption(f"Method: {method}. " + (f"Hold-out error on the last 30 days: trend {fmt(st_stats['mae_trend'])} vs flat {fmt(st_stats['mae_flat'])} per day. "
                                            if not math.isnan(st_stats["mae_trend"]) else "")
                   + "The projection is a planning range, not a promise; the synthetic data shows no proven trend or seasonality.")

# =============================================================================================
# TAB 7: ML LAB
# =============================================================================================
with tabs[6]:
    st.markdown("#### ML Lab: can we predict which orders will fail?")
    lab = run_ml_lab(D["folder"])
    mx_ = lab["matrix"]; best = mx_.sort_values("Test AUC", ascending=False).iloc[0]
    st.caption(f"Time-aware split: train Jan to Sep ({lab['train_n']:,} orders), test Oct to Dec ({lab['test_n']:,} unseen orders). "
               f"Target: order cancelled or returned ({lab['fail_rate']:.1%} of test orders). Engine: {lab['engine']}.")
    banner(f"<b>Transparent science.</b> The best model reaches a test AUC of <b>{best['Test AUC']:.2f}</b>. An AUC of 0.50 is a coin flip, and accuracy "
           f"({best['Accuracy']:.1%}) only equals the \"always predict success\" baseline ({best['Majority-class baseline']:.1%}). "
           "So we <b>suppress</b> the model's risk scores instead of selling a fake prediction, and act on deterministic rules that are provably real: "
           "products priced below unit cost (arithmetic) and pending orders older than 30 days (a queue rule). To make failures predictable, record cancel and return reasons.",
           "banner-warn")
    ml1, ml2 = st.columns([1, 1.15])
    with ml1:
        st.markdown("##### Model performance comparison")
        table(mx_)
        st.markdown(metric_card("Best test AUC", f"{best['Test AUC']:.2f}", f"{best['Model']}: no better than chance", "text-alert"), unsafe_allow_html=True)
    with ml2:
        fig = go.Figure()
        for (name, (fx, tx_)), col in zip(lab["curves"].items(), [SKY, GREEN, AMBER]):
            auc_ = float(mx_.loc[mx_.Model == name, "Test AUC"].iloc[0])
            fig.add_trace(go.Scatter(x=fx, y=tx_, mode="lines", name=f"{name} (AUC {auc_:.2f})", line=dict(color=col, width=2.2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin flip (0.50)", line=dict(color=RED, dash="dash")))
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1])
        plot(style_fig(fig, "ROC curves on unseen Oct to Dec orders", height=430))

# =============================================================================================
# TAB 8: ACTION PLAN
# =============================================================================================
with tabs[7]:
    st.markdown("#### Prioritized action plan")
    st.caption("Tick items as your team completes them. Impacts overlap and are upper bounds, so do not add them up.")
    actions = [
        dict(key="act_price", title=f"Reprice the {N_BELOW} products priced below unit cost", impact=GAIN_NOW, conf="HIGH", owner="Pricing and Finance",
             how=f"Raise list prices to cost / (1 - {TM}%); start with the 20 worst products. Impact shown for the {TM}% floor and {VL}% volume loss set in the Pricing tab."),
        dict(key="act_stuck", title=f"Clear {K['stuck_n']:,} stuck pending orders", impact=K["stuck_recoverable"], conf="MEDIUM", owner="Order operations",
             how=f"Work the queue daily with an owner and escalate after 7 days. {money(K['stuck_value'])} is locked; impact applies the {K['completion_rate']:.0%} historical completion rate."),
        dict(key="act_stock", title=f"Reorder the {len(HIGH)} critical products before they run out", impact=float(HIGH.revenue_at_risk.sum()), conf="MEDIUM", owner="Supply chain",
             how="Place purchase orders now, and rebuild the stockout flag from stock level, reorder point and restock time."),
        dict(key="act_ads", title="Audit campaign tracking, then cut spend above benchmark cost per conversion", impact=CAMP_STATS["excess"], conf="LOW", owner="Marketing",
             how=f"Fix {CAMP_STATS['invalid_rows']} impossible campaign rows first. Then move budget away from the highest cost-per-conversion campaigns as a test."),
        dict(key="act_fail", title="Record cancellation and return reasons", impact=K["failed_rev"] * 0.0, conf="MEDIUM", owner="Customer operations",
             how=f"{K['failed_orders']:,} orders failed with no real driver in the data. Reason codes make this fixable; do not build targeting on the current data."),
    ]
    done = 0
    for i, a in enumerate(actions, 1):
        ca, cb = st.columns([0.06, 0.94])
        checked = ca.checkbox(" ", key=a["key"], label_visibility="collapsed")
        done += int(bool(checked))
        impact_txt = f"up to {money(a['impact'])}" if a["impact"] > 0 else "unlocks every other failure fix"
        cb.markdown(f'<div class="metric-card" style="padding:14px 18px;margin-bottom:6px;"><div class="alert-head"><div><span class="alert-rank">#{i}</span>'
                    f'<span class="alert-title">{a["title"]}</span></div><div>{badge(a["conf"])}</div></div>'
                    f'<div class="alert-body"><b style="color:#38bdf8">Impact:</b> {impact_txt} &nbsp;|&nbsp; <b style="color:#38bdf8">Owner:</b> {a["owner"]}<br>{a["how"]}</div></div>',
                    unsafe_allow_html=True)
    st.progress(done / len(actions), text=f"{done} of {len(actions)} actions complete")

    st.markdown("#### Download work queues")
    floor_mult = 1 / (1 - TM / 100)
    rp = PT[PT.list_price < PT.unit_cost * floor_mult].sort_values("margin")
    repricing = pd.DataFrame(dict(product_id=rp.product_id, current_price=rp.list_price.round(2), cost=rp.unit_cost.round(2),
                                  recommended_price=(rp.unit_cost * floor_mult).round(2)))
    sq = F[F.stale].sort_values("revenue", ascending=False)
    stuck = pd.DataFrame(dict(transaction_id=sq.transaction_id, customer_id=sq.customer_id, days_pending=sq.age_days.astype(int), value=sq.revenue.round(2)))
    ad = CF.sort_values("spend", ascending=False)
    ad_audit = pd.DataFrame(dict(campaign_id=ad.campaign_id, channel=ad.channel, spend=ad.spend.round(2),
                                 realized_conversions=ad.conversions.where(ad.valid).astype("Int64")))
    d1_, d2_, d3_ = st.columns(3)
    with d1_:
        st.markdown(metric_card("Repricing list", f"{len(repricing)} products", f"floor = cost / (1 - {TM}%)", "text-success"), unsafe_allow_html=True)
        st.download_button("Download_Repricing_List.csv", csv_bytes(repricing), "Download_Repricing_List.csv", "text/csv", key="dl_reprice")
    with d2_:
        st.markdown(metric_card("Stuck orders queue", f"{len(stuck):,} orders", f"{money(stuck.value.sum())} pending over {STALE_DAYS} days", "text-warning"), unsafe_allow_html=True)
        st.download_button("Download_Stuck_Orders_Queue.csv", csv_bytes(stuck), "Download_Stuck_Orders_Queue.csv", "text/csv", key="dl_stuck")
    with d3_:
        st.markdown(metric_card("Ad spend audit", f"{len(ad_audit):,} campaigns", f"{CAMP_STATS['invalid_rows']} unverifiable rows left blank", "text-alert"), unsafe_allow_html=True)
        st.download_button("Download_Ad_Spend_Audit.csv", csv_bytes(ad_audit), "Download_Ad_Spend_Audit.csv", "text/csv", key="dl_ads")
    st.caption("realized_conversions is the recorded conversion count for campaigns with valid tracking (conversions <= clicks <= impressions); "
               "impossible rows are left blank because their conversions cannot be verified.")

    with st.expander("Data quality, cleaning log and assumptions"):
        dq = pd.DataFrame([
            dict(Issue="List price below unit cost", Detail=f"{N_BELOW} of {len(PRODUCTS)} products; {int(F[F.unit_price < F.unit_cost].product_id.nunique())} products have at least one order sold below cost after discounts", Severity="High"),
            dict(Issue="Impossible campaign rows", Detail=f"{int((~CAMP.valid).sum())} rows with clicks > impressions or conversions > clicks", Severity="High"),
            dict(Issue="Stockout flag disagrees with stock", Detail=f"{int(((INV.stockout_flag == 1) & (INV.stock_on_hand > 0)).sum())} flagged records have stock; {int(((INV.stock_on_hand == 0) & (INV.stockout_flag == 0)).sum())} zero-stock records are not flagged", Severity="Medium"),
            dict(Issue="Duplicate product-date inventory records", Detail=f"{int(INV.duplicated(['product_id', 'date']).sum())} duplicates; latest record per product used", Severity="Low"),
            dict(Issue="Pending orders never resolved", Detail=f"{int((FACT.status == 'Pending').sum())} pending orders; {int(FACT.stale.sum())} older than {STALE_DAYS} days", Severity="Medium"),
        ])
        table(dq)
        table(pd.DataFrame(D["log"], columns=["Cleaning step", "Rows affected"]))
        st.markdown(
            "- Completed orders are realized revenue; cancelled and returned orders are not (goods restocked, no cost incurred).\n"
            "- unit_price already includes the discount (verified on every row), so discounts are never applied twice.\n"
            f"- Pending orders older than {STALE_DAYS} days at the end of the data ({AS_OF.date()}) are treated as stuck.\n"
            "- Currency is not stated in the data; $ is assumed for prices and campaign spend.\n"
            "- The dataset is synthetic: costs and prices were generated independently, so finance should confirm margins before repricing.")

st.markdown(
    '<div class="site-footer"><span><strong>LeakFinder</strong> / Live local analysis</span>'
    '<span>W V P S SRIRAJ (23BCE8414)</span></div>',
    unsafe_allow_html=True,
)
