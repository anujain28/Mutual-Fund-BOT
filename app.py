import os
import json
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st

# ==========================
# Basic Setup & Config
# ==========================

st.set_page_config(
    page_title="💹 AI Mutual Fund Analysis Bot",
    page_icon="💹",
    layout="wide",
)

# --- Global styles, including black table --- #
st.markdown(
    """
<style>
/* Black styled table for projections and any custom HTML tables */
.dark-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #020617;
    color: #f9fafb;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 8px;
    margin-bottom: 12px;
}
.dark-table th, .dark-table td {
    padding: 8px 10px;
    border: 1px solid #1f2937;
    font-size: 0.85rem;
}
.dark-table th {
    background-color: #111827;
    font-weight: 600;
    text-align: left;
}

/* Darken dataframes */
div[data-testid="stDataFrame"] table {
    background-color: #020617 !important;
    color: #f9fafb !important;
}
div[data-testid="stDataFrame"] th,
div[data-testid="stDataFrame"] td {
    background-color: #020617 !important;
    color: #f9fafb !important;
    border-color: #1f2937 !important;
    font-size: 0.85rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

IST = pytz.timezone("Asia/Kolkata")

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "notify_enabled": False,
}


def load_config() -> Dict:
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config():
    cfg = {
        "telegram_bot_token": st.session_state.get("telegram_bot_token", ""),
        "telegram_chat_id": st.session_state.get("telegram_chat_id", ""),
        "notify_enabled": st.session_state.get("notify_enabled", False),
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        st.sidebar.warning(f"Could not save config: {e}")


# Initialise session state keys
for key, default in [
    ("telegram_bot_token", ""),
    ("telegram_chat_id", ""),
    ("notify_enabled", False),
    ("last_reco_notify", {}),
    ("send_now_flag", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================
# Telegram Helpers
# ==========================

def send_telegram_message(text: str) -> Dict:
    token = st.session_state.get("telegram_bot_token", "")
    chat_id = st.session_state.get("telegram_chat_id", "")
    if not token or not chat_id:
        return {"ok": False, "error": "Telegram not configured"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.get(url, params={"chat_id": chat_id, "text": text})
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


def generate_telegram_reco_text(df_norm: pd.DataFrame) -> str:
    if df_norm is None or df_norm.empty:
        return ""
    # Only Super Core / Core / Satellite for TG
    buckets_order = ["Super Core", "Core", "Satellite"]
    lines = []
    now = datetime.now(IST)
    header = f"📊 MF Auto Recommendations\n🕒 {now.strftime('%d-%m-%Y %I:%M %p')} IST\n"
    lines.append(header)

    for b in buckets_order:
        sub = df_norm[df_norm["Bucket"] == b]
        if sub.empty:
            continue
        sub = sub.sort_values("AI Score", ascending=False).head(3)
        lines.append(f"\n⭐ {b}:")
        for _, row in sub.iterrows():
            nm = row["Scheme Name"]
            xirr = row.get("XIRR (%)", np.nan)
            rec = row.get("Recommendation", "")
            tgt = row.get("Target Year", "")
            msg = f"• {nm}"
            if not pd.isna(xirr):
                msg += f" | XIRR ~ {xirr:.1f}%"
            if tgt:
                msg += f" | 🎯 {tgt}"
            if rec:
                msg += f" | {rec}"
            lines.append(msg)

    return "\n".join(lines).strip()


def handle_scheduled_notifications(df_norm: Optional[pd.DataFrame]):
    if df_norm is None or df_norm.empty:
        return
    if not st.session_state.get("notify_enabled", False):
        return
    if not st.session_state.get("telegram_bot_token") or not st.session_state.get("telegram_chat_id"):
        return

    now = datetime.now(IST)
    today_str = now.strftime("%Y-%m-%d")

    # Slots: key -> (hour, minute)
    slots = {
        "morning": (9, 30),
        "midday": (13, 30),
        "close": (15, 0),
    }

    last_map: Dict = st.session_state.get("last_reco_notify", {}) or {}

    for key, (h, m) in slots.items():
        if now.hour == h and abs(now.minute - m) <= 2:
            last = last_map.get(key, "")
            if last.startswith(today_str):
                continue  # already sent this slot today
            msg = generate_telegram_reco_text(df_norm)
            if msg:
                send_telegram_message(msg)
                last_map[key] = today_str + " " + now.strftime("%H:%M")
                st.caption(f"📬 Telegram recommendations sent for slot: {key} at {now.strftime('%H:%M')} IST")

    st.session_state["last_reco_notify"] = last_map


# ==========================
# Sidebar
# ==========================

def render_sidebar():
    st.sidebar.title("⚙️ Settings & Links")

    # Stocks app link
    st.sidebar.markdown("### 🔗 Stocks Analysis")
    st.sidebar.markdown("[📈](https://airobots.streamlit.app/)")
    st.sidebar.markdown("---")

    # Load config into state
    cfg = load_config()
    if not st.session_state.get("telegram_bot_token"):
        st.session_state["telegram_bot_token"] = cfg.get("telegram_bot_token", "")
    if not st.session_state.get("telegram_chat_id"):
        st.session_state["telegram_chat_id"] = cfg.get("telegram_chat_id", "")
    if "notify_enabled" not in st.session_state or st.session_state["notify_enabled"] is False:
        st.session_state["notify_enabled"] = cfg.get("notify_enabled", False)

    st.sidebar.subheader("📨 Telegram Recommendations")

    st.sidebar.checkbox(
        "Enable auto recommendations (9:30, 13:30, 15:00 IST)",
        value=st.session_state.get("notify_enabled", False),
        key="notify_enabled",
    )
    st.sidebar.text_input(
        "Bot Token",
        value=st.session_state.get("telegram_bot_token", ""),
        key="telegram_bot_token",
    )
    st.sidebar.text_input(
        "Chat ID",
        value=st.session_state.get("telegram_chat_id", ""),
        key="telegram_chat_id",
    )

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Save", key="btn_save_tg_sidebar"):
            save_config()
            st.sidebar.success("Saved config.json")
    with c2:
        if st.button("📤 Send Now", key="btn_send_now_sidebar"):
            st.session_state["send_now_flag"] = True


# ==========================
# Data Loading & Mapping
# ==========================

def load_portfolio_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Only CSV, XLS, or XLSX files are supported.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()


def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)

    def find_col_by_keywords(keywords):
        for c in cols:
            low = str(c).lower()
            if any(k in low for k in keywords):
                return c
        return None

    mapping = {
        "scheme": find_col_by_keywords(["scheme name", "fund name", "scheme", "plan name"]),
        "category": find_col_by_keywords(["category"]),
        "subcategory": find_col_by_keywords(["sub category", "sub-category", "subcategory", "sub cat"]),
        "invested": find_col_by_keywords(
            ["invested", "investment", "cost value", "purchase amount", "amount invested", "inv amount",
             "purchase cost", "total investment"]
        ),
        "current": find_col_by_keywords(
            ["current value", "current", "market value", "value (₹)", "value (rs)", "current value (rs)", "current amount"]
        ),
        "xirr": find_col_by_keywords(["xirr"]),
        "dividend": find_col_by_keywords(["dividend yield", "dividend (%)", "dividend %"]),
    }

    if mapping["scheme"] is None:
        st.error("Could not auto-detect 'Scheme Name' column. Please ensure it contains text like 'Scheme Name' or 'Fund Name'.")
        st.write("Detected columns:", cols)
        return mapping

    if mapping["invested"] is None:
        st.warning("⚠️ Could not detect Invested Amount column. Will treat Invested (₹) as 0 unless found.")

    if mapping["current"] is None:
        st.warning("⚠️ Could not detect Current Value column. Will treat Current Value (₹) as 0 unless found.")

    return mapping


# ==========================
# XIRR from Internet (mfapi.in)
# ==========================

def fetch_scheme_xirr_from_mfapi_by_name(scheme_name: str) -> Optional[float]:
    """
    Approximate XIRR using mfapi.in NAV history (CAGR style).
    """
    try:
        q = scheme_name.strip()
        if not q:
            return None
        search_url = "https://api.mfapi.in/mf/search"
        r = requests.get(search_url, params={"q": q}, timeout=8)
        if not r.ok:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        scheme_code = data[0].get("schemeCode")
        if not scheme_code:
            return None

        nav_url = f"https://api.mfapi.in/mf/{scheme_code}"
        r2 = requests.get(nav_url, timeout=10)
        if not r2.ok:
            return None
        j2 = r2.json()
        nav_list = j2.get("data", [])
        if not nav_list:
            return None

        # Ensure sorted by date ascending
        def _parse_date(d):
            return datetime.strptime(d["date"], "%d-%m-%Y")

        nav_sorted = sorted(nav_list, key=_parse_date)
        first = nav_sorted[0]
        last = nav_sorted[-1]
        nav_start = float(first["nav"])
        nav_end = float(last["nav"])
        if nav_start <= 0:
            return None
        d0 = _parse_date(first)
        d1 = _parse_date(last)
        years = max((d1 - d0).days / 365.0, 1.0)
        cagr = (nav_end / nav_start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return None


def enhance_xirr_with_online_data(df_norm: pd.DataFrame) -> pd.DataFrame:
    if "XIRR (%)" not in df_norm.columns:
        df_norm["XIRR (%)"] = np.nan

    needs_idx = df_norm.index[df_norm["XIRR (%)"].isna()].tolist()
    if needs_idx:
        st.info("🔍 Fetching XIRR online from mfapi.in for schemes missing XIRR (approximate CAGR-style XIRR).")
        cache: Dict[str, Optional[float]] = {}
        prog = st.progress(0.0)
        for i, idx in enumerate(needs_idx):
            name = str(df_norm.at[idx, "Scheme Name"])
            key = name.upper()
            if key in cache:
                xirr_val = cache[key]
            else:
                xirr_val = fetch_scheme_xirr_from_mfapi_by_name(name)
                cache[key] = xirr_val
            if xirr_val is not None:
                df_norm.at[idx, "XIRR (%)"] = xirr_val
            prog.progress((i + 1) / len(needs_idx))
        prog.empty()

    # If still NaN, assume 8% as default XIRR
    df_norm["XIRR (%)"] = df_norm["XIRR (%)"].fillna(8.0)
    return df_norm


# ==========================
# AI Classification & Buckets
# ==========================

def classify_bucket(category: str, subcat: str, xirr: float, pnl_pct: float, scheme_name: str) -> str:
    cat_lower = (category or "").lower()
    sub_lower = (subcat or "").lower()
    name_lower = (scheme_name or "").lower()

    is_index = "index" in cat_lower or "index" in sub_lower or "nifty" in name_lower or "sensex" in name_lower
    is_large = "large" in cat_lower or "large cap" in sub_lower or "bluechip" in sub_lower
    is_smallmid = any(k in (cat_lower + sub_lower) for k in ["small", "mid"])
    is_thematic = any(k in (cat_lower + sub_lower) for k in [
        "sector", "theme", "thematic", "psu", "banking", "infra", "pharma",
        "energy", "auto", "technology", "digital", "gold", "it"
    ])

    if np.isnan(xirr):
        xirr = 8.0
    if np.isnan(pnl_pct):
        pnl_pct = 0.0

    # Super Core: index / large diversified with good XIRR
    if (is_index or is_large) and xirr >= 11:
        return "Super Core"
    # Core: diversified equity / flexi cap / large-mid with decent XIRR
    if (is_index or is_large) and xirr >= 9:
        return "Core"
    if (not is_smallmid and not is_thematic) and xirr >= 10:
        return "Core"
    # Satellite: mid/small / thematic / factor funds with good XIRR
    if is_smallmid and xirr >= 12:
        return "Satellite"
    if is_thematic and xirr >= 10:
        return "Satellite"
    # Medium: okay-ish performers
    if xirr >= 7 and pnl_pct > -25:
        return "Medium"
    # Exit: long-term poor + deep drawdown
    if xirr < 3 and pnl_pct < -20:
        return "Exit"
    # Default
    return "Weak"


def target_year_and_horizon(bucket: str):
    if bucket == "Super Core":
        return 2045, "20+ years"
    if bucket == "Core":
        return 2040, "10–15 years"
    if bucket == "Satellite":
        return 2035, "7–10 years"
    if bucket == "Medium":
        return 2030, "5–7 years"
    # Weak / Exit
    return 2026, "0–3 years / Review"


def recommendation_from_bucket(bucket: str) -> str:
    if bucket == "Super Core":
        return "BUY & HOLD 20+ yrs"
    if bucket == "Core":
        return "BUY / HOLD 10–15 yrs"
    if bucket == "Satellite":
        return "BUY (Aggressive 7–10 yrs)"
    if bucket == "Medium":
        return "HOLD / REVIEW"
    if bucket == "Exit":
        return "EXIT / SWITCH GRADUALLY"
    return "REVIEW / AVOID NEW"


def bucket_reason(row: pd.Series) -> str:
    bucket = row.get("Bucket", "")
    xirr = row.get("XIRR (%)", np.nan)
    cat = row.get("Category", "")
    pnl = row.get("P&L (%)", np.nan)

    parts = []
    if bucket == "Super Core":
        parts.append("Low-cost diversified or index-style core holding.")
    elif bucket == "Core":
        parts.append("Strong long-term core equity candidate.")
    elif bucket == "Satellite":
        parts.append("Higher-risk satellite bet for extra returns.")
    elif bucket == "Medium":
        parts.append("Decent but not standout performance.")
    elif bucket == "Exit":
        parts.append("Persistently weak risk–reward profile.")
    else:
        parts.append("Mixed signals; keep under periodic review.")

    if not pd.isna(xirr):
        parts.append(f"Scheme XIRR ≈ {xirr:.1f}% p.a.")
    if not pd.isna(pnl):
        parts.append(f"Total P&L ≈ {pnl:.1f}%")
    if cat:
        parts.append(f"Category: {cat}")

    return " ".join(parts)


def compute_ai_score(row: pd.Series) -> float:
    score = 50.0
    xirr = row.get("XIRR (%)", np.nan)
    pnl = row.get("P&L (%)", np.nan)
    divy = row.get("Dividend Yield (%)", 0.0)
    bucket = row.get("Bucket", "")

    if not pd.isna(xirr):
        if xirr >= 15:
            score += 20
        elif xirr >= 12:
            score += 15
        elif xirr >= 10:
            score += 10
        elif xirr >= 8:
            score += 5
        elif xirr < 0:
            score -= 10

    if not pd.isna(pnl):
        if pnl < -15:
            score += 3  # possibly undervalued
        elif pnl > 40:
            score -= 3  # already run up a lot

    try:
        if float(divy) > 0.5:
            score += 3
    except Exception:
        pass

    if bucket == "Super Core":
        score += 10
    elif bucket == "Core":
        score += 5
    elif bucket == "Exit":
        score -= 10

    return float(round(max(0.0, min(100.0, score)), 1))


def build_normalised_df(df_raw: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    if mapping.get("scheme") is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Scheme Name"] = df_raw[mapping["scheme"]].astype(str).str.strip()

    out["Category"] = (
        df_raw[mapping["category"]].astype(str).str.strip()
        if mapping.get("category") and mapping["category"] in df_raw.columns
        else "Unknown"
    )
    out["Sub Category"] = (
        df_raw[mapping["subcategory"]].astype(str).str.strip()
        if mapping.get("subcategory") and mapping["subcategory"] in df_raw.columns
        else "Unknown"
    )

    # Invested & Current values (₹)
    if mapping.get("invested") and mapping["invested"] in df_raw.columns:
        invested = pd.to_numeric(df_raw[mapping["invested"]], errors="coerce").fillna(0.0)
    else:
        invested = pd.Series(0.0, index=df_raw.index)

    if mapping.get("current") and mapping["current"] in df_raw.columns:
        current = pd.to_numeric(df_raw[mapping["current"]], errors="coerce").fillna(0.0)
    else:
        current = pd.Series(0.0, index=df_raw.index)

    out["Invested (₹)"] = invested.round(2)
    out["Current Value (₹)"] = current.round(2)
    out["P&L (₹)"] = (out["Current Value (₹)"] - out["Invested (₹)"]).round(2)
    out["P&L (%)"] = np.where(
        out["Invested (₹)"] > 0,
        (out["P&L (₹)"] / out["Invested (₹)"]) * 100.0,
        np.nan,
    )
    out["P&L (%)"] = out["P&L (%)"].round(1)

    # XIRR from file if present
    if mapping.get("xirr") and mapping["xirr"] in df_raw.columns:
        xirr = pd.to_numeric(df_raw[mapping["xirr"]], errors="coerce")
        out["XIRR (%)"] = xirr
    else:
        out["XIRR (%)"] = np.nan

    # Dividend Yield
    if mapping.get("dividend") and mapping["dividend"] in df_raw.columns:
        divy = pd.to_numeric(df_raw[mapping["dividend"]], errors="coerce").fillna(0.0)
        out["Dividend Yield (%)"] = divy.round(2)
    else:
        out["Dividend Yield (%)"] = 0.0

    # Enhance XIRR from internet when missing, then fill with 8%
    out = enhance_xirr_with_online_data(out)
    out["XIRR (%)"] = out["XIRR (%)"].round(1)

    # Buckets & AI fields
    buckets = []
    tgt_years = []
    horizons = []
    recos = []
    reasons = []
    scores = []

    for _, r in out.iterrows():
        bucket = classify_bucket(
            r.get("Category", ""),
            r.get("Sub Category", ""),
            r.get("XIRR (%)", np.nan),
            r.get("P&L (%)", np.nan),
            r.get("Scheme Name", ""),
        )
        tgt, horizon = target_year_and_horizon(bucket)
        reco = recommendation_from_bucket(bucket)
        reason = bucket_reason(r)
        temp_row = r.to_dict()
        temp_row["Bucket"] = bucket
        score = compute_ai_score(pd.Series(temp_row))

        buckets.append(bucket)
        tgt_years.append(tgt)
        horizons.append(horizon)
        recos.append(reco)
        reasons.append(reason)
        scores.append(score)

    out["Bucket"] = buckets
    out["Target Year"] = tgt_years
    out["Suggested Horizon"] = horizons
    out["Recommendation"] = recos
    out["Bucket Reason"] = reasons
    out["AI Score"] = scores

    # Sort by Target Year then Bucket strength
    bucket_rank = {"Super Core": 1, "Core": 2, "Satellite": 3, "Medium": 4, "Weak": 5, "Exit": 6}
    out["_bucket_rank"] = out["Bucket"].map(bucket_rank).fillna(9)
    out = out.sort_values(["Target Year", "_bucket_rank", "AI Score"], ascending=[True, True, False]).reset_index(drop=True)
    out.drop(columns=["_bucket_rank"], inplace=True)

    return out


# ==========================
# NFO Fetch
# ==========================

def fetch_nfo_data() -> pd.DataFrame:
    """
    Try a couple of public pages to get NFO list via read_html.
    If fails, return empty df and we show links instead.
    """
    urls = [
        "https://www.valueresearchonline.com/funds/new-fund-offers/",
        "https://www.amfiindia.com/new-fund-offer",
    ]
    for url in urls:
        try:
            tables = pd.read_html(url)
            if tables:
                df = tables[0]
                return df
        except Exception:
            continue
    return pd.DataFrame()


# ==========================
# Display Helpers
# ==========================

def portfolio_snapshot(df_norm: pd.DataFrame):
    total_inv = float(df_norm["Invested (₹)"].sum())
    total_curr = float(df_norm["Current Value (₹)"].sum())
    total_pnl = total_curr - total_inv
    pnl_pct = (total_pnl / total_inv * 100.0) if total_inv > 0 else np.nan

    # Weighted XIRR (approx). If missing, assume 8%.
    if "XIRR (%)" in df_norm.columns and not df_norm["XIRR (%)"].isna().all():
        xirr_series = df_norm["XIRR (%)"].fillna(8.0)
        if total_inv > 0:
            weights = df_norm["Invested (₹)"]
            if weights.sum() > 0:
                weights = weights / weights.sum()
                portfolio_xirr = float((xirr_series * weights).sum())
            else:
                portfolio_xirr = 8.0
        else:
            portfolio_xirr = 8.0
    else:
        portfolio_xirr = 8.0  # default XIRR if not found anywhere

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Investment (₹)", f"₹{total_inv:,.0f}")
    with c2:
        st.metric("Current Value (₹)", f"₹{total_curr:,.0f}")
    with c3:
        if not pd.isna(pnl_pct):
            st.metric("Total P&L (₹)", f"₹{total_pnl:,.0f}", f"{pnl_pct:.1f}%")
        else:
            st.metric("Total P&L (₹)", f"₹{total_pnl:,.0f}")
    with c4:
        st.metric("Portfolio XIRR (approx)", f"{portfolio_xirr:.1f}% p.a.")

    return total_curr, portfolio_xirr


def show_projection_table(current_value: float, portfolio_xirr: float):
    st.markdown("### 🔮 Portfolio Projections (INR)")

    if current_value <= 0:
        st.info("Not enough data to compute projections (current value is 0).")
        return

    # If XIRR somehow NaN, assume 8%
    if pd.isna(portfolio_xirr):
        portfolio_xirr = 8.0

    rate = portfolio_xirr / 100.0
    years_list = [5, 10, 15, 20]
    rows = []
    for y in years_list:
        fv = current_value * ((1 + rate) ** y)
        gain = fv - current_value
        rows.append(
            {
                "Years": y,
                "Current Value (₹)": f"₹{current_value:,.0f}",
                "Projected Value (₹)": f"₹{fv:,.0f}",
                "Gain (₹)": f"₹{gain:,.0f}",
            }
        )

    df_proj = pd.DataFrame(rows)
    st.markdown(
        df_proj.to_html(classes="dark-table", index=False, escape=False),
        unsafe_allow_html=True,
    )
    st.caption(f"Projection uses portfolio XIRR ≈ {portfolio_xirr:.1f}% p.a. (if XIRR was missing, 8% was assumed).")


def show_bucket_tables(df_norm: pd.DataFrame):
    st.markdown("### 🧠 AI Buckets & Plans (Sub Category hidden, all values in ₹)")

    # Display columns: Recommendation second column, no Sub Category
    display_cols = [
        "Scheme Name",
        "Recommendation",
        "Category",
        "Target Year",
        "Bucket",
        "Suggested Horizon",
        "Invested (₹)",
        "Current Value (₹)",
        "P&L (₹)",
        "P&L (%)",
        "XIRR (%)",
        "Dividend Yield (%)",
        "Bucket Reason",
    ]
    display_cols = [c for c in display_cols if c in df_norm.columns]

    # Super Core + Core table
    st.subheader("🏛 Core Portfolio (Super Core + Core)")
    core = df_norm[df_norm["Bucket"].isin(["Super Core", "Core"])]
    if core.empty:
        st.info("No Super Core / Core funds detected yet.")
    else:
        df_show = core[display_cols].copy()
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.subheader("🚀 Satellite & Thematic Bets")
    sat = df_norm[df_norm["Bucket"] == "Satellite"]
    if sat.empty:
        st.info("No Satellite funds detected.")
    else:
        df_show = sat[display_cols].copy()
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.subheader("🧊 Medium / Weak / Exit (Review Zone)")
    rev = df_norm[df_norm["Bucket"].isin(["Medium", "Weak", "Exit"])]
    if rev.empty:
        st.info("No Medium / Weak / Exit funds detected.")
    else:
        df_show = rev[display_cols].copy()
        st.dataframe(df_show, use_container_width=True, hide_index=True)


def show_category_allocation(df_norm: pd.DataFrame):
    st.markdown("### 🧩 Category Allocation (by Current Value in ₹)")
    if "Category" not in df_norm.columns:
        st.info("Category column not present.")
        return
    total = df_norm["Current Value (₹)"].sum()
    if total <= 0:
        st.info("No non-zero current value for allocation.")
        return

    grp = (
        df_norm.groupby("Category")["Current Value (₹)"]
        .sum()
        .reset_index()
        .sort_values("Current Value (₹)", ascending=False)
    )
    grp["Weight (%)"] = (grp["Current Value (₹)"] / total * 100.0).round(1)

    st.dataframe(grp, use_container_width=True, hide_index=True)


def show_full_table(df_norm: pd.DataFrame):
    st.markdown("### 📋 Full Normalised Table (₹ values)")

    display_cols = [
        "Scheme Name",
        "Recommendation",
        "Category",
        "Target Year",
        "Bucket",
        "Suggested Horizon",
        "Invested (₹)",
        "Current Value (₹)",
        "P&L (₹)",
        "P&L (%)",
        "XIRR (%)",
        "Dividend Yield (%)",
        "Bucket Reason",
        "AI Score",
    ]
    display_cols = [c for c in display_cols if c in df_norm.columns]

    df_show = df_norm[display_cols].copy()
    st.dataframe(df_show, use_container_width=True, hide_index=True)


def show_top10_scanner(df_norm: Optional[pd.DataFrame]):
    st.markdown("### 🏆 Top 10 Mutual Funds (AI Scanner on Your Portfolio – ₹ values)")
    if df_norm is None or df_norm.empty:
        st.info("Upload your portfolio file in the main section to see AI-ranked Top 10 funds from your holdings.")
        return

    # Top 10 by AI Score, but favour Super Core / Core / Satellite
    bucket_weight = {"Super Core": 3, "Core": 2, "Satellite": 1, "Medium": 0, "Weak": -1, "Exit": -2}
    df = df_norm.copy()
    df["bucket_weight"] = df["Bucket"].map(bucket_weight).fillna(0)
    df["scanner_score"] = df["AI Score"] + df["bucket_weight"] * 3.0
    df = df.sort_values("scanner_score", ascending=False).head(10)

    display_cols = [
        "Scheme Name",
        "Recommendation",  # second column
        "Bucket",
        "Target Year",
        "XIRR (%)",
        "P&L (%)",
        "AI Score",
        "Category",
        "Suggested Horizon",
        "Invested (₹)",
        "Current Value (₹)",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    st.caption("⚠️ This scanner ranks funds *within your portfolio* based on AI-style scoring: XIRR, P&L, category type and long-term role (core vs satellite). All values are in ₹ where applicable.")


def show_nfo_tab():
    st.markdown("### 🆕 New Fund Offers (NFO)")

    df_nfo = fetch_nfo_data()
    if df_nfo is not None and not df_nfo.empty:
        st.info("Live NFO list fetched from public MF sites (AMFI / ValueResearch style).")
        st.dataframe(df_nfo.head(30), use_container_width=True, hide_index=True)
    else:
        st.warning("Could not automatically fetch NFO data right now.")
        st.markdown(
            """
You can check the latest New Fund Offers here (values in ₹ as per their sites):

- AMFI NFO list  
- ValueResearch New Fund Offers  
- 5paisa / ICICIDirect NFO pages  

(Links intentionally not clickable here to keep the app lightweight.)
"""
        )


# ==========================
# Main App
# ==========================

def main():
    render_sidebar()

    st.markdown(
        """
    <div style="padding:16px;border-radius:16px;background:linear-gradient(120deg,#4f46e5,#0ea5e9);color:white;margin-bottom:12px;">
        <h2 style="margin:0 0 6px 0;">💹 AI Mutual Fund Analysis Bot</h2>
        <p style="margin:0;font-size:0.9rem;">
            Upload your MF portfolio • Get AI-based buckets & horizon • Auto Telegram recommendations • NFO watcher
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📁 Upload Mutual Fund Portfolio (values in ₹)")

    uploaded = st.file_uploader("Upload portfolio export (CSV / Excel)", type=["csv", "xls", "xlsx"])
    df_raw = None
    df_norm = None

    if uploaded is not None:
        df_raw = load_portfolio_file(uploaded)
        if df_raw is not None and not df_raw.empty:
            st.markdown("#### 🔍 Raw Preview (first 10 rows)")
            st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

            mapping = auto_map_columns(df_raw)
            if mapping.get("scheme") is not None:
                df_norm = build_normalised_df(df_raw, mapping)
            else:
                st.stop()
        else:
            st.stop()

    # Tabs: portfolio = only present + future; recos in separate tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Overview",
        "🤖 AI Recommendations",
        "🏆 AI Top 10 Scanner",
        "🆕 NFO Radar",
    ])

    with tab1:
        if df_norm is None or df_norm.empty:
            st.info("Upload your mutual fund portfolio file above to see present and future portfolio value here.")
        else:
            current_val, port_xirr = portfolio_snapshot(df_norm)
            show_projection_table(current_val, port_xirr)

    with tab2:
        if df_norm is None or df_norm.empty:
            st.info("Upload your portfolio to see AI-based recommendations, buckets, and detailed tables.")
        else:
            show_bucket_tables(df_norm)
            st.markdown("---")
            show_category_allocation(df_norm)
            st.markdown("---")
            show_full_table(df_norm)

    with tab3:
        show_top10_scanner(df_norm)

    with tab4:
        show_nfo_tab()

    # Telegram: manual send
    if df_norm is not None and st.session_state.get("send_now_flag", False):
        msg = generate_telegram_reco_text(df_norm)
        if msg:
            resp = send_telegram_message(msg)
            if resp.get("ok"):
                st.success("Telegram recommendations sent successfully.")
            else:
                st.error(f"Telegram error: {resp}")
        else:
            st.info("Nothing to send yet (no bucketed funds).")
        st.session_state["send_now_flag"] = False

    # Auto scheduled notifications (needs portfolio)
    if df_norm is not None:
        handle_scheduled_notifications(df_norm)


if __name__ == "__main__":
    main()



