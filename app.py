import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, List

st.set_page_config(
    page_title="🧠 MF Analysis Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- STYLES ------------------------- #
st.markdown(
    """
<style>
.stApp { background-color: #020617; color: #e5e7eb; }
body { background-color: #020617; color: #e5e7eb; }

.main-header {
    background: linear-gradient(120deg, #4f46e5 0%, #0ea5e9 100%);
    padding: 18px; border-radius: 18px; color: white; margin-bottom: 12px;
    box-shadow: 0 12px 28px rgba(15,23,42,0.55); border: 1px solid rgba(255,255,255,0.18);
}
.main-header h1 { margin-bottom: 4px; font-size: clamp(1.6rem, 3vw, 2.3rem); }
.main-header p { margin: 0; font-size: 0.9rem; opacity: 0.96; }
.status-badge {
    display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.07em; background: rgba(15,23,42,0.35);
    border: 1px solid rgba(226,232,240,0.7); margin-top: 6px;
}

.metric-card {
    padding: 12px; border-radius: 14px; background: #020617;
    border: 1px solid #1f2937;
    box-shadow: 0 4px 14px rgba(15,23,42,0.7);
    margin-bottom: 10px; color: #e5e7eb;
}
.metric-card h3 { font-size: 0.9rem; color: #e5e7eb; margin-bottom: 4px; }
.metric-card .value { font-size: 1.15rem; font-weight: 600; color: #f9fafb; }
.metric-card .sub { font-size: 0.8rem; color: #9ca3af; }

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

/* make all dataframes dark */
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

/* uploader button */
div[data-testid="stFileUploader"] section button {
    background-color: #111827 !important;
    color: #f9fafb !important;
    border: 1px solid #374151 !important;
    font-weight: 500 !important;
}
div[data-testid="stFileUploader"] section button:hover {
    background-color: #1f2937 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------- HELPERS ------------------------- #


def fmt_inr(x: float) -> str:
    try:
        return f"₹{x:,.2f}"
    except Exception:
        return "₹0.00"


def fmt_lakhs(x: float) -> str:
    try:
        return f"₹{x/1e5:.2f} L"
    except Exception:
        return "₹0.00 L"


def load_mf_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded, sep=None, engine="python")
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded)
        else:
            st.error("Only CSV, XLS, XLSX files supported.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()


def map_mf_columns(df: pd.DataFrame) -> (Optional[Dict[str, str]], Optional[str]):
    """
    Try to map Groww / CAMS / generic MF headers to logical keys.
    Keep it flexible using multiple possible header names.
    """
    lowered = {c.lower().strip(): c for c in df.columns}

    def find_col(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in lowered:
                return lowered[cand]
        return None

    mapping = {
        "scheme": find_col(["scheme name", "scheme", "fund name", "scheme_name"]),
        "category": find_col(["category", "scheme category", "fund category"]),
        "sub_category": find_col(["sub category", "scheme sub category", "sub_category"]),
        "invested": find_col(["invested amount", "investment amount", "amount invested", "cost value"]),
        "current": find_col(["current value", "value", "current amount", "market value"]),
        "xirr": find_col(["xirr", "xirr (since inception)", "return % (xirr)", "returns xirr"]),
        "ret_1y": find_col(["1y return", "1 year return", "returns 1y", "return 1y"]),
        "ret_3y": find_col(["3y return", "3 year return", "returns 3y", "return 3y"]),
        "ret_5y": find_col(["5y return", "5 year return", "returns 5y", "return 5y"]),
        "exp_ratio": find_col(["expense ratio", "expense_ratio", "ter"]),
        "risk": find_col(["risk", "risk label", "riskometer", "risk level"]),
    }

    required = ["scheme", "invested", "current"]
    missing = [k for k in required if mapping[k] is None]

    if missing:
        return None, (
            "Could not detect essential columns. I need at least: "
            "Scheme / Fund Name, Invested Amount, Current Value.\n\n"
            "Detected columns:\n" + ", ".join(df.columns.astype(str))
        )
    return mapping, None


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


# ------------------------- MF RULE ENGINE ------------------------- #


def classify_mf_row(row: pd.Series) -> (str, str):
    """
    Return (bucket, short_reason)
    Buckets: Super Core, Core, Satellite, Weak, Exit
    """
    scheme = str(row.get("Scheme Name", "")).upper()
    cat = str(row.get("Category", "")).upper()
    sub = str(row.get("Sub Category", "")).upper()
    risk = str(row.get("Risk", "")).upper()

    xirr = float(row.get("XIRR (%)", 0.0))
    r3 = float(row.get("3Y Return (%)", 0.0))
    r5 = float(row.get("5Y Return (%)", 0.0))
    gain_pct = float(row.get("Gain (%)", 0.0))

    is_index = any(k in scheme for k in ["NIFTY", "SENSEX", "INDEX"])
    is_large = "LARGE" in cat or "LARGE" in sub
    is_mid = "MID" in cat or "MID" in sub
    is_small = "SMALL" in cat or "SMALL" in sub
    is_sector = any(k in cat for k in ["SECTOR", "THEME"]) or any(
        k in sub for k in ["SECTOR", "THEME"]
    )

    high_risk = any(k in risk for k in ["HIGH", "VERY HIGH"])
    low_risk = any(k in risk for k in ["LOW", "MODERATE"])

    # Super Core: Broad market compounding
    if (is_index or is_large) and not is_sector and xirr >= 12 and r5 >= 12 and low_risk:
        return "Super Core", "Broad market compounder for 15–20+ years"

    # Core: Diversified equity with decent history
    if (is_large or is_mid or is_index) and xirr >= 10 and r3 >= 10 and not is_sector:
        return "Core", "Diversified equity suitable for 10–15 years"

    # Satellite: thematic / small & mid for aggression
    if is_small or is_sector or high_risk:
        if xirr >= 12 or r3 >= 15:
            return "Satellite", "High growth potential; limit allocation; 7–10 year bet"
        return "Weak", "Risky / thematic with average history"

    # Weak & Exit decisions
    if xirr < 6 and r3 < 8 and r5 < 8 and gain_pct < 10:
        return "Exit", "Long-term returns weak; consider switch to better funds"

    if xirr < 8 and r3 < 10 and gain_pct < 12:
        return "Weak", "Below-par compounding; keep under watch"

    # Default fallback
    return "Core", "Reasonable long-term candidate"


def suggest_horizon_mf(bucket: str) -> str:
    if bucket == "Super Core":
        return "20+ years (retirement / wealth core)"
    if bucket == "Core":
        return "10–15 years (main growth engine)"
    if bucket == "Satellite":
        return "5–10 years (aggressive satellite)"
    if bucket == "Weak":
        return "Review in 1–3 years; consider switch"
    return "0–1 year (gradual exit & redeploy)"


# ------------------------- APP ------------------------- #


def main():
    st.markdown(
        """
<div class="main-header">
  <h1>🧠 Mutual Fund Analysis Bot</h1>
  <p>Upload your MF portfolio (Groww / CAMS / KFin) • Get 20-year view • Core / Satellite buckets</p>
  <div class="status-badge">Long-term • Compounding • India</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📂 Upload Mutual Fund Portfolio File")
    st.write(
        "Supported: **CSV / XLS / XLSX**. Export from Groww / CAMS / KFin with columns like "
        "`Scheme Name, Category, Invested Amount, Current Value, XIRR, 1Y/3Y/5Y Returns, Expense Ratio, Risk`."
    )

    uploaded = st.file_uploader(
        "Drop your MF holdings file here", type=["csv", "xls", "xlsx"], key="mf_file"
    )

    if uploaded is None:
        st.info("Upload your mutual fund holdings file to begin analysis.")
        return

    df_raw = load_mf_file(uploaded)
    if df_raw.empty:
        return

    mapping, err = map_mf_columns(df_raw)
    if err:
        st.error(err)
        st.write("Raw preview:")
        st.dataframe(df_raw.head(), use_container_width=True)
        return

    st.markdown("#### 🔍 Raw Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    # Normalised DF
    df = pd.DataFrame()
    df["Scheme Name"] = df_raw[mapping["scheme"]].astype(str)
    df["Category"] = (
        df_raw[mapping["category"]].astype(str)
        if mapping["category"]
        else "NA"
    )
    df["Sub Category"] = (
        df_raw[mapping["sub_category"]].astype(str)
        if mapping["sub_category"]
        else "NA"
    )

    df["Invested (₹)"] = to_num(df_raw[mapping["invested"]])
    df["Current (₹)"] = to_num(df_raw[mapping["current"]])
    df["Gain (₹)"] = df["Current (₹)"] - df["Invested (₹)"]
    df["Gain (%)"] = np.where(
        df["Invested (₹)"] > 0,
        df["Gain (₹)"] / df["Invested (₹)"] * 100.0,
        0.0,
    )

    if mapping["xirr"]:
        df["XIRR (%)"] = to_num(df_raw[mapping["xirr"]])
    else:
        df["XIRR (%)"] = np.nan

    if mapping["ret_1y"]:
        df["1Y Return (%)"] = to_num(df_raw[mapping["ret_1y"]])
    else:
        df["1Y Return (%)"] = np.nan

    if mapping["ret_3y"]:
        df["3Y Return (%)"] = to_num(df_raw[mapping["ret_3y"]])
    else:
        df["3Y Return (%)"] = np.nan

    if mapping["ret_5y"]:
        df["5Y Return (%)"] = to_num(df_raw[mapping["ret_5y"]])
    else:
        df["5Y Return (%)"] = np.nan

    if mapping["exp_ratio"]:
        df["Expense Ratio (%)"] = to_num(df_raw[mapping["exp_ratio"]])
    else:
        df["Expense Ratio (%)"] = np.nan

    if mapping["risk"]:
        df["Risk"] = df_raw[mapping["risk"]].astype(str)
    else:
        df["Risk"] = "NA"

    total_current = float(df["Current (₹)"].sum())
    total_invested = float(df["Invested (₹)"].sum())
    total_gain = float(df["Gain (₹)"].sum())
    overall_gain_pct = (
        total_gain / total_invested * 100.0 if total_invested > 0 else 0.0
    )

    df["Allocation (%)"] = np.where(
        total_current > 0, df["Current (₹)"] / total_current * 100.0, 0.0
    )

    # Simple portfolio "blended" XIRR approximation using Allocation
    if df["XIRR (%)"].notna().any():
        df["_w"] = df["Allocation (%)"] / 100.0
        blended_xirr = float((df["XIRR (%)"].fillna(0.0) * df["_w"]).sum())
    else:
        blended_xirr = np.nan

    # Bucketing
    buckets = []
    reasons = []
    horizons = []
    for _, r in df.iterrows():
        b, reason = classify_mf_row(r)
        buckets.append(b)
        reasons.append(reason)
        horizons.append(suggest_horizon_mf(b))

    df["Bucket"] = buckets
    df["Bucket Reason"] = reasons
    df["Suggested Horizon"] = horizons

    # ----------------- Portfolio Snapshot ----------------- #
    st.markdown("### 📈 Portfolio Snapshot")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-card'><h3>Total Invested</h3>"
            f"<div class='value'>{fmt_inr(total_invested)}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><h3>Current Value</h3>"
            f"<div class='value'>{fmt_inr(total_current)}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><h3>Total Gain</h3>"
            f"<div class='value'>{fmt_inr(total_gain)} ({overall_gain_pct:.2f}%)</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        if not np.isnan(blended_xirr):
            st.markdown(
                f"<div class='metric-card'><h3>Blended XIRR</h3>"
                f"<div class='value'>{blended_xirr:.2f}%</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='metric-card'><h3>Blended XIRR</h3>"
                "<div class='value'>N/A</div><div class='sub'>XIRR not available in file</div></div>",
                unsafe_allow_html=True,
            )

    snap_data = [
        {"Metric": "Total Invested", "Value": fmt_inr(total_invested)},
        {"Metric": "Current Value", "Value": fmt_inr(total_current)},
        {"Metric": "Total Gain", "Value": f"{fmt_inr(total_gain)} ({overall_gain_pct:.2f}%)"},
        {"Metric": "Blended XIRR", "Value": f"{blended_xirr:.2f}%" if not np.isnan(blended_xirr) else "N/A"},
    ]
    snap_df = pd.DataFrame(snap_data)
    st.markdown(snap_df.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

    # -------------- Category allocation -------------- #
    st.markdown("### 🧩 Category Allocation (by Current Value)")
    cat_alloc = df.groupby("Category", as_index=False)["Current (₹)"].sum()
    cat_alloc["Allocation (%)"] = np.where(
        total_current > 0, cat_alloc["Current (₹)"] / total_current * 100.0, 0.0
    )
    cat_alloc = cat_alloc.sort_values("Allocation (%)", ascending=False)
    st.dataframe(cat_alloc, use_container_width=True, hide_index=True)

    # -------------- Buckets -------------- #
    st.markdown("### 🤖 AI Buckets – Super Core / Core / Satellite / Weak / Exit")

    base_cols = [
        "Scheme Name",
        "Category",
        "Sub Category",
        "Invested (₹)",
        "Current (₹)",
        "Gain (₹)",
        "Gain (%)",
        "Allocation (%)",
        "XIRR (%)",
        "3Y Return (%)",
        "5Y Return (%)",
        "Risk",
        "Bucket Reason",
        "Suggested Horizon",
    ]

    bucket_info = [
        ("Super Core", "🌟 20+ year core compounding funds"),
        ("Core", "✅ Main 10–15 year growth funds"),
        ("Satellite", "🚀 Aggressive 5–10 year satellite bets"),
        ("Weak", "⚠️ Underperformers; keep under review"),
        ("Exit", "❌ Likely exit / switch candidates"),
    ]

    for bucket, desc in bucket_info:
        sub = df[df["Bucket"] == bucket]
        if sub.empty:
            continue
        st.markdown(f"#### {bucket} — {desc}")
        st.dataframe(
            sub[base_cols].sort_values("Allocation (%)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # Raw table at end if user wants full export view
    with st.expander("📋 Full Normalised Table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
