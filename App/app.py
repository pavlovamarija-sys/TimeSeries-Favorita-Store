import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Time Series Project", layout="wide")
st.title("📈 Time Series Forecasting Project")

# -----------------------------
# Load data
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "cleaned_timeseries.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

# Make duplicate column names unique
if df.columns.duplicated().any():
    counts = {}
    new_cols = []
    for c in df.columns:
        n = counts.get(c, 0)
        new_cols.append(c if n == 0 else f"{c}_{n}")
        counts[c] = n + 1
    df.columns = new_cols

# Required columns
if "date" not in df.columns:
    st.error("Missing required column: 'date'")
    st.stop()
if "unit_sales" not in df.columns:
    st.error("Missing required column: 'unit_sales'")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["unit_sales"] = pd.to_numeric(df["unit_sales"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

# -----------------------------
# Helpers
# -----------------------------
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def is_binary_01(s: pd.Series) -> bool:
    s_num = to_num(s).dropna()
    if s_num.empty:
        return False
    return set(s_num.unique()).issubset({0, 1})

def is_numeric(s: pd.Series, threshold: float = 0.8) -> bool:
    s_num = to_num(s)
    return s_num.notna().mean() >= threshold

def safe_binary(s: pd.Series) -> pd.Series:
    """Return 0/1 series (NaN -> 0)."""
    s_num = to_num(s).fillna(0)
    return (s_num == 1).astype(int)

# -----------------------------
# Compact overview
# -----------------------------
with st.expander("Dataset overview"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")
    st.dataframe(df.head(50), use_container_width=True)

st.header("Interactive Visualization")

left, right = st.columns([1, 2], gap="large")

# -----------------------------
# LEFT: controls
# -----------------------------
with left:
    st.subheader("Controls")

    # Date slider
    min_d, max_d = df["date"].min(), df["date"].max()
    start, end = st.slider(
        "Date range",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(min_d.to_pydatetime(), max_d.to_pydatetime()),
    )

    df_filt = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()
    st.caption(f"Rows after date filter: {len(df_filt):,}")

    st.markdown("---")
    st.markdown("**Default plot**")
    st.write("- X-axis: `date`")
    st.write("- Y-axis: `unit_sales`")

    st.markdown("---")
    st.markdown("**Add overlays (any other columns)**")

    overlay_candidates = [c for c in df_filt.columns if c not in ["date", "unit_sales"]]

    selected_overlays = st.multiselect(
        "Select columns to overlay",
        options=overlay_candidates,
        default=[],
    )

    # Quick summary of selected overlay types
    if selected_overlays:
        st.markdown("**Overlay types (auto):**")
        for col in selected_overlays:
            if is_binary_01(df_filt[col]):
                st.write(f"- {col}: binary → markers (value=1)")
            elif is_numeric(df_filt[col]):
                st.write(f"- {col}: numeric → 2nd axis line")
            else:
                nunique = df_filt[col].astype(str).nunique(dropna=True)
                st.write(f"- {col}: categorical ({nunique} unique) → may be skipped")

# -----------------------------
# RIGHT: plot
# -----------------------------
with right:
    st.subheader("Plot")

    base = df_filt[["date", "unit_sales"]].dropna().sort_values("date")
    x = base["date"]
    y = base["unit_sales"]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Base sales line
    ax1.plot(x, y)
    ax1.set_xlabel("date")
    ax1.set_ylabel("unit_sales")
    ax1.set_title("unit_sales over time (with overlays)")

    # Second axis for numeric overlays
    ax2 = None
    any_legend = False

    skipped = []

    for col in selected_overlays:
        s = df_filt[col]

        # 1) Binary overlay -> markers where 1
        if is_binary_01(s):
            mask = safe_binary(s) == 1
            marked = df_filt.loc[mask, ["date", "unit_sales"]].dropna().sort_values("date")
            if not marked.empty:
                ax1.scatter(marked["date"], marked["unit_sales"], s=30, label=col)
                any_legend = True
            continue

        # 2) Numeric overlay -> line on right y-axis
        if is_numeric(s):
            if ax2 is None:
                ax2 = ax1.twinx()
                ax2.set_ylabel("overlay (numeric)")

            overlay_series = df_filt[["date", col]].copy()
            overlay_series[col] = to_num(overlay_series[col])
            overlay_series = overlay_series.dropna().sort_values("date")

            if not overlay_series.empty:
                ax2.plot(overlay_series["date"], overlay_series[col], label=col)
                any_legend = True
            continue

        # 3) Categorical overlay -> only if very few unique values (<= 5)
        nunique = s.astype(str).nunique(dropna=True)
        if nunique <= 5:
            # Mark dates where category is not empty
            # (simple: mark every row; for many points it gets messy, so we keep it limited)
            cat_df = df_filt.loc[s.notna(), ["date", "unit_sales", col]].dropna().sort_values("date")
            # To avoid clutter: sample a bit if too many
            if len(cat_df) > 200:
                cat_df = cat_df.iloc[:: max(1, len(cat_df)//200)]
            ax1.scatter(cat_df["date"], cat_df["unit_sales"], s=15, label=f"{col} (cat)")
            any_legend = True
        else:
            skipped.append(f"{col} (categorical, {nunique} unique)")

    # Legends: combine ax1 + ax2 if both exist
    if any_legend:
        handles1, labels1 = ax1.get_legend_handles_labels()
        if ax2 is not None:
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
        else:
            ax1.legend(loc="upper left")

    st.pyplot(fig)

    if skipped:
        st.warning("Skipped overlays (too many categories):\n- " + "\n- ".join(skipped))

    with st.expander("Show filtered data preview"):
        cols = ["date", "unit_sales"] + selected_overlays
        cols = [c for c in cols if c in df_filt.columns]
        st.dataframe(df_filt[cols].head(200), use_container_width=True)