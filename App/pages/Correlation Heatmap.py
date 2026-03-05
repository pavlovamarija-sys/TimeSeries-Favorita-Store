import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Correlation Heatmap", layout="wide")
st.title("🔥 Correlation Heatmap (Drivers vs Sales)")

st.markdown(
    """
This page shows the **Pearson correlation** between `unit_sales` and all **numeric driver features**.
It helps quickly identify which external variables move most closely with sales (positive or negative).
"""
)

# -----------------------------
# Load data
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../TimeSeries
DATA_PATH = ROOT / "Data" / "cleaned_timeseries.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

# Make duplicate column names unique (safe guard)
if df.columns.duplicated().any():
    counts = {}
    new_cols = []
    for c in df.columns:
        n = counts.get(c, 0)
        new_cols.append(c if n == 0 else f"{c}_{n}")
        counts[c] = n + 1
    df.columns = new_cols

# Ensure main columns exist
if "unit_sales" not in df.columns:
    st.error("Missing required column: unit_sales")
    st.stop()

# Convert numeric columns
for c in df.columns:
    if c != "date":
        df[c] = pd.to_numeric(df[c], errors="ignore")

# Parse date if present
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# -----------------------------
# Controls
# -----------------------------
st.subheader("Controls")

min_nonnull = st.slider(
    "Minimum % of non-missing values (numeric columns)",
    0.2, 1.0, 0.8, 0.05
)

# pick numeric columns with enough non-null
numeric_cols = []
for c in df.columns:
    if c in ["date"]:
        continue
    s = pd.to_numeric(df[c], errors="coerce")
    if s.notna().mean() >= min_nonnull:
        numeric_cols.append(c)

if "unit_sales" not in numeric_cols:
    numeric_cols.append("unit_sales")

# optional: allow excluding some columns
exclude_cols = st.multiselect(
    "Exclude columns",
    options=[c for c in numeric_cols if c != "unit_sales"],
    default=[]
)

cols_for_corr = [c for c in numeric_cols if c not in exclude_cols]

# -----------------------------
# Correlations with sales
# -----------------------------
st.subheader("Correlations with unit_sales")

corr_series = (
    df[cols_for_corr]
    .apply(pd.to_numeric, errors="coerce")
    .corr(numeric_only=True)["unit_sales"]
    .drop("unit_sales", errors="ignore")
    .dropna()
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

if corr_series.empty:
    st.warning("No numeric columns available for correlation under the current filters.")
    st.stop()

top_n = st.slider("How many top drivers to show", 5, min(30, len(corr_series)), min(15, len(corr_series)))
top_features = corr_series.head(top_n).index.tolist()

heat_df = (
    df[["unit_sales"] + top_features]
    .apply(pd.to_numeric, errors="coerce")
    .corr()
)

# -----------------------------
# Heatmap plot (matplotlib)
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heat_df.values, aspect="auto")

ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns, rotation=45, ha="right")
ax.set_yticks(range(len(heat_df.index)))
ax.set_yticklabels(heat_df.index)

# Annotate values
for i in range(heat_df.shape[0]):
    for j in range(heat_df.shape[1]):
        ax.text(j, i, f"{heat_df.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Correlation Heatmap (unit_sales + top drivers)")

st.pyplot(fig)

# -----------------------------
# Table + quick interpretation
# -----------------------------
st.subheader("Top correlations (absolute)")

corr_table = corr_series.head(top_n).reset_index()
corr_table.columns = ["Feature", "Correlation_with_unit_sales"]
st.dataframe(corr_table, use_container_width=True)

best_feature = corr_series.index[0]
best_corr = float(corr_series.iloc[0])

direction = "positive" if best_corr > 0 else "negative"
strength = abs(best_corr)

if strength < 0.1:
    strength_txt = "very weak"
elif strength < 0.3:
    strength_txt = "weak"
elif strength < 0.5:
    strength_txt = "moderate"
else:
    strength_txt = "strong"

st.info(
    f"**Quick takeaway:** The strongest linear relationship is **{best_feature}** "
    f"with a **{strength_txt} {direction}** correlation (≈ {best_corr:.2f}). "
    "Remember: correlation is not causation, and non-linear relationships may not show up here."
)