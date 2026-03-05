import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Time Series Project", layout="wide")
st.title("📈 Time Series Forecasting Project")

# -----------------------------
# Load data
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent  # App/ -> project root
DATA_PATH = ROOT / "Data" / "cleaned_timeseries.csv"

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

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

@st.cache_resource
def load_rf_model(model_path: Path):
    return joblib.load(model_path)

def get_rf_feature_columns(model, df_source: pd.DataFrame):
    """
    Prefer the exact training feature list (model.feature_names_in_).
    Fallback: all numeric columns except date/unit_sales/forecast.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    numeric_cols = []
    for c in df_source.columns:
        if c in ["date", "unit_sales", "forecast", "rf_forecast", "sarima_forecast"]:
            continue
        s = df_source[c]
        if is_numeric(s, threshold=0.9):
            numeric_cols.append(c)
    return numeric_cols

@st.cache_data(show_spinner=False)
def load_sarima_forecast(csv_path: Path) -> pd.Series | None:
    """
    Loads models/sarima_forecast.csv created in the notebook.
    Expects a datetime index column and a 'sarima' column.
    Returns a pd.Series with DateTimeIndex or None if not usable.
    """
    if not csv_path.exists():
        return None
    df_s = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if "sarima" not in df_s.columns:
        return None
    s = df_s["sarima"].copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

# -----------------------------
# Overview
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

    # Forecast demo controls
    st.markdown("**Forecast demo**")
    forecast_mode = st.checkbox("Forecast demo mode", value=True)

    horizon = st.slider("Forecast horizon (days)", 7, 60, 30, step=1, disabled=not forecast_mode)
    method = st.selectbox(
        "Forecast method",
        ["Last value", "Rolling mean (7 days)"],
        disabled=not forecast_mode,
    )

    # Model comparison controls (RF + SARIMA)
    st.markdown("---")
    st.markdown("**Model comparison**")

    use_rf = st.checkbox("Add Random Forest forecast", value=True, disabled=not forecast_mode)
    st.caption("Requires: models/rf_model.pkl")

    use_sarima = st.checkbox("Add SARIMA forecast (loaded)", value=True, disabled=not forecast_mode)
    st.caption("Requires: models/sarima_forecast.csv")

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

# -----------------------------
# RIGHT: plot
# -----------------------------
with right:
    st.subheader("Plot")

    # Keep a 'model base' that contains ALL columns for RF feature creation,
    # and a 'plot base' with just date/unit_sales for the main line.
    model_base = df_filt.dropna(subset=["date", "unit_sales"]).sort_values("date").copy()
    plot_base = model_base[["date", "unit_sales"]].copy()

    # --- Create the RF features that the saved model expects ---
    model_base = model_base.sort_values("date").copy()
    model_base["unit_sales"] = pd.to_numeric(model_base["unit_sales"], errors="coerce")

    # lags
    model_base["lag_1"] = model_base["unit_sales"].shift(1)
    model_base["lag_7"] = model_base["unit_sales"].shift(7)
    model_base["lag_14"] = model_base["unit_sales"].shift(14)

    # rolling features (use shift(1) to avoid leakage)
    model_base["rolling_mean_7"] = model_base["unit_sales"].shift(1).rolling(7).mean()
    model_base["rolling_std_7"] = model_base["unit_sales"].shift(1).rolling(7).std()

    # weekend flag
    model_base["is_weekend"] = (pd.to_datetime(model_base["date"]).dt.dayofweek >= 5).astype(int)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Forecast split
    train_df = plot_base.copy()
    test_df = None
    cutoff = None

    # Preload SARIMA series (if user wants it) so we can align later
    SARIMA_PATH = ROOT / "models" / "sarima_forecast.csv"
    sarima_series = load_sarima_forecast(SARIMA_PATH) if forecast_mode and use_sarima else None

    if forecast_mode:
        if len(plot_base) > horizon + 10:
            cutoff = plot_base["date"].max() - pd.Timedelta(days=horizon)
            train_df = plot_base[plot_base["date"] <= cutoff].copy()
            test_df = plot_base[plot_base["date"] > cutoff].copy()

            if not train_df.empty and not test_df.empty:
                # Demo forecast (baseline)
                if method == "Last value":
                    yhat = train_df["unit_sales"].iloc[-1]
                else:
                    yhat = train_df["unit_sales"].tail(7).mean()

                test_df["forecast"] = yhat

                # Random Forest forecast
                if use_rf:
                    MODEL_PATH = ROOT / "models" / "rf_model.pkl"
                    if MODEL_PATH.exists():
                        try:
                            model_obj = load_rf_model(MODEL_PATH)

                            # handle case where pickle stores a dict
                            if isinstance(model_obj, dict):
                                model = model_obj.get("model", None)
                                if model is None:
                                    st.warning("Random Forest model not found inside the saved dictionary.")
                            else:
                                model = model_obj

                            # Align model_base with same cutoff split, so RF sees the right rows
                            model_test = model_base[model_base["date"] > cutoff].copy()
                            model_test = model_test.iloc[: len(test_df)].copy()

                            feat_cols = get_rf_feature_columns(model, model_base)

                            missing_feats = [c for c in feat_cols if c not in model_test.columns]
                            if missing_feats:
                                st.warning(
                                    "Random Forest forecast skipped because these feature columns are missing:\n"
                                    + ", ".join(missing_feats)
                                )
                            else:
                                X = model_test[feat_cols].copy()
                                for c in X.columns:
                                    X[c] = pd.to_numeric(X[c], errors="coerce")
                                X = X.fillna(0)

                                rf_pred = model.predict(X)
                                test_df["rf_forecast"] = rf_pred
                        except Exception as e:
                            st.warning(f"Could not compute Random Forest forecast: {e}")
                    else:
                        st.warning(f"Random Forest model file not found: {MODEL_PATH}")

                # SARIMA forecast (loaded from CSV; align to this test window)
                if use_sarima:
                    if sarima_series is None:
                        st.warning(f"SARIMA forecast file not found or invalid: {SARIMA_PATH}")
                    else:
                        try:
                            sar = sarima_series.reindex(pd.to_datetime(test_df["date"]))
                            # If too many NaNs, it likely comes from a different split/horizon
                            nan_rate = float(sar.isna().mean())
                            if nan_rate > 0.25:
                                st.warning(
                                    "SARIMA forecast loaded but does not align well with the current test window "
                                    "(likely saved with a different split). Re-save SARIMA with the same horizon."
                                )
                            test_df["sarima_forecast"] = sar.values
                        except Exception as e:
                            st.warning(f"Could not align SARIMA forecast: {e}")

            else:
                test_df = None
                cutoff = None
        else:
            st.warning("Not enough rows in the selected date range for the chosen forecast horizon.")
            forecast_mode = False
            train_df = plot_base
            test_df = None
            cutoff = None

    # Plot actuals + forecasts
    ax1.plot(train_df["date"], train_df["unit_sales"], label="Actual (train)")
    if test_df is not None and not test_df.empty:
        ax1.plot(test_df["date"], test_df["unit_sales"], label="Actual (test)")

        # Baseline demo forecast (black dashed)
        ax1.plot(
            test_df["date"],
            test_df["forecast"],
            linestyle="--",
            color="black",
            linewidth=2,
            label="Forecast (demo)",
        )

        # Random Forest forecast (dashed)
        if "rf_forecast" in test_df.columns:
            ax1.plot(
                test_df["date"],
                test_df["rf_forecast"],
                linestyle="--",
                linewidth=2,
                label="Forecast (Random Forest)",
            )

        # SARIMA forecast (dotted)
        if "sarima_forecast" in test_df.columns:
            ax1.plot(
                test_df["date"],
                test_df["sarima_forecast"],
                linestyle=":",
                linewidth=2,
                label="Forecast (SARIMA)",
            )

        # cutoff line
        ax1.axvline(cutoff, linestyle=":", linewidth=1)
        ax1.text(cutoff, ax1.get_ylim()[1], " cutoff", va="top", ha="left", fontsize=9)

    ax1.set_xlabel("date")
    ax1.set_ylabel("unit_sales")
    ax1.set_title("unit_sales over time (with overlays)")

    # Overlay colors
    overlay_colors = itertools.cycle([
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # grey
    ])

    ax2 = None
    skipped = []

    for col in selected_overlays:
        s = df_filt[col]
        color = next(overlay_colors)

        # Binary overlay -> markers where 1
        if is_binary_01(s):
            mask = safe_binary(s) == 1
            marked = df_filt.loc[mask, ["date", "unit_sales"]].dropna().sort_values("date")
            if not marked.empty:
                ax1.scatter(marked["date"], marked["unit_sales"], s=30, label=col, color=color)
            continue

        # Numeric overlay -> line on right y-axis
        if is_numeric(s):
            if ax2 is None:
                ax2 = ax1.twinx()
                ax2.set_ylabel("overlay (numeric)")

            overlay_series = df_filt[["date", col]].copy()
            overlay_series[col] = to_num(overlay_series[col])
            overlay_series = overlay_series.dropna().sort_values("date")

            if not overlay_series.empty:
                ax2.plot(
                    overlay_series["date"],
                    overlay_series[col],
                    label=col,
                    color=color,
                    linewidth=2,
                )
            continue

        # Categorical overlay -> only if very few unique values (<= 5)
        nunique = s.astype(str).nunique(dropna=True)
        if nunique <= 5:
            cat_df = df_filt.loc[s.notna(), ["date", "unit_sales", col]].dropna().sort_values("date")
            if len(cat_df) > 200:
                cat_df = cat_df.iloc[:: max(1, len(cat_df)//200)]
            ax1.scatter(
                cat_df["date"],
                cat_df["unit_sales"],
                s=15,
                label=f"{col} (cat)",
                color=color,
            )
        else:
            skipped.append(f"{col} (categorical, {nunique} unique)")

    # Legends: combine ax1 + ax2 if both exist
    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    st.pyplot(fig)

    # Forecast metrics
    if test_df is not None and not test_df.empty:
        # Demo metrics
        mae_demo = mean_absolute_error(test_df["unit_sales"], test_df["forecast"])
        rmse_demo = rmse(test_df["unit_sales"], test_df["forecast"])

        m1, m2 = st.columns(2)
        m1.metric("MAE (demo forecast)", f"{mae_demo:.2f}")
        m2.metric("RMSE (demo forecast)", f"{rmse_demo:.2f}")

        # Random Forest metrics (if available)
        if "rf_forecast" in test_df.columns:
            mae_rf = mean_absolute_error(test_df["unit_sales"], test_df["rf_forecast"])
            rmse_rf = rmse(test_df["unit_sales"], test_df["rf_forecast"])

            m3, m4 = st.columns(2)
            m3.metric("MAE (Random Forest)", f"{mae_rf:.2f}")
            m4.metric("RMSE (Random Forest)", f"{rmse_rf:.2f}")

        # SARIMA metrics (if available)
        if "sarima_forecast" in test_df.columns:
            valid = pd.notna(test_df["sarima_forecast"])
            if valid.any():
                st.info(
                    "SARIMA forecast is loaded from the statistical model notebook and shown for visual comparison only.")

                m5, m6 = st.columns(2)
                m5.metric("MAE (SARIMA)", f"{mae_sar:.2f}")
                m6.metric("RMSE (SARIMA)", f"{rmse_sar:.2f}")

        with st.expander("Show forecast table"):
            cols = ["date", "unit_sales", "forecast"]
            if "rf_forecast" in test_df.columns:
                cols.append("rf_forecast")
            if "sarima_forecast" in test_df.columns:
                cols.append("sarima_forecast")
            st.dataframe(test_df[cols], use_container_width=True)

    if skipped:
        st.warning("Skipped overlays (too many categories):\n- " + "\n- ".join(skipped))

    with st.expander("Show filtered data preview"):
        cols = ["date", "unit_sales"] + selected_overlays
        cols = [c for c in cols if c in df_filt.columns]
        st.dataframe(df_filt[cols].head(200), use_container_width=True)