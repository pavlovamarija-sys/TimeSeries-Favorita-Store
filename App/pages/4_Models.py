import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Model Evaluation")

st.markdown("""
This table presents the final model comparison between statistical forecasting methods and tuned machine learning models.

### Statistical models
These models rely only on the historical time series to capture trend and seasonality.
- Naive seasonal baseline
- ETS (Exponential Smoothing)
- ARIMA
- SARIMA
- Theta
- Prophet (Additive)

### Machine learning models
These models incorporate external drivers such as weather conditions, holidays, and macroeconomic indicators through engineered features.
- Random Forest (tuned)
- XGBoost (tuned)

### Evaluation metrics
Models are evaluated using standard forecasting metrics:
- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **sMAPE** – Symmetric Mean Absolute Percentage Error
The final model is selected based on the lowest RMSE, which penalizes larger forecasting errors more strongly.
""")

st.divider()
st.subheader("Model Evaluation Metrics")
st.caption("🥇 Best overall model | 📊 Best statistical model | 🤖 Best machine learning model")

ROOT = Path(__file__).resolve().parents[2]
metrics_path = ROOT / "models" / "metrics_tuned.csv"

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)

    # Identify best models
    overall_best_idx = metrics["RMSE"].idxmin()

    stat_keywords = ["Naive", "ETS", "ARIMA", "SARIMA", "Theta", "Prophet"]
    ml_keywords = ["Regression", "Forest", "XGBoost"]

    stat_mask = metrics["Model"].astype(str).str.contains("|".join(stat_keywords), case=False, na=False)
    ml_mask = metrics["Model"].astype(str).str.contains("|".join(ml_keywords), case=False, na=False)

    stat_best_idx = metrics[stat_mask]["RMSE"].idxmin() if stat_mask.any() else None
    ml_best_idx   = metrics[ml_mask]["RMSE"].idxmin() if ml_mask.any() else None

    metrics_display = metrics.copy()

    # Add icons
    metrics_display.loc[overall_best_idx, "Model"] += " 🥇"
    if stat_best_idx is not None:
        metrics_display.loc[stat_best_idx, "Model"] += " 📊"
    if ml_best_idx is not None:
        metrics_display.loc[ml_best_idx, "Model"] += " 🤖"

    # Sort + format
    metrics_display = metrics_display.sort_values("RMSE")
    for c in ["MAE", "RMSE", "sMAPE"]:
        metrics_display[c] = pd.to_numeric(metrics_display[c], errors="coerce").round(1)

    # Display styled table
    st.dataframe(
        metrics_display.style.highlight_min(subset=["RMSE"], color="#d4edda"),
        use_container_width=True
    )
else:
    st.warning("models/metrics_tuned.csv not found. Add the file to display model results.")

st.success(
    "Final model selection: The tuned Random Forest model achieved the lowest RMSE "
    "and is selected as the final forecasting model. "
    "Among statistical models, SARIMA provides the best performance."
)
st.divider()
st.subheader("Key Insights")

st.markdown("""
- Statistical models capture **trend and seasonality** in the historical time series.
- Machine learning models incorporate **external drivers** such as weather conditions, holidays, and macroeconomic indicators through engineered features.
- Forecast accuracy is evaluated using **MAE, RMSE, and sMAPE**.
- In the final comparison, the **tuned Random Forest model achieved the lowest RMSE and is selected as the final forecasting model**, while **SARIMA performs best among the statistical approaches**.
""")