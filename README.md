# 📈 Time Series Forecasting – Favorita Store

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pavlovamarija-sys-timeseries-favorita-store-appapp-ho3sqq.streamlit.app/)

Time series forecasting project combining statistical models, machine learning models, and external data sources to analyze and predict retail sales.
The project includes an interactive Streamlit dashboard for exploring demand patterns and forecasting results.

### Project Overview

This project analyzes daily retail sales data to understand demand patterns and improve forecasting accuracy.

The analysis integrates several external drivers that may influence retail demand, including:

* weather conditions
* holidays
* macroeconomic indicators
* oil prices

The workflow includes:

* exploratory time series analysis 
* feature engineering 
* statistical forecasting models 
* machine learning models 
* hyperparameter tuning 
* an interactive Streamlit dashboard for visualization

The final selected model is a tuned Random Forest model, while statistical models such as SARIMA are included for comparison.

### Data Sources

The dataset combines several sources:

#### Retail sales data 
* daily unit sales

#### Weather data (NASA POWER API)
* solar radiation
* temperature
* precipitation

#### Holiday data
* national and local holidays 
#### Macroeconomic indicators
* consumer price index
* minimum wage 
#### Oil prices
* global oil price index

### Feature Engineering

Several features were created to capture temporal structure and external drivers:

#### Lag features

* lag_1
* lag_7
* lag_14

#### Rolling statistics

* rolling_mean_7 
* rolling_std_7

#### Calendar variables

* day_of_week 
* month 
* is_weekend

#### External variables

* weather indicators (is_sunny, is_rainy) 
* oil price 
* holiday indicators
* macroeconomic variables (CPI, minimum wage)

These features allow machine learning models to capture short-term dynamics and weekly seasonality.

### Models Implemented
#### Statistical Time Series Models

* Naive baseline 
* Exponential Smoothing (ETS) 
* ARIMA 
* SARIMA 
* Theta Model 
* Prophet 

#### Machine Learning Models

* Linear Regression (with engineered features) 
* Random Forest 
* XGBoost

Hyperparameter tuning was performed using Hyperopt, and experiments were tracked with MLflow.

### Results Summary

Models capturing trend and weekly seasonality produced the best forecasts.

Machine learning models improved performance when lag features and external variables were included.

| Model                 | RMSE |
| --------------------- | ---- |
| Naive baseline        | ~173 |
| Random Forest (tuned) | ~127 |
| SARIMA                | ~108 |


#### Key insight:

* The time series shows strong weekly seasonality 
* Lag features significantly improve machine learning models 
* External variables provide additional explanatory power 
* Random Forest performed best overall when combining engineered features and external drivers

## Streamlit Interactive Dashboard

An interactive dashboard was built using Streamlit to explore the dataset.

Features include:

* interactive time series visualization 
* overlay of weather, holiday, and macroeconomic variables 
* dynamic filtering by date 
* forecast comparison between models 
* display of model evaluation metrics

### Project Files

Important project files:

### Streamlit Application

- Live dashboard  
  [Open Streamlit App](https://pavlovamarija-sys-timeseries-favorita-store-appapp-ho3sqq.streamlit.app/)

- Main dashboard  
  [App/app.py](App/app.py)

- Streamlit pages  
  [App/pages](App/pages)

### Dataset

- Cleaned time series dataset  
  [Data/cleaned_timeseries.csv](Data/cleaned_timeseries.csv)

### Analysis Notebooks

- Exploratory Data Analysis  
  [notebooks/EDA.ipynb](notebooks/EDA.ipynb)

- Baseline model comparison  
  [notebooks/baseline_comparison.ipynb](notebooks/baseline_comparison.ipynb)

- Feature engineering methods  
  [notebooks/feature_engineering_methods.ipynb](notebooks/feature_engineering_methods.ipynb)

- Machine learning feature engineering  
  [notebooks/feature_engineering_models.ipynb](notebooks/feature_engineering_models.ipynb)

- Statistical forecasting models  
  [notebooks/statistical_models.ipynb](notebooks/statistical_models.ipynb)

- Tuned model comparison  
  [notebooks/tuned_comparison.ipynb](notebooks/tuned_comparison.ipynb)

- Hyperparameter tuning with MLflow  
  [notebooks/week3_hyperopt_mlflow.ipynb](notebooks/week3_hyperopt_mlflow.ipynb)

### Running the Project

#### Clone the repository:

git clone https://github.com/pavlovamarija-sys/TimeSeries-Favorita-Store.git

cd TimeSeries-Favorita-Store

#### Install dependencies:

pip install -r requirements.txt

Run the Streamlit dashboard:

streamlit run App/app.py

streamlit run app.py

### Project Structure

TimeSeries-Favorita-Store
│
├── App                     # Streamlit dashboard
│   ├── app.py
│   └── pages
│
├── Data                    # Cleaned dataset
│
├── models                  # Saved models and evaluation results
│
├── notebooks               # Analysis and model development
│
├── requirements.txt
└── README.md

### Technologies Used

* Python 
* Pandas 
* Scikit-learn 
* Statsmodels 
* Darts 
* Hyperopt 
* MLflow 
* Matplotlib 
* Streamlit



