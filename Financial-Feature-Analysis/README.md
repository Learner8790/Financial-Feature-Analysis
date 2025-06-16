# Quantitative Feature Analysis for Stock Return Prediction

This project is an end-to-end data science workflow designed to identify key predictive features for stock returns in the Indian market. It demonstrates a complete pipeline from data acquisition and feature engineering to automated feature selection, model training, and dynamic analysis.

## Key Objectives & Findings
* **Automated Data Pipeline:** Programmatically ingests daily stock data and relevant Indian market indicators using `yfinance`.
* **Advanced Feature Selection:** Implements Recursive Feature Elimination with Cross-Validation (RFECV) to systematically identify the most powerful predictors from a larger set. Our analysis proved that a concise set of 6 features related to volatility, trend, and price channels was optimal.
* **Dynamic Regime Analysis:** The project includes an analysis showing how feature importances shift dramatically between "Bullish" and "Bearish" market regimes.
* **Optimized Classification Model:** The final deliverable is an `LGBMClassifier` tuned with regularization, which shows a statistically significant predictive edge in forecasting directional stock movements.

## Interactive Dashboard
An interactive dashboard for this project has been built with Streamlit and is available for live demonstration.

## How to Run
1.  Clone this repository.
2.  Install the required libraries: `pip install -r requirements.txt`
3.  To run the interactive dashboard: `streamlit run app.py`