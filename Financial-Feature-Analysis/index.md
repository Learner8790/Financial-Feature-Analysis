# Project Report: A Quantitative Approach to Feature Selection and Return Prediction in the Indian Stock Market

**Version:** 1.0 
**Author:** Mohan Sai Balusu

---

## 1. Executive Summary

This project outlines a complete, end-to-end data science workflow designed to identify the key drivers of stock returns in the Indian equity market. Our primary objective was not simply to build a predictive model, but to develop a sophisticated, data-driven framework for understanding **which** market factors are most influential and **how** their importance shifts under different market conditions.

We successfully implemented a fully automated pipeline for data ingestion, feature engineering, and model training. The key achievement was the use of **Recursive Feature Elimination (RFECV)** to systematically prove that a concise set of 6 features was optimal for prediction. Our final, tuned classification model demonstrated a **statistically significant predictive edge (55.4% accuracy, 0.57 ROC AUC)** on unseen data. The project culminates in a dynamic regime analysis that provides deep, actionable insights into market behavior.

---

## 2. Phase 1: Data Acquisition & Pipeline

### 2.1. Objective
To programmatically source and prepare all necessary historical financial data, ensuring a repeatable, scalable, and reliable foundation for analysis.

### 2.2. Technical Stack
- **Data Sourcing:** `yfinance`
- **Data Manipulation:** `pandas`, `numpy`
- **Modeling & Evaluation:** `scikit-learn`, `lightgbm`
- **Visualization:** `matplotlib`, `seaborn`

### 2.3. Data Sourcing Strategy
Our final, robust pipeline sources two key datasets exclusively from Yahoo Finance:

* **Equity Price Data:** Daily OHLCV data for a diversified portfolio of 10 NIFTY 50 stocks, representing a cross-section of the Indian economy.
* **Indian Market Indicators:** Daily data for the `NIFTY 50`, `NIFTY Bank`, `NIFTY IT`, and the `USD/INR` exchange rate to provide essential market context.

A critical cleaning step was performed to remove non-trading days, ensuring data integrity for all subsequent calculations.

---

## 3. Phase 2: Feature Engineering & Target Definition

### 3.1. Objective
To transform the raw time-series data into a rich, informative feature set that our machine learning models can learn from.

### 3.2. Target Variable
We defined our prediction goal. Initially a regression problem, it was later refined to a more tractable classification problem:
* **`target_class`:** A binary variable. `1` if the stock's 21-day forward return was > 1%, and `0` otherwise.

### 3.3. Feature Creation
We generated a universe of 24 potential features for each stock on each day, including:
* **Momentum:** Returns over 21, 42, and 63-day periods.
* **Volatility:** Rolling standard deviation of daily returns over 21 and 63-day windows.
* **Technical Indicators:** A comprehensive set from the `pandas_ta` library, including RSI, MACD, Bollinger Bands, ADX, and Aroon Oscillator.
* **Market Context:** Daily returns of the NIFTY 50, NIFTY Bank, NIFTY IT, and USD/INR exchange rate.

---

## 4. Phase 3: Automated Feature Selection (RFECV)

### 4.1. The Core Problem
A model with too many features can be confused by noise, leading to poor performance on new data (overfitting). Our goal was to find the "vital few" features from the 24 we created.

### 4.2. Methodology
We employed **Recursive Feature Elimination with Cross-Validation (RFECV)**. This algorithm systematically:
1.  Trains a model on all features.
2.  Eliminates the weakest feature.
3.  Re-trains the model and evaluates its performance using time-series cross-validation.
4.  Repeats this process until a specified minimum is reached.

### 4.3. Results
The RFECV process provided data-driven proof that the model's performance peaked when using exactly **6 features**. This answered the question of "how many?"

The **6 "Golden Features"** identified were:
- `volatility_21d`
- `volatility_63d`
- `MACDs_12_26_9`
- `BBL_20_2.0` (Lower Bollinger Band)
- `BBU_20_2.0` (Upper Bollinger Band)
- `ADX_14` (Average Directional Index)

This was the most critical analytical step, providing a concise, powerful, and evidence-based feature set for our final model.

---

## 5. Phase 4: Model Training, Tuning & Evaluation

### 5.1. Initial Regression Model
Our first model attempted to predict the exact future return. It yielded a negative R-squared, confirming the immense difficulty of this task and providing the rationale to pivot to a classification approach.

### 5.2. Baseline Classification Model
Using our 6 golden features, we trained an `LGBMClassifier`.
* **Result:** This model achieved **54.0% accuracy** and a **0.56 ROC AUC score**, demonstrating a real, albeit small, predictive edge.

### 5.3. Dynamic Regime Analysis
This was a key analytical output. We trained separate models on "Bullish" (market up) and "Bearish" (market down) days and found that feature importances changed dramatically:
* **On Bullish Days:** The **Upper Bollinger Band** (a measure of price ceiling) was the most important feature.
* **On Bearish Days:** **Long-term Volatility** (a measure of risk) became the dominant factor.

### 5.4. Hyperparameter Tuning & Final Model
We conducted a final tuning step using `GridSearchCV` with regularization to prevent overfitting.
* **Result:** The final, optimized model's performance improved.
* **Final Accuracy:** **55.4%**
* **Final ROC AUC Score:** **0.57**

This concluded the modeling process, yielding a robust classifier with a proven statistical edge on unseen data.
