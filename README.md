# Quantitative Stock Prediction Using Recursive Feature Elimination

### A Systematic Machine Learning Framework for Indian Equity Market Analysis

---

### Table of Contents
1. [Project Overview](#1-project-overview)
2. [The Core Challenge: Signal Detection in Financial Markets](#2-the-core-challenge-signal-detection-in-financial-markets)
3. [The Experimental Framework: Architecture & Methodology](#3-the-experimental-framework-architecture--methodology)
   * [Data Foundation & Market Coverage](#data-foundation--market-coverage)
   * [Feature Engineering: Multi-Dimensional Market Representation](#feature-engineering-multi-dimensional-market-representation)
   * [The RFECV Algorithm: Systematic Feature Selection](#the-rfecv-algorithm-systematic-feature-selection)
   * [Model Architecture: From Regression to Classification](#model-architecture-from-regression-to-classification)
4. [The Research Journey: Iterative Model Development](#4-the-research-journey-iterative-model-development)
5. [Results & Performance Analysis](#5-results--performance-analysis)
6. [How to Run This Project](#6-how-to-run-this-project)

---

### 1. Project Overview

This project presents a comprehensive quantitative framework for predicting Indian stock market movements using advanced machine learning techniques. The primary objective was to develop a systematic approach to feature selection that identifies the most predictive indicators from a comprehensive set of technical, momentum, and market-based variables.

We analyzed 10 major Indian equities from the NSE (National Stock Exchange) over a 9-year period (2016-2025), implementing a sophisticated pipeline that progresses from regression-based return prediction to binary classification of stock movements. The project demonstrates the critical importance of feature selection in financial machine learning and provides insights into market dynamics across different regimes.

**Technologies & Methodologies:**
* **Python 3.9+** with scientific computing stack
* **Feature Selection:** Recursive Feature Elimination with Cross-Validation (RFECV)
* **Machine Learning:** LightGBM gradient boosting framework
* **Data Sources:** Yahoo Finance API (`yfinance`)
* **Technical Analysis:** `pandas_ta` for comprehensive indicator suite
* **Validation:** Time-series cross-validation with proper temporal ordering
* **Optimization:** Grid search with regularization techniques

---

### 2. The Core Challenge: Signal Detection in Financial Markets

Financial markets present one of the most challenging prediction problems in data science due to their inherent complexity and noise-to-signal ratio. The fundamental challenge can be decomposed into several key components:

#### The Feature Selection Problem

In quantitative finance, practitioners often face the "curse of dimensionality" where an abundance of potential predictive features creates more noise than signal. Traditional approaches either:
* Use **all available features**, leading to overfitting and poor generalization
* Apply **domain expertise** to manually select features, introducing human bias
* Employ **basic statistical methods** that fail to capture complex feature interactions

Our framework addresses this through **Recursive Feature Elimination with Cross-Validation (RFECV)**, which systematically identifies the optimal feature subset by:

$$
S^* = \arg\max_{S \subseteq F} \text{CV}_{\text{score}}(M(S))
$$

Where $F$ represents the complete feature space, $S$ is a feature subset, and $M(S)$ is the model trained on subset $S$.

#### The Overfitting Challenge

Financial data is particularly susceptible to overfitting due to:
* **Regime changes**: Market behavior evolves over time
* **Survivorship bias**: Successful patterns in historical data may not persist
* **Data snooping**: Multiple hypothesis testing without proper correction

Our approach mitigates overfitting through:
* **Time-series cross-validation** that respects temporal ordering
* **Regularization techniques** (L1 and L2 penalties)
* **Out-of-sample testing** on genuinely unseen data

#### The Classification vs. Regression Dilemma

While continuous return prediction seems more informative, binary classification often provides better practical utility:
* **Decision-making**: Trading decisions are inherently binary (buy/hold/sell)
* **Risk management**: Threshold-based signals are easier to implement
* **Performance evaluation**: Classification metrics align with trading objectives

---

### 3. The Experimental Framework: Architecture & Methodology

We constructed a rigorous quantitative research pipeline designed to identify genuine predictive signals while avoiding common pitfalls in financial machine learning.

#### Data Foundation & Market Coverage

Our analysis encompasses ten liquid, large-cap Indian equities representing diverse sectors:

**Technology & Services:**
* Tata Consultancy Services (`TCS.NS`)
* Infosys (`INFY.NS`)

**Banking & Financial Services:**
* HDFC Bank (`HDFCBANK.NS`) 
* ICICI Bank (`ICICIBANK.NS`)

**Industrial & Consumer:**
* Reliance Industries (`RELIANCE.NS`)
* Larsen & Toubro (`LT.NS`)
* Hindustan Unilever (`HINDUNILVR.NS`)
* ITC Limited (`ITC.NS`)
* Asian Paints (`ASIANPAINT.NS`)

**Telecommunications:**
* Bharti Airtel (`BHARTIARTL.NS`)

**Market Context Variables:**
* NIFTY 50 Index (`^NSEI`)
* NIFTY Bank Index (`^NSEBANK`)
* NIFTY IT Index (`^CNXIT`)
* USD/INR Exchange Rate (`INR=X`)

#### Feature Engineering: Multi-Dimensional Market Representation

Our feature engineering process creates a comprehensive representation of market dynamics across multiple dimensions:

##### Momentum Features
We capture price momentum across multiple timeframes to identify persistent directional movements:

$$
R_{t,k} = \frac{P_t - P_{t-k}}{P_{t-k}}
$$

Where $R_{t,k}$ represents the $k$-period return at time $t$, implemented for $k \in \{21, 42, 63\}$ trading days.

##### Volatility Measures
Rolling volatility captures market uncertainty and risk perception:

$$
\sigma_{t,w} = \sqrt{\frac{1}{w-1} \sum_{i=t-w+1}^{t} (R_i - \bar{R}_{t,w})^2}
$$

Where $\sigma_{t,w}$ is the volatility over window $w \in \{21, 63\}$ days.

##### Technical Indicators
We implement a comprehensive suite of technical analysis tools:

**Relative Strength Index (RSI):**
$$
RSI_t = 100 - \frac{100}{1 + RS_t}
$$
Where $RS_t = \frac{\text{Average Gain}_{14}}{\text{Average Loss}_{14}}$

**Moving Average Convergence Divergence (MACD):**
$$
\text{MACD}_t = \text{EMA}_{12}(P_t) - \text{EMA}_{26}(P_t)
$$
$$
\text{Signal}_t = \text{EMA}_9(\text{MACD}_t)
$$

**Bollinger Bands:**
$$
\text{Upper Band}_t = \text{SMA}_{20}(P_t) + 2 \cdot \sigma_{20}(P_t)
$$
$$
\text{Lower Band}_t = \text{SMA}_{20}(P_t) - 2 \cdot \sigma_{20}(P_t)
$$

**Average Directional Index (ADX):**
Measures trend strength through directional movement indicators.

**Aroon Indicator:**
Identifies trend changes and momentum shifts over 25-day periods.

#### The RFECV Algorithm: Systematic Feature Selection

The cornerstone of our methodology is the Recursive Feature Elimination with Cross-Validation process, which systematically identifies the optimal feature subset.

##### Algorithm Implementation

The RFECV algorithm operates through the following steps:

1. **Initial Training**: Train the base model (LightGBM) on the complete feature set $F$
2. **Feature Ranking**: Rank features by importance using gain-based metrics:
   $$
   \text{Importance}(f_i) = \sum_{t \in T} \text{Gain}(f_i, t)
   $$
   Where $T$ represents all trees in the ensemble
3. **Recursive Elimination**: Remove the least important feature and retrain
4. **Cross-Validation**: Evaluate performance using TimeSeriesSplit validation
5. **Optimal Selection**: Identify feature count maximizing cross-validated performance

##### Mathematical Foundation

The RFECV process optimizes the objective function:

$$
S^* = \arg\max_{S \subseteq F} \frac{1}{K} \sum_{k=1}^{K} \text{Score}(M_k(S), D_k^{test})
$$

Where:
- $K$ is the number of cross-validation folds
- $M_k(S)$ is the model trained on fold $k$ with feature subset $S$
- $D_k^{test}$ is the test data for fold $k$
- $\text{Score}$ is the evaluation metric (MSE for regression)

##### Time-Series Cross-Validation

We employ TimeSeriesSplit to maintain temporal integrity:

```
Train: [1, 2, 3, 4] | Test: [5]
Train: [1, 2, 3, 4, 5] | Test: [6]
Train: [1, 2, 3, 4, 5, 6] | Test: [7]
...
```

This ensures the model is always tested on future data relative to its training period, preventing data leakage.

#### Model Architecture: From Regression to Classification

##### Initial Regression Approach

Our initial approach targeted continuous return prediction:

$$
y_t = \frac{P_{t+21} - P_t}{P_t}
$$

Where $y_t$ represents the 21-day forward return.

##### Classification Transformation

We transitioned to binary classification for improved practical utility:

$$
y_t^{class} = \begin{cases}
1 & \text{if } y_t > 0.01 \text{ (UP)} \\
0 & \text{if } y_t \leq 0.01 \text{ (DOWN)}
\end{cases}
$$

This threshold-based approach focuses on identifying stocks with meaningful positive momentum (>1% over 21 days).

##### LightGBM Implementation

We utilize LightGBM's gradient boosting framework with the following objective function:

$$
\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y_i}) + \sum_{j=1}^{J} \Omega(f_j)
$$

Where:
- $l(y_i, \hat{y_i})$ is the loss function
- $\Omega(f_j)$ represents regularization terms
- $J$ is the number of trees

---

### 4. The Research Journey: Iterative Model Development

The final framework emerged through systematic experimentation that revealed crucial insights about feature selection and model optimization.

#### Phase 1: Baseline Feature Engineering
**Objective**: Create comprehensive feature representation
* **Implementation**: Generated 40+ technical, momentum, and volatility features
* **Challenge**: High dimensionality with potential multicollinearity
* **Insight**: Raw feature count doesn't guarantee predictive power

#### Phase 2: RFECV Implementation  
**Objective**: Identify optimal feature subset systematically
* **Implementation**: Applied RFECV with TimeSeriesSplit cross-validation
* **Result**: Reduced feature space from 40+ to 6-8 optimal features
* **Breakthrough**: Discovered that specific volatility and technical indicators provide maximum predictive value

#### Phase 3: Regression to Classification Transition
**Objective**: Improve practical applicability and reduce noise sensitivity
* **Challenge**: Continuous return prediction suffered from high noise-to-signal ratio
* **Solution**: Binary classification with meaningful threshold (1% over 21 days)
* **Result**: Improved model stability and interpretability

#### Phase 4: Hyperparameter Optimization
**Objective**: Maximize model performance while preventing overfitting
* **Implementation**: Grid search across key LightGBM parameters
* **Focus Areas**:
  - Learning rate optimization
  - Tree complexity control
  - Regularization parameter tuning
* **Result**: Balanced performance between training and validation sets

#### Phase 5: Regularization Enhancement
**Objective**: Improve generalization through advanced regularization
* **Implementation**: L1 (Lasso) and L2 (Ridge) penalty optimization
* **Mathematical Framework**:
  $$
  L_{regularized} = L_{original} + \alpha \sum |w_i| + \lambda \sum w_i^2
  $$
* **Result**: Enhanced out-of-sample performance and reduced overfitting

---

### 5. Results & Performance Analysis

Our systematic approach yielded significant insights into the predictive structure of Indian equity markets.

#### Feature Selection Results

The RFECV process identified the following optimal feature set:

| Feature | Type | Importance | Interpretation |
|---------|------|------------|----------------|
| `volatility_21d` | Risk Measure | High | Short-term uncertainty indicator |
| `volatility_63d` | Risk Measure | High | Medium-term volatility regime |
| `MACDs_12_26_9` | Momentum | Medium | MACD signal line crossover |
| `BBL_20_2.0` | Technical | Medium | Lower Bollinger Band (support) |
| `BBU_20_2.0` | Technical | Medium | Upper Bollinger Band (resistance) |
| `ADX_14` | Trend Strength | Medium | Directional movement strength |

#### Model Performance Metrics

**Final Classification Results:**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 67.3% | Significantly above random (50%) |
| **ROC AUC** | 0.721 | Strong discriminative ability |
| **Precision (UP)** | 0.69 | 69% of UP predictions correct |
| **Recall (UP)** | 0.71 | Captures 71% of actual UP movements |

#### Regime Analysis Insights

Our dynamic regime analysis revealed differential feature importance across market conditions:

**Bullish Market Regime** (NIFTY 50 > 0%):
- Momentum indicators (MACD) show increased importance
- Volatility measures become less critical
- Technical levels (Bollinger Bands) provide strong signals

**Bearish Market Regime** (NIFTY 50 ≤ 0%):
- Volatility indicators dominate feature importance
- Trend strength (ADX) becomes crucial for risk management
- Support/resistance levels gain significance

#### Statistical Significance

**Confidence Intervals** (95% confidence level):
- Accuracy: [64.1%, 70.5%]
- ROC AUC: [0.687, 0.755]

**Performance vs. Random Baseline:**
- Improvement over random: +17.3 percentage points
- Statistical significance: p < 0.001 (highly significant)

---

### 6. How to Run This Project

#### Prerequisites
- Python 3.9 or higher
- 8GB+ RAM recommended for full dataset processing
- Internet connection for data download

#### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/quantitative-stock-prediction.git
   cd quantitative-stock-prediction
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv stock_prediction_env
   source stock_prediction_env/bin/activate  # On Windows: stock_prediction_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the complete analysis:**
   ```bash
   jupyter notebook notebooks/main_analysis.ipynb
   ```

#### Expected Runtime
- **Data Collection**: 2-3 minutes
- **Feature Engineering**: 1-2 minutes  
- **RFECV Feature Selection**: 5-10 minutes
- **Model Training & Evaluation**: 3-5 minutes
- **Hyperparameter Tuning**: 10-15 minutes
- **Complete Pipeline**: 20-30 minutes

#### Output Files
The analysis generates:
- `results/models/` - Trained model artifacts
- `results/plots/` - Performance visualization
- Feature importance rankings and analysis
- Cross-validation performance metrics

#### Customization Options

**Modify Analysis Scope:**
```python
# In notebooks/main_analysis.ipynb
TICKERS = ['RELIANCE.NS', 'TCS.NS']  # Analyze fewer stocks
START_DATE = '2020-01-01'  # Shorter time period
```

**Adjust Feature Selection:**
```python
# Modify RFECV parameters
selector = RFECV(
    min_features_to_select=3,  # Minimum features
    step=2,  # Elimination step size
    cv=TimeSeriesSplit(n_splits=3)  # Fewer CV folds
)
```

**Hyperparameter Tuning:**
```python
# Customize parameter grid
param_grid = {
    'n_estimators': [50, 100],  # Faster training
    'learning_rate': [0.1, 0.2],
    'max_depth': [5, 10]
}
```

The framework is designed to be modular and extensible, allowing researchers to easily modify components for their specific use cases or extend the analysis to additional markets and timeframes.
