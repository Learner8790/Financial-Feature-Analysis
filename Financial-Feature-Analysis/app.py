import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Prediction Analysis",
    page_icon="📈",
    layout="wide"
)

# --- Caching Functions to Avoid Re-running ---
# @st.cache_data is a powerful Streamlit feature that caches the output of a function,
# preventing it from re-running every time you interact with the app.

@st.cache_data
def fetch_data(start_date, end_date):
    """Fetches and prepares all necessary data."""
    TICKERS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS', 'ASIANPAINT.NS'
    ]
    INDIAN_MARKET_INDICATORS = {
        'NIFTY50': '^NSEI', 'NIFTY_BANK': '^NSEBANK',
        'NIFTY_IT': '^CNXIT', 'USD_INR': 'INR=X'
    }
    
    stock_data = yf.download(TICKERS, start=start_date, end=end_date, progress=False, ignore_tz=True)
    stock_data = stock_data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    stock_data.dropna(subset=['Close'], inplace=True)

    market_data = yf.download(list(INDIAN_MARKET_INDICATORS.values()), start=start_date, end=end_date, progress=False, ignore_tz=True)
    market_data = market_data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    market_data.dropna(subset=['Close'], inplace=True)
    
    ticker_map = {v: k for k, v in INDIAN_MARKET_INDICATORS.items()}
    market_data['Ticker'] = market_data['Ticker'].replace(ticker_map)
    
    return stock_data, market_data

@st.cache_data
def engineer_features(_stock_data, _market_data):
    """Generates features and the final modeling dataframe."""
    base_df = _stock_data.copy()
    base_df['Price'] = base_df['Close']
    base_df = base_df.sort_values(by=['Ticker', 'Date'])

    future_price = base_df.groupby('Ticker')['Price'].shift(-21)
    base_df['target_21d_return'] = (future_price / base_df['Price']) - 1

    grouped = base_df.groupby('Ticker')
    for lag in [21, 42, 63]:
        base_df[f'return_{lag}d'] = grouped['Price'].pct_change(lag)

    daily_returns = grouped['Price'].pct_change(1)
    for window in [21, 63]:
        base_df[f'volatility_{window}d'] = daily_returns.transform(lambda x: x.rolling(window).std())
    
    def apply_ta(group):
        group.ta.rsi(length=14, append=True)
        group.ta.macd(append=True)
        group.ta.bbands(length=20, append=True)
        group.ta.adx(length=14, append=True)
        group.ta.aroon(length=25, append=True)
        return group
    base_df = base_df.groupby('Ticker', group_keys=False).apply(apply_ta)
    
    market_pivot = _market_data.pivot(index='Date', columns='Ticker', values='Close')
    market_pivot.columns = [f'market_{col}' for col in market_pivot.columns]
    for col in market_pivot.columns:
        market_pivot[f'{col}_return_1d'] = market_pivot[col].pct_change(1)
    
    base_df = base_df.set_index('Date').join(market_pivot)
    
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price', 'Adj Close']
    market_price_cols = [f'market_{name}' for name in _market_data['Ticker'].unique()]
    base_df = base_df.drop(columns=[col for col in cols_to_drop if col in base_df.columns] + market_price_cols)

    final_df = base_df.dropna().reset_index()
    final_df['Ticker'] = final_df['Ticker'].astype('category')
    le = LabelEncoder()
    final_df['Ticker_Encoded'] = le.fit_transform(final_df['Ticker'])
    final_df['target_class'] = (final_df['target_21d_return'] > 0.01).astype(int)
    
    return final_df

@st.cache_data
def get_model_results(_df):
    """Trains the final model and returns all necessary results."""
    selected_features = [
        'volatility_21d', 'volatility_63d', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBU_20_2.0', 'ADX_14', 'Ticker_Encoded'
    ]
    
    X = _df[selected_features]
    y = _df['target_class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Using the best parameters we found from regularization tuning
    best_params = {'learning_rate': 0.1, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 0.5}
    model = lgb.LGBMClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, pred_probs)
    cm = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return accuracy, roc_auc, cm, fpr, tpr, importance_df, model

# --- Main App Logic ---
st.title(" Quantitative Feature & Model Analysis")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    START_DATE = st.date_input("Start Date", value=pd.to_datetime("2016-01-01"))
    END_DATE = st.date_input("End Date", value=pd.to_datetime("today"))
    
    if st.button("Run Analysis"):
        st.session_state.run = True
    
    st.markdown("---")
    st.markdown(
        """
        This app demonstrates an end-to-end quantitative analysis pipeline:
        1.  **Data Ingestion:** Fetches stock data.
        2.  **Feature Engineering:** Creates predictive features.
        3.  **Model Training:** Trains a classification model.
        4.  **Evaluation:** Displays performance metrics and insights.
        """
    )


# --- Main Content ---
if 'run' in st.session_state and st.session_state.run:
    with st.spinner("Running Analysis... This may take a few minutes."):
        # 1. Fetch Data
        stock_data, market_data = fetch_data(START_DATE, END_DATE)
        
        # 2. Engineer Features
        final_df = engineer_features(stock_data, market_data)
        
        # 3. Get Model Results
        accuracy, roc_auc, cm, fpr, tpr, importance_df, model = get_model_results(final_df)

    st.header(" Final Model Performance")
    st.success("Analysis Complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
    with col2:
        st.metric(label="ROC AUC Score", value=f"{roc_auc:.4f}")

    st.header(" Deeper Insights")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "ROC Curve"])

    with tab1:
        st.subheader("Top Predictive Features")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df, ax=ax, palette='viridis')
        ax.set_title("Feature Importance (Gain)")
        st.pyplot(fig)
        st.write("""
        This chart shows the features that the final model found most useful for making its predictions. 
        A higher importance score means the feature was more influential.
        """)
        
    with tab2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted DOWN', 'Predicted UP'],
                    yticklabels=['Actual DOWN', 'Actual UP'])
        ax.set_title('Model Prediction Accuracy')
        ax.set_ylabel('Actual Class')
        ax.set_xlabel('Predicted Class')
        st.pyplot(fig)
        st.write("""
        The confusion matrix shows us where the model succeeded and where it failed.
        - **Top-Left (True Negative):** Correctly predicted 'DOWN'.
        - **Bottom-Right (True Positive):** Correctly predicted 'UP'.
        - **Top-Right (False Positive):** Incorrectly predicted 'UP'.
        - **Bottom-Left (False Negative):** Incorrectly predicted 'DOWN'.
        """)

    with tab3:
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Model Discriminative Power')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        st.write("""
        The ROC curve illustrates the model's ability to distinguish between the 'UP' and 'DOWN' classes.
        A curve further away from the dashed diagonal line indicates better performance. The Area Under the Curve (AUC) summarizes this into a single score.
        """)

else:
    st.info(" Welcome! Please configure your analysis dates in the sidebar and click 'Run Analysis' to begin.")
