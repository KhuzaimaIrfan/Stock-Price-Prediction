# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import date, timedelta

# # If using Keras
# try:
#     from tensorflow.keras.models import load_model
#     has_keras = True
# except Exception:
#     has_keras = False

# For fetching historical prices (optional)
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- Load artifacts ---
st.sidebar.title("Model & Settings")

ARTIFACT_DIR = "."  # change if using subfolder
metadata_path = os.path.join(ARTIFACT_DIR, "metadata.json")
if not os.path.exists(metadata_path):
    st.error("metadata.json not found in project folder. Please add it.")
    st.stop()

with open(metadata_path, "r") as f:
    metadata = json.load(f)

model_type = metadata.get("model_type", "sklearn")  # 'sklearn' or 'lstm'
st.sidebar.write(f"Detected model type: **{model_type}**")

# Load model
model = None
if model_type == "sklearn":
    model_path = os.path.join(ARTIFACT_DIR, "stock_model.pkl")
    if not os.path.exists(model_path):
        st.warning("stock_model.pkl not found. Upload it to this folder.")
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
elif model_type == "lstm":
    model_path = os.path.join(ARTIFACT_DIR, "stock_lstm_model.h5")
    if not os.path.exists(model_path):
        st.warning("stock_lstm_model.h5 not found. Upload it to this folder.")
    # else:
    #     if not has_keras:
    #         st.error("TensorFlow not installed. Add tensorflow to requirements.txt.")
    #     else:
    #         model = load_model(model_path)
else:
    st.error("Unknown model type in metadata.json (use 'sklearn' or 'lstm').")
    st.stop()

# Load scaler if exists
scaler = None
scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

# --- UI inputs ---
st.title("📈 Stock Price Prediction (Streamlit)")

col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker (for historical data, optional)", value="AAPL")
    start_date = st.date_input("Start date", value=date.today() - timedelta(days=365))
    end_date = st.date_input("End date", value=date.today())

with col2:
    predict_days = st.number_input("Days to predict (if model supports horizon)", min_value=1, max_value=365, value=7)
    run_predict = st.button("Run Prediction")

# Fetch data (optional)
@st.cache_data
def fetch_history(sym, start, end):
    df = yf.download(sym, start=start, end=end, progress=False)
    return df

df = None
if ticker:
    try:
        df = fetch_history(ticker, start_date, end_date + timedelta(days=1))
    except Exception as e:
        st.warning(f"Could not fetch data: {e}")

# Show data
if df is not None and not df.empty:
    st.subheader("Historical Prices")
    st.dataframe(df.tail(10))
    st.line_chart(df['Close'])

# --- Prediction logic ---
def predict_sklearn(model, scaler, df, predict_days=7):
    # Naive approach: predict next N points using last known features.
    # This assumes model expects features like ['Close', 'Open', 'Volume', ...]
    features = metadata.get("features", ["Close"])
    last_row = df[features].iloc[-1].values.reshape(1, -1)
    if scaler:
        last_row = scaler.transform(last_row)
    preds = []
    cur = last_row.copy()
    for i in range(predict_days):
        p = model.predict(cur)
        preds.append(p[0])
        # update cur in a naive way: shift and replace last feature with p if single feature
        if len(features) == 1:
            cur = np.array([[p[0]]])
            if scaler:
                cur = scaler.transform(cur)
        else:
            # more sophisticated update could be implemented depending on your model
            pass
    return preds

def predict_lstm(model, scaler, df, window_size, predict_days=7):
    features = metadata.get("features", ["Close"])
    data = df[features].values
    if scaler:
        data = scaler.transform(data)
    # take last window_size points
    seq = data[-window_size:].reshape(1, window_size, len(features))
    preds = []
    cur_seq = seq.copy()
    for i in range(predict_days):
        p = model.predict(cur_seq)
        # p shape depends; assume it returns 1 value or vector of features
        preds.append(p.flatten()[0])
        # append p and drop first
        new_step = p.reshape(1,1,-1) if p.ndim==2 else np.array([[[p.flatten()[0]]]])
        cur_seq = np.concatenate([cur_seq[:,1:,:], new_step], axis=1)
    # inverse scale if scaler used (assumes single feature)
    if scaler and len(features) == 1:
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten().tolist()
    return preds

if run_predict:
    if df is None or df.empty:
        st.error("No historical data found — predictions require historical data.")
    else:
        st.subheader("Predictions")
        if model_type == "sklearn":
            preds = predict_sklearn(model, scaler, df, predict_days)
        else:
            window_size = metadata.get("window_size", 60)
            preds = predict_lstm(model, scaler, df, window_size, predict_days)

        # Build a DataFrame for plot
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=predict_days, freq='B')
        preds_df = pd.DataFrame({"Predicted": preds}, index=future_dates)
        st.line_chart(pd.concat([df['Close'].rename('Actual'), preds_df['Predicted']], axis=0))

        st.write("Predicted values (next days):")
        st.dataframe(preds_df)

st.markdown("---")
st.markdown("**Notes:** The app assumes your model and scaler align with `metadata.json`.")
