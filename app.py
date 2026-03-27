import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import get_stock_data
from preprocess import prepare_data
from model import build_model
from predict import predict_future
from profit_analysis import analyze_profit

st.set_page_config(page_title="Stock Profit Predictor", layout="wide")
st.title("📈 Stock Profit Prediction Bot")
st.caption("Uses LSTM deep learning on historical data to forecast future prices and profit potential.")

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.selectbox("Select Stock", ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA", "META"])

with col2:
    years = st.selectbox("Years of History", [2, 4, 6, 8, 10])

with col3:
    future_days = st.slider("Days to Predict", 7, 90, 30)

if st.button("🔍 Analyze Stock"):
    with st.spinner("Loading data and training model..."):
        data = get_stock_data(ticker, years)
        X, y, scaler = prepare_data(data)
        model = build_model(X)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        predictions = predict_future(model, data["Close"], scaler, future_days)

    close_series = data["Close"].squeeze()
    current_price = float(close_series.iloc[-1])
    analysis = analyze_profit(current_price, predictions)

    # Signal banner
    signal = analysis["signal"]
    color = "green" if signal == "BUY" else ("red" if "SELL" in signal else "orange")
    st.markdown(
        f"<h2 style='color:{color}; text-align:center;'>Signal: {signal}</h2>",
        unsafe_allow_html=True,
    )

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${analysis['current_price']}")
    m2.metric("Predicted Final", f"${analysis['predicted_final']}", f"{analysis['expected_return_pct']}%")
    m3.metric("Predicted Peak", f"${analysis['predicted_peak']}", f"{analysis['peak_return_pct']}%")
    m4.metric("Predicted Low", f"${analysis['predicted_low']}")

    # Chart
    st.subheader("Price Forecast Chart")
    fig, ax = plt.subplots(figsize=(12, 4))

    # Historical (last 90 days)
    hist = data["Close"].tail(90)
    ax.plot(hist.values, label="Historical Price", color="steelblue")

    # Predicted
    pred_x = range(len(hist), len(hist) + future_days)
    ax.plot(pred_x, predictions, label="Predicted Price", color=color, linestyle="--")
    ax.axvline(x=len(hist) - 1, color="gray", linestyle=":", label="Today")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    st.caption("⚠️ This is not financial advice. Predictions are based on historical patterns only.")
