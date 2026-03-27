def analyze_profit(current_price, predictions):
    """
    Returns a signal and expected return % based on predicted prices.
    """
    future_max = max(predictions)
    future_min = min(predictions)
    final_price = predictions[-1]

    expected_return = ((final_price - current_price) / current_price) * 100
    peak_return = ((future_max - current_price) / current_price) * 100

    if expected_return > 5:
        signal = "BUY"
    elif expected_return < -5:
        signal = "SELL / AVOID"
    else:
        signal = "HOLD / NEUTRAL"

    return {
        "signal": signal,
        "current_price": round(current_price, 2),
        "predicted_final": round(final_price, 2),
        "predicted_peak": round(future_max, 2),
        "predicted_low": round(future_min, 2),
        "expected_return_pct": round(expected_return, 2),
        "peak_return_pct": round(peak_return, 2),
    }
