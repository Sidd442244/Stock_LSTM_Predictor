import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data):

    close = data['Close']
    if hasattr(close, 'squeeze'):
        close = close.squeeze()
    close_prices = close.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    X = []
    y = []

    window = 60

    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    return X, y, scaler