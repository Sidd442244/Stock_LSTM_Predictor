import numpy as np

def predict_future(model, data, scaler, days):
    close = data[-60:]
    if hasattr(close, 'squeeze'):
        close = close.squeeze()
    inputs = close.values.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    predictions = []

    for i in range(days):
        X_test = inputs[-60:].reshape(1, 60, 1)
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred[0][0])
        inputs = np.append(inputs, pred).reshape(-1, 1)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions.flatten()
