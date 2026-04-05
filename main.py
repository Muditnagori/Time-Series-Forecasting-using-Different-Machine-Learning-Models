import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.preprocessing import scale_data, create_sequences
from src.arima_model import train_arima, forecast_arima
from src.lstm_model import build_lstm
from src.transformer_model import build_transformer
from src.evaluation import evaluate


# ================================
# LOAD DATA
# ================================
dataset = sm.datasets.get_rdataset("AirPassengers", "datasets")
data = dataset.data
series = data['value'].values

# ================================
# ARIMA
# ================================
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

arima_model = train_arima(train)
arima_pred = forecast_arima(arima_model, len(test))


# ================================
# PREPROCESS FOR DL
# ================================
scaled_data, scaler = scale_data(series)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ================================
# LSTM
# ================================
lstm = build_lstm((seq_length, 1))
lstm.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

lstm_pred = lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)
y_test_inv = scaler.inverse_transform(y_test)


# ================================
# TRANSFORMER
# ================================
transformer = build_transformer(seq_length)
transformer.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

transformer_pred = transformer.predict(X_test)
transformer_pred = scaler.inverse_transform(transformer_pred)


# ================================
# EVALUATION
# ================================
print("\n📊 Model Performance:")
evaluate(test, arima_pred, "ARIMA")
evaluate(y_test_inv, lstm_pred, "LSTM")
evaluate(y_test_inv, transformer_pred, "Transformer")


# ================================
# PLOT
# ================================
plt.figure(figsize=(12,6))
plt.plot(test, label="Actual", color='black')
plt.plot(arima_pred, label="ARIMA")
plt.plot(lstm_pred, label="LSTM")
plt.plot(transformer_pred, label="Transformer")

plt.legend()
plt.title("Time Series Forecasting Comparison")
plt.show()