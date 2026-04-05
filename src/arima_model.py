from statsmodels.tsa.arima.model import ARIMA

def train_arima(train, order=(5,1,0)):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps):
    return model_fit.forecast(steps=steps)