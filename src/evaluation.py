import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return rmse, mae