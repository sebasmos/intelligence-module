# import os, sys, types, warnings
import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../oasis')))
import torch
import bentoml
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf  # For keras/tf models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# tag = "icos_nkua_rnn_lstm_cell4:svvoqfdg76b46war"
# tag = "metrics_utilization_model_lstm:pcpxsptg76yrkwar"
# tag = "metrics_utilization_model_arima:qcprbvtg76mhywar"
tag = "metrics_utilization_model_xgb:pwx5cwdg76vpuwar"

import os

def _configure_tf_runtime(force_cpu: bool = True):
 
    if force_cpu:                              
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:                                      
        import tensorflow as tf                
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:                   
                tf.config.experimental.set_memory_growth(gpu, True)


_configure_tf_runtime(force_cpu=True)

def load_bento_and_runner(model_tag: str):
    """
    Returns: (bento_model, scaler, runner [, extra_dict])
    """
    if "xgb" in model_tag:
        bm = bentoml.xgboost.get(model_tag)
    elif "arima" in model_tag:
        bm = bentoml.picklable_model.get(model_tag)
    elif "nkua" in model_tag:
        bm = bentoml.keras.get(model_tag)  
    else:  # Torch LSTM
        bm = bentoml.picklable_model.get(model_tag)

    scaler = bm.custom_objects.get("scaler_obj")

    if "arima" in model_tag:
        api_config = {
            "model_parameters": {"arima_model_parameters": {"p": 5, "d": 1, "q": 0}},
            "history_sample_size": 500,
        }

        global metrics_utilization_model_arima
        global metrics_utilization_model_arima_scaler
        metrics_utilization_model_arima = bm
        metrics_utilization_model_arima_scaler = scaler

        history = bm.custom_objects["historical_data"]

        class ARIMAForecastRunnable(bentoml.Runnable):
            SUPPORTED_RESOURCES = ("cpu",)
            SUPPORTS_CPU_MULTI_THREADING = False

            def __init__(self):
                p = api_config["model_parameters"]["arima_model_parameters"]["p"]
                d = api_config["model_parameters"]["arima_model_parameters"]["d"]
                q = api_config["model_parameters"]["arima_model_parameters"]["q"]
                self.order = (p, d, q)
                self.history = history[-api_config["history_sample_size"]:]

            @bentoml.Runnable.method(batchable=False)
            def forecast(self, test_val):
                model = ARIMA(self.history, order=self.order).fit()
                y_hat = float(model.forecast()[0])
                self.history.append(test_val)
                self.history = self.history[-api_config["history_sample_size"]:]
                return y_hat

        runner = bentoml.Runner(ARIMAForecastRunnable)
        runner.init_local(quiet=True)
        return bm, scaler, runner

    runner = bm.to_runner(embedded=True)
    runner.init_local(quiet=True)
    return bm, scaler, runner


bm, scaler, runner = load_bento_and_runner(tag)

if "xgb" in tag:
    n_feat = 6
    X = np.random.randn(1, n_feat).astype(np.float32)
    Xs = X  
    print("Input shape to XGBoost runner:", Xs.shape)
    y_s = runner.run(Xs)
    pred = y_s.ravel()

elif "arima" in tag:
    test_val = float(np.random.randn())
    test_val_s = scaler.transform([[test_val]])[0][0] if scaler is not None else test_val
    y_s = runner.forecast.run(test_val_s)
    pred = scaler.inverse_transform([[y_s]]).ravel() if scaler else np.asarray([y_s])

elif "keras" in tag or "tf" in tag or "nkua" in tag:
    "NKUA models don't have a scaler, so we skip it."
    n_feat = 1
    X = np.random.randn(1, 1).astype(np.float32)
    y_s = runner.run(X)
    pred = y_s.ravel()

else:  # Torch LSTM
    n_feat = scaler.n_features_in_
    seq_len = 10
    df = pd.DataFrame(np.random.randn(seq_len, n_feat), columns=[f"c{i}" for i in range(n_feat)])
    Xs = scaler.transform(df.values).astype(np.float32)
    Xt = torch.from_numpy(Xs[np.newaxis, :, :])  # (1, seq, feat)
    with torch.no_grad():
        y_s_t = runner.run(Xt)
    y_s = y_s_t.detach().cpu().numpy()
    if y_s.ndim == 3:
        y_s = y_s[:, -1, :]
    if scaler and y_s.shape[1] == n_feat:
        pred = scaler.inverse_transform(y_s).ravel()
    else:
        pred = y_s.ravel()

print(f"\nModel tag : {tag}")
print("Prediction:", pred)