"""
ICOS Intelligence Coordination
----------------------------------------
The ICOS Coordination API has two goals:
a) First, models can be pre-built and added to the API as specified in a Developer guide. The API outputs model predictions or information about a new model trained in this scenario. This is performed for easy integration of ML models with automated functions of the OS developed in ICOS.
b) Second, part of this API is targeted to extend ML libraries to make them available to a technical user to save storage resources in devices with access to the API. In this context, the API returns a framework environment to allow users easy plug-and-play with the environment already available in the API.

Copyright © 2022-2025 CeADAR Ireland

This program is free software: you can redistribute it and/or modify it under the terms of the Apache License 2.0 as published by the Free Software Foundation.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Apache License v2.0 for more details.
You should have received a copy of the Apache License v2.0 along with this program. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

This work has received funding from the European Union's HORIZON research
and innovation programme under grant agreement No. 101070177.
----------------------------------------
"""

import numpy as np
import pandas as pd
import shap
import sys
from pandas import DataFrame, concat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import mlflow
from torch.utils.data import Dataset, DataLoader

def save_results(bentoml_logger, results, datasplit, bento_model, metrics):
        bentoml_logger.info(f"Model saved: {bento_model}")
        results["Trained Model"] = str(bento_model.tag)
        results["Dataset Splits"] = datasplit
        results["Results with test dataset"] = metrics
        bentoml_logger.info(results)
        return results

# Delete a single mlrun folder from mlflow experiment runs
def delete_mlrun(bentoml_logger, run_id):
    try:
        bentoml_logger.info(f"Deleting experiment: {run_id}")
        mlflow.delete_run(run_id)
    except Exception as e:
        raise Exception(e,sys)


def create_dataloaders(train_dataset, test_dataset, batch_size):
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Return shap plot
def shap_plot(shap_values, shap_plots=shap.plots.bar):
    fig = plt.figure(figsize=(15,6))
    shap_plots(shap_values, max_display=12, show=False)
    plt.tight_layout()
    plt.close(fig)
    return fig

# Perform data cleaning
def data_clean(df):
    # Keep only the relevant columns in the DataFrame
    df = df[['Date', 'CPU Core 1 Usage (%)', 'CPU Core 2 Usage (%)']]

    # Rename the columns for better readability
    cols = {'Date': 'date', 'CPU Core 1 Usage (%)': 'TARGET', 'CPU Core 2 Usage (%)': 'RAM'}
    df = df.rename(columns=cols, inplace=False)

    return df

def ts_supervised_structure(data, n_in=1, n_out=1, dropnan=True, autoregressive=True):
        no_autoregressive = not(autoregressive)
        if no_autoregressive:
            n_in = n_in - 1

        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            if no_autoregressive:
                cols.append(df.shift(i).iloc[:,:-1])
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars-1)]
            else:
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
# Perform train, test and split for time-series dataset
def data_simple_split(data_df,test_size=0.2):
    size = int(len(data_df) * 0.8)              
    train_df, test_df = data_df[0:size], data_df[size:len(data_df)] 
    train_df = train_df.reset_index(drop=True)  
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

# Calculate SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# Performs normalisation on complete dataset 
def scale_data(train_df, test_df, scaler):
    if scaler == "MinMax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()
    
    scaled_data_train = scaler.fit_transform(train_df)
    scaled_data_test = scaler.transform(test_df)

    return scaled_data_train, scaled_data_test, scaler

# Performs normalisation separately on input and output
def normalize_data(X_train, X_test, y_train, y_test):
    # Convert input data to pandas Series if they are not
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=X_test.columns if isinstance(X_test, pd.DataFrame) else None)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name=y_test.name if isinstance(y_test, pd.Series) else None)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name=y_train.name if isinstance(y_train, pd.Series) else None)

    # Normalize X data
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    X_train_normalized = pd.DataFrame(scaler_X.transform(X_train), columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns)

    # Normalize y data
    scaler_y = StandardScaler()
    scaler_y.fit(y_train.values.reshape(-1, 1))
    y_train_normalized = pd.Series(scaler_y.transform(y_train.values.reshape(-1, 1)).flatten(), name=y_train.name)
    y_test_normalized = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name)

    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_y
