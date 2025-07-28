"""
ICOS Intelligence Coordination
----------------------------------------
The ICOS Coordination API has two goals:
a) First, models can be pre-built and added to the API as specified in a Developer guide. The API outputs model predictions or information about a new model trained in this scenario. This is performed for easy integration of ML models with automated functions of the OS developed in ICOS.
b) Second, part of this API is targeted to extend ML libraries to make them available to a technical user to save storage resources in devices with access to the API. In this context, the API returns a framework environment to allow users easy plug-and-play with the environment already available in the API.

Copyright © 2022-2024 CeADAR Ireland

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

This work has received funding from the European Union's HORIZON research 
and innovation programme under grant agreement No. 101070177.
----------------------------------------
"""

# Importing the libraries
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# to be able to call `dataset/`
import numpy as np
import pandas as pd
import torch

from xgboost import XGBRegressor
from dataclay import DataClayObject, activemethod
from typing import Any
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm as tqdm_progress
from .lstm_model import *
from processing.utils import *
from .metrics import *
import os
from pandas import DataFrame, concat
import logging

bentoml_logger = logging.getLogger("bentoml")

class ModelMetricsDataClay(DataClayObject):
    data_components: Any
    def __init__(self, data_components):
        self.data_components = data_components

    # Function to train XGB model
    @activemethod
    def train_xgb(self):
        model = XGBRegressor()
        model.set_params(**self.data_components['model_parameters'])
        bentoml_logger.info(model)
        model.fit(self.data_components['X_train'], self.data_components['y_train'])
        y_pred = model.predict(self.data_components['X_test'])
        # Inverse scale the predicted values
        y_pred = self.data_components['scaler_obj'].inverse_transform(y_pred.reshape(-1,1))
        # Reshape the test data same as the predicted data i.e. (N,1)
        y_test = np.array(self.data_components['y_test']).reshape(-1,1) 

        return metrics(y_test, y_pred), model
    
    # Function to train ARIMA model
    @activemethod
    def train_arima(self):
        train = self.data_components['X_train'].squeeze(1)
        test = self.data_components['X_test'].squeeze(1)
        # Create a list to store the predicted values
        predictions = []

        # Define the ARIMA tuning parameters
        p = self.data_components['model_parameters']["p"]  # AR order
        d = self.data_components['model_parameters']["d"]  # I order (degree of differencing)
        q = self.data_components['model_parameters']["q"]  # MA order

        # Create a list of records to train ARIMA
        history = [x for x in train]

        # Iterate over the test data with a progress bar
        for t in tqdm_progress(range(len(test))):
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        
        y_test = test
        y_pred = predictions
        y_pred = self.data_components['scaler_obj'].inverse_transform(np.array(y_pred).reshape(-1,1))
        y_test = self.data_components['scaler_obj'].inverse_transform(self.data_components['X_test'].reshape(-1,1))

        return metrics(y_test, y_pred), model, history, y_test,y_pred
        
    def train_lstm_model(self, model, train_loader, criterion, optimizer, num_epochs, device):
        step_epoch = []
        step_loss = []

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)# adding cuda support

                optimizer.zero_grad()

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)
            step_epoch.append(epoch + 1)
            step_loss.append(avg_loss)

            bentoml_logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

        return model, step_epoch, step_loss
    
    def eval_lstm_model(self, model, val_loader, criterion, num_epochs, device):
        
        step_epoch = []
        step_loss = []

        for epoch in range(num_epochs):
            model.eval()
            eval_loss = 0
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device) # adding cuda support
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                eval_loss += loss.item()
            avg_eval_loss = eval_loss / len(val_loader)
            step_epoch.append(epoch + 1)
            step_loss.append(avg_eval_loss)
            bentoml_logger.info(f"Epoch [{epoch+1}/{num_epochs}], Eval Loss: {avg_eval_loss:.4f}")
        
        return model, step_epoch, step_loss


    def train_knowledge_distillation_regression(self,teacher, student, train_loader, epochs, T, soft_target_loss_weight, ce_loss_weight, optimizer,method, device):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        teacher.eval()  
        student.train() 
        step_epoch = []
        step_loss = []

        mse_criterion = nn.MSELoss()
        ce_criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)#cuda support
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)

                student_outputs = student(inputs)
                if method==1:
                    mse_loss = mse_criterion(student_outputs, teacher_outputs)
                    loss = mse_loss
                if method==2:
                    mse_loss = mse_criterion(student_outputs, teacher_outputs)
                    soft_target_loss = ce_criterion(student_outputs / T, teacher_outputs / T) * (T * T)  # Temperature scaling
                    loss = soft_target_loss_weight * soft_target_loss + ce_loss_weight * mse_loss
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                training_loss = running_loss/len(train_loader)
            step_epoch.append(epoch + 1)
            step_loss.append(training_loss)

            if (epoch + 1) % 10 == 0:
                bentoml_logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

        trained_student_model = student
        return trained_student_model,step_epoch,step_loss

    def distilled_process(self,teacher_model_trained, train_loader, val_loader,  input_size=1, output_size=1,num_epochs=100,criterion=None, device=None):
        hidden_size = 4
        studentmodel = LighterStudentLSTMModel(input_size, hidden_size, output_size).to(device)
        student_optimizer = optim.Adam(studentmodel.parameters(), lr=0.001)
        
        student_model_trained, step_epoch_s, step_loss_s = self.train_lstm_model(studentmodel, train_loader, criterion, student_optimizer,num_epochs, device)
        student_optimizer = optim.Adam(studentmodel.parameters(), lr=0.1)

        distilled_student_model,step_epoch_d,step_loss_d = self.train_knowledge_distillation_regression(
            teacher=teacher_model_trained,
            student=student_model_trained, 
            train_loader=train_loader,
            epochs=50,
            T=2,
            soft_target_loss_weight=0.25,
            ce_loss_weight=0.75,

            optimizer=student_optimizer,
            method = 2,
            device=device
        )
        return distilled_student_model, step_epoch_s, step_loss_s,step_epoch_d,step_loss_d


    @activemethod
    def train_pytorch(self, device="cpu"):

        device = torch.device(device)
        bentoml_logger.info(f"Device: {device}")
        print("MODEL METRICS self.data_components['model_parameters']: ", self.data_components['model_parameters'])
        pytorch_params = self.data_components['model_parameters']

        input_size = pytorch_params['input_size']
        output_size = pytorch_params['output_size']
        hidden_size = pytorch_params['hidden_size']
        num_epochs = pytorch_params['num_epochs']
        quantize = pytorch_params['quantize']
        distill = pytorch_params['distill']

        batch_size = self.data_components['batch_size']
        train_loader, val_loader = create_dataloaders(self.data_components['train_dataset'], self.data_components['test_dataset'], batch_size=batch_size)
        model = LSTMModel(input_size, hidden_size, output_size).to(device)
        bentoml_logger.info("Here is the floating point version of this module:")
        bentoml_logger.info(model)
        f=print_size_of_model(model,"fp32")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model, step_epoch_train, step_loss_train = self.train_lstm_model(model, train_loader, criterion, optimizer, num_epochs, device)
        model, step_epoch_eval, step_loss_eval = self.eval_lstm_model(model, val_loader, criterion, num_epochs, device)
        
        if distill:
            model, step_epoch_s, step_loss_s, step_epoch_d, step_loss_d = self.distilled_process(
                model, train_loader, val_loader, input_size, output_size, num_epochs, criterion, device
            )

        outputs = model(self.data_components['X_test'].to(device))
        
        y_pred = outputs.cpu().detach().numpy().reshape(outputs.shape[0], outputs.shape[1])

        y_pred = self.data_components['scaler_obj'].inverse_transform(y_pred)
        y_test = self.data_components['scaler_obj'].inverse_transform(self.data_components['y_test'])
        
        # Read iteratively given the N input variables, save on dict for scalability 
        metrics = {}
        num_outputs = y_test.shape[1]

        for i in range(num_outputs):
            y_test_col = y_test[:, i].reshape(-1, 1)
            y_pred_col = y_pred[:, i].reshape(-1, 1)
            metrics[f"metric_{i}"] = metrics_pytorch(model, y_test_col, y_pred_col)

        # Move model to CPU before saving
        bentoml_logger.info("Moving torch objects to CPU")
        model.cpu()

        return metrics, model, step_epoch_train, step_loss_train
