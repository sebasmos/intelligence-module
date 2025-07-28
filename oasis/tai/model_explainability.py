import shap
import matplotlib.pyplot as plt
import matplotlib
import torch

from analytics.lstm_model import LSTMModel
import numpy as np
matplotlib.use('Agg')  # Use non-GUI backend

def save_shap_explanations(model, data_components, args, mlflow, bentoml_logger, model_type="default"):
    """
    Calculate and save SHAP explanations for the model using MLflow.

    Parameters:
    - model: The trained model for which SHAP explanations are calculated.
    - data_components: Data used for SHAP explanation.
    - args: Additional arguments passed to the SHAP calculation function.
    - mlflow: MLflow module for logging artifacts and explanations.
    - bentoml_logger: Logger for tracking exceptions or info.
    - model_type: The type of the model (e.g., 'default', 'arima').
    """
    try:
        shap_explainer = shap.Explainer(model.predict, data_components['X_test'][:args.shap_samples])
        shap_values = shap_explainer(data_components['X_test'][:args.shap_samples])
        
        plots = [
            (shap.plots.waterfall, shap_values[1], f"model_explanations_shap/summary_waterfall_plot_{model_type}.png"),
            (shap.plots.bar, shap_values, f"model_explanations_shap/summary_bar_plot_{model_type}.png"),
            (shap.plots.heatmap, shap_values, f"model_explanations_shap/summary_beeswarm_plot_{model_type}.png")
        ]
        for plot_func, shap_data, file_name in plots:
            fig, _ = plt.subplots(figsize=(10, 8))  
            plot_func(shap_data, show=False)
            mlflow.log_figure(figure=fig, artifact_file=file_name)
        mlflow.shap.log_explainer(explainer=shap_explainer,artifact_path="model_explanations_shap",serialize_model_using_mlflow=False)
        bentoml_logger.info("Saving model explainability for XGBoost model")
    except Exception as e:
                    bentoml_logger.exception(e)


def save_shap_explanations_pytorch(model, data_components, args, mlflow, bentoml_logger, model_type="default"):
    """
    Calculate and save SHAP explanations for a PyTorch model using MLflow.

    Parameters:
    - model: The trained PyTorch model for which SHAP explanations are calculated.
    - data_components: Data used for SHAP explanation.
    - args: Additional arguments passed to the SHAP calculation function.
    - mlflow: MLflow module for logging artifacts and explanations.
    - bentoml_logger: Logger for tracking exceptions or info.
    - model_type: The type of the model (e.g., 'default', 'pytorch').
    """
    try:
        pass
        # model.eval()
        # X_test_tensor = torch.tensor(data_components['X_test'][:args.shap_samples], dtype=torch.float32)
        # input_size = 1
        # output_size = 1
        # hidden_size = data_components['model_parameters']['hidden_size']
        # model = LSTMModel(input_size, hidden_size, output_size)
        # shap_explainer = shap.DeepExplainer(model, X_test_tensor)
        # shap_values = shap_explainer(X_test_tensor) 
        # print("SHAP values [pytorch]: ", shap_values.shape)
        # plots = [
        #     # (shap.plots.waterfall, shap_values[1], f"model_explanations_shap/summary_waterfall_plot_{model_type}.png"),
        #     (shap.plots.bar, shap_values, f"model_explanations_shap/summary_bar_plot_{model_type}.png"),
        #     (shap.plots.heatmap, shap_values, f"model_explanations_shap/summary_beeswarm_plot_{model_type}.png")
        # ]
        # for plot_func, shap_data, file_name in plots:
        #     fig, _ = plt.subplots(figsize=(10, 8))  
        #     plot_func(shap_data, show=False)
        #     mlflow.log_figure(figure=fig, artifact_file=file_name)

        # mlflow.shap.log_explainer(explainer=shap_explainer, artifact_path="model_explanations_shap", serialize_model_using_mlflow=False)
        # bentoml_logger.info(f"Saving SHAP explanations for {model_type} model")

    except Exception as e:
        bentoml_logger.exception(f"Error in saving SHAP explanations: {e}")