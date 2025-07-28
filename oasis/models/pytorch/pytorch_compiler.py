import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # to be able to call `dataset/`
from tai.monitoring import *
from tai.model_explainability import *
from processing.process import *

def execute_pytorch(results, bentoml_logger, ModelMetricsDataClay, args, mlflow, bentoml, datasplit, data_components):
    with mlflow.start_run():
        bentoml_logger.info("Training Pytorch model")
        
        if args.dataclay:
            bentoml_logger.info("Running model training inside dataclay")
            try:
                bentoml_logger.info(f"Connecting to dataclay client")
                client = Client(proxy_host=args.dataclay_host, username=args.dataclay_hostname, 
                                password=args.dataclay_password, dataset=args.dataclay_dataset)
                client.start()
            except Exception as ex:        
                bentoml_logger.exception(f"Connecting to dataclay client failed due to exception {ex}")
            else:
                bentoml_logger.info("Successfully connected to dataclay client!")
        else:
            bentoml_logger.info("Running model training locally in device")
        
        model = ModelMetricsDataClay(data_components)

        if args.dataclay:
            model.make_persistent()
            bentoml_logger.info("Persistent PyTorch model object created!")

        # bentoml_logger.info("[BEFORE] MODEL TRAINED SUCCESFULLY")                            
        metrics_dict, model, step_epoch, step_loss = model.train_pytorch(args.device)
        # bentoml_logger.info("[FOUND] MODEL TRAINED SUCCESFULLY")

        mlflow.log_param("model_params", data_components['model_parameters'])
        for epoch, loss in zip(step_epoch, step_loss):
            mlflow.log_metric("loss", float(loss), step=epoch)

        # Log metrics for each variable dynamically
        for name, metric in metrics_dict.items():
            prefix = f"Metrics_{name}"
            mlflow.log_metric(f"{prefix}/MAE", metric['mae'])
            mlflow.log_metric(f"{prefix}/MSE", metric['mse'])
            mlflow.log_metric(f"{prefix}/RMSE", metric['rmse'])
            mlflow.log_metric(f"{prefix}/MAPE", metric['mape'])
            mlflow.log_metric(f"{prefix}/SMAPE", metric['smape'])

        # Move model to CPU before saving
        # model.cpu()

        # Save model with metrics into BentoML store
        custom_objects = {
            "scaler_obj": data_components['scaler_obj'],
            "test_sample_size": len(data_components['X_test']),
            "model_metrics": metrics_dict
        }

        bento_model = bentoml.picklable_model.save_model(
            'metrics_utilization_model_lstm',
            model,
            custom_objects=custom_objects
        )

        mlflow.log_params(datasplit)
        mlflow.set_tag("bentoml.model", bento_model.tag)
        
        save_shap_explanations_pytorch(model, data_components, args, mlflow, bentoml_logger, args.model_type)

    mlflow.end_run()

    results["Trained Model"] = str(bento_model.tag)
    results["Dataset Splits"] = datasplit
    results["Model Metrics"] = metrics_dict

    bentoml_logger.info(f"Model saved: {bento_model}")
    bentoml_logger.info(f"\nSaved Model in BentoML: {bento_model}")
    bentoml_logger.info(f"Trained model\n: {model}")
    print(results)

    if args.dataclay:
        bentoml_logger.info("Stopping dataclay client")
        client.stop()

    return save_results(bentoml_logger, results, datasplit, bento_model, metrics_dict)