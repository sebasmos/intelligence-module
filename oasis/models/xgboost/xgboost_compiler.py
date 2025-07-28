import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# to be able to call `dataset/`
from tai.monitoring import create_drift_detector
from tai.model_explainability import *
from processing.process import *

def execute_xgboost(results, bentoml_logger, ModelMetricsDataClay,args, mlflow,bentoml,datasplit, data_components, train_df, test_df):
    with mlflow.start_run():
        bentoml_logger.info("Training XGBoost model")
        if args.dataclay:
            # Run model training inside dataclay
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
            # Run model training outside dataclay 
            bentoml_logger.info("Running model training locally in device")
        
        model = ModelMetricsDataClay(data_components)
        if args.dataclay:
                model.make_persistent()
                bentoml_logger.info("Persistent XGBoost model object created!")

        metrics, model = model.train_xgb()
        mlflow.log_param("model_params", data_components['model_parameters'])
            
        reference_df = create_nanny_ml_df(data_components, test_df, model, args, "test")
        drift_detector_obj = create_drift_detector(["y_pred"], args.steps_back, reference_df)
            
        bento_model = bentoml.sklearn.save_model('metrics_utilization_model_xgb', model,
                                                    custom_objects={"scaler_obj": data_components['scaler_obj'],
                                                                    "drift_detector_obj": drift_detector_obj,
                                                                    "model_metrics": metrics,
                                                                    "test_sample_size": len(data_components['X_test'])})
            
        mlflow.log_metrics(metrics)
        mlflow.log_params(datasplit)
        mlflow.set_tag("bentoml.model", bento_model.tag)
        # SHAP explanations
        save_shap_explanations(model, data_components, args, mlflow, bentoml_logger, args.model_type)   
    mlflow.end_run()            
    if args.dataclay:
        bentoml_logger.info("Stopping dataclay client")
        client.stop()

    return save_results(bentoml_logger, results, datasplit, bento_model, metrics)

