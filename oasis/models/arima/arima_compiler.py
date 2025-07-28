import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# to be able to call `dataset/`
from tai.monitoring import create_drift_detector
from processing.process import *

def execute_arima(results, bentoml_logger, ModelMetricsDataClay,args, mlflow,bentoml,datasplit, data_components, train_df, test_df):
    with mlflow.start_run():
        bentoml_logger.info("Training ARIMA model")
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
            bentoml_logger.info("Persistent ARIMA model object created!")
        
        mlflow.log_param("model_params", data_components['model_parameters'])
        metrics, model, history, y_test, y_pred = model.train_arima()
        data_components["y_test"] = y_test
        data_components["y_pred"] = y_pred
        # Handle drift detection and save model
        reference_df = create_nanny_ml_df(data_components, test_df, model, args, "test")
        drift_detector_obj = create_drift_detector(["y_pred"], args.steps_back, reference_df)
        
        bento_model = bentoml.picklable_model.save_model('metrics_utilization_model_arima', model,
                                                         custom_objects={"scaler_obj": data_components['scaler_obj'],
                                                                        "historical_data": history,
                                                                        "drift_detector_obj": drift_detector_obj,
                                                                        "model_metrics": metrics,
                                                                        "test_sample_size": len(data_components['X_test'])},
                                                         signatures={"predict": {"batchable": True}})
        
        mlflow.log_metrics(metrics)
        mlflow.log_params(datasplit)
        mlflow.set_tag("bentoml.model", bento_model.tag)
    mlflow.end_run() 
        # mlflow.log_param("dataset_path", DATASET_PATH)# do we need this?
    if args.dataclay:
        bentoml_logger.info("Stopping dataclay client")
        client.stop()

    return save_results(bentoml_logger, results, datasplit, bento_model, metrics)
