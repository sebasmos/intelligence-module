# Test POST request API to perform model training

import requests

url = 'http://localhost:3000/train'

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

test_cases = {
    'test_model_training_locally_xgb' : {
                                        'model_name': 'METRICS',
                                        'model_type': 'XGB',
                                        'test_size': 0.2,
                                        'dataset_name': 'cpu_sample_dataset_orangepi.csv',
                                        'steps_back': 12,
                                        "num_variables": 1,
                                        'max_models_count': 5,
                                        'max_mlruns_count': 10,
                                        'model_parameters': {
                                            'xgboost_model_parameters': {
                                                'n_estimators': 1000,
                                                'max_depth': 7,
                                                'eta': 0.1,
                                                'subsample': 0.7,
                                                'colsample_bytree': 0.8,
                                                'alpha': 0,
                                            },
                                            },
                                        },
    'test_model_training_locally_arima': {
                                        'model_name': 'METRICS',
                                        'model_type': 'ARIMA',
                                        'test_size': 0.2,
                                        'dataset_name': 'cpu_sample_dataset_orangepi.csv',
                                        "num_variables": 1,
                                        'steps_back': 12,
                                        'max_models_count': 5,
                                        'max_mlruns_count': 10,
                                        'dataclay': False,
                                        'model_parameters': {
                                            'arima_model_parameters': {
                                                'p': 5,
                                                'd': 1,
                                                'q': 0,
                                            },
                                        },
                                    },
   'test_model_training_locally_lstm': {
                                        "model_name": "METRICS",
                                        "model_type": "PYTORCH",
                                        "test_size": 0.2,
                                        "dataset_name": "node_3_utilisation_sample_dataset.csv",
                                        "num_variables": 2,
                                        "steps_back": 6,
                                        "batch_size": 64,
                                        "max_models_count": 5,
                                        "max_mlruns_count": 10,
                                        "shap_samples": 100,
                                        "dataclay": False,
                                        "dataclay_host": "127.0.0.1",
                                        "dataclay_hostname": "testuser",
                                        "dataclay_password": "s3cret",
                                        "dataclay_dataset": "testdata",
                                        "model_parameters": {
                                            "pytorch_model_parameters": {
                                            "distill": False,
                                            "hidden_size": 64,
                                            "input_size": 2,
                                            "num_epochs": 50,
                                            "output_size": 2,
                                            "quantize": False
                                            }
                                        },
                                        "device": "cpu",
                                        "federated": {
                                                "use_federated": False,
                                                "federated_params": {
                                                    "run_config": {},
                                                    "use_stream": True
                                                }
                                        }
                                    }
    # 'test_model_training_dataclay' : {
    #                                     'model_name': 'METRICS',
    #                                     'model_type': 'XGB',
    #                                     'test_size': 0.2,
    #                                     'dataset_name': 'cpu_sample_dataset_orangepi.csv',
    #                                     'steps_back': 12,
    #                                     'max_models_count': 5,
    #                                     'max_mlruns_count': 10,
    #                                     'dataclay': 'true',
    #                                     'dataclay_hostname': 'testuser',
    #                                     'dataclay_password': 's3cret',
    #                                     'dataclay_dataset': 'testdata',
    #                                     'dataclay_host': '127.0.0.1',
    #                                     'model_parameters': {
    #                                         'arima_model_parameters': {
    #                                             'p': 5,
    #                                             'd': 1,
    #                                             'q': 0,
    #                                         },
    #                                         'xgboost_model_parameters': {
    #                                             'n_estimators': 1000,
    #                                             'max_depth': 7,
    #                                             'eta': 0.1,
    #                                             'subsample': 0.7,
    #                                             'colsample_bytree': 0.8,
    #                                             'alpha': 0,
    #                                         },
    #                                     },
    #                                 },                           
}


def launch_test_training():
    for test in test_cases:
        print('-'*50)
        print(f"Running test: {test}")
        response = requests.post(url=url, headers=headers, json=test_cases[test])
        if response.status_code == 200:
            print('Test passed')
            print(response.json())

        else:
            print(response)
            print('Test failed')
        print('-'*50)

if __name__ == "__main__":
    launch_test_training()
