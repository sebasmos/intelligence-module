# Test POST request to perform model iurl = 'http://localhost:3000/#/core/analytics__predict'nference

import requests

url = 'http://localhost:3000/predict'

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

test_cases = { 
            'test_model_inference_xgb' : {
                                'model_tag': 'metrics_utilization_model_xgb:latest',
                                'model_type': 'XGB',
                                "metric_type": 2,
                                'steps_back': 6,
                                'input_series': {'input_1': [79.4, 67.9, 71.2, 46.5, 67.3, 65.7]}
                            },
            'test_model_inference_arima': {
                                    'model_tag': 'metrics_utilization_model_arima:latest',
                                    "metric_type": 2,
                                    'model_type': 'ARIMA',
                                    'input_series': {'input_1': [79.4]},
                                    'history_sample_size': 500,
                                    'data_interruption': False,
                                    'history_data': [],
                                },
            'test_model_inference_arima_device_interrupt': {
                                            'model_tag': 'metrics_utilization_model_arima:latest',
                                            'model_type': 'ARIMA',
                                            'input_series': {'input_1': [79.4]},
                                            'history_sample_size': 10,
                                            'data_interruption': True,
                                            'history_data': [
                                                0.843,
                                                0.835,
                                                0.838,
                                                0.8370000000000001,
                                                0.794,
                                                0.843,
                                                0.835,
                                                0.838,
                                                0.8370000000000001,
                                                0.70,
                                            ],
                                        },
            'test_model_inference_lstm': {
                                            'model_tag': 'metrics_utilization_model_lstm:latest',
                                            "metric_type": 2,
                                            "steps_back": 6,
                                            "input_series": {
                                                "input_1": [
                                                0.79,
                                                0.83,
                                                0.84,
                                                0.82,
                                                0.84,
                                                0.88
                                                ],
                                                "input_2": [
                                                13.3,
                                                13.29,
                                                13.27,
                                                13.37,
                                                13.24,
                                                13.15
                                                ]
                                            }
                                        }
}

def launch_test_inference():
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
    launch_test_inference()
