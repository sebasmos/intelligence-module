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
