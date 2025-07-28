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

# Test cases to test mlflow, jupyterlab and lomos api services

import requests

url = 'http://localhost:3000/'
test_urls = {
            'mlflow' : 'http://localhost:3000/mlflowui',
            'jupyterlab' : 'http://localhost:3000/jupyterlab_service',
            'lomos' : 'http://localhost:3000/get_anomalies_from_lomos_api'
}

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

test_cases = {
    'mlflowui' : '',
    'jupyterlab_service' : {
                        'jupyterlab_token': 'icos_1234',
                        'jupyterlab_port': '8888',
                    },
    'get_anomalies_from_lomos_api' : {
                    'lomos_api_url': 'http://10.160.3.227:25001/api/top_anomaly',
                    'lomos_api_params': {
                        'min_anomaly_score': 0.1,
                        'from_timestamp': '2024-01-08T00:00:01.200000Z',
                        'to_timestamp': '2024-01-18T00:00:01.200000Z',
                    },
                }
}

def launch_misc_tests():
    for test in test_cases:
        print('-'*50)
        print(f"Running test: {test}")
        if test == 'mlflowui':
            response = requests.get(url=url+test, headers=headers)
        else:
            response = requests.post(url=url+test, headers=headers, json=test_cases[test])

        if response.status_code == 200:
            print('Test passed')
            print(response.json())

        else:
            print(response)
            print('Test failed')
        print('-'*50)

if __name__ == "__main__":
    launch_misc_tests()