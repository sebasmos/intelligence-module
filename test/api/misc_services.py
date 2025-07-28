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