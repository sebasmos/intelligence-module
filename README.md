# ICOS Intelligence Coordination API 

## Credits

- The Innovation and Development Group of CeADAR, Ireland's National Centre for Applied AI, based in University College Dublin, has developed the Open source AI framework as a service its full API. Jaydeep Samanta, Sebastian Cajas Ordoñez, Romila Ghosh, Dr. Andrés L. Suárez-Cetrulo and Dr. Ricardo Simón Carbajo have performed this work.
- The National and Kapodistrian University of Athens (NKUA) has contributed with anomaly detection and LSTM models for the current version.
- This work has been performed as part of partial fulfillment of the Intelligence layer of the EU HORIZON project No. 101070177 with name IoT to Cloud Operating System (ICOS).

## Documentation Index

Below is a list of related documentation sections to help you navigate the Intelligence Layer and its capabilities:

1. [**API Documentation**](api.md)  
2. [**Backend Services**](backend.md)
3. [**Deployment Guide**](deployment.md)  
4. [**Usage**](usage.md)  
5. [**Development & Contribution**](development.md)  

## License

- This project, developed by CeADAR Ireland, is licensed under the Apache License v2.0 (Apache-2.0) (https://www.apache.org/licenses/LICENSE-2.0). This includes the source code of both the API and models contained.
- Energy consumption prediction, Federeated Learning, LSTM models and classification models for anomaly detection provided by the National and Kapodistrian University of Athens (NKUA) are included in the repository. However, their source code is not included. The licensing terms of NKUA's source code, used to generate these runnable files, are not distributed with this project and may or may not be similar to GPLv3.

## Dependencies

This project uses the following external dependencies:

* `DataClay` (BSD 3-Clause "New" or "Revised" License, https://pypi.org/project/dataclay/)

Please note that `DataClay` is not included in this repository; it is only imported as an external dependency.


## Final Version

The AI coordination module facilitates optimisation, predictive analytics, and applying machine learning
models across the edge-cloud continuum. It entails implementing policies for utilising, sharing, and
updating models. This acts as an interface and provides coordination between the meta-kernel and user
layers providing and requesting services. This component helps to coordinate with other intelligence
layers of the same domain in the continuum, providing ICOS with the ability to learn collaboratively.

In this version, the AI coordination API will offer a mechanism for estimating the CPU/Memory consumption of the
ICOS agents. This API endpoint service will save a reference to the saved model in the AI analytics module,
which is a CPU-usage prediction model in this case. When a request is made to the API, this model will
forecast CPU usage one minute in the future. This version will also provide mechanisms for detecting drift
and explain model predictions to imporove AI trustworthiness. Moreover there are techiques to monitor model performance, 
model compression and scoring the model predictions. Finally, the API now has the ability to perform Federated 
Learning for trust and privacy related applications.

## Files and Folder structure

```
├── .gitignore  
├── bentofile.yaml   
├── LICENSE
├── README.md         
├── oasis
│   ├── analytics
│   │   ├── dataframes.py
│   │   ├── lstm_model.py
│   │   ├── metrics.py
│   │   └── model_metrics.py
│   ├── dataset
│   │   ├── cpu_sample_dataset_orangepi.csv
│   │   ├── cpu_sample_dataset.csv
│   │   ├── CPU_usage_data_joined.csv
│   │   ├── node_3_utilisation_sample_dataset.csv
│   │   ├── sample_1.csv
│   │   └── sample_orange_pi.csv
│   ├── api_service.py
│   ├── api_service_configs.json
│   ├── api_train.py
│   ├── bentofile.yaml
│   ├── clean_dockers.sh
│   ├── configuration.yaml
│   ├── model_sync.py
│   ├── organizer_fl.py
│   ├── requirements.txt
│   ├── models
│   │   ├── arima
│   │   │   └── arima_compiler.py
│   │   ├── management
│   │   │   └──ai_model_repo.py
│   │   │   └── registry.py
│   │   ├── pytorch
│   │   │   └── pytorch_compiler.py
│   │   └── xgboost
│   │       └── xgboost_compiler.py
│   ├── processing
│   │   ├── process.py
│   │   └── utils.py
│   └── tai
│       ├── model_explainability.py
│       └── monitoring.py   
├── test
│   ├── api
        ├── misc_services.py
        ├── model_inference.py
        ├── model_training.py
```
* .gitignore:
    -  Files that should be ignored by git. Add seperate .gitignore files in sub folders if needed

* bentofile.yaml:
    - Configuration file to create a bento.

* apis: 
    - This folder includes a openapi.yaml which will contain the definitions of the API.
    - This file describes an API in its entirety, including: Endpoints: which are available and operations on each (GET /users, POST /users) Authentication methods.

* oasis/src:
    - This folder contains the source code for running the API
    1. dataset:
        - This folder will contain the dataset that will be used to train the cpu utilisation models
        - An init file for initialising the dataset using a mapper function
    2. api_train_model.py:
        - This Python module encompasses the necessary functionalities for training ML algorithms. These functions handle crucial tasks such as data ingestion, data preprocessing, feature selection, and data splitting, ensuring that the input training and testing data are appropriately formatted for the training model. Once the data is preprocessed, it can be passed as input to the training function, which executes the model training and generates model metrics.
        - Furthermore, these trained models are saved using BentoML along with custom objects that may be utilised during the model prediction.
    3. api_service.py:
        - This is an API service module that provides API endpoints for training and inference using trained models.
        - This module calls the api_train_model.py file internally to retrieve trained models or construct new ones.
        - The predict function, which receives the best-trained model, plays a crucial role in returning the predicted values to their original format. This function can accept various input parame ters such as data input, trained model, and preprocessors, if required, and ultimately delivers the results.
    4. api_service_configs.json:
        - This is a configuration files that is used to store default parameters.
        - This may be changed by a user by sending POST request through the API endpoints.
    5. requirements.txt:
        - This contains all necessary libraries along with versions required for the API

* test: 
    - Tests for the API and its models.

## Starting the API service

A docker image has been provided which can be found [here](https://drive.google.com/file/d/1dIpbUn8GnhN93AHQLLA4vW6GZNMWuvsq/view?usp=sharing ). This image is intended to be run in the ICOS Controller with a x86 architecture in the alpha version. 

- Run the below command to launch the service as a docker
`docker run --network host -it --rm -p 3000:3000 -p 5000:5000 --cpus 15 --memory 20g -e BENTOML_CONFIG_OPTIONS='api_server.traffic.timeout=1200 runners.resources.cpu=0.5 runners.resources."nvidia.com/gpu"=0' analytics:latest serve`

The above command would enable the ICOS Intelligence Layer Coordination API server to be launched at http://0.0.0.0:3000, which could be accessed using any browser.


## Integrating new models in the API

* Update dataset: 
    - Add the dataset that you would like to use while training the model in dataset directory and also update init file for mapping the relavant dataset.

* Model training:
    - The training script is where the model is trained, and the trained model is saved in the bentoml repository along with any scalar objects that were utilised during the model training.
    - In api_train_model.py file create a function definition inside `class ModelTrain` and call that function through `initiate_train` function. 

* API Service & model inference:
    - Once the trained model is saved in Bentoml repository, this can now be used for inference using the Bentoml service api script

## Tests

The current code has been tested in below setups:

1. ARM_64 Orange Pi board
    
    - **Hardware Information:**

        - CPU: rockchip-rk3588 8-cores (Cortex-A76, Cortex-A55)
    
    - **Operating System:** Linux orangepi5 5.10.110 (Debian) 

1. x86_64 TR AMD server.

    - **Hardware Information:**

        - CPU: AMD Ryzen Threadripper PRO 5975WX 32-Cores

        - GPU: NVIDIA RTX 4090

    - **Operating System:**: Ubuntu 22.04

1.  x86_64 AMD Server in NCSRD Infrastructure

    - **Hardware Information:**
        - CPU: AMD Opteron 240 (Gen 1 Class Opteron), 32-bit, 64-bit
  
    - **Operating System:**
        - ubuntu-20.04.6-live-server-amd64.iso
  
    - **Virtualization Platform:**
        - Proxmox

    - **VM Configuration in OpenVPN:**
        - [VM Configuration Link](https://newrepository.atosresearch.eu/index.php/f/1271920), located under the WP5 folder.

    - **Access Information:**
        - SSH Command:
            ```bash
            ssh -l icos -L 3000:localhost:3000 10.160.3.200
            ```
        - Username: `icos`
        - Password: `icosmeta`
        - VM name: `icosapi`
    - **Conda environment**:
        - Activate environment: `conda activate icos`
    - **Remarks:**
        - Port 3000 is port-forwarded to enable BentoML to compile inside the VM.


## Requirements

The current containerised version of the API requires a minimum of 20GB so we recommend
an ICOS Controller with atleast 40GB of RAM.


# Legal
The ICOS Coordination API is released under the Apache License v2.
Copyright © 2022-2025 CeADAR Ireland. All rights reserved.

🇪🇺 This work has received funding from the European Union's HORIZON research and innovation programme under grant agreement No. 101070177.
