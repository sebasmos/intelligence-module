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

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from oasis.models.management.huggingface import push_bentoml_model_to_huggingface, pull_bentoml_model


if __name__ == "__main__":
    hf_token = "your_hf_token"
    repo_id = "ICOS-AI/icos_models"
    model_tag = "metrics_utilization_model_lstm:yth7kgc5t6mpiwar"
    # model_tag = "metrics_utilization_model_arima:bc65igrmzwxgcwar"
    commit_message = "new model with arima"
    model_name = model_tag
    push_bentoml_model_to_huggingface(
        bentoml_model_tag=model_tag,
        repo_id=repo_id,
        hf_token=hf_token,
        commit_message=commit_message,
        model_name=model_name
    )
    # pulled_path = pull_bentoml_model(repo_id, model_tag, model_tag, hf_token)
    # print("Pulled to:", pulled_path)
