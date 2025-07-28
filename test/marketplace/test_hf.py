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
