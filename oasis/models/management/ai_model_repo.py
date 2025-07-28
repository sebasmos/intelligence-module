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

from __future__ import annotations
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub import (
    login,
    snapshot_download
)
import bentoml
import shutil
import tempfile
from datetime import datetime
from bentoml._internal.models.model import ModelContext
import torch
import shutil
import pickle                          
from typing import Any                 


def push(
    bentoml_model_tag: str,
    repo_id: str,
    hf_token: str,
    commit_message: str = "Add/Update model",
) -> None:
    login(token=hf_token)
    api = HfApi()
    model = bentoml.models.get(bentoml_model_tag) 
    model_dir = Path(model.path).parent
    if not model_dir.exists():
        raise FileNotFoundError(f"Local BentoML model directory not found: {model_dir}")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=model_dir.name,
        commit_message=commit_message,
    )

def _detect_framework(obj: Any) -> str:
    """
    Very small heuristic – extend if you have more frameworks.
    Returns one of: 'torch', 'sklearn', 'xgboost', 'arima', 'tensorflow', 'generic'
    """
    mod = obj.__class__.__module__
    if mod.startswith("torch"):
        return "torch"
    if mod.startswith(("sklearn", "lightgbm")):
        return "sklearn"
    if mod.startswith("xgboost"):
        return "xgboost"
    if mod.startswith("statsmodels"):          
        return "arima"
    if mod.startswith(("tensorflow", "keras")):
        return "tensorflow"
    return "generic"

def pull(repo_id: str, bentoml_model_tag: str, hf_token: str | None = None) -> str:
    if hf_token:
        login(token=hf_token)

    model_name_prefix, tag_suffix = bentoml_model_tag.split(":")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=f"{model_name_prefix}/{tag_suffix}/**",
            local_dir=tmp_dir,
            token=hf_token,
        )

        model_dir   = tmp_dir / model_name_prefix / tag_suffix
        saved_pkl   = model_dir / "saved_model.pkl"   
        saved_pb    = model_dir / "saved_model.pb"    
        custom_path = model_dir / "custom_objects.pkl"

        custom_objects = {}
        if custom_path.exists():
            with custom_path.open("rb") as f:
                custom_objects = pickle.load(f)

        if saved_pkl.exists():                       
            try:
                import torch
                obj = torch.load(saved_pkl, map_location="cpu")
            except Exception:
                with saved_pkl.open("rb") as f:
                    obj = pickle.load(f)
            fw = _detect_framework(obj)              

        elif saved_pb.exists():                      
            import tensorflow as tf                  
            obj = tf.keras.models.load_model(
                model_dir,
                custom_objects=custom_objects or None,
            )
            fw = "tensorflow"

        else:
            raise FileNotFoundError(
                f"Neither 'saved_model.pkl' nor 'saved_model.pb' found in {model_dir}"
            )

        if fw == "torch":
            tag = bentoml.picklable_model.save_model(
                model_name_prefix, obj, custom_objects=custom_objects
            ).tag
        elif fw == "sklearn":
            tag = bentoml.sklearn.save_model(
                model_name_prefix, obj, custom_objects=custom_objects
            ).tag
        elif fw == "xgboost":
            tag = bentoml.sklearn.save_model(
                model_name_prefix, obj, custom_objects=custom_objects
            ).tag
        elif fw == "arima":
            tag = bentoml.picklable_model.save_model(
                model_name_prefix, obj, custom_objects=custom_objects
            ).tag
        elif fw == "tensorflow" or fw=="keras":                   
            tag = bentoml.keras.save_model(
                model_name_prefix, obj, custom_objects=custom_objects
            ).tag
        else:                                        # fallback
            tag = bentoml.picklable_model.save_model(
                model_name_prefix, obj, custom_objects=custom_objects
            ).tag

        return tag
