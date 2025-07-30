#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Sync Service - Automatic Model Synchronization from DataClay to BentoML

This service provides continuous background synchronization of trained machine learning
models from the DataClay distributed storage to the local BentoML model repository.
It runs as a daemon thread within the Intelligence API container and monitors for
new models at configurable intervals.

The service uses hash-based change detection to ensure only genuinely new or modified
models are synchronized, preventing unnecessary storage usage and processing overhead.

Copyright (c) 2025 National and Kapodistrian University of Athens (NKUA)
Author: Anastasios Kaltakis
License: Apache 2.0

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This work has received funding from the European Union's HORIZON research
and innovation programme under grant agreement No. 101070177.
"""

import threading
import time
import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import bentoml
import torch

__author__ = "Anastasios Kaltakis"
__copyright__ = "Copyright (c) 2025 NKUA"
__license__ = "Apache 2.0"
__version__ = "1.0.0"
__maintainer__ = "Anastasios Kaltakis"
__email__ = "anastasioskaltakis@gmail.com"
__status__ = "Production"

# Configure logging
logger = logging.getLogger("ModelSync")


class ModelSyncService:
    """
    Background service for automatic model synchronization from DataClay to BentoML.

    This service continuously monitors DataClay storage for new trained models and
    automatically synchronizes them to the local BentoML repository, making them
    available for serving predictions through the Intelligence API.

    The service uses hash-based change detection to ensure efficient synchronization
    by only processing models that have actually changed. This prevents duplicate
    storage and unnecessary processing overhead.

    The service runs in a separate daemon thread and performs the following tasks:
    1. Loads information about existing models in BentoML on startup
    2. Periodically checks DataClay for models (default: every 30 seconds)
    3. Calculates a hash of each model's weights and configuration
    4. Only downloads and saves models that have actually changed
    5. Maintains sync history with model hashes to ensure efficiency

    Attributes:
        check_interval (int): Seconds between synchronization checks
        is_running (bool): Flag indicating if the service is active
        thread (threading.Thread): The background thread running the sync loop
        synced_models (Dict[str, str]): Maps model identifiers to their content hashes
        metrics (List[str]): List of supported metric types to monitor

    Example:
        >>> sync_service = ModelSyncService(check_interval=30)
        >>> sync_service.start()  # Starts background monitoring
        >>> status = sync_service.get_status()  # Check service status
        >>> sync_service.stop()  # Gracefully stop the service
    """

    def __init__(self, check_interval: int = 30):
        """
        Initialize the Model Sync Service.

        Args:
            check_interval (int): Interval in seconds between synchronization checks.
                                 Defaults to 30 seconds. Minimum value is 10 seconds.

        Raises:
            ValueError: If check_interval is less than 10 seconds.
        """
        if check_interval < 10:
            raise ValueError("Check interval must be at least 10 seconds")

        self.check_interval = check_interval
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        # Track synced models with their content hashes for change detection
        self.synced_models: Dict[str, str] = {}  # identifier -> model_hash

        # Supported metrics for ICOS system
        self.metrics = ["cpu_usage", "memory_usage", "power_consumption"]

        logger.info(f"Model Sync Service v{__version__} initialized (interval: {check_interval}s)")
        logger.info(f"Monitoring metrics: {', '.join(self.metrics)}")

    def start(self) -> None:
        """
        Start the background synchronization service.

        This method starts a daemon thread that continuously monitors DataClay
        for new models. The thread will automatically terminate when the main
        program exits. On startup, it loads information about existing models
        to prevent re-syncing already synchronized models.

        Note:
            If the service is already running, this method logs a warning
            and returns without creating a new thread.
        """
        if self.is_running:
            logger.warning("Model Sync Service is already running")
            return

        self.is_running = True
        self.thread = threading.Thread(
            target=self._run_forever,
            daemon=True,
            name="ModelSyncThread"
        )
        self.thread.start()
        
        # Load existing models info on startup to avoid duplicates
        self._load_existing_models()
        
        logger.info("Model Sync Service started successfully ✓")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the synchronization service gracefully.

        This method signals the background thread to stop and waits for it
        to complete any ongoing synchronization before terminating.

        Args:
            timeout (float): Maximum seconds to wait for thread termination.
                           Defaults to 5.0 seconds.
        """
        if not self.is_running:
            logger.warning("Model Sync Service is not running")
            return

        logger.info("Stopping Model Sync Service...")
        self.is_running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning(f"Thread did not stop within {timeout}s timeout")
            else:
                logger.info("Model Sync Service stopped successfully ✓")

    def _load_existing_models(self) -> None:
        """
        Load information about existing models in BentoML repository.
        
        This method scans the BentoML repository for previously synced models
        and extracts their content hashes. This prevents re-syncing models
        that haven't changed since the last sync operation.
        """
        try:
            # Check existing models for each name pattern we might save
            model_names = ["global_lstm_model"] + [f"lstm_{m}" for m in self.metrics]
            
            for model_name in model_names:
                try:
                    # Get latest version of this model
                    model = bentoml.models.get(f"{model_name}:latest")
                    
                    # Extract metric and hash from custom objects if available
                    if hasattr(model, 'custom_objects'):
                        metric = model.custom_objects.get('metric')
                        model_hash = model.custom_objects.get('model_hash')
                        
                        if metric and model_hash:
                            identifier = f"{metric}_latest"
                            self.synced_models[identifier] = model_hash
                            logger.info(f"Found existing model for {identifier} with hash {model_hash[:8]}...")
                            
                except bentoml.exceptions.NotFound:
                    # Model doesn't exist yet, that's expected for new deployments
                    pass
                except Exception as e:
                    logger.debug(f"Error checking model {model_name}: {e}")
                    
            logger.info(f"Loaded {len(self.synced_models)} existing models from BentoML repository")
            
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")

    def _run_forever(self) -> None:
        """
        Main synchronization loop that runs continuously in the background.

        This private method contains the core logic of the service:
        1. Waits for initial system stabilization (5 seconds)
        2. Continuously checks each configured metric for new models
        3. Uses hash-based comparison to detect actual changes
        4. Handles errors gracefully to prevent service disruption
        5. Sleeps between checks according to check_interval

        The loop continues until is_running is set to False.
        """
        logger.info("Model Sync background thread started")

        # Initial delay to allow system components to stabilize
        time.sleep(5)
        logger.debug("Initial stabilization period completed")

        while self.is_running:
            try:
                # Check each metric for new models
                for metric in self.metrics:
                    if not self.is_running:  # Check if stopped during iteration
                        break

                    logger.debug(f"Checking for updates to {metric} model...")
                    self._sync_model_if_new(metric)

            except Exception as e:
                # Log error but continue running
                logger.error(f"Unexpected error in sync loop: {e}", exc_info=True)

            # Wait before next check cycle
            if self.is_running:
                logger.debug(f"Sleeping for {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        logger.info("Model Sync background thread terminated")

    def _sync_model_if_new(self, metric: str) -> None:
        """
        Check DataClay for a specific metric's model and sync ONLY if it has changed.

        This method performs intelligent synchronization for a single metric:
        1. Queries DataClay for the latest model
        2. Calculates a hash of the model's weights and configuration
        3. Compares with previously synced model hash
        4. Only proceeds with sync if the model has actually changed
        5. Creates a PyTorch model from the stored state
        6. Saves it to BentoML repository with hash metadata

        Args:
            metric (str): The metric type to check (e.g., "cpu_usage")

        Note:
            Models are saved with two names:
            - "global_lstm_model": Standard name expected by API
            - "lstm_{metric}": Metric-specific name for clarity
        """
        identifier = f"{metric}_latest"

        try:
            # Import here to avoid circular dependencies at module load time
            from icos_fl.utils.model_storage import ModelStorageManager

            # Attempt to load model from DataClay
            logger.debug(f"Querying DataClay for {identifier}...")
            manager = ModelStorageManager()
            result = manager.load_model_from_dataclay(identifier)

            if result is None:
                logger.debug(f"No model found in DataClay for {identifier}")
                return

            # Model found - unpack the result
            state_dict, model_config = result
            
            # Calculate hash of this model to detect changes
            model_hash = self._calculate_model_hash(state_dict, model_config)
            logger.debug(f"Model {identifier} hash: {model_hash[:8]}...")
            
            # Check if we already have this exact model
            existing_hash = self.synced_models.get(identifier)
            if existing_hash == model_hash:
                logger.debug(f"Model {identifier} already synced (hash matches), skipping")
                return
                
            # This is a new or updated model!
            logger.info(f"Found NEW model for {identifier} (hash: {model_hash[:8]}...)")

            # Create PyTorch model from state dict
            model = self._create_model(state_dict, model_config)
            if model is None:
                logger.error(f"Failed to create PyTorch model for {identifier}")
                return

            # Save to BentoML repository with hash information
            self._save_to_bentoml(model, metric, model_config, model_hash)

            # Update our tracking registry
            self.synced_models[identifier] = model_hash
            logger.info(f"✓ Successfully synchronized NEW model {identifier}")

        except ImportError as e:
            logger.error(f"Import error - ensure icos-fl package is installed: {e}")
        except Exception as e:
            logger.error(f"Failed to sync {identifier}: {e}", exc_info=True)

    def _calculate_model_hash(self, state_dict: dict, model_config: dict) -> str:
        """
        Calculate a unique hash for a model based on its weights and configuration.
        
        This method creates a deterministic hash that uniquely identifies a model's
        content, allowing detection of actual changes versus re-saves of identical models.
        
        Args:
            state_dict (dict): Model weights and parameters
            model_config (dict): Model architecture configuration
            
        Returns:
            str: SHA256 hash string uniquely identifying this model version
        """
        # Create a deterministic representation of the model
        hash_data = {
            'config': model_config,
            'weights': {}
        }
        
        # Process each weight tensor to create a compact representation
        for key, tensor in state_dict.items():
            if hasattr(tensor, 'shape'):
                # For tensors, use shape and a checksum of values
                tensor_np = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor
                checksum = hashlib.md5(tensor_np.tobytes()).hexdigest()
                hash_data['weights'][key] = {
                    'shape': list(tensor.shape),
                    'checksum': checksum
                }
            else:
                # For non-tensors, just include the value
                hash_data['weights'][key] = str(tensor)
                
        # Create deterministic JSON string with sorted keys
        json_str = json.dumps(hash_data, sort_keys=True)
        
        # Return SHA256 hash for security and uniqueness
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _create_model(self, state_dict: dict, model_config: dict) -> Optional[torch.nn.Module]:
        """
        Create a PyTorch LSTM model from saved state and configuration.

        Args:
            state_dict (dict): Model weights and parameters
            model_config (dict): Model architecture configuration

        Returns:
            Optional[torch.nn.Module]: Initialized model or None if creation fails
        """
        try:
            from icos_fl.models.lstm import LSTMModel

            # Extract model parameters with safe defaults
            model = LSTMModel(
                hidden_layer_size=model_config.get('hidden_layer_size', 64),
                time_step=model_config.get('time_step', 10),
                num_layers=model_config.get('num_layers', 1)
            )

            # Load saved weights
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode for inference

            logger.debug(f"Created LSTM model with config: {model_config}")
            return model

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return None

    def _save_to_bentoml(self, model: torch.nn.Module, metric: str,
                         model_config: dict, model_hash: str) -> None:
        """
        Save a PyTorch model to BentoML repository with hash metadata.

        This method saves the model with comprehensive metadata including
        the content hash, allowing future runs to detect if a model has
        already been synchronized.

        Args:
            model (torch.nn.Module): The model to save
            metric (str): Metric type (e.g., "cpu_usage")
            model_config (dict): Model configuration for metadata
            model_hash (str): Content hash of the model for change detection
        """
        # Save with standard name for API compatibility
        saved_model = bentoml.pytorch.save_model(
            "global_lstm_model",  # Standard name expected by API
            model,
            custom_objects={
                "metric": metric,
                "model_config": model_config,
                "model_hash": model_hash,  # Save hash for future comparisons
                "synced_from": "dataclay",
                "synced_at": datetime.now().isoformat(),
                "sync_service_version": __version__
            },
            labels={
                "metric": metric,
                "source": "model_sync",
                "author": __author__
            }
        )
        logger.debug(f"Saved as primary model: {saved_model.tag}")

        # Also save metric-specific version for clarity and debugging
        metric_model = bentoml.pytorch.save_model(
            f"lstm_{metric}",
            model,
            custom_objects={
                "metric": metric,
                "model_config": model_config,
                "model_hash": model_hash,
                "synced_at": datetime.now().isoformat()
            }
        )
        logger.debug(f"Saved as metric-specific model: {metric_model.tag}")

    def get_status(self) -> Dict[str, any]:
        """
        Get current status and statistics of the sync service.

        Returns:
            Dict[str, any]: Status information including:
                - is_running: Whether service is active
                - thread_alive: Whether background thread is alive
                - check_interval: Current check interval in seconds
                - metrics: List of monitored metrics
                - synced_models: Dictionary of synced models with hash previews
                - version: Service version
                - author: Service author

        Example:
            >>> status = sync_service.get_status()
            >>> print(f"Service running: {status['is_running']}")
            >>> print(f"Synced models: {status['synced_models']}")
        """
        status = {
            "is_running": self.is_running,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "check_interval": self.check_interval,
            "metrics": self.metrics,
            "synced_models": {
                k: {
                    "hash": v[:8] + "...",  # Show first 8 chars of hash
                    "full_hash": v
                } for k, v in self.synced_models.items()
            },
            "version": __version__,
            "author": __author__
        }

        # Add thread information if available
        if self.thread:
            status["thread_name"] = self.thread.name
            status["thread_daemon"] = self.thread.daemon

        return status


# Module-level convenience functions

def create_and_start_sync_service(check_interval: int = 30) -> ModelSyncService:
    """
    Convenience function to create and start a sync service.

    Args:
        check_interval (int): Sync interval in seconds

    Returns:
        ModelSyncService: Running service instance

    Example:
        >>> sync_service = create_and_start_sync_service(30)
        >>> # Service is now running in background
    """
    service = ModelSyncService(check_interval)
    service.start()
    return service


if __name__ == "__main__":
    # Module test when run directly
    print(f"Model Sync Service v{__version__}")
    print(f"Author: {__author__}")
    print(f"Copyright: {__copyright__}")
    print("=" * 70)
    print("Features:")
    print("- ✓ Hash-based change detection for efficient synchronization")
    print("- ✓ Only syncs models that have actually changed")
    print("- ✓ Prevents duplicate models in BentoML repository")
    print("- ✓ Loads existing models on startup")
    print("-" * 70)

    # Run a simple test
    service = ModelSyncService(check_interval=10)

    try:
        service.start()
        print("\nService started successfully")
        print("Press Ctrl+C to stop...")

        while True:
            time.sleep(5)
            status = service.get_status()
            print(f"\nStatus: Running={status['is_running']}, "
                  f"Thread={status['thread_alive']}, "
                  f"Synced={len(status['synced_models'])}")
            
            # Show synced models if any
            if status['synced_models']:
                print("Synced models:")
                for model_id, info in status['synced_models'].items():
                    print(f"  - {model_id}: {info['hash']}")

    except KeyboardInterrupt:
        print("\n\nStopping service...")
        service.stop()
        print("Service stopped")

