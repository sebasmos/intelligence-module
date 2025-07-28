import mlflow
import sys

def delete_mlrun(run_id, bentoml_logger):
        try:
            bentoml_logger.info(f"Deleting experiment: {run_id}")
            mlflow.delete_run(run_id)
        except Exception as e:
            raise Exception(e,sys)