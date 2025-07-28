This is a readme for testing the Intelligence API

* Installation:

- Make sure that the Intelligence API docker is up and running as specified in the main README file.

* Test cases:

- Model training test: This script tests the API for all model training related scenarios including model offloading to dataclay server.
- Model Inference test: This script tests the API request to perform model inference and related scenarios.
- Misc Services test: This script has API request to test mlflow ui, jupyter lab and lomos api services.

* Running the tests:

- All the tests are written in python and can be launched from the command line for instance:
```
    python model_training.py
```

* Expected result:

- Test should run correctly and print PASS.

