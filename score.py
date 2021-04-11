import json
import joblib
import os
import numpy as np


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "knn.pkl")
    model = joblib.load(model_path)


def run(raw_data):
    return raw_data
    # data = json.loads(raw_data)
    # data = np.array(data)
    # pred = model.predict(data)
    # return json.dumps(pred)
