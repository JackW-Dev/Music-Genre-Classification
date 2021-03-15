import json
import joblib
import os
import pandas as pd


def init():
    global model

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "knn.pkl")
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = pd.read_csv(raw_data)
        data.drop("TrackNum", 1, inplace=True)
        raw_data = data.iloc[:, 1:5].values
        labels = data.iloc[:, 0].values
        prediction = model.predict(raw_data)
        return json.dumps(prediction.toList())

    except Exception as ex:
        error = str(ex)
        return error
