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
        raw_data = json.loads(raw_data)
        raw_data = pd.DataFrame(raw_data)
        data = raw_data.iloc[:, 1:].values
        labels = raw_data.iloc[:, 0].values
        print(data[0])
        print(labels[0])
        print(data.iloc[0, :])
        prediction = model.predict(data)
        return json.dumps(prediction.toList())

    except Exception as ex:
        error = str(ex)
        return error
