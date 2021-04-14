import json
import joblib
import os
import pandas as pd


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "knn.pkl")
    model = joblib.load(model_path)


def run(raw_data):
    json_data = raw_data

    json_dict = json.loads(json_data)

    json_df = pd.DataFrame(eval(json_dict))

    loaded_json_data = json_df.iloc[:, 1:].values

    pred = model.predict(loaded_json_data)

    return json.dumps(pred.tolist())
