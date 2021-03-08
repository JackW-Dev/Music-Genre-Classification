import joblib
from azureml.core import Model


def init():
    global model
    model_path = Model.get_model_path("models/knn")
    print("Model path: ", model_path)
    model = joblib.load(model_path)

def run(data):
    try:
        print(data)
        result = model.predict(data)
        return {"data" : result.tolist(), "message": "Successfully classified"}
    except Exception as ex:
        error = str(ex)
        return {"data": error, "message": "Failed Classification"}