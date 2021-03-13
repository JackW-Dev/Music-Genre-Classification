from azureml.core import Model
import joblib


def init():
    global model
    model_path = Model.get_model_path("models/knn")
    model = joblib.load(model_path)


def run(data):
    try:
        data.drop('TrackNum', 1, inplace = True)
        raw_data = data.iloc[:, 1:5].values
        labels = data.iloc[:, 0].values
        prediction = model.predict(raw_data)
        return prediction

    except Exception as ex:
        error = str(ex)
        return error
