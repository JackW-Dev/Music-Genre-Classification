import json
import joblib
import pandas as pd
from azureml.core import Model
# from sklearn.metrics import classification_report


def init():
    global knn_model
    global svm_model

    knn_model_path = Model.get_model_path("knn")
    svm_model_path = Model.get_model_path("svm")

    knn_model = joblib.load(knn_model_path)
    svm_model = joblib.load(svm_model_path)


def run(json_data):
    json_dict = json.loads(json_data)
    json_df = pd.DataFrame(eval(json_dict))
    loaded_json_data = json_df.iloc[:, 1:].values
    loaded_json_labels = json_df.iloc[:, 0].values

    knn_pred = knn_model.predict(loaded_json_data)
    svm_pred = svm_model.predict(loaded_json_data)

    # knn_accuracy = classification_report(loaded_json_labels, knn_pred)
    # svm_accuracy = classification_report(loaded_json_labels, svm_pred)

    return {"KNN": knn_pred.tolist(), "SVM": svm_pred.tolist()}

# , "KNN_Accuracy": knn_accuracy, "SVM_Accuracy": svm_accuracy
