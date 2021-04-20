import json
import joblib
import pandas as pd
from azureml.core import Model
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


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

    knn_accuracy = accuracy_score(loaded_json_labels, knn_pred)
    svm_accuracy = accuracy_score(loaded_json_labels, svm_pred)

    # knn_roc = roc_auc_score(loaded_json_labels, knn_pred, multi_class="ovr")
    # svm_roc = roc_auc_score(loaded_json_labels, svm_pred, multi_class="ovr")

    # knn_f1 = f1_score(loaded_json_labels, knn_pred.tolist())
    # svm_f1 = f1_score(loaded_json_labels, svm_pred.tolist())

    # The backslash after the comma allows for multiline return statements
    return {"KNN Prediction": knn_pred.tolist(), "SVM Prediction": svm_pred.tolist()},\
           {"KNN Accuracy": knn_accuracy, "SVM Accuracy": svm_accuracy}
    # return { "KNN AUROC": knn_roc, "SVM AUROC": svm_roc}
    # return {"KNN F1": knn_f1, "SVM F1": svm_f1}
