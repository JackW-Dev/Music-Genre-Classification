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

    knn_pred_proba = knn_model.predict_proba(loaded_json_data)
    svm_pred_proba = svm_model.predict_proba(loaded_json_data)

    knn_accuracy = accuracy_score(loaded_json_labels, knn_pred)
    svm_accuracy = accuracy_score(loaded_json_labels, svm_pred)

    knn_auroc = roc_auc_score(loaded_json_labels, knn_pred_proba, multi_class="ovr")
    svm_auroc = roc_auc_score(loaded_json_labels, svm_pred_proba, multi_class="ovr")

    knn_f1 = f1_score(loaded_json_labels, knn_pred, average="micro")
    svm_f1 = f1_score(loaded_json_labels, svm_pred, average="micro")

    # The backslash after the comma allows for multiline return statements
    return {"KNN Prediction": knn_pred.tolist(), "SVM Prediction": svm_pred.tolist()}, \
           {"KNN Accuracy": knn_accuracy, "SVM Accuracy": svm_accuracy},\
           {"KNN AUROC": knn_auroc, "SVM AUROC": svm_auroc},\
           {"KNN F1": knn_f1, "SVM F1": svm_f1}
