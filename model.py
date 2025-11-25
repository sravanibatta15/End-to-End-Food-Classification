import os
import sys
import json
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from common import Model
from log_code import setup_logging
logger=setup_logging('model')

class Performance:
    def custom(model_obj, path):
        try:
            logger("Started..........")
            with open('model.json', 'r') as f:
                label_map = json.load(f)
            labels = list(label_map.keys())
            custom_json = 'custom_model.json'
            y_preds, y_true = [], []

            for class_name in labels:
                class_dir = os.path.join(path, class_name)
                for img in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, img)
                    if image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                        y_pred = Model.model(model_obj, image_path)
                        y_preds.append(y_pred)
                        y_true.append(label_map[class_name])

            cm = confusion_matrix(y_true, y_preds)
            precision = precision_score(y_true, y_preds, average=None)
            recall = recall_score(y_true, y_preds, average=None)
            f1 = f1_score(y_true, y_preds, average=None)
            acc = accuracy_score(y_true, y_preds)

            results = {}
            for i, cls in enumerate(labels):
                TP = cm[i, i]
                FN = np.sum(cm[i, :]) - TP
                FP = np.sum(cm[:, i]) - TP
                TN = np.sum(cm) - (TP + FP + FN)
                class_acc = (TP + TN) / np.sum(cm)
                results[cls] = {
                    "TP": int(TP),
                    "TN": int(TN),
                    "FP": int(FP),
                    "FN": int(FN),
                    "Precision": round(float(precision[i]), 3),
                    "Recall": round(float(recall[i]), 3),
                    "F1_Score": round(float(f1[i]), 3),
                    "Class_Accuracy": round(float(class_acc), 3)
                }

            results["Overall_Accuracy"] = round(float(acc), 3)
            with open(custom_json, 'w') as f:
                json.dump(results, f, indent=4)
            return results
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            logger.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_value}")

    def vgg16(model_obj, path):
        try:
            with open('model.json', 'r') as f:
                label_map = json.load(f)
            labels = list(label_map.keys())
            custom_json = 'vgg_model.json'
            y_preds, y_true = [], []

            for class_name in labels:
                class_dir = os.path.join(path, class_name)
                for img in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, img)
                    if image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                        y_pred = Model.model(model_obj, image_path)
                        y_preds.append(y_pred)
                        y_true.append(label_map[class_name])

            cm = confusion_matrix(y_true, y_preds)
            precision = precision_score(y_true, y_preds, average=None)
            recall = recall_score(y_true, y_preds, average=None)
            f1 = f1_score(y_true, y_preds, average=None)
            acc = accuracy_score(y_true, y_preds)

            results = {}
            for i, cls in enumerate(labels):
                TP = cm[i, i]
                FN = np.sum(cm[i, :]) - TP
                FP = np.sum(cm[:, i]) - TP
                TN = np.sum(cm) - (TP + FP + FN)
                class_acc = (TP + TN) / np.sum(cm)
                results[cls] = {
                    "TP": int(TP),
                    "TN": int(TN),
                    "FP": int(FP),
                    "FN": int(FN),
                    "Precision": round(float(precision), 3),
                    "Recall": round(float(recall), 3),
                    "F1_Score": round(float(f1), 3),
                    "Class_Accuracy": round(float(class_acc), 3)
                }

            results["Overall_Accuracy"] = round(float(acc), 3)
            with open(custom_json, 'w') as f:
                json.dump(results, f, indent=4)
            return results
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            logger.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_value}")