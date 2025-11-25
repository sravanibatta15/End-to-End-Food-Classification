import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

test_dir = './testing_dataset'
models_dir = './models/vgg16_models'
output_json = './model_evaluation_results_vgg16.json'

img_size = (256, 256)
batch_size = 32

# ----------------------
# MODEL → CLASS MAPPING
# ----------------------
model_class_map = {
    1: {'apple_pie': 0, 'Baked Potato': 1, 'burger': 2},
    2: {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    3: {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    4: {'Crispy Chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    5: {'Donut': 0, 'fried_rice': 1, 'Fries': 2},
    6: {'Hot Dog': 0, 'ice_cream': 1, 'idli': 2},
    7: {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    8: {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    9: {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    10: {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    11: {'Sandwich': 0, 'sushi': 1, 'Taco': 2, 'Taquito': 3}
}

test_datagen = ImageDataGenerator(rescale=1. / 255)
results = {}

for model_num, class_map in model_class_map.items():

    model_path = os.path.join(models_dir, f"vgg16_model_{model_num}.h5")
    if not os.path.exists(model_path):
        print(f"Missing model: {model_path}")
        continue

    print(f"\nLoading: {model_path}")
    model = load_model(model_path)

    # Load ONLY folders this model was trained on
    allowed_folders = list(class_map.keys())

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        classes=allowed_folders,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    preds = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes

    # Multi-class confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    total_samples = np.sum(conf_matrix)

    # Convert matrix to list for JSON
    conf_matrix_full = conf_matrix.tolist()

    model_name = f"vgg16_model_{model_num}"  # <<< Only model name

    # -------------------------
    # Compute metrics per class
    # -------------------------
    for class_name, class_idx in class_map.items():
        TP = conf_matrix[class_idx, class_idx]
        FP = np.sum(conf_matrix[:, class_idx]) - TP
        FN = np.sum(conf_matrix[class_idx, :]) - TP
        TN = total_samples - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / total_samples if total_samples > 0 else 0

        print(f" Class: {class_name}")
        print(f" Acc: {accuracy:.4f}  Prec: {precision:.4f}  Recall: {recall:.4f}")

        results[class_name] = {
            "model_used": model_name,   # <<< Only model name saved here
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "true_positive": int(TP),
            "false_positive": int(FP),
            "false_negative": int(FN),
            "true_negative": int(TN),
            "confusion_matrix_full": conf_matrix_full
        }

with open(output_json, "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved:", output_json)