import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

from tensorflow.keras.models import load_model
from tensorflow.nn import softmax

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# Config / Files
# -------------------------------
CUSTOM_JSON_FILE = 'model_evaluation_results_custom.json'
RESNET_JSON_FILE = 'model_evaluation_results_resnet.json'
VGG_JSON_FILE = 'model_evaluation_results_vgg16.json'
FOOD_JSON_DIR = 'food_json'   # <-- folder you confirmed is in project root

os.makedirs(FOOD_JSON_DIR, exist_ok=True)

# Admin key for update endpoint (use environment variable in production)
NUTRITION_ADMIN_KEY = os.environ.get("NUTRITION_ADMIN_KEY", "changeme123")

# -------------------------------
# Fallback classes (your list)
# -------------------------------
FALLBACK_CLASSES = [
    'apple_pie', 'Baked Potato', 'burger', 'butter_naan', 'chai', 'chapati',
    'cheesecake', 'chicken_curry', 'chole_bhature', 'Crispy Chicken',
    'dal_makhani', 'dhokla', 'Donut', 'fried_rice', 'Fries', 'Hot Dog',
    'ice_cream', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer',
    'kulfi', 'masala_dosa', 'momos', 'omelette', 'paani_puri',
    'pakode', 'pav_bhaji', 'pizza', 'samosa', 'Sandwich',
    'sushi', 'Taco', 'Taquito'
]

# Build CLASS_LIST by scanning food_json directory (preferred)
def build_class_list_from_folder():
    names = []
    try:
        for fn in sorted(os.listdir(FOOD_JSON_DIR)):
            if fn.lower().endswith('.json'):
                name = fn[:-5]  # remove .json
                names.append(name)
    except Exception:
        pass
    return names if names else FALLBACK_CLASSES

CLASS_LIST = build_class_list_from_folder()

# -------------------------------
# Nutrition file helpers
# -------------------------------
def normalize_for_key(name: str) -> str:
    """Normalized underscore lowercase key used for storing new files."""
    if not name:
        return ""
    return name.strip().lower().replace(" ", "_").replace("-", "_")

def titleize_spaces(name: str) -> str:
    """Title-case with spaces: 'apple_pie' -> 'Apple Pie'"""
    if not name:
        return ""
    return name.replace("_", " ").title()

def try_file_variants(class_name: str):
    """
    Try multiple filename variants to find the correct JSON file inside FOOD_JSON_DIR
    Returns parsed JSON dict if found, else None.
    """
    if not class_name:
        return None

    # candidate name variants (most-likely order)
    variants = []

    # raw as provided
    variants.append(class_name)
    # normalized underscores lower (apple pie -> apple_pie)
    variants.append(normalize_for_key(class_name))
    # title-case with spaces (apple_pie -> Apple Pie)
    variants.append(titleize_spaces(class_name))
    # lowercase with spaces (apple_pie -> apple pie)
    variants.append(normalize_for_key(class_name).replace("_", " "))
    # lower-case raw
    variants.append(class_name.lower())
    # title case raw
    variants.append(class_name.title())
    # remove duplicate variants
    seen = []
    candidates = [v for v in variants if v and v not in seen and not seen.append(v)]

    for cand in candidates:
        fn = os.path.join(FOOD_JSON_DIR, f"{cand}.json")
        if os.path.exists(fn):
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    else:
                        # if file exists but not dict, still try to return it
                        return data
            except Exception:
                # try next variant if JSON parse fails
                continue

    # If nothing matched, attempt fuzzy: search filenames whose normalized form matches class_name normalized
    norm_target = normalize_for_key(class_name)
    try:
        for fn in os.listdir(FOOD_JSON_DIR):
            if not fn.lower().endswith(".json"):
                continue
            base = fn[:-5]
            if normalize_for_key(base) == norm_target:
                full = os.path.join(FOOD_JSON_DIR, fn)
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data
                except Exception:
                    continue
    except Exception:
        pass

    return None

def NutritionLoader(class_name):
    """
    Returns nutrition dict for class_name by reading the corresponding file in FOOD_JSON_DIR.
    If not found, returns a consistent N/A structure.
    """
    data = try_file_variants(class_name)
    if isinstance(data, dict) and data:
        return data
    # fallback N/A
    return {
        "calories": "N/A",
        "protein": "N/A",
        "fat": "N/A",
        "carbohydrates": "N/A",
        "fiber": "N/A"
    }

# -------------------------------
# Model mapping & eval loader (unchanged)
# -------------------------------
RAW_MODEL_CLASS_INDEX = {
    "custom_model_1":  {'apple_pie': 0, 'Baked Potato': 1, 'burger': 2},
    "custom_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "custom_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "custom_model_4":  {'Crispy Chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "custom_model_5":  {'Donut': 0, 'fried_rice': 1, 'Fries': 2},
    "custom_model_6":  {'Hot Dog': 0, 'ice_cream': 1, 'idli': 2},
    "custom_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "custom_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "custom_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "custom_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "custom_model_11": {'Sandwich': 0, 'sushi': 1, 'Taco': 2, 'Taquito': 3},

    "resnet_model_1":  {'apple_pie': 0, 'Baked Potato': 1, 'burger': 2},
    "resnet_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "resnet_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "resnet_model_4":  {'Crispy Chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "resnet_model_5":  {'Donut': 0, 'fried_rice': 1, 'Fries': 2},
    "resnet_model_6":  {'Hot Dog': 0, 'ice_cream': 1, 'idli': 2},
    "resnet_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "resnet_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "resnet_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "resnet_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "resnet_model_11": {'Sandwich': 0, 'sushi': 1, 'Taco': 2, 'Taquito': 3},

    "vgg16_model_1":  {'apple_pie': 0, 'Baked Potato': 1, 'burger': 2},
    "vgg16_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "vgg16_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "vgg16_model_4":  {'Crispy Chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "vgg16_model_5":  {'Donut': 0, 'fried_rice': 1, 'Fries': 2},
    "vgg16_model_6":  {'Hot Dog': 0, 'ice_cream': 1, 'idli': 2},
    "vgg16_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "vgg16_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "vgg16_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "vgg16_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "vgg16_model_11": {'Sandwich': 0, 'sushi': 1, 'Taco': 2, 'Taquito': 3}
}

NORMALIZED_MODEL_CLASS_INDEX = {
    model.lower(): {normalize_for_key(cls): idx for cls, idx in mapping.items()}
    for model, mapping in RAW_MODEL_CLASS_INDEX.items()
}
MODEL_CLASS_INDEX = NORMALIZED_MODEL_CLASS_INDEX

# -------------------------------
# Eval JSON loader (unchanged)
# -------------------------------
def load_eval_json(model_type):
    if model_type == "custom_models":
        file = CUSTOM_JSON_FILE
    elif model_type == "resnet_models":
        file = RESNET_JSON_FILE
    elif model_type == "vgg16_models":
        file = VGG_JSON_FILE
    else:
        return {}

    if not os.path.exists(file):
        return {}

    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def find_model_from_json(eval_json, classname):
    if not eval_json or not classname:
        return None, None
    cname = normalize_for_key(classname)
    for key, val in eval_json.items():
        if normalize_for_key(key) == cname:
            return val.get("model_used"), val
    return None, None

def get_model_path(model_type, model_used):
    if not model_used:
        return None
    model_base = os.path.splitext(model_used)[0].lower()

    folder_map = {
        "custom_models": "./models/custom_models",
        "resnet_models": "./models/resnet_models",
        "vgg16_models": "./models/vgg16_models"
    }

    folder = folder_map.get(model_type)
    if not folder:
        return None

    exact = os.path.join(folder, model_base + ".h5")
    if os.path.exists(exact):
        return exact

    # fuzzy search
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if model_base in f.lower():
                return os.path.join(folder, f)
    return None

_model_cache = {}
def load_model_cached(path):
    if path not in _model_cache:
        _model_cache[path] = load_model(path)
    return _model_cache[path]

# -------------------------------
# Preprocess
# -------------------------------
def preprocess_dynamic(img, w, h):
    arr = np.array(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    arr = cv2.resize(arr, (w, h))
    arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

def get_class_from_index(model_used_base, idx):
    if not model_used_base:
        return None
    key = model_used_base.lower()
    mapping = MODEL_CLASS_INDEX.get(key, {})
    for c, v in mapping.items():
        if v == idx:
            return c
    return None

# -------------------------------
# HTML route
# -------------------------------
@app.route("/")
def index():
    # re-build class list in case files changed
    global CLASS_LIST
    CLASS_LIST = build_class_list_from_folder()
    return render_template("index.html", classes=CLASS_LIST)

# -------------------------------
# GET Nutrition API
# -------------------------------
@app.route("/get_nutrition")
def get_nutrition():
    class_name = request.args.get("class")
    return jsonify(NutritionLoader(class_name))

# -------------------------------
# Update nutrition endpoint (ADMIN)
# -------------------------------
@app.route("/update_nutrition", methods=["POST"])
def update_nutrition():
    """
    POST JSON:
    {
      "admin_key": "changeme123",
      "class_key": "idli",
      "nutrition": { "calories": "58 kcal", ... },
      "replace": true  # optional
    }
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400

    admin_key = body.get("admin_key", "")
    if NUTRITION_ADMIN_KEY and admin_key != NUTRITION_ADMIN_KEY:
        return jsonify({"success": False, "error": "Invalid admin key"}), 403

    class_key = body.get("class_key")
    nutrition_values = body.get("nutrition")
    replace = body.get("replace", True)

    if not class_key or not isinstance(nutrition_values, dict):
        return jsonify({"success": False, "error": "class_key and nutrition dict required"}), 400

    # store file using normalized underscore-lowercase name for consistency
    store_name = normalize_for_key(class_key)
    filename = f"{store_name}.json"
    filepath = os.path.join(FOOD_JSON_DIR, filename)

    # if merge requested and file exists, merge keys
    if os.path.exists(filepath) and not replace:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}
        existing.update(nutrition_values)
        nutrition_values = existing

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nutrition_values, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to write file: {str(e)}"}), 500

    # update CLASS_LIST for UI
    global CLASS_LIST
    CLASS_LIST = build_class_list_from_folder()

    return jsonify({"success": True, "stored": store_name, "nutrition": nutrition_values})

# -------------------------------
# PREDICT
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file"})

    file = request.files["file"]
    selected_class = request.form.get("selected_class")
    model_type = request.form.get("model_type")

    if not file:
        return jsonify({"success": False, "error": "Uploaded file missing"})
    if not selected_class:
        return jsonify({"success": False, "error": "Selected class missing"})
    if not model_type:
        return jsonify({"success": False, "error": "Model type missing"})

    filename = secure_filename(file.filename)
    if filename == "":
        return jsonify({"success": False, "error": "Invalid filename"})
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load eval json
    eval_json = load_eval_json(model_type)
    model_used, class_entry = find_model_from_json(eval_json, selected_class)

    if not model_used:
        return jsonify({"success": False, "error": "Model not found for this class"})

    model_path = get_model_path(model_type, model_used)
    if not model_path:
        return jsonify({"success": False, "error": "Model file missing"})

    # Load model
    try:
        model = load_model_cached(model_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to load model: {str(e)}"})

    # determine expected input shape
    try:
        shape = model.input_shape  # often (None, H, W, C)
        if len(shape) == 4:
            _, h, w, _ = shape
            if h is None or w is None:
                h, w = 224, 224
        else:
            h, w = 224, 224
    except Exception:
        w, h = 224, 224

    try:
        img = Image.open(filepath).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to open image: {str(e)}"})

    x = preprocess_dynamic(img, w, h)

    try:
        pred = model.predict(x).squeeze()
    except Exception as e:
        return jsonify({"success": False, "error": f"Model prediction failed: {str(e)}"})

    # if raw outputs are not probabilities, apply softmax
    try:
        if np.any(pred < 0) or np.sum(pred) == 0:
            pred = softmax(pred).numpy()
    except Exception:
        try:
            pred = np.array(pred)
        except:
            pass

    try:
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
    except Exception:
        idx = 0
        conf = 0.0

    model_key = os.path.splitext(model_used)[0].lower()
    pred_label = get_class_from_index(model_key, idx)

    # If no label mapping found, fallback to selected class
    if pred_label is None:
        pred_label = normalize_for_key(selected_class) if selected_class else "unknown"

    # Build response extras: classification report / confusion matrix if present in class_entry
    classification_report = None
    confusion_matrix = None
    confusion_matrix_full = None
    model_labels_order = None

    if class_entry and isinstance(class_entry, dict):
        classification_report = class_entry.get("classification_report") or class_entry.get("class_report") or class_entry.get("report")
        confusion_matrix = class_entry.get("confusion_matrix")
        confusion_matrix_full = class_entry.get("confusion_matrix_full") or confusion_matrix
        model_labels_order = class_entry.get("labels") or class_entry.get("label_order") or class_entry.get("model_labels_order")

    # prepare nutrition for both selected (user-chosen) and predicted label using file-per-class loader
    nutrition_selected = NutritionLoader(selected_class)
    nutrition_predicted = NutritionLoader(pred_label)

    # Debug prints (remove if noisy)
    print("PREDICT -> selected_class:", selected_class)
    print("PREDICT -> predicted_label:", pred_label)
    # optional sample print
    try:
        print("PREDICT -> nutrition_predicted keys:", list(nutrition_predicted.keys())[:6] if isinstance(nutrition_predicted, dict) else nutrition_predicted)
    except Exception:
        pass

    response = {
        "success": True,
        "selected_class": selected_class,
        "predicted_label": pred_label,
        "confidence": conf,
        "model_used": model_used,
        "model_type": model_type,
        "class_metrics": class_entry or {},
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "confusion_matrix_full": confusion_matrix_full,
        "model_labels_order": model_labels_order,
        # include both nutrition variants (frontend will prefer predicted)
        "nutrition_selected": nutrition_selected,
        "nutrition_predicted": nutrition_predicted
    }

    return jsonify(response)

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
