import os
import json
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import tensorflow as tf

app = Flask(__name__)

# -------------------------------
# Folders
# -------------------------------
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models_s'
FOOD_JSON_DIR = 'food_json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(FOOD_JSON_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
# Nutrition / Admin Key
# -------------------------------
NUTRITION_ADMIN_KEY = os.environ.get("NUTRITION_ADMIN_KEY", "changeme123")

# -------------------------------
# Fallback classes
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

# -------------------------------
# Helper functions for nutrition
# -------------------------------
def normalize_for_key(name: str) -> str:
    if not name:
        return ""
    return name.strip().lower().replace(" ", "").replace("-", "")

def titleize_spaces(name: str) -> str:
    if not name:
        return ""
    return name.replace("_", " ").title()

def try_file_variants(class_name: str):
    if not class_name:
        return None
    variants = [
        class_name,
        normalize_for_key(class_name),
        titleize_spaces(class_name),
        normalize_for_key(class_name).replace("_", " "),
        class_name.lower(),
        class_name.title()
    ]
    seen = []
    candidates = [v for v in variants if v and v not in seen and not seen.append(v)]
    for cand in candidates:
        fn = os.path.join(FOOD_JSON_DIR, f"{cand}.json")
        if os.path.exists(fn):
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            except Exception:
                continue
    # Fuzzy fallback
    norm_target = normalize_for_key(class_name)
    try:
        for fn in os.listdir(FOOD_JSON_DIR):
            if fn.lower().endswith(".json") and normalize_for_key(fn[:-5]) == norm_target:
                with open(os.path.join(FOOD_JSON_DIR, fn), "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception:
        pass
    return None

def NutritionLoader(class_name):
    data = try_file_variants(class_name)
    if isinstance(data, dict) and data:
        return data
    return {"calories":"N/A","protein":"N/A","fat":"N/A","carbohydrates":"N/A","fiber":"N/A"}

# -------------------------------
# Build class list
# -------------------------------
def build_class_list_from_folder():
    names = []
    try:
        for fn in sorted(os.listdir(FOOD_JSON_DIR)):
            if fn.lower().endswith('.json'):
                names.append(fn[:-5])
    except Exception:
        pass
    return names if names else FALLBACK_CLASSES

CLASS_LIST = build_class_list_from_folder()

# -------------------------------
# Load JSON for models
# -------------------------------
CUSTOM_JSON_FILE = 'model_evaluation_results_custom.json'
RESNET_JSON_FILE = 'model_evaluation_results_resnet.json'
VGG_JSON_FILE = 'model_evaluation_results_vgg16.json'

def load_eval_json(model_type):
    if model_type=="custom_models": file=CUSTOM_JSON_FILE
    elif model_type=="resnet_models": file=RESNET_JSON_FILE
    elif model_type=="vgg16_models": file=VGG_JSON_FILE
    else: return {}
    if not os.path.exists(file): return {}
    try:
        with open(file,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def find_model_from_json(eval_json, classname):
    if not eval_json or not classname: return None, None
    cname = normalize_for_key(classname)
    for key,val in eval_json.items():
        if normalize_for_key(key) == cname:
            return val.get("model_used"), val
    return None, None

# -------------------------------
# Google Drive TFLite links mapping (example)
# -------------------------------
# Replace the URLs below with your real Google Drive download links
MODEL_GDRIVE_LINKS = {
    # custom models
    "custom_model_1": "https://drive.google.com/uc?export=download&id=1Zrvq9hxlxWwKHcIvII7GniV-7o78Jscr",
    "custom_model_2": "https://drive.google.com/uc?export=download&id=13MZvLhwwhbdR2pk0CftTaS6sZVGL4AIK",
    "custom_model_3": "https://drive.google.com/uc?export=download&id=1gFL37ZAHFh34_fbjIK5ojCWUNjeMY36b",
    "custom_model_4": "https://drive.google.com/uc?export=download&id=14P3AMKW2fy8VtkTpV39y-PnAL2PDzsGT",
    "custom_model_5": "https://drive.google.com/uc?export=download&id=1JO7PHqdutMiFOe5M4ORxgy1oKoiDPvCX",
    "custom_model_6": "https://drive.google.com/uc?export=download&id=1BY7gF-zD23VKSabKTerJjl4OoOB3-g0U",
    "custom_model_7": "https://drive.google.com/uc?export=download&id=1FqqQ2TlSJQV58eQ5sPYYKnFUzI590RDt",
    "custom_model_8": "https://drive.google.com/uc?export=download&id=1R37IkfcUyMgvof1NmtJeSZoMp2Xz0B4D",
    "custom_model_9": "https://drive.google.com/uc?export=download&id=1GTMkZyHoEA5frHDmKzeoyiiKl1eNDy3K",
    "custom_model_10": "https://drive.google.com/uc?export=download&id=19-CtzGr8da5eW7LY9L3w0jqZFJ9Q1uC2",
    "custom_model_11": "https://drive.google.com/uc?export=download&id=1G0w35ZDU-x7KVmJ266U8ENGseZluXuq0",
    # vgg16 models
    "vgg16_model_1": "https://drive.google.com/uc?export=download&id=1OX9k5zM8QVAC9Jompv9cDPhSaNgHKKR8",
    "vgg16_model_2": "https://drive.google.com/uc?export=download&id=1tGwg0vGl6ALhKMd_kXv0ftMnpvNbWUzA",
    "vgg16_model_3": "https://drive.google.com/uc?export=download&id=1JkJkJASyjSmCZqoPT4sj5ebM6946d967",
    "vgg16_model_4": "https://drive.google.com/uc?export=download&id=1ii1HQpIciRih0Oc4jX6RfWrKovMJQSoG",
    "vgg16_model_5": "https://drive.google.com/uc?export=download&id=1WH6xsoXtjJ0yy0BXxf2ac6dD3WKtL6PQ",
    "vgg16_model_6": "https://drive.google.com/uc?export=download&id=1vkYjMmGDGNBuJR3lMdf_KR5McrcFe_z7",
    "vgg16_model_7": "https://drive.google.com/uc?export=download&id=1HveIEB9ToHQC5tXJxHEhzEMHqy4M-uPk",
    "vgg16_model_8": "https://drive.google.com/uc?export=download&id=1tP7Z7mBKT3DglenicexM1ZuuN8dUqna3",
    "vgg16_model_9": "https://drive.google.com/uc?export=download&id=1mAISDlaWVwSOL_Ut9HfYwa59ptWZ1F3q",
    "vgg16_model_10": "https://drive.google.com/uc?export=download&id=1-KxBRORaBIfd3ow58LOr8jFZamaCtgbn",
    "vgg16_model_11": "https://drive.google.com/uc?export=download&id=17hM6FQOQoNFaK5IQq3MpC9__qMVnIKJY",
    # resnet models
    "resnet_model_1": "https://drive.google.com/uc?export=download&id=1M8gj9XWDbxaal-lkoqeXMPz__45czeML",
    "resnet_model_2": "https://drive.google.com/uc?export=download&id=14OCv2K2SuQogCmiag8qgnRcCPuA1gaiR",
    "resnet_model_3": "https://drive.google.com/uc?export=download&id=1dfPUCzJWu4MoCCQ7HElPfcywJ_gaD2vm",
    "resnet_model_4": "https://drive.google.com/uc?export=download&id=1rFvFyLTmx07XSxw7sU4x9kIBdONv2EKP",
    "resnet_model_5": "https://drive.google.com/uc?export=download&id=1orzSUgM_Hyz1v0KLNSv8sVV4Sg4i0Vj-",
    "resnet_model_6": "https://drive.google.com/uc?export=download&id=17o2CB4Gj-l3W0pNbMeM5RmCPqzdlPh0E",
    "resnet_model_7": "https://drive.google.com/uc?export=download&id=1_eLdyWCceJER7_ktRSD4d5dnO8zTh_Gi",
    "resnet_model_8": "https://drive.google.com/uc?export=download&id=185K0iWqiKJTs82-wCwRDB9NoO2MhuJYA",
    "resnet_model_9": "https://drive.google.com/uc?export=download&id=1rEM5Mdnj1bmtXAsztuTTEshB079TiNVR",
    "resnet_model_10": "https://drive.google.com/uc?export=download&id=1krENZeNw8-v1uUEF6kks6G0xlSCSZJun",
    "resnet_model_11": "https://drive.google.com/uc?export=download&id=1e9bifJUWTW0wPSgZmShMmW6JnALqzZkb"
}

def download_model(model_name):
    """Download model from Google Drive if not exists"""
    if not model_name or model_name not in MODEL_GDRIVE_LINKS: return None
    # Remove existing file
    for f in os.listdir(MODEL_FOLDER):
        os.remove(os.path.join(MODEL_FOLDER,f))
    model_path = os.path.join(MODEL_FOLDER, model_name+".tflite")
    if os.path.exists(model_path):
        return model_path
    url = MODEL_GDRIVE_LINKS[model_name]
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return model_path
    except Exception as e:
        print("Failed to download model:", e)
        return None
    return None

# -------------------------------
# Flask routes
# -------------------------------
@app.route("/")
def index():
    global CLASS_LIST
    CLASS_LIST = build_class_list_from_folder()
    return render_template("index.html", classes=CLASS_LIST)

@app.route("/get_nutrition")
def get_nutrition():
    class_name = request.args.get("class")
    return jsonify(NutritionLoader(class_name))

@app.route("/update_nutrition", methods=["POST"])
def update_nutrition():
    body = request.get_json(force=True, silent=True)
    if not body: return jsonify({"success":False,"error":"Invalid JSON"}),400
    if body.get("admin_key") != NUTRITION_ADMIN_KEY:
        return jsonify({"success":False,"error":"Invalid admin key"}),403
    class_key = body.get("class_key")
    nutrition_values = body.get("nutrition")
    replace = body.get("replace", True)
    if not class_key or not isinstance(nutrition_values, dict):
        return jsonify({"success":False,"error":"class_key & nutrition dict required"}),400
    store_name = normalize_for_key(class_key)
    filepath = os.path.join(FOOD_JSON_DIR, store_name+".json")
    if os.path.exists(filepath) and not replace:
        try:
            with open(filepath,"r",encoding="utf-8") as f:
                existing=json.load(f)
            existing.update(nutrition_values)
            nutrition_values=existing
        except: pass
    try:
        with open(filepath,"w",encoding="utf-8") as f:
            json.dump(nutrition_values,f,indent=2)
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),500
    global CLASS_LIST
    CLASS_LIST=build_class_list_from_folder()
    return jsonify({"success":True,"stored":store_name,"nutrition":nutrition_values})

# -------------------------------
# Prediction
# -------------------------------
def preprocess_image_for_tflite(img, h, w):
    arr = np.array(img)
    if arr.ndim==2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2]==4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    arr = cv2.resize(arr, (w,h))
    arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr,0)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file"}), 400

    file = request.files["file"]
    selected_class = request.form.get("selected_class")
    model_type = request.form.get("model_type")

    if not file or not selected_class or not model_type:
        return jsonify({"success": False, "error": "Missing required parameters"}), 400

    # Save uploaded file (only one file in uploads folder)
    filename = secure_filename(file.filename)
    if filename == "":
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    # Clear uploads folder
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load evaluation JSON for model type
    eval_json = load_eval_json(model_type)
    model_used, class_entry = find_model_from_json(eval_json, selected_class)
    if not model_used:
        return jsonify({"success": False, "error": "Model not found for this class"}), 400

    # Download model if not exists
    model_path = download_model(model_used)
    if not model_path:
        return jsonify({"success": False, "error": "Failed to download model"}), 500

    # Load TFLite model
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to load TFLite model: {str(e)}"}), 500

    # Preprocess image
    try:
        img = Image.open(filepath).convert("RGB")
        x = preprocess_image_for_tflite(img, h, w)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to open image: {str(e)}"}), 500

    # Run prediction
    try:
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index']).squeeze()
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
    except Exception as e:
        return jsonify({"success": False, "error": f"TFLite prediction failed: {str(e)}"}), 500

    # Map output index to correct class label
    # Try class_index_map from JSON; fallback to selected_class
    class_index_map = class_entry.get("class_index_map", None)
    if class_index_map:
        pred_label = next((k for k, v in class_index_map.items() if v == idx), selected_class)
    else:
        pred_label = selected_class

    # Use metrics for predicted label (if exists in JSON)
    pred_metrics = eval_json.get(pred_label, {}) or {}

    # Nutrition
    nutrition_selected = NutritionLoader(selected_class)
    nutrition_predicted = NutritionLoader(pred_label)

    response = {
        "success": True,
        "selected_class": selected_class,
        "predicted_label": pred_label,
        "confidence": conf,
        "model_used": model_used,
        "model_type": model_type,
        "class_metrics": pred_metrics,
        "classification_report": pred_metrics.get("classification_report"),
        "confusion_matrix": pred_metrics.get("confusion_matrix"),
        "confusion_matrix_full": pred_metrics.get("confusion_matrix_full"),
        "model_labels_order": pred_metrics.get("labels"),
        "nutrition_selected": nutrition_selected,
        "nutrition_predicted": nutrition_predicted
    }

    # Clean uploaded file
    # try:
    #     os.remove(filepath)
    # except:
    #     pass

    return jsonify(response)


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__=="__main__":
    app.run(debug=True)
