import os
import json

nut_data = {
    "Baked Potato":     {"fat": 0.2,  "carbohydrates": 37,  "protein": 3,   "calories": 161, "fiber": 3.8},
    "Crispy Chicken":   {"fat": 14,   "carbohydrates": 8,   "protein": 17,  "calories": 246, "fiber": 0},
    "Donut":            {"fat": 10,   "carbohydrates": 22,  "protein": 3,   "calories": 195, "fiber": 0.5},
    "Fries":            {"fat": 15,   "carbohydrates": 41,  "protein": 3.5, "calories": 312, "fiber": 3.8},
    "Hot Dog":          {"fat": 16,   "carbohydrates": 2,   "protein": 11,  "calories": 189, "fiber": 0},
    "Sandwich":         {"fat": 12,   "carbohydrates": 30,  "protein": 13,  "calories": 290, "fiber": 2},
    "Taco":             {"fat": 9,    "carbohydrates": 13,  "protein": 12,  "calories": 226, "fiber": 2},
    "Taquito":          {"fat": 7,    "carbohydrates": 15,  "protein": 7,   "calories": 187, "fiber": 1},
    "apple_pie":        {"fat": 14,   "carbohydrates": 47,  "protein": 2,   "calories": 296, "fiber": 2},
    "burger":           {"fat": 15,   "carbohydrates": 35,  "protein": 17,  "calories": 350, "fiber": 1},
    "butter_naan":      {"fat": 8,    "carbohydrates": 42,  "protein": 5,   "calories": 310, "fiber": 1},
    "chai":             {"fat": 1.5,  "carbohydrates": 8,   "protein": 1,   "calories": 60,  "fiber": 0},
    "chapati":          {"fat": 4,    "carbohydrates": 18,  "protein": 3,   "calories": 120, "fiber": 2},
    "cheesecake":       {"fat": 18,   "carbohydrates": 22,  "protein": 4,   "calories": 320, "fiber": 0},
    "chicken_curry":    {"fat": 9,    "carbohydrates": 6,   "protein": 22,  "calories": 210, "fiber": 1},
    "chole_bhature":    {"fat": 12,   "carbohydrates": 45,  "protein": 9,   "calories": 356, "fiber": 3},
    "dal_makhani":      {"fat": 8,    "carbohydrates": 20,  "protein": 8,   "calories": 215, "fiber": 2.5},
    "dhokla":           {"fat": 2,    "carbohydrates": 11,  "protein": 3,   "calories": 80,  "fiber": 1},
    "fried_rice":       {"fat": 4,    "carbohydrates": 44,  "protein": 8,   "calories": 270, "fiber": 1},
    "ice_cream":        {"fat": 11,   "carbohydrates": 16,  "protein": 2,   "calories": 190, "fiber": 0},
    "idli":             {"fat": 0.2,  "carbohydrates": 13,  "protein": 2,   "calories": 39,  "fiber": 0.9},
    "jalebi":           {"fat": 10,   "carbohydrates": 36,  "protein": 1,   "calories": 310, "fiber": 0},
    "kaathi_rolls":     {"fat": 16,   "carbohydrates": 38,  "protein": 12,  "calories": 380, "fiber": 2},
    "kadai_paneer":     {"fat": 21,   "carbohydrates": 9,   "protein": 11,  "calories": 300, "fiber": 2},
    "kulfi":            {"fat": 10,   "carbohydrates": 15,  "protein": 4,   "calories": 195, "fiber": 0},
    "masala_dosa":      {"fat": 12,   "carbohydrates": 30,  "protein": 5,   "calories": 260, "fiber": 2.5},
    "momos":            {"fat": 5,    "carbohydrates": 24,  "protein": 6,   "calories": 180, "fiber": 1},
    "omelette":         {"fat": 10,   "carbohydrates": 1,   "protein": 12,  "calories": 154, "fiber": 0},
    "paani_puri":       {"fat": 6,    "carbohydrates": 17,  "protein": 1,   "calories": 120, "fiber": 1},
    "pakode":           {"fat": 9,    "carbohydrates": 11,  "protein": 3,   "calories": 130, "fiber": 1},
    "pav_bhaji":        {"fat": 14,   "carbohydrates": 38,  "protein": 6,   "calories": 320, "fiber": 4},
    "pizza":            {"fat": 12,   "carbohydrates": 40,  "protein": 11,  "calories": 280, "fiber": 2},
    "samosa":           {"fat": 11,   "carbohydrates": 24,  "protein": 3,   "calories": 262, "fiber": 2},
    "sushi":            {"fat": 1,    "carbohydrates": 28,  "protein": 6,   "calories": 130, "fiber": 1},
}

folder='food_json'
for i in os.listdir(folder):
    if i.endswith('.json'):
        class_name=i.replace('.json','')
        if class_name in nut_data:
            file=os.path.join(folder,i)
            with open(file,'r') as f:
                data=json.load(f)
            data.update(nut_data[class_name])
            with open(file, "w") as f1:
                json.dump(data,f1, indent=4)
print(f'Json files updated successfully')