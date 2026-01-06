import json
import pickle
import numpy as np
import os
import shap
import pandas as pd

MODEL_PATH = "/checkpoint/RandomForestClassifier_cc1.pkl"
METADATA_PATH = "/checkpoint/cc1_metadata.json"

def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Không tìm thấy file model hoặc metadata!")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    return model, metadata

def preprocess_input(user_input, metadata):
    col_names = metadata['col_name']
    cat_idx = set(metadata['cat_idx'])
    cat_strs = metadata['cat_str']
    
    feature_vector = []
    current_cat_pointer = 0 
    
    for i, col_name in enumerate(col_names):
        if col_name not in user_input:
            raise ValueError(f"Thiếu thông tin đầu vào cho trường: {col_name}")
            
        raw_val = user_input[col_name]
        
        if i in cat_idx:
            valid_options = cat_strs[current_cat_pointer]
            try:
                processed_val = valid_options.index(raw_val)
            except ValueError:
                processed_val = 0
            
            feature_vector.append(processed_val)
            current_cat_pointer += 1
            
        else:
            try:
                processed_val = float(raw_val)
                feature_vector.append(processed_val)
            except ValueError:
                raise ValueError(f"Giá trị cho '{col_name}' phải là số!")

    return np.array([feature_vector])

def explain_prediction(real_model, input_vector, feature_names):
    print("\n--- PHÂN TÍCH SHAP VALUES ---")
    
    explainer = shap.TreeExplainer(real_model)
    shap_values = explainer.shap_values(input_vector)
    
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
        if len(shap_values_class1.shape) > 2:
            shap_values_class1 = shap_values_class1[:, :, 1]

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Value_Input": input_vector[0],
        "SHAP_Value": shap_values_class1[0]
    })
    
    feature_importance["Abs_SHAP"] = feature_importance["SHAP_Value"].abs()
    feature_importance = feature_importance.sort_values(
        by="Abs_SHAP",
        ascending=False
    ).drop(columns=["Abs_SHAP"])
    
    print(feature_importance.to_string(index=False))
    
    return explainer, shap_values

if __name__ == "__main__":
    model, metadata = load_resources()

    sample_input = {
        "age": 45,
        "balance": 100000.0,
        "vintage": 0.5,
        "transaction_status": 0,
        "credit_card": 0,
        "gender": "female",
        "income": "less_than_5L",
        "product_holdings": "1",
        "credit_type": "poor"
    }

    try:
        input_vector = preprocess_input(sample_input, metadata)

        real_model = model
        if hasattr(model, "clf"):
            real_model = model.clf
        elif hasattr(model, "model"):
            real_model = model.model

        prediction = real_model.predict(input_vector)
        probability = real_model.predict_proba(input_vector)
        
        print(f"Label dự đoán: {prediction[0]}")
        print(f"Xác suất Rời bỏ: {probability[0][1]:.4f}")

        explainer, shap_values = explain_prediction(
            real_model,
            input_vector,
            metadata["col_name"]
        )

    except Exception as e:
        print(f"\nLỖI: {e}")
