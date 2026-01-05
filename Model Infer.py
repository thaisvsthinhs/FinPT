import json
import pickle
import numpy as np
import os

MODEL_PATH = "/kaggle/working/saved_models/RandomForestClassifier_cd2.pkl"
METADATA_PATH = "/kaggle/working/saved_models/cd2_metadata.json"

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
    
    print("\n--- BẮT ĐẦU XỬ LÝ INPUT ---")
    
    for i, col_name in enumerate(col_names):
        if col_name not in user_input:
            raise ValueError(f"Thiếu thông tin đầu vào cho trường: {col_name}")
            
        raw_val = user_input[col_name]
        
        if i in cat_idx:
            valid_options = cat_strs[current_cat_pointer]
            try:
                processed_val = valid_options.index(raw_val)
            except ValueError:
                print(f"Cảnh báo: Giá trị '{raw_val}' không hợp lệ cho '{col_name}'. Mặc định về 0 ({valid_options[0]}).")
                processed_val = 0
                
            print(f"Mapping '{col_name}': '{raw_val}' -> {processed_val}")
            feature_vector.append(processed_val)
            current_cat_pointer += 1
            
        else:
            try:
                processed_val = float(raw_val)
                feature_vector.append(processed_val)
            except ValueError:
                raise ValueError(f"Giá trị cho '{col_name}' phải là số!")

    return np.array([feature_vector])

if __name__ == "__main__":
    model, metadata = load_resources()

    sample_input = {
        "credit_limit_in_NT_dollars": 10000,
        "gender": "male",
        "education": "high_school",
        "marriage": "other",
        "age": 40,
        "repayment_status_in_September_2005": "payment delay for eight months",
        "repayment_status_in_August_2005": "payment delay for seven months",
        "repayment_status_in_July_2005": "payment delay for six months",
        "repayment_status_in_June_2005": "payment delay for five months",
        "repayment_status_in_May_2005": "payment delay for four months",
        "repayment_status_in_April_2005": "payment delay for three months",
        "amount_of_bill_statement_in_September_2005": 15000,
        "amount_of_bill_statement_in_August_2005": 14800,
        "amount_of_bill_statement_in_July_2005": 14500,
        "amount_of_bill_statement_in_June_2005": 14000,
        "amount_of_bill_statement_in_May_2005": 13500,
        "amount_of_bill_statement_in_April_2005": 13000,
        "amount_of_previous_payment_in_September_2005": 0,
        "amount_of_previous_payment_in_August_2005": 0,
        "amount_of_previous_payment_in_July_2005": 0,
        "amount_of_previous_payment_in_June_2005": 0,
        "amount_of_previous_payment_in_May_2005": 0,
        "amount_of_previous_payment_in_April_2005": 0
    }

    try:
        input_vector = preprocess_input(sample_input, metadata)

        real_model = model
        if hasattr(model, 'clf'):
            print(">>> Đã tìm thấy model thật trong biến '.clf'")
            real_model = model.clf
        elif hasattr(model, 'model'):
            print(">>> Đã tìm thấy model thật trong biến '.model'")
            real_model = model.model

        prediction = real_model.predict(input_vector)
        print(f"Label dự đoán: {prediction[0]}")

        try:
            probability = real_model.predict_proba(input_vector)
            print(f"Xác suất (0 vs 1): {probability[0]}")
        except AttributeError:
            print("Model này không hỗ trợ predict_proba().")

    except Exception as e:
        print(f"\nLỖI: {e}")
