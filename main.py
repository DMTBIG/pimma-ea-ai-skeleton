from flask import Flask, jsonify, send_file
- import traceback
- import requests
import os
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import schedule
import threading
import time
import sys
import subprocess
import io
import base64

# Standard Library Imports first, then third-party, then local
# (Order is generally good, just consolidated)

# === Auto-install and Import Core Dependencies ===

# Requests (Imported once at the top, auto-install if missing)
try:
    import requests
except ImportError:
    print("Requests library not found, attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests
    print("Requests library installed and imported.")

# Matplotlib (Imported once at the top, auto-install if missing)
try:
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend for non-interactive plotting in threads
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Arial' # Set default font
except ImportError:
    print("Matplotlib not found, attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Arial'
    print("Matplotlib installed and imported.")

# stable-baselines3 and gym
try:
    import gym
    import stable_baselines3

    print("Gym, Stable-Baselines3, and Shimmy are already installed.")
except ImportError:
    print("stable-baselines3, gym, or shimmy not found, attempting to install all...")
    # Ensure 'subprocess' and 'sys' are imported at the very top of your file
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gym", "stable-baselines3", "shimmy~=2.0"])
    import gym
    import stable_baselines3
    print("gym, stable-baselines3, and shimmy installed and imported successfully.")
    
- from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv # อาจจำเป็นถ้า PIMMAEnv ไม่ได้เป็น VecEnv โดยตรง
- from gym import Env, spaces

# SHAP
try:
    import shap
    print("SHAP is already installed.")
except ImportError:
    print("SHAP library not found, attempting to install...")
    # SHAP can sometimes have specific dependencies, simple install first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    print("SHAP library installed and imported successfully.")
    
# === Global Configuration & Initialization ===
# Create necessary folders
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("presets", exist_ok=True) # Ensure presets folder exists
baseline_mean = None
explainer = None

# Logging Configuration
logging.basicConfig(
    filename="logs/ai_service.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Flask App Initialization
app = Flask(__name__)

# --- Constants ---
MODEL_PATH = "models/trained_model.xgb"
BEST_MODEL_PATH = "models/best_model.xgb"
TRADE_DATA_LOG_FILE = "logs/trade_data.csv"
LAST_RETRAIN_JSON = "logs/last_retrain.json"
WFT_RESULTS_CSV = "logs/walking_forward_results.csv"
EXECUTION_FAILURES_CSV = "logs/execution_failures.csv"
EXIT_DECISIONS_CSV = "logs/exit_decisions.csv"
AI_FAIL_ENTRY_CSV = "logs/ai_fail_entry.csv"

# --- Global Variables ---
USE_FAKE_MODEL = os.environ.get("USE_FAKE_MODEL", "False").lower() == "true" # Control via env variable
model = None # Will hold the loaded XGBoost model
ENFORCE_LICENSE = False # Toggle for license enforcement
ALLOWED_LICENSE_KEYS = {"ABC123", "YOUR_VALID_KEY_HERE"}

# === Utility Functions ===
def parse_feature_string(feature_str):
    """Parses a comma-separated string of features into a list of floats."""
    if not feature_str or not isinstance(feature_str, str):
        return []
    return [float(f) for f in feature_str.strip("[]").split(',') if f.strip()]

def _save_retrain_status(status, samples_used, duration_seconds, file_path=LAST_RETRAIN_JSON):
    """Saves the retraining status to a JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "samples_used": int(samples_used),
        "duration_seconds": float(duration_seconds),
        "status": str(status)
    }
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"[RETRAIN_STATUS_SAVE] Status saved: {status}, Samples: {samples_used}, Duration: {duration_seconds}s")
    except IOError as e:
        logging.error(f"[RETRAIN_STATUS_SAVE] Error saving retrain status to {file_path}: {e}", exc_info=True)

# === Reinforcement Learning (RL) Setup ===

def _initialize_shap_explainer():
    global model, explainer # ใช้ model ที่เป็น global XGBoost model ของเรา
    if model is not None and isinstance(model, xgb.XGBModel): # ตรวจสอบว่าเป็น XGBoost model
        try:
            explainer = shap.TreeExplainer(model)
            logging.info("[SHAP_INIT] SHAP TreeExplainer initialized successfully for the XGBoost model.")
        except Exception as e:
            logging.error(f"[SHAP_INIT] Failed to initialize SHAP TreeExplainer: {e}", exc_info=True)
            explainer = None
    elif model is not None:
        logging.warning(f"[SHAP_INIT] Model type ({type(model)}) is not directly XGBModel. SHAP TreeExplainer might not be optimal or work. Consider other SHAP explainers if needed.")
        # อาจจะลองใช้ KernelExplainer หรือ DeepExplainer สำหรับโมเดลประเภทอื่น
        # หรือปล่อยให้เป็น None ถ้า TreeExplainer ไม่เหมาะสม
        explainer = None # หรือจะลอง explainer = shap.Explainer(model) ถ้า SHAP รองรับอัตโนมัติ
    else:
        explainer = None
        logging.info("[SHAP_INIT] Main model not loaded. SHAP explainer not initialized.")


MODEL_FOLDER = "models" # ตรวจสอบว่าตรงกับที่คุณใช้งาน
os.makedirs(MODEL_FOLDER, exist_ok=True) # สร้างโฟลเดอร์ models ถ้ายังไม่มี

# สร้าง environment แบบง่าย
class PIMMAEnv(Env):
    def __init__(self, feature_dim):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(feature_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.current_obs = np.zeros(self.observation_space.shape, dtype=np.float32) # Ensure float32
        self._current_step = 0
        self._max_steps = 200 # ตัวอย่าง: กำหนดจำนวน step สูงสุดต่อ episode

    def reset(self):
        self.current_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._current_step = 0
        return self.current_obs

    def step(self, action):
        # ในการใช้งานจริง ส่วนนี้จะซับซ้อนกว่านี้มาก
        # ต้องมีการอัปเดต self.current_obs ด้วยข้อมูลใหม่จากตลาด
        # และคำนวณ reward จากผลของการ action นั้นๆ
        
        # สำหรับตอนนี้ ใช้ dummy reward และ next_obs คือ obs เดิม
        # และ done จะเป็น True ถ้าถึง _max_steps (เพื่อให้ learn() จบได้)
        self._current_step += 1
        reward = np.random.rand() - 0.5 # Dummy reward: random -0.5 to 0.5
        
        # ถ้า MQL ส่ง next_obs มาใน /rl/store, อาจจะไม่ต้องอัปเดต current_obs ที่นี่มากนัก
        # แต่ PPO.learn() จะเรียก step() นี้ซ้ำๆ เพื่อเก็บ rollouts
        # ดังนั้น PIMMAEnv ควรจะสามารถ simulate environment ได้ในระดับหนึ่ง
        # หรือคุณต้องมีวิธี map ข้อมูลจาก rl_buffer เข้าไปใน PPO.learn() โดยตรง (ซึ่งซับซ้อนกว่า)

        # Placeholder: สร้าง next_obs แบบสุ่มเล็กน้อยจาก current_obs
        # ในการใช้งานจริง next_obs ควรมาจากข้อมูลตลาดหลังจาก action
        next_obs = self.current_obs + np.random.normal(0, 0.1, self.current_obs.shape).astype(np.float32)
        self.current_obs = next_obs # อัปเดต current_obs สำหรับ step ถัดไป

        done = self._current_step >= self._max_steps
        info = {}
        return self.current_obs, reward, done, info

    def render(self, mode='human'): # Optional
        pass

    def close(self): # Optional
        pass

# โหลดหรือสร้าง policy
feature_dim = 45  # เท่ากับจำนวนฟีเจอร์
rl_model_path = os.path.join(MODEL_FOLDER, "rl_model.zip")

# สร้าง VecEnv ก่อนส่งให้ PPO
# PPO ใน stable-baselines3 มักจะทำงานกับ VecEnv
# ถ้า PIMMAEnv ของคุณซับซ้อน อาจจะต้อง wrap ด้วย DummyVecEnv หรือ SubprocVecEnv
# แต่ถ้า PIMMAEnv ถูกออกแบบให้ PPO.learn() เรียกใช้โดยตรง ก็อาจจะไม่ต้อง wrap ตอนสร้างโมเดล
# อย่างไรก็ตาม การใช้ VecEnv เป็น practice ที่ดี
# rl_env_instance = PIMMAEnv(feature_dim)
# rl_env = DummyVecEnv([lambda: rl_env_instance]) # Wrap it

if os.path.isfile(rl_model_path):
    try:
        # rl_model = PPO.load(rl_model_path, env=rl_env) # ถ้า PIMMAEnv ไม่ถูก save/load พร้อม model
        rl_model = PPO.load(rl_model_path) # ถ้า env ถูก save ไปกับ model หรือจะ set ทีหลัง
        # ถ้า env ไม่ได้ถูก load มาด้วย, ต้อง set ใหม่:
        if rl_model.get_env() is None:
            temp_env = PIMMAEnv(feature_dim)
            rl_model.set_env(DummyVecEnv([lambda: temp_env]))
        logging.info(f"[RL_SETUP] Loaded RL model from {rl_model_path}")
    except Exception as e:
        logging.error(f"[RL_SETUP] Error loading RL model from {rl_model_path}: {e}. Creating new model.", exc_info=True)
        temp_env = PIMMAEnv(feature_dim)
        rl_env_for_new_model = DummyVecEnv([lambda: temp_env])
        rl_model = PPO("MlpPolicy", rl_env_for_new_model, verbose=0, n_steps=128) # n_steps เป็น hyperparam สำคัญ
        rl_model.save(rl_model_path)
        logging.info(f"[RL_SETUP] Created and saved new RL model to {rl_model_path}")
else:
    temp_env = PIMMAEnv(feature_dim)
    rl_env_for_new_model = DummyVecEnv([lambda: temp_env])
    rl_model = PPO("MlpPolicy", rl_env_for_new_model, verbose=0, n_steps=128) # ปรับ n_steps ตามความเหมาะสม
    rl_model.save(rl_model_path)
    logging.info(f"[RL_SETUP] New RL model created and saved to {rl_model_path}")


# buffer เก็บ transitions
rl_buffer = []
buffer_lock = threading.Lock()

# === Model Loading ===
def load_model(path=MODEL_PATH):
    """Loads the XGBoost model from the specified path."""
    global model
    if os.path.exists(path) and not USE_FAKE_MODEL:
        try:
            model = joblib.load(path)
            logging.info(f"[MODEL STATUS] ✅ Model loaded successfully from {path}.")
            return True
        except Exception as e:
            logging.error(f"[MODEL STATUS] ❌ Failed to load model from {path}", exc_info=True)
            model = None
            return False
    elif USE_FAKE_MODEL:
        logging.warning("[MODEL STATUS] ⚠️ Using FAKE model as per configuration.")
        model = None # Ensure model is None if fake is used, predict_entry will handle it
        return True # Faking is a valid state
    else:
        logging.warning(f"[MODEL STATUS] ⚠️ Model file not found at {path}. Waiting for retrain or WFT best model activation.")
        model = None
        return False
   
load_model() # Initial attempt to load the model

def load_model(path=MODEL_PATH):
    global model # 'explainer' จะถูกจัดการโดย _initialize_shap_explainer ซึ่งใช้ global 'explainer'
    
    model_actually_loaded = False # ตั้งค่าเริ่มต้นสถานะการโหลดโมเดลจริง

    if os.path.exists(path) and not USE_FAKE_MODEL:
        try:
            model = joblib.load(path)
            logging.info(f"[MODEL STATUS] ✅ Real model loaded successfully from {path}.")
            model_actually_loaded = True
        except Exception as e:
            logging.error(f"[MODEL STATUS] ❌ Failed to load real model from {path}", exc_info=True)
            model = None # หากโหลดไม่สำเร็จ ให้ model เป็น None
            model_actually_loaded = False
            
    elif USE_FAKE_MODEL:
        logging.warning("[MODEL STATUS] ⚠️ Using FAKE model as per configuration.")
        model = None # ถ้าใช้ FAKE model ก็ไม่มีโมเดลจริงโหลด
        # สำหรับการ return ของฟังก์ชัน อาจจะถือว่า "สำเร็จ" เพราะเราตั้งใจใช้ fake model
        # หรือจะ return ค่าที่บ่งบอกว่าใช้ fake model ก็ได้ ขึ้นอยู่กับการออกแบบ
        # ในที่นี้จะให้ model_actually_loaded เป็น False เพราะไม่มี "โมเดลจริง" โหลด
        model_actually_loaded = False # หรือ True ถ้าคุณมองว่าการใช้ fake model คือ "โหลดสำเร็จ" แบบหนึ่ง

    else: # กรณีไฟล์ไม่พบ และไม่ได้ใช้ fake model
        logging.warning(f"[MODEL STATUS] ⚠️ Model file not found at {path} and not using FAKE model.")
        model = None
        model_actually_loaded = False
    
    # เรียก _initialize_shap_explainer() หนึ่งครั้งในตอนท้าย
    # เพื่อให้ explainer ถูกตั้งค่าตามสถานะล่าสุดของ global 'model'
    _initialize_shap_explainer()
    
    # การ return ค่าของฟังก์ชัน load_model:
    # อาจจะ return boolean ที่บอกว่า "มีโมเดลพร้อมใช้งานหรือไม่ (รวม fake model)"
    # หรือ "โมเดลจริงถูกโหลดสำเร็จหรือไม่"
    # ตัวอย่าง: ถ้าต้องการให้ฟังก์ชันบอกว่ามีโมเดล (จริงหรือ fake) พร้อมใช้งานหรือไม่:
    if model is not None or USE_FAKE_MODEL:
        return True # มีโมเดลจริงโหลด หรือ ตั้งใจใช้ fake model
    return False # ไม่มีโมเดルจริง และไม่ได้ตั้งใจใช้ fake model
    
# === Core AI Logic ===
def predict_entry_logic(features):
    """Core logic for making a prediction using the loaded model or a fake one."""
    if USE_FAKE_MODEL or model is None:
        logging.info("[PREDICT_LOGIC] Using fake model or model not loaded. Returning random prediction.")
        return round(np.random.uniform(0, 1), 4)
    try:
        # Ensure features is a list of floats
        if not isinstance(features, list) or not all(isinstance(f, (int, float)) for f in features):
            logging.error(f"[PREDICT_LOGIC] Invalid features format: {features}. Expected list of numbers.")
            # Fallback to fake prediction if features are malformed
            return round(np.random.uniform(0, 1), 4)

        prediction = model.predict(np.array([features]))[0] # XGBoost expects 2D array
        return float(prediction) # Ensure it's a Python float
    except Exception as e:
        logging.error("[PREDICT_LOGIC] Error during prediction", exc_info=True)
        # Fallback to fake prediction on error
        return round(np.random.uniform(0, 1), 4)

def _retrain_model_core(data_path=TRADE_DATA_LOG_FILE, output_model_path=MODEL_PATH):
    """Core logic for retraining the model."""
    start_time = time.time() # ย้าย start_time มาอยู่นอก try หรือบรรทัดแรกใน try ก็ได้
                           # ถ้าอยู่นอก try จะทำให้คำนวณ duration ได้แม้เกิด error ก่อนเริ่ม try

    try: # <--- try block หลักของฟังก์ชัน
        # --- ส่วนที่ 1: โหลดและตรวจสอบข้อมูล ---
        if not os.path.exists(data_path):
            _save_retrain_status("failed_no_file", 0, 0) # duration เป็น 0 เพราะยังไม่ได้เริ่ม
            logging.error(f"[RETRAIN_CORE] Data file not found at {data_path}")
            return False, "Data file not found", 0, 0

        df = pd.read_csv(data_path)
        df = df[df['command'].str.upper() == "PREDICT_ENTRY"]
        df = df[df['features'].notna() & df['result'].notna() & (df['result'] != "")]
        
        try:
            df['result'] = df['result'].astype(float)
        except ValueError as e:
            duration_err = round(time.time() - start_time, 2)
            _save_retrain_status("failed_invalid_result_data", 0, duration_err)
            logging.error(f"[RETRAIN_CORE] Error converting 'result' to float: {e}. Check data in {data_path}.", exc_info=True)
            return False, "Invalid data in 'result' column", 0, duration_err

        # --- ส่วนที่ 2: เตรียม X, y (ส่วน Robust X,y Preparation ที่เคยมี) ---
        processed_data = []
        for index, row in df.iterrows():
            features = parse_feature_string(row['features'])
            if features:
                processed_data.append({'X': features, 'y': row['result']})
        
        if not processed_data:
            duration_err = round(time.time() - start_time, 2)
            _save_retrain_status("failed_no_valid_features", 0, duration_err)
            logging.error("[RETRAIN_CORE] No valid features data after parsing to train the model.")
            return False, "No valid features to train on", 0, duration_err

        feature_length = len(processed_data[0]['X'])
        X_final = []
        y_final = []
        for item in processed_data:
            if len(item['X']) == feature_length:
                X_final.append(item['X'])
                y_final.append(item['y'])
            else:
                logging.warning(f"[RETRAIN_CORE] Inconsistent feature length found. Expected {feature_length}, got {len(item['X'])}. Skipping row.")

        if not X_final:
            duration_err = round(time.time() - start_time, 2)
            _save_retrain_status("failed_no_consistent_features", 0, duration_err)
            logging.error("[RETRAIN_CORE] No features with consistent length to train the model.")
            return False, "No features with consistent length", 0, duration_err
        # --- สิ้นสุด Robust X,y Preparation ---

        # --- ส่วนที่ 3: Train model ใหม่ ---
        model_new = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model_new.fit(np.array(X_final), np.array(y_final))
        joblib.dump(model_new, output_model_path)

        duration = round(time.time() - start_time, 2)
        _save_retrain_status("success", len(X_final), duration)
        logging.info(f"[RETRAIN_CORE] Model retrained successfully. Samples: {len(X_final)}, Duration: {duration}s. Saved to {output_model_path}")
        
        # --- ส่วนที่ 4: อัปเดต global model และ SHAP explainer ---
        if output_model_path == MODEL_PATH:
            global model # ใช้ global model ที่ประกาศไว้นอกฟังก์ชัน
            model = model_new
            logging.info("[RETRAIN_CORE] Global model updated with newly retrained model.")
            _initialize_shap_explainer() # <--- เรียกตรงนี้
            
        return True, "Retrain completed", len(X_final), duration

    except Exception as e: # <--- except block ที่สอดคล้องกับ try ด้านบน
        # หาก start_time ถูกกำหนดนอก try block หลัก เรายังสามารถคำนวณ duration ได้
        # หาก start_time อยู่ใน try และ error เกิดก่อนถึง start_time, duration อาจจะไม่ถูก define
        # ดังนั้น ควรประกาศ duration_err เริ่มต้น หรือ ย้าย start_time ออกมา
        current_duration_on_error = round(time.time() - start_time, 2) if 'start_time' in locals() else 0
        _save_retrain_status("failed_exception", 0, current_duration_on_error)
        logging.error("[RETRAIN_CORE] Exception during model retraining", exc_info=True)
        return False, str(e), 0, current_duration_on_error
        
# === Data Logging Functions ===
def log_trade_data(command, features_str, result=None, ticket=None, file_path=TRADE_DATA_LOG_FILE):
    """Logs trade-related data to a CSV file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "command": str(command),
        "features": str(features_str), # Store as string
        "result": result if result is not None else "",
        "ticket": ticket if ticket is not None else ""
    }
    df_new_log = pd.DataFrame([log_entry])
    try:
        if not os.path.exists(file_path):
            df_new_log.to_csv(file_path, index=False)
        else:
            df_new_log.to_csv(file_path, mode="a", header=False, index=False)
    except IOError as e:
        logging.error(f"[LOG_TRADE_DATA] Error writing to {file_path}: {e}", exc_info=True)

def log_generic_csv(file_path, data_dict):
    """Logs a dictionary to a specified CSV file."""
    df_entry = pd.DataFrame([data_dict])
    try:
        if not os.path.exists(file_path):
            df_entry.to_csv(file_path, index=False)
        else:
            df_entry.to_csv(file_path, mode="a", header=False, index=False)
    except IOError as e:
        logging.error(f"[LOG_GENERIC_CSV] Error writing to {file_path}: {e}", exc_info=True)

# === License Key System ===
def validate_license(http_request):
    """Validates the license key from the request headers."""
    if not ENFORCE_LICENSE:
        return True
    license_key = http_request.headers.get("X-License-Key", "")
    return license_key in ALLOWED_LICENSE_KEYS

# === Flask Routes ===
@app.route("/download_csv", methods=["GET"])
def download_csv():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    try:
        return send_file(TRADE_DATA_LOG_FILE, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"[DOWNLOAD_CSV] File not found: {TRADE_DATA_LOG_FILE}", exc_info=True)
        return jsonify({"error": "Trade data log file not found."}), 404
    except Exception as e:
        logging.error("[DOWNLOAD_CSV] Error", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    try:
        payload = request.get_json()
        if not payload or 'data' not in payload:
            return jsonify({"error": "Missing 'data' in payload"}), 400
        
        features_str = payload.get('data', "")
        features_list = parse_feature_string(features_str)

        if not features_list:
            logging.warning(f"[PREDICT_ROUTE] Received empty or invalid features string: {features_str}")
            return jsonify({"error": "Invalid or empty features string provided"}), 400

        logging.info(f"[PREDICT_ROUTE] Received features (sample): {features_list[:5]}… total={len(features_list)}")
        
        ai_score = predict_entry_logic(features_list)
        
        logging.info(f"[PREDICT_ROUTE] AI Score: {ai_score:.4f}")
        log_trade_data("PREDICT_ENTRY", features_str, ai_score) # Log original string and score
        return jsonify(ai_score), 200 # ai_score is already float

    except Exception as e:
        logging.error("[PREDICT_ROUTE] Exception occurred", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    try:
        raw_data = request.get_data()
        json_str = raw_data.decode('utf-8')
        payload = json.loads(json_str)
        params = payload.get("current_params", payload)

        optimized = {}
        for k, v_str in params.items():
            try:
                v = float(v_str)
                new_val = v * np.random.uniform(0.9, 1.1)
                optimized[k] = round(new_val, 3) # Python float
            except ValueError:
                logging.warning(f"[OPTIMIZER] Could not convert param {k} value '{v_str}' to float. Skipping.")
                optimized[k] = v_str # Keep original if not convertible

        logging.info(f"[OPTIMIZER] Optimized result: {optimized}")
        return jsonify(optimized), 200
    except json.JSONDecodeError as e:
        logging.error("[OPTIMIZER] JSON decode error", exc_info=True)
        return jsonify({"error": f"Invalid JSON: {e}", "raw_data_sample": raw_data[:100].decode('utf-8', errors='replace')}), 400
    except Exception as e:
        logging.error("[OPTIMIZER] Error occurred", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/monitor', methods=['GET'])
def monitor_route():
    # No license check for basic monitoring usually
    try:
        global model
        model_status_str = "NOT_LOADED"
        if USE_FAKE_MODEL:
            model_status_str = "FAKE"
        elif model is not None:
            model_status_str = "REAL"
            
        info = {
            "model_status": model_status_str,
            "model_path_configured": MODEL_PATH,
            "model_file_exists": os.path.exists(MODEL_PATH),
            "latency_ms": int(np.random.randint(20, 80)), # Python int
            "enforce_license": ENFORCE_LICENSE,
            "timestamp": datetime.now().isoformat()
        }
        logging.info("[MONITOR] Health check requested.")
        return jsonify(info), 200
    except Exception as e:
        logging.error("[MONITOR] Exception", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/retrain_status", methods=["GET"])
def retrain_status_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    try:
        with open(LAST_RETRAIN_JSON, "r") as f:
            data = json.load(f)
        
        html = f"""
        <html><head><title>🧠 Retrain Status</title></head><body>
            <h2>🧠 AI Model Retrain Status</h2>
            <ul>
                <li><b>📅 Timestamp:</b> {data.get("timestamp", "-")}</li>
                <li><b>✅ Samples Used:</b> {data.get("samples_used", "-")}</li>
                <li><b>⏱️ Duration:</b> {data.get("duration_seconds", "-")} seconds</li>
                <li><b>📈 Status:</b> {data.get("status", "-")}</li>
            </ul>
            <p>
                <a href="/visualize_summary"><button>📊 View Profit Summary Chart</button></a>
                <a href="/wft_summary"><button>📈 View WFT Summary Chart</button></a>
            </p>
            <p>
                <a href="/download_csv"><button>⬇ Download trade_data.csv</button></a>
                <a href="/download_report"><button>📋 Download Latest Report PDF</button></a>
            </p>
            <p><a href="/dashboard"><button>🏠 Back to Dashboard</button></a></p>
        </body></html>"""
        return html
    except FileNotFoundError:
        logging.warning(f"[RETRAIN_STATUS] {LAST_RETRAIN_JSON} not found.")
        return f"<p style='color:orange;'>⚠️ Retrain status file not found. Run a retrain cycle first.</p><p><a href='/dashboard'>Back to Dashboard</a></p>", 404
    except Exception as e:
        logging.error("[RETRAIN_STATUS] Error", exc_info=True)
        return f"<p style='color:red;'>❌ Error loading retrain status: {e}</p>", 500

@app.route('/explain', methods=['POST'])
def explain_shap_route(): # เปลี่ยนชื่อฟังก์ชันเล็กน้อยเพื่อไม่ให้ซ้ำกับของเดิม (ถ้ายังไม่ได้ลบ)
    global explainer
    if explainer is None:
        return jsonify({"error": "SHAP explainer not initialized. Model might not be loaded or compatible."}), 503 # Service Unavailable

    try:
        payload = request.get_json()
        if not payload or "features" not in payload:
            return jsonify({"error": "Missing 'features' in payload"}), 400

        feats_input = payload['features']
        if not isinstance(feats_input, list):
             return jsonify({"error": "'features' must be a list of numbers"}), 400
        
        # SHAP TreeExplainer คาดหวัง input เป็น 2D array (n_samples, n_features)
        # ในที่นี้เรารับ features สำหรับ 1 instance
        try:
            feats = np.array(feats_input, dtype=np.float32).reshape(1, -1)
        except ValueError as ve:
            return jsonify({"error": f"Invalid feature format or type: {ve}"}), 400

        # ตรวจสอบจำนวน features ถ้า model มี attribute นี้ (XGBoost model มักจะมี n_features_in_)
        if hasattr(model, 'n_features_in_') and feats.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Feature mismatch. Expected {model.n_features_in_} features, got {feats.shape[1]}."}), 400

        # คำนวณ SHAP values
        # สำหรับ XGBoost regressor (single output), explainer.shap_values(feats) จะคืน NumPy array shape (1, n_features)
        shap_values_for_instance = explainer.shap_values(feats) # ผลลัพธ์ควรเป็น (1, n_features)

        # เราต้องการ SHAP values สำหรับ instance เดียว (คือแถวแรก) และแปลงเป็น list
        # และ base_value (หรือ expected_value)
        if isinstance(shap_values_for_instance, list):
            # กรณี multi-output/multi-class (จากที่คุณโน้ตไว้ว่า regression ใช้ index [0])
            # แต่สำหรับ single output regression จาก TreeExplainer มักจะไม่ใช่ list
            # อย่างไรก็ตาม เพื่อให้ครอบคลุม:
            shap_values_to_send = shap_values_for_instance[0].tolist()
            # expected_value สำหรับ multi-class อาจเป็น array, สำหรับ regression เป็น float
            base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        elif isinstance(shap_values_for_instance, np.ndarray) and shap_values_for_instance.ndim == 2:
            shap_values_to_send = shap_values_for_instance[0].tolist() # เอาแถวแรก (instance เดียว)
            base_val = explainer.expected_value # สำหรับ regression มักเป็น float เดียว
        else:
            # กรณีที่ไม่คาดคิด
            logging.error(f"[EXPLAIN_SHAP] Unexpected shap_values format: {type(shap_values_for_instance)}")
            return jsonify({"error": "Unexpected SHAP values format from explainer."}), 500

        return jsonify({
            "shap_values": shap_values_to_send,
            "base_value": float(base_val) # explainer.expected_value มักจะเป็น float สำหรับ regression
        })

    except Exception as e:
        logging.error(f"[EXPLAIN_SHAP] Error calculating SHAP values: {e}", exc_info=True)
        return jsonify({"error": f"Could not calculate SHAP values: {str(e)}"}), 500

@app.route('/summary', methods=['GET'])
def summary_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    try:
        if not os.path.exists(TRADE_DATA_LOG_FILE):
            return jsonify({"error": f"{TRADE_DATA_LOG_FILE} not found. No data to summarize."}), 404
            
        df = pd.read_csv(TRADE_DATA_LOG_FILE)
        # Filter for rows where 'result' is not NaN and not an empty string
        df = df[df['result'].notna() & (df['result'] != "")]
        
        # Attempt to convert 'result' to float, coercing errors to NaN, then drop NaN
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df.dropna(subset=['result'], inplace=True)

        if df.empty:
            return jsonify({"message": "No valid trade results found to summarize."}), 200

        total_trades = len(df)
        avg_result = float(round(df['result'].mean(), 4)) if not df['result'].empty else 0.0
        std_dev_result = float(round(df['result'].std(), 4)) if not df['result'].empty else 0.0
        max_win = float(round(df['result'].max(), 4)) if not df['result'].empty else 0.0
        max_loss = float(round(df['result'].min(), 4)) if not df['result'].empty else 0.0
        # Ensure total_trades is not zero before division
        winrate = float(round((df['result'] > 0).sum() / total_trades * 100, 2)) if total_trades > 0 else 0.0

        stats = {
            "total_trades": total_trades,
            "average_result": avg_result,
            "std_deviation_result": std_dev_result,
            "max_win": max_win,
            "max_loss": max_loss,
            "winrate_percent": winrate,
            "last_updated": datetime.now().isoformat(),
            "data_source": TRADE_DATA_LOG_FILE
        }
        logging.info("[SUMMARY] Generated summary stats.")
        return jsonify(stats), 200
    except Exception as e:
        logging.error("[SUMMARY] Error generating summary", exc_info=True)
        return jsonify({"error": str(e)}), 500

def _generate_plot_base64(plot_function, *args, **kwargs):
    """Helper to generate a plot and return its base64 encoded string."""
    fig, ax = plt.subplots(figsize=(10, 5)) # Standardized figure size
    plot_function(ax, *args, **kwargs) # Call the specific plot drawing function
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig) # Close the figure to free memory
    return img64

def _plot_cumulative_pnl(ax, pnl_series):
    """Plots cumulative P&L on a given Matplotlib Axes object."""
    ax.plot(np.cumsum(pnl_series), label="Cumulative Result", color='green')
    ax.set_title("AI Result Summary (Cumulative)")
    ax.set_xlabel("Number of Trades/Events")
    ax.set_ylabel("Cumulative Result Value")
    ax.grid(True)
    ax.legend()

@app.route('/visualize_summary')
def visualize_summary_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    error_html_template = """
        <html><head><title>AI Profit Summary</title></head><body>
            <h1>📊 AI Profit Summary</h1>
            <p style="color:{color};font-weight:bold;">{message}</p>
            <p><a href="/download_csv"><button>⬇ Download trade_data.csv</button></a>
            <a href="/dashboard"><button>🏠 Back to Dashboard</button></a></p>
        </body></html>"""
    try:
        if not os.path.exists(TRADE_DATA_LOG_FILE):
            return error_html_template.format(color="orange", message=f"⚠️ Data file '{TRADE_DATA_LOG_FILE}' not found."), 404

        df = pd.read_csv(TRADE_DATA_LOG_FILE)
        if "result" not in df.columns:
            return error_html_template.format(color="red", message="❌ 'result' column not found in data."), 200
        
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        pnl_series = df["result"].dropna()

        if len(pnl_series) < 2:
            return error_html_template.format(color="orange", message="⚠️ Not enough valid result data (at least 2 points required) to generate a chart."), 200

        img64 = _generate_plot_base64(_plot_cumulative_pnl, pnl_series)

        return f"""
        <html><head><title>AI Profit Summary</title></head><body>
            <h1>📊 AI Profit Summary</h1>
            <img src="data:image/png;base64,{img64}" alt="Summary Chart"/>
            <br/><br/>
            <p><a href="/download_csv"><button>⬇ Download trade_data.csv</button></a>
            <a href="/dashboard"><button>🏠 Back to Dashboard</button></a></p>
        </body></html>""", 200
    except Exception as e:
        logging.error("[VISUALIZE_SUMMARY] Error generating chart", exc_info=True)
        return error_html_template.format(color="red", message=f"❌ An error occurred: {e}"), 500

# --- Placeholder/Simple Data Routes ---
@app.route('/trends')
def trends_route():
    kw = request.args.get('kw', 'general_topic')
    # Placeholder: In a real app, query Google Trends API or similar
    return jsonify(keyword=kw, trend_value=round(np.random.uniform(30, 90), 1), source="Placeholder Trends API"), 200

@app.route('/social')
def social_route():
    src = request.args.get('src', 'twitter')
    # Placeholder: In a real app, query Twitter/Reddit API and perform sentiment analysis
    return jsonify(source_platform=src, sentiment_score=round(np.random.uniform(-0.8, 0.8), 2), analysis_type="Placeholder Sentiment"), 200

@app.route('/macro')
def macro_route():
    name = request.args.get('name', 'PMI')
    # Placeholder: In a real app, query a financial data API
    return jsonify(indicator_name=name, value=round(np.random.uniform(45, 55), 1), source="Placeholder Macro Data API"), 200

@app.route('/orderbook')
def orderbook_route():
    symbol = request.args.get('symbol', 'BTCUSD')
    # Placeholder: In a real app, connect to exchange API
    return jsonify(trading_symbol=symbol, imbalance_ratio=round(np.random.uniform(-0.5, 0.5), 3), source="Placeholder Orderbook API"), 200

@app.route('/onchain')
def onchain_route():
    metric = request.args.get('metric', 'active_addresses')
    sym = request.args.get('symbol', 'ETH')
    # Placeholder: In a real app, query on-chain data provider
    return jsonify(crypto_symbol=sym, onchain_metric=metric, value=int(np.random.randint(1000, 100000)), source="Placeholder On-chain API"), 200

@app.route("/cot", methods=["GET"])
def cot_route():
    logging.info("[COT] COT Data requested")
    # Placeholder for Commitment of Traders data
    data = {"asset": "Gold", "net_long_commercials": round(np.random.uniform(50, 85), 2), "source": "Placeholder COT"}
    return jsonify(data), 200

@app.route("/openinterest", methods=["GET"])
def openinterest_route():
    logging.info("[OI] Open Interest requested")
    # Placeholder for Open Interest
    oi_value = int(np.random.randint(50000, 200000))
    return jsonify({"symbol": "XAUUSD_Futures", "open_interest": oi_value, "source": "Placeholder OI"}), 200

@app.route("/news", methods=["GET"])
def news_route():
    logging.info("[NEWS] News requested")
    # Placeholder for News Impact
    news_data = {
        "event_name": "FOMC Meeting Minutes",
        "impact_level": "high" if np.random.rand() > 0.5 else "medium",
        "affected_pairs": ["USDJPY", "EURUSD", "XAUUSD"],
        "scheduled_time": (datetime.now() + pd.Timedelta(hours=np.random.randint(1,5))).isoformat(),
        "source": "Placeholder News API"
    }
    return jsonify(news_data), 200

@app.route("/correlation", methods=["GET"])
def correlation_route():
    logging.info("[CORRELATION] Correlation data requested")
    # Placeholder for Correlation Data
    correlation_data = {
        "target_asset": "XAUUSD",
        "correlations": {
            "DXY": round(np.random.uniform(-0.9, -0.2), 2),
            "US10Y_BondYield": round(np.random.uniform(-0.5, 0.5), 2),
            "OIL": round(np.random.uniform(-0.3, 0.3), 2)
        },
        "source": "Placeholder Correlation Engine"
    }
    return jsonify(correlation_data), 200

@app.route("/vwap", methods=["GET"])
def vwap_route():
    symbol = request.args.get('symbol', 'XAUUSD')
    logging.info(f"[VWAP] VWAP requested for {symbol}")
    data = {"symbol": symbol, "vwap": round(np.random.uniform(1900, 2100), 2), "timeframe": "1H", "source": "Placeholder VWAP"}
    return jsonify(data), 200

@app.route("/volumeprofile", methods=["GET"])
def volumeprofile_route():
    symbol = request.args.get('symbol', 'XAUUSD')
    logging.info(f"[VolumeProfile] Volume Profile requested for {symbol}")
    profile = {
        "symbol": symbol,
        "point_of_control": round(np.random.uniform(1950, 1970), 2),
        "high_volume_node": round(np.random.uniform(1920, 1980), 2),
        "low_volume_node": round(np.random.uniform(1850, 1900), 2),
        "timeframe": "Daily",
        "source": "Placeholder Volume Profile"
    }
    return jsonify(profile), 200

@app.route("/harmonics", methods=["GET"])
def harmonics_route():
    symbol = request.args.get('symbol', 'EURUSD')
    logging.info(f"[Harmonics] Harmonic Pattern scan requested for {symbol}")
    patterns = ["Gartley", "Bat", "Butterfly", "Crab", "Shark"]
    data = {
        "symbol": symbol,
        "pattern_detected": np.random.choice(patterns),
        "status": "developing" if np.random.rand() > 0.3 else "valid_entry_zone",
        "entry_zone_start": round(np.random.uniform(1.0500, 1.0600), 4),
        "entry_zone_end": round(np.random.uniform(1.0601, 1.0700), 4),
        "timeframe": "H4",
        "source": "Placeholder Harmonics Scanner"
    }
    return jsonify(data), 200

# === Retraining and Model Management Routes ===
@app.route('/retrain', methods=['GET']) # Can be POST if parameters are sent, GET for simple trigger
def retrain_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    logging.info("[RETRAIN_ROUTE] Manual retrain triggered.")
    
    # Run core logic in a thread to make the HTTP request return faster for manual triggers
    # For scheduled tasks, it runs in the scheduler's thread already.
    def _run_and_log():
        with app.app_context(): # Need app context if core logic uses Flask specific things (not much here)
             success, message, samples, duration = _retrain_model_core()
             # Further logging or actions after retrain if needed
             if success:
                 logging.info(f"[RETRAIN_ROUTE_THREAD] Retrain successful via route: {message}")
             else:
                 logging.error(f"[RETRAIN_ROUTE_THREAD] Retrain failed via route: {message}")

    thread = threading.Thread(target=_run_and_log)
    thread.start()
    
    return jsonify({
        "status": "triggered",
        "message": "Retrain process initiated in the background. Check /retrain_status for updates."
    }), 202


@app.route('/log_result', methods=['POST'])
def log_result_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    try:
        payload = request.get_json()
        ticket = payload.get("ticket")
        result_val = payload.get("result") # Renamed to avoid conflict with 'result' variable from list comprehension

        if ticket is None or result_val is None:
            return jsonify({"error": "Missing 'ticket' or 'result' in payload"}), 400

        try:
            ticket = int(ticket)
            result_val = float(result_val)
        except ValueError:
            return jsonify({"error": "Ticket must be an integer and result must be a float."}), 400

        # This logic updates the last PREDICT_ENTRY without a ticket.
        # It assumes that PREDICT_ENTRY is logged first, then its outcome (ticket, result) is logged later.
        if not os.path.exists(TRADE_DATA_LOG_FILE):
            # If file doesn't exist, we can't update. Log as a new entry perhaps?
            # Or, more simply, state that no prior entry exists to update.
            # For now, let's ensure the file can be created by log_trade_data if needed,
            # but updating requires a prior entry.
            log_trade_data("RESULT_LOGGED_NO_PRIOR_PREDICT", f"Ticket: {ticket}", result_val, ticket)
            return jsonify({"status": "logged_as_new", "message": "No prior predict entry to update, logged as new result."}), 201

        df = pd.read_csv(TRADE_DATA_LOG_FILE)
        updated = False
        # Iterate backwards to find the most recent PREDICT_ENTRY without a ticket
        for idx in range(len(df) - 1, -1, -1):
            # Ensure 'ticket' column exists and handle potential float NaNs from CSV read
            # Also ensure 'command' column exists
            if 'command' in df.columns and df.at[idx, "command"].upper() == "PREDICT_ENTRY":
                current_ticket = df.at[idx, "ticket"] if 'ticket' in df.columns else np.nan
                if pd.isna(current_ticket) or str(current_ticket).strip() == "":
                    df.loc[idx, "ticket"] = ticket
                    df.loc[idx, "result"] = result_val # Update result as well, might be P/L
                    updated = True
                    break
        
        if updated:
            df.to_csv(TRADE_DATA_LOG_FILE, index=False)
            logging.info(f"[LOG_RESULT] Successfully updated trade log for ticket {ticket} with result {result_val}.")
            return jsonify({"status": "success", "message": f"Result for ticket {ticket} logged."}), 200
        else:
            # If no suitable PREDICT_ENTRY was found to update, log this as a new "RESULT_ONLY" entry.
            log_trade_data("RESULT_LOGGED_ORPHANED", f"Ticket: {ticket}", result_val, ticket)
            logging.warning(f"[LOG_RESULT] No matching PREDICT_ENTRY found to update for ticket {ticket}. Logged as orphaned result.")
            return jsonify({"status": "not_found_or_already_logged", "message": "No unlogged PREDICT_ENTRY found to associate this result with, or it was already logged. Logged as new/orphaned."}), 202

    except Exception as e:
        logging.error("[LOG_RESULT] Exception occurred", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/log_execution_failure", methods=["POST"])
def log_execution_failure_route():
    # No license for internal logging typically
    raw_data = request.get_data()
    try:
        # Handle potential null terminators from MQL4/5
        cleaned_raw_data = raw_data.rstrip(b'\x00')
        json_str = cleaned_raw_data.decode('utf-8')
        payload = json.loads(json_str)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        logging.error(f"[LOG_EXEC_FAIL] JSON parse/decode error: {e}. Raw (first 100 bytes): {raw_data[:100]!r}", exc_info=True)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "PayloadParseError",
            "details": str(e),
            "raw_payload_sample": raw_data[:200].decode('ascii', errors='replace') # Save a sample
        }
        log_generic_csv(EXECUTION_FAILURES_CSV, log_entry)
        return jsonify({"error": f"Invalid JSON or encoding: {e}", "raw_bytes_sample": list(raw_data[:64])}), 400

    ts = datetime.now().isoformat()
    log_data = {"timestamp": ts}
    log_file_target = None

    if "reason" in payload and "ticket" in payload: # More specific for execution failures
        log_file_target = EXECUTION_FAILURES_CSV
        log_data["ticket"] = payload.get("ticket", "")
        log_data["reason"] = payload.get("reason", "Unknown reason")
        log_data["details"] = payload.get("details", "") # Optional extra details
        logging.info(f"[LOG_EXEC_FAIL] Logging execution failure for ticket {log_data['ticket']}: {log_data['reason']}")
    elif "score" in payload and "method" in payload and "ticket" in payload: # For exit decisions
        log_file_target = EXIT_DECISIONS_CSV
        log_data["ticket"] = payload.get("ticket", "")
        log_data["score"] = payload.get("score", 0.0)
        log_data["method"] = payload.get("method", "Unknown method")
        logging.info(f"[LOG_EXIT_DECISION] Logging exit decision for ticket {log_data['ticket']}: Score {log_data['score']}, Method {log_data['method']}")
    else:
        logging.warning(f"[LOG_EXEC_FAIL] Unknown payload structure for logging: {payload}")
        # Log to a generic error log or a specific "malformed_payloads.csv"
        log_generic_csv("logs/malformed_failure_logs.csv", {"timestamp": ts, "payload_received": json.dumps(payload)})
        return jsonify({"error": "Unknown payload structure. Required fields: (ticket, reason) or (ticket, score, method).", "received_payload": payload}), 400

    log_generic_csv(log_file_target, log_data)
    return jsonify({"status": "ok", "message": "Log received."}), 200

# This route seems redundant if /log_execution_failure handles exit decisions.
# Kept for compatibility if MQL uses it directly, but logic is similar.
@app.route("/log_exit_decision", methods=["POST"])
def log_exit_decision_route_specific():
    # This acts as a more specific endpoint, essentially a subset of /log_execution_failure
    raw_bytes = request.get_data()
    logging.info(f"[LOG_EXIT_DECISION_SPECIFIC] Raw bytes: {raw_bytes!r}")
    try:
        s = raw_bytes.rstrip(b'\x00').decode('utf-8')
        payload = json.loads(s)
    except Exception as e:
        logging.error(f"[LOG_EXIT_DECISION_SPECIFIC] JSON parse error: {e}, raw string: {s!r}", exc_info=True)
        return jsonify({"error": f"Invalid JSON: {e}", "raw_string_content": s}), 400

    if not all(k in payload for k in ["ticket", "score", "method"]):
        return jsonify({"error": "Missing required fields: ticket, score, method", "received_payload": payload}), 400

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "ticket": payload.get("ticket", ""),
        "score": payload.get("score", 0.0),
        "method": payload.get("method", "")
    }
    log_generic_csv(EXIT_DECISIONS_CSV, log_data)
    logging.info(f"[LOG_EXIT_DECISION_SPECIFIC] Logged: {log_data}")
    return jsonify({"status": "ok", "message": "Exit decision logged."}), 200


@app.route("/log_fail_entry", methods=["POST"])
def log_fail_entry_route():
    # No license for internal logging
    try:
        payload = request.get_json()
        if not payload: return jsonify({"error": "Empty payload"}), 400

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "score": payload.get("score", 0.0),
            "features": str(payload.get("features", "")), # Store as string
            "reason": payload.get("reason", "Unknown reason")
        }
        log_generic_csv(AI_FAIL_ENTRY_CSV, log_data)
        logging.info(f"[FAIL_ENTRY_LOG] Logged AI entry failure: {log_data['reason']}")
        return jsonify({"status": "logged"}), 200
    except Exception as e:
        logging.error("[FAIL_ENTRY_LOG] Error logging failed entry", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- HTML View Routes for Logs ---
def _generate_html_table_page(csv_path, page_title, error_message_not_found):
    """Helper to generate an HTML page displaying a CSV file as a table."""
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(error_message_not_found)
        
        df = pd.read_csv(csv_path)
        # Sanitize column names for HTML display if necessary (e.g., replace underscores)
        # df.columns = [col.replace('_', ' ').title() for col in df.columns]
        html_table = df.to_html(classes="table table-striped table-hover", index=False, border=0, escape=True)

        return f"""
        <html>
        <head>
            <title>{page_title}</title>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
            <style> body {{ padding: 20px; font-family: Arial, sans-serif; }} h1 {{ margin-bottom: 20px; }} </style>
        </head>
        <body>
            <h1>{page_title}</h1>
            {html_table if not df.empty else '<p>No data logged yet.</p>'}
            <br><p><a href="/dashboard" class="btn btn-primary">⬅ Back to Dashboard</a></p>
        </body></html>"""
    except FileNotFoundError:
        logging.warning(f"[HTML_TABLE_VIEW] File not found: {csv_path}")
        return f"<p style='color:orange;'>⚠️ {error_message_not_found}</p><p><a href='/dashboard'>Back to Dashboard</a></p>", 404
    except Exception as e:
        logging.error(f"[HTML_TABLE_VIEW] Error loading log for {csv_path}", exc_info=True)
        return f"<p style='color:red;'>❌ Error loading log: {e}</p><p><a href='/dashboard'>Back to Dashboard</a></p>", 500

@app.route("/fail_log", methods=["GET"])
def fail_log_view_route():
    return _generate_html_table_page(EXECUTION_FAILURES_CSV, "❌ Execution Failure Log", "Execution failure log not found.")

@app.route("/exit_decision_log", methods=["GET"]) # New route for specific exit decision log
def exit_decision_log_view_route():
    return _generate_html_table_page(EXIT_DECISIONS_CSV, "🚪 Exit Decision Log", "Exit decision log not found.")

@app.route("/fail_entry_log", methods=["GET"])
def fail_entry_log_view_route():
    return _generate_html_table_page(AI_FAIL_ENTRY_CSV, "🚫 AI Entry Fail Log", f"{AI_FAIL_ENTRY_CSV} not found.")


# --- Walking Forward Test (WFT) ---
def _run_walking_forward_test_core(
    data_path=TRADE_DATA_LOG_FILE,
    output_path=WFT_RESULTS_CSV,
    train_window=300, # Number of samples
    test_window=50   # Number of samples
):
    try:
        logging.info("[WFT_CORE] 🔁 Starting Walking Forward Test")
        if not os.path.exists(data_path):
            logging.error(f"[WFT_CORE] Data file not found at {data_path}")
            return {"status": "error", "message": f"Data file not found: {data_path}"}

        df = pd.read_csv(data_path)
        df = df[df['command'].str.upper() == "PREDICT_ENTRY"]
        df = df[df['features'].notna() & df['result'].notna() & (df['result'] != "")]
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df.dropna(subset=['result'], inplace=True)
        
        # Parse features and keep only valid ones along with their results
        parsed_data = []
        for index, row in df.iterrows():
            features = parse_feature_string(row['features'])
            timestamp = pd.to_datetime(row['timestamp'], errors='coerce') # Add timestamp parsing
            if features and pd.notna(timestamp):
                parsed_data.append({'X': features, 'y': row['result'], 'timestamp': timestamp})
        
        if not parsed_data:
            logging.error("[WFT_CORE] No valid feature/result pairs after parsing for WFT.")
            return {"status": "error", "message": "No valid feature/result data for WFT."}

        # Ensure all feature lists have consistent length
        expected_len = len(parsed_data[0]['X'])
        df_wft_data = pd.DataFrame([d for d in parsed_data if len(d['X']) == expected_len])

        if len(df_wft_data) < train_window + test_window:
            msg = f"Not enough consistent data for WFT (required: {train_window + test_window}, found: {len(df_wft_data)})"
            logging.error(f"[WFT_CORE] {msg}")
            return {"status": "error", "message": msg}
        
        df_wft_data = df_wft_data.sort_values("timestamp").reset_index(drop=True)

        wft_results_list = []
        total_rounds = (len(df_wft_data) - train_window) // test_window

        if total_rounds <= 0:
            msg = f"Not enough data for even one WFT round with train_window={train_window}, test_window={test_window}."
            logging.error(f"[WFT_CORE] {msg}")
            return {"status": "error", "message": msg}

        for i in range(total_rounds):
            train_start_idx = i * test_window
            train_end_idx = train_start_idx + train_window
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_window

            if test_end_idx > len(df_wft_data): # Ensure test data does not exceed bounds
                break 

            train_set = df_wft_data.iloc[train_start_idx:train_end_idx]
            test_set = df_wft_data.iloc[test_start_idx:test_end_idx]
            
            if train_set.empty or test_set.empty:
                logging.warning(f"[WFT_CORE] Skipping round {i+1} due to empty train/test set.")
                continue

            X_train = np.array(train_set['X'].tolist())
            y_train = np.array(train_set['y'].tolist())
            X_test = np.array(test_set['X'].tolist())
            # y_test_actual = np.array(test_set['y'].tolist()) # Actual results for evaluation

            wft_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=i) # Use round for different random_state
            wft_model.fit(X_train, y_train)
            
            # Predictions are the model's output, treat these as the "simulated trade results/scores"
            # The 'result' column in trade_data.csv is the target for the regressor.
            # If this target is already a P&L or performance metric, then 'preds' are predicted P&L/metric.
            preds = wft_model.predict(X_test) 

            avg_pred_metric = np.mean(preds)
            std_pred_metric = np.std(preds)
            # "Win rate" based on predicted metric being positive (if metric > 0 is "win")
            winrate_on_pred = np.mean(preds > 0) * 100 if len(preds) > 0 else 0
            # Drawdown on the sequence of predicted metrics
            mdd_on_pred = np.min(np.cumsum(preds)) if len(preds) > 0 else 0

            round_result_data = {
                "round": i + 1,
                "train_start_ts": train_set['timestamp'].iloc[0].isoformat(),
                "test_start_ts": test_set['timestamp'].iloc[0].isoformat(),
                "avg_predicted_metric": round(avg_pred_metric, 4),
                "std_predicted_metric": round(std_pred_metric, 4),
                "winrate_on_predicted_metric_percent": round(winrate_on_pred, 2),
                "max_drawdown_on_predicted_metric": round(mdd_on_pred, 4),
                "samples_train": len(X_train),
                "samples_test": len(X_test)
            }
            wft_results_list.append(round_result_data)
            logging.info(f"[WFT_CORE] Round {i+1}/{total_rounds} complete: AvgPred={avg_pred_metric:.2f}, WinRatePred={winrate_on_pred:.1f}%")

        if not wft_results_list:
            logging.warning("[WFT_CORE] No WFT rounds were completed.")
            return {"status": "warning", "message": "WFT completed but no rounds generated results."}
            
        results_df = pd.DataFrame(wft_results_list)
        results_df.to_csv(output_path, index=False)
        logging.info(f"[WFT_CORE] ✅ Walking Forward Test finished. Results saved to {output_path}")
        return {"status": "success", "message": f"WFT finished. Results saved to {output_path}", "rounds_completed": len(wft_results_list)}

    except Exception as e:
        logging.error("[WFT_CORE] ❌ Error in walking forward test", exc_info=True)
        return {"status": "error", "message": f"WFT failed: {str(e)}"}

@app.route("/wft", methods=["GET"]) # Or POST if params
def wft_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    logging.info("[WFT_ROUTE] WFT initiated via route.")
    # Run WFT in a background thread
    # Get params from request.args if needed, e.g., /wft?train_window=400
    train_win = request.args.get('train_window', 300, type=int)
    test_win = request.args.get('test_window', 50, type=int)

    thread = threading.Thread(target=_run_walking_forward_test_core, args=(TRADE_DATA_LOG_FILE, WFT_RESULTS_CSV, train_win, test_win))
    thread.daemon = True # Allow main program to exit even if threads are running
    thread.start()
    return jsonify({"status": "triggered", "message": f"WFT process initiated in background with train_window={train_win}, test_window={test_win}. Check logs or WFT summary page."}), 202

def _plot_wft_summary(ax, df_wft):
    """Plots WFT summary on a given Matplotlib Axes."""
    ax.plot(df_wft['round'], df_wft['avg_predicted_metric'], label='Avg Predicted Metric', marker='o')
    ax.plot(df_wft['round'], df_wft['winrate_on_predicted_metric_percent'], label='Winrate on Predicted Metric (%)', marker='x')
    ax.set_title("📊 Walking Forward Test (WFT) Summary")
    ax.set_xlabel("Test Round Number")
    ax.set_ylabel("Performance Metric Value")
    ax.grid(True)
    ax.legend()

@app.route("/wft_summary")
def wft_summary_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    error_html_template = """
        <html><head><title>WFT Summary</title></head><body>
            <h1>📈 WFT Summary</h1>
            <p style="color:{color};font-weight:bold;">{message}</p>
            <p><a href="/wft"><button>🔁 Run WFT Now</button></a>
            <a href="/dashboard"><button>🏠 Back to Dashboard</button></a></p>
        </body></html>"""
    try:
        if not os.path.exists(WFT_RESULTS_CSV):
            return error_html_template.format(color="orange", message=f"⚠️ WFT results file ('{WFT_RESULTS_CSV}') not found. Please run WFT first."), 404

        df_wft = pd.read_csv(WFT_RESULTS_CSV)
        if df_wft.empty or 'round' not in df_wft.columns or \
           'avg_predicted_metric' not in df_wft.columns or \
           'winrate_on_predicted_metric_percent' not in df_wft.columns:
            return error_html_template.format(color="red", message="❌ WFT results file is empty or has incorrect columns."), 200
        
        # Ensure data types for plotting
        df_wft['round'] = pd.to_numeric(df_wft['round'], errors='coerce')
        df_wft['avg_predicted_metric'] = pd.to_numeric(df_wft['avg_predicted_metric'], errors='coerce')
        df_wft['winrate_on_predicted_metric_percent'] = pd.to_numeric(df_wft['winrate_on_predicted_metric_percent'], errors='coerce')
        df_wft.dropna(subset=['round', 'avg_predicted_metric', 'winrate_on_predicted_metric_percent'], inplace=True)

        if len(df_wft) < 1 : # Need at least one point to plot meaningfully
             return error_html_template.format(color="orange", message=f"⚠️ Not enough valid data in WFT results to generate chart."), 200

        img64 = _generate_plot_base64(_plot_wft_summary, df_wft)
        
        return f"""
        <html><head><title>WFT Summary</title></head><body>
            <h1>📈 Walking Forward Test Summary</h1>
            <img src="data:image/png;base64,{img64}" alt="WFT Summary Chart"/>
            <br/><br/>
            <p><a href="/wft"><button>🔁 Run WFT Again</button></a>
            <a href="/download_csv"><button>⬇ trade_data.csv</button></a>
            <a href="/dashboard"><button>🏠 Back to Dashboard</button></a></p>
        </body></html>""", 200
    except Exception as e:
        logging.error("[WFT_SUMMARY_ROUTE] Error generating WFT summary chart", exc_info=True)
        return error_html_template.format(color="red", message=f"❌ An error occurred: {e}"), 500

def _train_model_from_best_wft_round(
    wft_results_path=WFT_RESULTS_CSV,
    trade_data_path=TRADE_DATA_LOG_FILE,
    output_model_path=BEST_MODEL_PATH,
    optimize_metric='winrate_on_predicted_metric_percent' # or 'avg_predicted_metric'
):
    try:
        if not os.path.exists(wft_results_path):
            return False, f"WFT results file not found: {wft_results_path}"
        
        df_wft = pd.read_csv(wft_results_path)
        if df_wft.empty or optimize_metric not in df_wft.columns:
            return False, f"WFT results empty or missing optimizing metric '{optimize_metric}'."

        # Find the best round based on the chosen metric (higher is better)
        best_row = df_wft.loc[df_wft[optimize_metric].idxmax()]
        
        best_train_start_ts = pd.to_datetime(best_row['train_start_ts'])
        num_train_samples_in_best_round = int(best_row["samples_train"])

        # Load all trade data
        df_all_trades = pd.read_csv(trade_data_path)
        df_all_trades = df_all_trades[df_all_trades['command'].str.upper() == "PREDICT_ENTRY"]
        df_all_trades['result'] = pd.to_numeric(df_all_trades['result'], errors='coerce')
        df_all_trades.dropna(subset=['features', 'result'], inplace=True)
        df_all_trades['timestamp'] = pd.to_datetime(df_all_trades['timestamp'])
        df_all_trades.sort_values('timestamp', inplace=True)

        # Filter data that was used for training in the best WFT round
        # This requires careful matching of timestamps and feature parsing as done in WFT
        training_data_for_best_model_list = []
        for index, row in df_all_trades.iterrows():
            if row['timestamp'] >= best_train_start_ts:
                features = parse_feature_string(row['features'])
                if features:
                    training_data_for_best_model_list.append({'X': features, 'y': row['result']})
                if len(training_data_for_best_model_list) >= num_train_samples_in_best_round:
                    break # Collected enough samples corresponding to the best WFT training set

        if len(training_data_for_best_model_list) < num_train_samples_in_best_round :
            return False, f"Could not gather enough training samples ({len(training_data_for_best_model_list)} found, {num_train_samples_in_best_round} expected) for the best WFT round."

        df_train_best = pd.DataFrame(training_data_for_best_model_list)
        X_train_best = np.array(df_train_best['X'].tolist())
        y_train_best = np.array(df_train_best['y'].tolist())

        # Train the final model using these parameters
        final_best_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=int(best_row['round']))
        final_best_model.fit(X_train_best, y_train_best)
        joblib.dump(final_best_model, output_model_path)

        msg = f"Best model trained from WFT round {best_row['round']} (Metric {optimize_metric}: {best_row[optimize_metric]:.2f}). Saved to {output_model_path}."
        logging.info(f"[WFT_BEST_MODEL] {msg}")
        return True, msg
    except Exception as e:
        logging.error("[WFT_BEST_MODEL] Error training best model from WFT", exc_info=True)
        return False, str(e)

@app.route("/activate_best_wft_model", methods=["GET"])
def activate_best_wft_model_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    global model # We will update the global model
    try:
        success, message = _train_model_from_best_wft_round()
        if not success:
            return jsonify({"status": "error", "message": f"Failed to train best model from WFT: {message}"}), 500

        # Load the newly trained best model as the active model
        if load_model(path=BEST_MODEL_PATH): # load_model updates global 'model'
            return jsonify({"status": "success", "message": f"Best model from WFT activated. {message}"}), 200
        else:
            return jsonify({"status": "error", "message": "Best model trained from WFT but failed to load it as active model."}), 500
            
    except Exception as e:
        logging.error("[ACTIVATE_BEST_WFT_MODEL] Failed to activate best WFT model", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# --- Report Generation ---
def _create_exit_report_placeholder(output_dir="."):
    """Creates a placeholder trading exit report (text file)."""
    try:
        filename_base = f"TradingReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Placeholder text file for now
        txt_filename = os.path.join(output_dir, f"{filename_base}.txt") # Changed to .txt for placeholder

        report_content = f"""
        Trading Exit Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ----------------------------------------------------
        This is a placeholder for the Trading Exit Report.
        
        Summary of Trades:
        - (Data from {TRADE_DATA_LOG_FILE} or {EXIT_DECISIONS_CSV} would go here)
        - Total Trades Considered: ...
        - Profitable Exits: ...
        - Losing Exits: ...

        Key Metrics:
        - Average P&L per Exit: ...
        - Win Rate of Exits: ...
        - Largest Winning Exit: ...
        - Largest Losing Exit: ...

        Notes:
        - (Any specific observations or issues encountered)
        
        Future PDF implementations should include charts and more detailed tables.
        """
        with open(txt_filename, "w") as f:
            f.write(report_content)
        logging.info(f"[REPORT_GEN] ✅ Placeholder report saved as {txt_filename}")
        return txt_filename
    except Exception as e:
        logging.error("[REPORT_GEN] ❌ Error generating placeholder report", exc_info=True)
        return None

@app.route("/download_report", methods=["GET"])
def download_report_route():
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    try:
        # Reports are currently text files, saved in the script's CWD
        reports_dir = os.path.abspath(".") # Assuming reports are in the root for now
        # Look for .txt reports, as PDF is placeholder
        files = [f for f in os.listdir(reports_dir) if f.startswith("TradingReport_") and f.endswith(".txt")]
        if not files:
            return "<p>❌ No reports (TradingReport_*.txt) found in the application directory.</p><p><a href='/dashboard'>Back to Dashboard</a></p>", 404

        latest_file = sorted(files, key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)), reverse=True)[0]
        return send_file(os.path.join(reports_dir, latest_file), as_attachment=True)
    except Exception as e:
        logging.error("[DOWNLOAD_REPORT] Error finding/sending report", exc_info=True)
        return f"<p style='color:red;'>❌ Error loading report: {e}</p>", 500

# --- Presets ---
@app.route("/preset/<name>")
def get_preset_route(name):
    if not validate_license(request): return jsonify({"error": "Invalid license"}), 403
    try:
        # Sanitize name to prevent directory traversal
        safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        if not safe_name:
            return jsonify({"error": "Invalid preset name"}), 400
            
        preset_file_path = os.path.join("./presets/", f"{safe_name}.json")
        with open(preset_file_path, "r") as f:
            preset_data = json.load(f)
        return jsonify(preset_data)
    except FileNotFoundError:
        logging.warning(f"[PRESET] Preset file not found: presets/{safe_name}.json")
        return jsonify({"error": f"Preset '{safe_name}' not found"}), 404
    except json.JSONDecodeError:
        logging.error(f"[PRESET] Error decoding JSON for preset: presets/{safe_name}.json", exc_info=True)
        return jsonify({"error": f"Preset '{safe_name}' is not valid JSON"}), 500
    except Exception as e:
        logging.error(f"[PRESET] Error loading preset '{name}'", exc_info=True)
        return jsonify({"error": f"Could not load preset: {e}"}), 500

@app.route('/drift', methods=['POST'])
def drift():
    global baseline_mean # ระบุว่าเราจะใช้ตัวแปร global baseline_mean
    try:
        if baseline_mean is None:
            # ควรจะโหลด baseline_mean จากไฟล์ที่บันทึกไว้ตอน train model
            # หรือคำนวณเมื่อเริ่ม app ถ้ายังไม่ได้ทำ
            # ตัวอย่าง: baseline_mean = np.load('models/baseline_feature_mean.npy')
            # หรือถ้าไม่มีจริงๆ อาจจะต้อง return error หรือค่า default
            logging.error("[DRIFT] baseline_mean is not initialized.")
            return "Error: Baseline mean not initialized on server.", 500, {'Content-Type':'text/plain'}

        payload = request.get_json()
        if not payload or 'features' not in payload:
            return "Error: Missing 'features' in JSON payload.", 400, {'Content-Type':'text/plain'}

        feats_input = payload['features']
        
        # ตรวจสอบว่า feats_input เป็น list ของตัวเลขหรือไม่
        if not isinstance(feats_input, list) or not all(isinstance(x, (int, float)) for x in feats_input):
            return "Error: 'features' must be a list of numbers.", 400, {'Content-Type':'text/plain'}

        feats = np.array(feats_input, dtype=float)

        # ตรวจสอบว่าจำนวน features ตรงกับ baseline_mean หรือไม่
        if feats.shape[0] != baseline_mean.shape[0]:
            logging.warning(f"[DRIFT] Feature length mismatch. Input: {feats.shape[0]}, Baseline: {baseline_mean.shape[0]}")
            return f"Error: Feature length mismatch. Expected {baseline_mean.shape[0]} features.", 400, {'Content-Type':'text/plain'}
        
        # Handle division by zero if len(feats) is 0 (though previous checks should prevent this)
        if len(feats) == 0:
            return "Error: Features array cannot be empty.", 400, {'Content-Type':'text/plain'}

        drift_value = np.linalg.norm(feats - baseline_mean) / len(feats)
        logging.info(f"[DRIFT] Calculated drift: {drift_value}")
        return str(float(drift_value)), 200, {'Content-Type':'text/plain'}

    except Exception as e:
        logging.error(f"[DRIFT] Error calculating drift: {e}", exc_info=True)
        return f"Error: Could not calculate drift - {str(e)}", 500, {'Content-Type':'text/plain'}

@app.route('/rl/predict', methods=['POST'])
def rl_predict():
    global rl_model # ตรวจสอบว่า rl_model ถูกโหลดหรือสร้างแล้ว
    if rl_model is None:
        return jsonify(error="RL model not loaded"), 500
    try:
        payload = request.get_json()
        if not payload or "features" not in payload:
            return jsonify(error="Missing 'features' in payload"), 400
        
        data = payload["features"]
        # ตรวจสอบว่า data เป็น list และมีจำนวน feature_dim
        if not isinstance(data, list) or len(data) != feature_dim:
            return jsonify(error=f"Features must be a list of {feature_dim} numbers"), 400

        obs = np.array(data, dtype=np.float32).reshape(1, -1) # PPO คาดหวัง obs ที่มี batch dimension
        action, _ = rl_model.predict(obs, deterministic=True)
        return jsonify(action=int(action))
    except Exception as e:
        logging.error(f"[RL_PREDICT] Error: {e}", exc_info=True)
        return jsonify(error=str(e)), 500


@app.route('/rl/store', methods=['POST'])
def rl_store():
    global rl_buffer, buffer_lock
    try:
        payload = request.get_json()
        # ตรวจสอบ payload ที่จำเป็น: {"obs": [...], "action":0, "reward":0.5, "next_obs":[...], "done":False}
        required_keys = ["obs", "action", "reward", "next_obs", "done"]
        if not payload or not all(key in payload for key in required_keys):
            return jsonify(error=f"Missing one or more required keys: {required_keys}"), 400
        
        # อาจจะมีการ validate data types เพิ่มเติมที่นี่

        with buffer_lock:
            rl_buffer.append(payload)
        # logging.info(f"[RL_STORE] Stored transition. Buffer size: {len(rl_buffer)}") # Optional: log buffer size
        return jsonify(status="stored", buffer_size=len(rl_buffer))
    except Exception as e:
        logging.error(f"[RL_STORE] Error: {e}", exc_info=True)
        return jsonify(error=str(e)), 500

@app.route('/rl/update', methods=['POST'])
def rl_update():
    global rl_model, rl_buffer, buffer_lock, rl_model_path
    if rl_model is None:
        return jsonify(error="RL model not loaded"), 500

    # ควรมีการทำ RL update ใน background thread เพื่อไม่ให้ request ค้างนาน
    # แต่สำหรับตัวอย่างนี้ จะทำแบบ synchronous ก่อน

    transitions_to_learn = []
    with buffer_lock:
        if not rl_buffer:
            return jsonify(status="no_new_data", message="No new transitions in buffer to update model."), 200
        transitions_to_learn = rl_buffer.copy()
        rl_buffer.clear()
    
    logging.info(f"[RL_UPDATE] Starting RL model update with {len(transitions_to_learn)} transitions.")

    # --- ส่วนนี้คือจุดที่ซับซ้อน และอาจจะต้องปรับปรุงให้เหมาะสมกับ PPO ---
    # PPO เป็น on-policy algorithm ปกติจะเรียนรู้จาก rollouts ที่สร้างขึ้นใหม่โดย policy ปัจจุบัน
    # การเรียนรู้จาก buffer ของ (s, a, r, s', d) ที่เก็บไว้ (offline data) สำหรับ PPO ไม่ตรงไปตรงมาเท่า off-policy algorithms
    # `rl_model.learn(total_timesteps=X)` จะสั่งให้โมเดล run environment ของมันเอง X timesteps
    # ไม่ได้ใช้ข้อมูล obs, acts, rews จาก buffer โดยตรงใน PPO.learn() แบบนี้

    # **วิธีที่เป็นไปได้ (แต่ซับซ้อนกว่า):**
    # 1. สร้าง Custom Replay Buffer หรือ Dataset จาก `transitions_to_learn`.
    # 2. ปรับ PPO algorithm หรือใช้ algorithm ที่เหมาะกับ offline learning มากกว่า ถ้าต้องการเรียนจาก buffer แบบนี้จริงๆ
    # 3. หรือ `PIMMAEnv` จะต้องถูกออกแบบให้ `step()` สามารถรับข้อมูลจาก `transitions_to_learn` แทนการ simulate เอง
    #    (เช่น MQL ส่งข้อมูลมาให้ PIMMAEnv ใช้ใน step ถัดไป)

    # **สำหรับตอนนี้ โค้ดเดิมของคุณคือ `rl_model.learn(total_timesteps=len(data))`**
    # นี่จะหมายถึงการให้ PPO model ปัจจุบัน run `PIMMAEnv` เป็นจำนวน `len(transitions_to_learn)` steps
    # ซึ่ง `PIMMAEnv` ที่เราสร้างมี dummy `step` function.
    # ถ้าต้องการให้มัน "เรียนรู้" อะไรบางอย่างจริงๆ `PIMMAEnv` จะต้องมีความหมายมากกว่านี้
    # หรือคุณอาจจะหมายถึงการทำ pre-training ด้วย behavioral cloning หรือเทคนิคอื่นจาก buffer นี้ก่อน
    # แล้วค่อย fine-tune ด้วย PPO.learn() แบบ online.

    # เพื่อให้โค้ดทำงานตามที่คุณให้มา (แม้ว่าอาจจะต้องปรับปรุง logic การ train ของ RL ในอนาคต):
    try:
        # ตรวจสอบว่า env ของ rl_model ถูกตั้งค่าแล้ว
        if rl_model.get_env() is None:
            logging.warning("[RL_UPDATE] RL model environment not set. Setting a default one.")
            temp_env = PIMMAEnv(feature_dim)
            rl_model.set_env(DummyVecEnv([lambda: temp_env]))

        # จำนวน timesteps ที่จะ train ในครั้งนี้
        # การ train PPO ด้วยจำนวน timesteps น้อยๆ ซ้ำๆ อาจจะไม่ค่อยมีประสิทธิภาพเท่า train ด้วย batch ใหญ่
        # ค่านี้ควรสัมพันธ์กับ n_steps ที่ตั้งไว้ตอนสร้าง PPO model
        # เช่น ถ้า n_steps=128, total_timesteps ควรเป็น مضاعفات ของ 128
        # แต่ len(transitions_to_learn) คือจำนวน transitions ที่เก็บมา
        # นี่เป็นจุดที่การออกแบบการ train ของคุณเข้ามาเกี่ยวข้อง
        num_timesteps_to_learn = max(128, len(transitions_to_learn)) # อย่างน้อย train 1 rollout (n_steps) หรือมากกว่า
        
        logging.info(f"[RL_UPDATE] Calling rl_model.learn() with total_timesteps={num_timesteps_to_learn}")
        rl_model.learn(total_timesteps=num_timesteps_to_learn, reset_num_timesteps=False) # reset_num_timesteps=False เพื่อให้นับต่อ
        rl_model.save(rl_model_path)
        logging.info(f"[RL_UPDATE] RL model updated and saved to {rl_model_path}")
        return jsonify(status="rl_model_updated", transitions_processed=len(transitions_to_learn))
    except Exception as e:
        logging.error(f"[RL_UPDATE] Error during RL model update: {e}", exc_info=True)
        # คืน transitions กลับเข้า buffer ถ้าการ train ล้มเหลว (อาจจะซับซ้อน ควรพิจารณา)
        # with buffer_lock:
        #     rl_buffer.extend(transitions_to_learn) # Caution: could lead to repeated failures
        return jsonify(error=f"Failed to update RL model: {str(e)}"), 500

# --- Dashboard ---
@app.route("/dashboard")
def dashboard_route():
    # Basic dashboard, no license check needed for view generally
    return """
    <html>
    <head>
        <title>📊 AI Trading Dashboard</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; font-family: Arial, sans-serif; }
            .task-btn { margin-bottom: 10px; min-width: 250px; }
            .container { max-width: 800px; }
            h1, h2 { margin-bottom: 20px; }
            .list-group-item a { text-decoration: none; }
            .list-group-item button { width: 100%; text-align: left; }
        </style>
        <script>
        async function runTask(path, btnId, method = 'GET', body = null) {
            const btn = document.getElementById(btnId);
            const originalText = btn.innerText;
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            let message = '';
            try {
                const options = { method: method };
                if (method === 'POST' && body) {
                    options.headers = {'Content-Type': 'application/json'};
                    options.body = JSON.stringify(body);
                }
                const res = await fetch(path, options);
                const data = await res.json(); // Expect JSON response
                if (res.ok) {
                    message = `✅ Success: ${data.message || JSON.stringify(data)}`;
                } else {
                    message = `❌ Error ${res.status}: ${data.error || JSON.stringify(data)}`;
                }
            } catch(e) {
                message = `❌ Network/Script Error: ${e}`;
            }
            alert(message); // Simple feedback
            btn.disabled = false;
            btn.innerText = originalText;
        }

        function viewPage(path) { window.location.href = path; }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>📊 AI Trading Dashboard</h1>

            <h2>🚀 Actions</h2>
            <div class="list-group mb-4">
                <button id="retrain-btn" class="list-group-item list-group-item-action task-btn btn btn-primary"
                        onclick="runTask('/retrain','retrain-btn')">
                    🔁 Trigger Model Retrain
                </button>
                <button id="wft-btn" class="list-group-item list-group-item-action task-btn btn btn-info"
                        onclick="runTask('/wft','wft-btn')">
                    📈 Trigger Walking Forward Test (WFT)
                </button>
                <button id="activate-wft-btn" class="list-group-item list-group-item-action task-btn btn btn-success"
                        onclick="runTask('/activate_best_wft_model','activate-wft-btn')">
                    🌟 Activate Best Model from WFT
                </button>
            </div>

            <h2>📈 Status & Visualizations</h2>
            <div class="list-group mb-4">
                <button class="list-group-item list-group-item-action task-btn btn btn-light" onclick="viewPage('/retrain_status')">🧠 View Retrain Status</button>
                <button class="list-group-item list-group-item-action task-btn btn btn-light" onclick="viewPage('/summary')">💹 View Trade Summary (JSON)</button>
                <button class="list-group-item list-group-item-action task-btn btn btn-light" onclick="viewPage('/visualize_summary')">📊 View Profit Summary Chart</button>
                <button class="list-group-item list-group-item-action task-btn btn btn-light" onclick="viewPage('/wft_summary')">📈 View WFT Summary Chart</button>
                <button class="list-group-item list-group-item-action task-btn btn btn-light" onclick="runTask('/monitor','monitor-btn-silent')">🩺 Check API Monitor (Alert)</button>
                 <button class="list-group-item list-group-item-action task-btn btn btn-light" onclick="viewPage('/explain')">💡 View Model Feature Importance (JSON)</button>
            </div>
            
            <h2>📄 Logs & Reports</h2>
            <div class="list-group mb-4">
                <button class="list-group-item list-group-item-action task-btn btn btn-light" onclick="viewPage('/download_csv')">⬇ Download Trade Data CSV</button>
                <button class="list-group-item list-group-item-action task-btn btn btn-light" onclick="viewPage('/download_report')">📋 Download Latest Report</button>
                <button class="list-group-item list-group-item-action task-btn btn btn-warning" onclick="viewPage('/fail_log')">❌ View Execution Failure Log</button>
                <button class="list-group-item list-group-item-action task-btn btn btn-warning" onclick="viewPage('/exit_decision_log')">🚪 View Exit Decision Log</button>
                <button class="list-group-item list-group-item-action task-btn btn btn-warning" onclick="viewPage('/fail_entry_log')">🚫 View AI Entry Fail Log</button>
            </div>
        </div>
    </body></html>"""


# === Background Scheduler ===
# --- Scheduled Tasks ---
def scheduled_model_retrain_task():
    logging.info("[SCHEDULER] Initiating scheduled model retrain...")
    with app.app_context(): # Ensure Flask app context is available if needed by core logic
        success, message, samples, duration = _retrain_model_core()
    if success:
        logging.info(f"[SCHEDULER] ✅ Scheduled model retrain completed. Samples: {samples}, Duration: {duration}s. Message: {message}")
    else:
        logging.error(f"[SCHEDULER] ❌ Scheduled model retrain failed. Message: {message}")

def scheduled_report_generation_task():
    logging.info("[SCHEDULER] Initiating scheduled report generation...")
    with app.app_context():
        report_file = _create_exit_report_placeholder() # Using placeholder
    if report_file:
        logging.info(f"[SCHEDULER] ✅ Scheduled report generated: {report_file}")
    else:
        logging.error("[SCHEDULER] ❌ Scheduled report generation failed.")

# --- Scheduler Runner ---
def run_scheduler():
    # Define schedules
    # Example: Retrain every day at 1:00 AM server time
    schedule.every().day.at("01:00").do(scheduled_model_retrain_task)
    # Example: Generate report every day at 00:00 server time
    schedule.every().day.at("00:00").do(scheduled_report_generation_task)
    
    # Example: Run WFT weekly on Sunday at 03:00 AM
    # schedule.every().sunday.at("03:00").do(
    #    lambda: _run_walking_forward_test_core(TRADE_DATA_LOG_FILE, WFT_RESULTS_CSV) # Use lambda for args
    # )

    logging.info("[SCHEDULER] Scheduler started with defined tasks.")
    while True:
        schedule.run_pending()
        time.sleep(60) # Check every minute

# === Main Execution ===
if __name__ == '__main__':
    logging.info("[SERVER_INIT] Flask API service is starting...")
    print("[SERVER_INIT] Flask API service is starting...")

    # Start the background scheduler thread (ONLY ONCE)
    # Ensure this block does not run again if Flask's reloader is active (use_reloader=False handles this)
    if not (app.debug and os.environ.get('WERKZEUG_RUN_MAIN') == 'true'): # Check if not in reloader's subprocess
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logging.info("[SERVER_INIT] Background scheduler thread started.")
        print("[SERVER_INIT] Background scheduler thread started.")
    else:
        logging.info("[SERVER_INIT] Scheduler thread already started by main Werkzeug process or skipped in reloader.")
        print("[SERVER_INIT] Scheduler thread already started by main Werkzeug process or skipped in reloader.")

    # Log server start
    print(f"[SERVER_INIT] Flask API is attempting to run on http://127.0.0.1:5000")
    logging.info(f"[SERVER_INIT] Flask API initialized, attempting to run on http://127.0.0.1:5000")

    # Run Flask app
    # use_reloader=False is important if you manage threads yourself to avoid them starting twice
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)