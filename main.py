from flask import Flask, request, jsonify, send_file
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
from gym import Env
from gym import spaces
from stable_baselines3 import PPO

# Standard Library Imports first, then third-party, then local

# === Auto-install and Import Core Dependencies ===

# Requests
try:
    import requests
except ImportError:
    print("Requests library not found, attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests
    print("Requests library installed and imported.")

# Matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'Arial'  # Set default font
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
    import gym  # noqa
    import stable_baselines3  # noqa
    print("Gym, Stable-Baselines3, and Shimmy are already installed.")
except ImportError:
    print("stable-baselines3, gym, or shimmy not found, attempting to install all...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gym", "stable-baselines3", "shimmy~=2.0"])
    import gym  # noqa
    import stable_baselines3  # noqa
    print("gym, stable-baselines3, and shimmy installed and imported successfully.")

from stable_baselines3.common.vec_env import DummyVecEnv


# SHAP
try:
    import shap
    print("SHAP is already installed.")
except ImportError:
    print("SHAP library not found, attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    print("SHAP library installed and imported successfully.")

# === Global Configuration & Initialization ===
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("presets", exist_ok=True)
baseline_mean = None
explainer = None

logging.basicConfig(
    filename="logs/ai_service.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = Flask(__name__)

MODEL_PATH = "models/trained_model.xgb"
BEST_MODEL_PATH = "models/best_model.xgb"
TRADE_DATA_LOG_FILE = "logs/trade_data.csv"
LAST_RETRAIN_JSON = "logs/last_retrain.json"
WFT_RESULTS_CSV = "logs/walking_forward_results.csv"
EXECUTION_FAILURES_CSV = "logs/execution_failures.csv"
EXIT_DECISIONS_CSV = "logs/exit_decisions.csv"
AI_FAIL_ENTRY_CSV = "logs/ai_fail_entry.csv"

USE_FAKE_MODEL = os.environ.get("USE_FAKE_MODEL", "False").lower() == "true"
model = None
ENFORCE_LICENSE = False
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
        logging.info(
            f"[RETRAIN_STATUS_SAVE] Status saved: {status}, "
            f"Samples: {samples_used}, Duration: {duration_seconds}s"
        )
    except IOError as e:
        logging.error(f"[RETRAIN_STATUS_SAVE] Error saving retrain status to {file_path}: {e}", exc_info=True)


# === Reinforcement Learning (RL) Setup ===

def _initialize_shap_explainer():
    global model, explainer
    if model is not None and isinstance(model, xgb.XGBModel):
        try:
            explainer = shap.TreeExplainer(model)
            logging.info("[SHAP_INIT] SHAP TreeExplainer initialized successfully for the XGBoost model.")
        except Exception as e:
            logging.error(f"[SHAP_INIT] Failed to initialize SHAP TreeExplainer: {e}", exc_info=True)
            explainer = None
    elif model is not None:
        logging.warning(
            f"[SHAP_INIT] Model type ({type(model)}) is not directly XGBModel. "
            "SHAP TreeExplainer might not be optimal or work. "
            "Consider other SHAP explainers if needed."
        )
        explainer = None
    else:
        explainer = None
        logging.info("[SHAP_INIT] Main model not loaded. SHAP explainer not initialized.")


MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)


class PIMMAEnv(Env):
    def __init__(self, feature_dim_env):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(feature_dim_env,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.current_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._current_step = 0
        self._max_steps = 200  # ตัวอย่าง: กำหนดจำนวน step สูงสุดต่อ episode

    def reset(self):
        self.current_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._current_step = 0
        return self.current_obs

    def step(self, action):
        self._current_step += 1
        reward = np.random.rand() - 0.5
        next_obs = self.current_obs + np.random.normal(0, 0.1, self.current_obs.shape).astype(np.float32)
        self.current_obs = next_obs
        done = self._current_step >= self._max_steps
        info = {}
        return self.current_obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


feature_dim = 45
rl_model_path = os.path.join(MODEL_FOLDER, "rl_model.zip")
rl_model = None

if os.path.isfile(rl_model_path):
    try:
        rl_model = PPO.load(rl_model_path)
        if rl_model.get_env() is None:
            temp_env_load = PIMMAEnv(feature_dim)
            rl_model.set_env(DummyVecEnv([lambda: temp_env_load]))
        logging.info(f"[RL_SETUP] Loaded RL model from {rl_model_path}")
    except Exception as e:
        logging.error(f"[RL_SETUP] Error loading RL model: {e}. Creating new model.", exc_info=True) # Shortened
        temp_env_create = PIMMAEnv(feature_dim)
        rl_env_for_new_model_load_fail = DummyVecEnv([lambda: temp_env_create])
        rl_model = PPO("MlpPolicy", rl_env_for_new_model_load_fail, verbose=0, n_steps=128)
        rl_model.save(rl_model_path)
        logging.info(f"[RL_SETUP] Created and saved new RL model to {rl_model_path}")
else:
    temp_env_new = PIMMAEnv(feature_dim)
    rl_env_for_new_model_no_file = DummyVecEnv([lambda: temp_env_new])
    rl_model = PPO("MlpPolicy", rl_env_for_new_model_no_file, verbose=0, n_steps=128)
    rl_model.save(rl_model_path)
    logging.info(f"[RL_SETUP] New RL model created and saved to {rl_model_path}")

rl_buffer = []
buffer_lock = threading.Lock()


# === Model Loading ===
def load_model(path=MODEL_PATH):
    global model  # This is correct as 'model' is assigned to in this function
    if os.path.exists(path) and not USE_FAKE_MODEL:
        try:
            model = joblib.load(path)
            logging.info(f"[MODEL STATUS] ✅ Real model loaded successfully from {path}.")
        except Exception as e:
            logging.error(f"[MODEL STATUS] ❌ Failed to load real model from {path}: {e}", exc_info=True)
            model = None
    elif USE_FAKE_MODEL:
        logging.warning("[MODEL STATUS] ⚠️ Using FAKE model as per configuration.")
        model = None
    else:
        logging.warning(f"[MODEL STATUS] ⚠️ Model file not found at {path} and not using FAKE model.")
        model = None
    _initialize_shap_explainer()
    return model is not None or USE_FAKE_MODEL


# The F824 error for 'global model' at approx line 134 was for a redundant global declaration
# at the module level *before* this call. It has been removed.
load_model()  # Initial attempt to load the model


# === Core AI Logic ===
def predict_entry_logic(features):
    """Core logic for making a prediction using the loaded model or a fake one."""
    if USE_FAKE_MODEL or model is None:
        logging.info("[PREDICT_LOGIC] Using fake model or model not loaded. Returning random prediction.")
        return round(np.random.uniform(0, 1), 4)
    try:
        if not isinstance(features, list) or not all(isinstance(f, (int, float)) for f in features):
            logging.error(f"[PREDICT_LOGIC] Invalid features format: {features}. Expected list of numbers.")
            return round(np.random.uniform(0, 1), 4)
        prediction = model.predict(np.array([features]))[0]
        return float(prediction)
    except Exception as e:
        logging.error(f"[PREDICT_LOGIC] Error during prediction: {e}", exc_info=True)
        return round(np.random.uniform(0, 1), 4)


def _retrain_model_core(data_path=TRADE_DATA_LOG_FILE, output_model_path=MODEL_PATH):
    """Core logic for retraining the model."""
    start_time = time.time()
    try:
        if not os.path.exists(data_path):
            _save_retrain_status("failed_no_file", 0, 0)
            logging.error(f"[RETRAIN_CORE] Data file not found at {data_path}")
            return False, "Data file not found", 0, 0

        df = pd.read_csv(data_path)
        df = df[df['command'].str.upper() == "PREDICT_ENTRY"]
        df = df[df['features'].notna() & df['result'].notna() & (df['result'] != "")]

        try:
            df['result'] = df['result'].astype(float)
        except ValueError as e_convert:
            duration_err = round(time.time() - start_time, 2)
            _save_retrain_status("failed_invalid_result_data", 0, duration_err)
            logging.error(
                f"[RETRAIN_CORE] Error converting 'result' to float: {e_convert}. Check data in {data_path}.",
                exc_info=True
            )
            return False, "Invalid data in 'result' column", 0, duration_err

        processed_data = []
        for _idx, row in df.iterrows():
            features = parse_feature_string(row['features'])
            if features:
                processed_data.append({'X': features, 'y': row['result']})

        if not processed_data:
            duration_err = round(time.time() - start_time, 2)
            _save_retrain_status("failed_no_valid_features", 0, duration_err)
            logging.error("[RETRAIN_CORE] No valid features data after parsing to train the model.")
            return False, "No valid features to train on", 0, duration_err

        feature_length = len(processed_data[0]['X'])  # Safe due to check above
        X_final = []
        y_final = []
        for item in processed_data:
            if len(item['X']) == feature_length:
                X_final.append(item['X'])
                y_final.append(item['y'])
            else:
                logging.warning(
                    f"[RETRAIN_CORE] Inconsistent feature length. Expected {feature_length}, "
                    f"got {len(item['X'])}. Skipping."
                )

        if not X_final:
            duration_err = round(time.time() - start_time, 2)
            _save_retrain_status("failed_no_consistent_features", 0, duration_err)
            logging.error("[RETRAIN_CORE] No features with consistent length to train.")  # Shortened log
            return False, "No features with consistent length", 0, duration_err

        model_new = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model_new.fit(np.array(X_final), np.array(y_final))
        joblib.dump(model_new, output_model_path)

        duration = round(time.time() - start_time, 2)
        _save_retrain_status("success", len(X_final), duration)
        logging.info(
            f"[RETRAIN_CORE] Model retrained. Samples: {len(X_final)}, "
            f"Duration: {duration}s. Saved: {output_model_path}"  # Shortened log
        )

        if output_model_path == MODEL_PATH:
            global model
            model = model_new
            logging.info("[RETRAIN_CORE] Global model updated with newly retrained model.")
            _initialize_shap_explainer()

        return True, "Retrain completed", len(X_final), duration

    except Exception as e_main:
        current_duration_on_error = round(time.time() - start_time, 2) if 'start_time' in locals() else 0
        _save_retrain_status("failed_exception", 0, current_duration_on_error)
        logging.error(f"[RETRAIN_CORE] Exception during model retraining: {e_main}", exc_info=True)
        return False, str(e_main), 0, current_duration_on_error


# === Data Logging Functions ===
def log_trade_data(command, features_str, result=None, ticket=None, file_path=TRADE_DATA_LOG_FILE):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "command": str(command),
        "features": str(features_str),
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
    if not ENFORCE_LICENSE:
        return True
    license_key = http_request.headers.get("X-License-Key", "")
    return license_key in ALLOWED_LICENSE_KEYS


# === Flask Routes ===
@app.route("/download_csv", methods=["GET"])
def download_csv():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    try:
        return send_file(TRADE_DATA_LOG_FILE, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"[DOWNLOAD_CSV] File not found: {TRADE_DATA_LOG_FILE}", exc_info=True)
        return jsonify({"error": "Trade data log file not found."}), 404
    except Exception as e:
        logging.error(f"[DOWNLOAD_CSV] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    try:
        payload = request.get_json()
        if not payload or 'data' not in payload:
            return jsonify({"error": "Missing 'data' in payload"}), 400

        features_str = payload.get('data', "")
        features_list = parse_feature_string(features_str)

        if not features_list:
            logging.warning(f"[PREDICT_ROUTE] Received empty/invalid features: {features_str}")
            return jsonify({"error": "Invalid or empty features string provided"}), 400

        logging.info(f"[PREDICT_ROUTE] Features (sample): {features_list[:5]}… total={len(features_list)}")
        ai_score = predict_entry_logic(features_list)
        logging.info(f"[PREDICT_ROUTE] AI Score: {ai_score:.4f}")
        log_trade_data("PREDICT_ENTRY", features_str, ai_score)
        return jsonify(ai_score), 200
    except Exception as e:
        logging.error(f"[PREDICT_ROUTE] Exception occurred: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/optimize', methods=['POST'])
def optimize_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
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
                optimized[k] = round(new_val, 3)
            except ValueError:
                logging.warning(f"[OPTIMIZER] Could not convert param {k} value '{v_str}' to float. Skipping.")
                optimized[k] = v_str
        logging.info(f"[OPTIMIZER] Optimized result: {optimized}")
        return jsonify(optimized), 200
    except json.JSONDecodeError as e:
        logging.error("[OPTIMIZER] JSON decode error", exc_info=True)
        raw_sample = raw_data[:100].decode('utf-8', errors='replace')
        return jsonify({"error": f"Invalid JSON: {e}", "raw_data_sample": raw_sample}), 400
    except Exception as e:
        logging.error(f"[OPTIMIZER] Error occurred: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/monitor', methods=['GET'])
def monitor_route():
    try:
        model_status_str = "NOT_LOADED"
        if USE_FAKE_MODEL:
            model_status_str = "FAKE"
        elif model is not None:
            model_status_str = "REAL"
        info = {
            "model_status": model_status_str,
            "model_path_configured": MODEL_PATH,
            "model_file_exists": os.path.exists(MODEL_PATH),
            "latency_ms": int(np.random.randint(20, 80)),
            "enforce_license": ENFORCE_LICENSE,
            "timestamp": datetime.now().isoformat()
        }
        logging.info("[MONITOR] Health check requested.")
        return jsonify(info), 200
    except Exception as e:
        logging.error(f"[MONITOR] Exception: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/retrain_status", methods=["GET"])
def retrain_status_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
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
        return (
            "<p style='color:orange;'>⚠️ Retrain status file not found. Run a retrain cycle first.</p>"
            "<p><a href='/dashboard'>Back to Dashboard</a></p>"
        ), 404
    except Exception as e:
        logging.error(f"[RETRAIN_STATUS] Error: {e}", exc_info=True)
        return f"<p style='color:red;'>❌ Error loading retrain status: {e}</p>", 500


@app.route('/explain', methods=['POST'])
def explain_shap_route():
    if explainer is None:
        return jsonify(
            {"error": "SHAP explainer not initialized. Model might not be loaded or compatible."}
        ), 503
    try:
        payload = request.get_json()
        if not payload or "features" not in payload:
            return jsonify({"error": "Missing 'features' in payload"}), 400
        feats_input = payload['features']
        if not isinstance(feats_input, list):
            return jsonify({"error": "'features' must be a list of numbers"}), 400
        try:
            feats = np.array(feats_input, dtype=np.float32).reshape(1, -1)
        except ValueError as ve:
            return jsonify({"error": f"Invalid feature format or type: {ve}"}), 400
        if hasattr(model, 'n_features_in_') and feats.shape[1] != model.n_features_in_:
            err_msg = (
                f"Feature mismatch. Expected {model.n_features_in_} features, "
                f"got {feats.shape[1]}."
            )
            return jsonify({"error": err_msg}), 400
        shap_values_for_instance = explainer.shap_values(feats)
        if isinstance(shap_values_for_instance, list):
            shap_values_to_send = shap_values_for_instance[0].tolist()
            base_val = explainer.expected_value[0] \
                if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        elif isinstance(shap_values_for_instance, np.ndarray) and shap_values_for_instance.ndim == 2:
            shap_values_to_send = shap_values_for_instance[0].tolist()
            base_val = explainer.expected_value
        else:
            logging.error(f"[EXPLAIN_SHAP] Unexpected shap_values format: {type(shap_values_for_instance)}")
            return jsonify({"error": "Unexpected SHAP values format from explainer."}), 500
        return jsonify({"shap_values": shap_values_to_send, "base_value": float(base_val)})
    except Exception as e:
        logging.error(f"[EXPLAIN_SHAP] Error calculating SHAP values: {e}", exc_info=True)
        return jsonify({"error": f"Could not calculate SHAP values: {str(e)}"}), 500


@app.route('/summary', methods=['GET'])
def summary_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    try:
        if not os.path.exists(TRADE_DATA_LOG_FILE):
            return jsonify({"error": f"{TRADE_DATA_LOG_FILE} not found. No data to summarize."}), 404
        df = pd.read_csv(TRADE_DATA_LOG_FILE)
        df = df[df['result'].notna() & (df['result'] != "")]
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df.dropna(subset=['result'], inplace=True)
        if df.empty:
            return jsonify({"message": "No valid trade results found to summarize."}), 200
        total_trades = len(df)
        avg_result = float(round(df['result'].mean(), 4)) if not df['result'].empty else 0.0
        std_dev_result = float(round(df['result'].std(), 4)) if not df['result'].empty else 0.0
        max_win = float(round(df['result'].max(), 4)) if not df['result'].empty else 0.0
        max_loss = float(round(df['result'].min(), 4)) if not df['result'].empty else 0.0
        winrate = float(round((df['result'] > 0).sum() / total_trades * 100, 2)) if total_trades > 0 else 0.0
        stats = {
            "total_trades": total_trades, "average_result": avg_result,
            "std_deviation_result": std_dev_result, "max_win": max_win,
            "max_loss": max_loss, "winrate_percent": winrate,
            "last_updated": datetime.now().isoformat(), "data_source": TRADE_DATA_LOG_FILE
        }
        logging.info("[SUMMARY] Generated summary stats.")
        return jsonify(stats), 200
    except Exception as e:
        logging.error(f"[SUMMARY] Error generating summary: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _generate_plot_base64(plot_function, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_function(ax, *args, **kwargs)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img64


def _plot_cumulative_pnl(ax, pnl_series):
    ax.plot(np.cumsum(pnl_series), label="Cumulative Result", color='green')
    ax.set_title("AI Result Summary (Cumulative)")
    ax.set_xlabel("Number of Trades/Events")
    ax.set_ylabel("Cumulative Result Value")
    ax.grid(True)
    ax.legend()


@app.route('/visualize_summary')
def visualize_summary_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    err_html = """
        <html><head><title>AI Profit Summary</title></head><body>
            <h1>📊 AI Profit Summary</h1>
            <p style="color:{color};font-weight:bold;">{message}</p>
            <p><a href="/download_csv"><button>⬇ Download trade_data.csv</button></a>
            <a href="/dashboard"><button>🏠 Back to Dashboard</button></a></p>
        </body></html>"""
    try:
        if not os.path.exists(TRADE_DATA_LOG_FILE):
            return err_html.format(color="orange", message=f"⚠️ Data file '{TRADE_DATA_LOG_FILE}' not found."), 404
        df = pd.read_csv(TRADE_DATA_LOG_FILE)
        if "result" not in df.columns:
            return err_html.format(color="red", message="❌ 'result' column not found in data."), 200
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        pnl_series = df["result"].dropna()
        if len(pnl_series) < 2:
            msg = "⚠️ Not enough valid result data (at least 2 points required) to generate a chart."
            return err_html.format(color="orange", message=msg), 200
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
        logging.error(f"[VISUALIZE_SUMMARY] Error generating chart: {e}", exc_info=True)
        return err_html.format(color="red", message=f"❌ An error occurred: {e}"), 500


# --- Placeholder/Simple Data Routes ---
@app.route('/trends')
def trends_route():
    kw = request.args.get('kw', 'general_topic')
    return jsonify(keyword=kw, trend_value=round(np.random.uniform(30, 90), 1), source="Placeholder Trends API"), 200


@app.route('/social')
def social_route():
    src = request.args.get('src', 'twitter')
    return jsonify(
        source_platform=src,
        sentiment_score=round(np.random.uniform(-0.8, 0.8), 2),
        analysis_type="Placeholder Sentiment"
    ), 200


@app.route('/macro')
def macro_route():
    name = request.args.get('name', 'PMI')
    return jsonify(
        indicator_name=name, value=round(np.random.uniform(45, 55), 1), source="Placeholder Macro Data API"
    ), 200


@app.route('/orderbook')
def orderbook_route():
    symbol = request.args.get('symbol', 'BTCUSD')
    return jsonify(
        trading_symbol=symbol, imbalance_ratio=round(np.random.uniform(-0.5, 0.5), 3),
        source="Placeholder Orderbook API"
    ), 200


@app.route('/onchain')
def onchain_route():
    metric = request.args.get('metric', 'active_addresses')
    sym = request.args.get('symbol', 'ETH')
    return jsonify(
        crypto_symbol=sym, onchain_metric=metric, value=int(np.random.randint(1000, 100000)),
        source="Placeholder On-chain API"
    ), 200


@app.route("/cot", methods=["GET"])
def cot_route():
    logging.info("[COT] COT Data requested")
    data = {"asset": "Gold", "net_long_commercials": round(np.random.uniform(50, 85), 2), "source": "Placeholder COT"}
    return jsonify(data), 200


@app.route("/openinterest", methods=["GET"])
def openinterest_route():
    logging.info("[OI] Open Interest requested")
    oi = int(np.random.randint(50000, 200000))
    return jsonify({"symbol": "XAUUSD_Futures", "open_interest": oi, "source": "Placeholder OI"}), 200


@app.route("/news", methods=["GET"])
def news_route():
    logging.info("[NEWS] News requested")
    news_data = {
        "event_name": "FOMC Meeting Minutes",
        "impact_level": "high" if np.random.rand() > 0.5 else "medium",
        "affected_pairs": ["USDJPY", "EURUSD", "XAUUSD"],
        "scheduled_time": (datetime.now() + pd.Timedelta(hours=np.random.randint(1, 5))).isoformat(),
        "source": "Placeholder News API"
    }
    return jsonify(news_data), 200


@app.route("/correlation", methods=["GET"])
def correlation_route():
    logging.info("[CORRELATION] Correlation data requested")
    corr_data = {
        "target_asset": "XAUUSD",
        "correlations": {
            "DXY": round(np.random.uniform(-0.9, -0.2), 2),
            "US10Y_BondYield": round(np.random.uniform(-0.5, 0.5), 2),
            "OIL": round(np.random.uniform(-0.3, 0.3), 2)
        },
        "source": "Placeholder Correlation Engine"
    }
    return jsonify(corr_data), 200


@app.route("/vwap", methods=["GET"])
def vwap_route():
    symbol = request.args.get('symbol', 'XAUUSD')
    logging.info(f"[VWAP] VWAP requested for {symbol}")
    data = {
        "symbol": symbol, "vwap": round(np.random.uniform(1900, 2100), 2),
        "timeframe": "1H", "source": "Placeholder VWAP"
    }
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
        "timeframe": "Daily", "source": "Placeholder Volume Profile"
    }
    return jsonify(profile), 200


@app.route("/harmonics", methods=["GET"])
def harmonics_route():
    symbol = request.args.get('symbol', 'EURUSD')
    logging.info(f"[Harmonics] Harmonic Pattern scan requested for {symbol}")
    patterns = ["Gartley", "Bat", "Butterfly", "Crab", "Shark"]
    data = {
        "symbol": symbol, "pattern_detected": np.random.choice(patterns),
        "status": "developing" if np.random.rand() > 0.3 else "valid_entry_zone",
        "entry_zone_start": round(np.random.uniform(1.0500, 1.0600), 4),
        "entry_zone_end": round(np.random.uniform(1.0601, 1.0700), 4),
        "timeframe": "H4", "source": "Placeholder Harmonics Scanner"
    }
    return jsonify(data), 200


# === Retraining and Model Management Routes ===
@app.route('/retrain', methods=['GET'])
def retrain_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    logging.info("[RETRAIN_ROUTE] Manual retrain triggered.")

    def _run_and_log():
        with app.app_context():
            success, message, _, _ = _retrain_model_core()
            if success:
                logging.info(f"[RETRAIN_ROUTE_THREAD] Retrain successful via route: {message}")
            else:
                logging.error(f"[RETRAIN_ROUTE_THREAD] Retrain failed via route: {message}")
    thread = threading.Thread(target=_run_and_log)
    thread.start()
    return jsonify({
        "status": "triggered",
        "message": "Retrain process initiated. Check /retrain_status for updates."
    }), 202


@app.route('/log_result', methods=['POST'])
def log_result_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    try:
        payload = request.get_json()
        ticket = payload.get("ticket")
        result_val = payload.get("result")
        if ticket is None or result_val is None:
            return jsonify({"error": "Missing 'ticket' or 'result' in payload"}), 400
        try:
            ticket = int(ticket)
            result_val = float(result_val)
        except ValueError:
            return jsonify({"error": "Ticket must be an integer and result must be a float."}), 400
        if not os.path.exists(TRADE_DATA_LOG_FILE):
            log_trade_data("RESULT_LOGGED_NO_PRIOR_PREDICT", f"Ticket: {ticket}", result_val, ticket)
            return jsonify(
                {"status": "logged_as_new", "message": "No prior predict entry to update, logged as new result."}
            ), 201
        df = pd.read_csv(TRADE_DATA_LOG_FILE)
        updated = False
        for idx in range(len(df) - 1, -1, -1):
            if 'command' in df.columns and df.at[idx, "command"].upper() == "PREDICT_ENTRY":
                current_ticket = df.at[idx, "ticket"] if 'ticket' in df.columns else np.nan
                if pd.isna(current_ticket) or str(current_ticket).strip() == "":
                    df.loc[idx, "ticket"] = ticket
                    df.loc[idx, "result"] = result_val
                    updated = True
                    break
        if updated:
            df.to_csv(TRADE_DATA_LOG_FILE, index=False)
            logging.info(f"[LOG_RESULT] Updated trade log for ticket {ticket} with result {result_val}.")
            return jsonify({"status": "success", "message": f"Result for ticket {ticket} logged."}), 200
        else:
            log_trade_data("RESULT_LOGGED_ORPHANED", f"Ticket: {ticket}", result_val, ticket)
            logging.warning(
                f"[LOG_RESULT] No matching PREDICT_ENTRY for ticket {ticket}. Logged as orphaned."
            )
            return jsonify(
                {"status": "not_found_or_already_logged",
                 "message": "No unlogged PREDICT_ENTRY to associate. Logged as new/orphaned."}
            ), 202
    except Exception as e:
        logging.error(f"[LOG_RESULT] Exception occurred: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/log_execution_failure", methods=["POST"])
def log_execution_failure_route():
    raw_data = request.get_data()
    try:
        cleaned_raw_data = raw_data.rstrip(b'\x00')
        json_str = cleaned_raw_data.decode('utf-8')
        payload = json.loads(json_str)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        logging.error(
            f"[LOG_EXEC_FAIL] JSON parse error: {e}. Raw (100B): {raw_data[:100]!r}", exc_info=True
        )
        log_entry = {
            "timestamp": datetime.now().isoformat(), "error_type": "PayloadParseError",
            "details": str(e), "raw_payload_sample": raw_data[:200].decode('ascii', errors='replace')
        }
        log_generic_csv(EXECUTION_FAILURES_CSV, log_entry)
        return jsonify({"error": f"Invalid JSON or encoding: {e}", "raw_bytes_sample": list(raw_data[:64])}), 400
    ts = datetime.now().isoformat()
    log_data = {"timestamp": ts}
    log_file_target = None
    if "reason" in payload and "ticket" in payload:
        log_file_target = EXECUTION_FAILURES_CSV
        log_data["ticket"] = payload.get("ticket", "")
        log_data["reason"] = payload.get("reason", "Unknown reason")
        log_data["details"] = payload.get("details", "")
        logging.info(f"[LOG_EXEC_FAIL] Logged exec failure for ticket {log_data['ticket']}: {log_data['reason']}")
    elif "score" in payload and "method" in payload and "ticket" in payload:
        log_file_target = EXIT_DECISIONS_CSV
        log_data["ticket"] = payload.get("ticket", "")
        log_data["score"] = payload.get("score", 0.0)
        log_data["method"] = payload.get("method", "Unknown method")
        logging.info(
            f"[LOG_EXIT_DECISION] Logged exit decision for ticket {log_data['ticket']}: "
            f"Score {log_data['score']}, Method {log_data['method']}"
        )
    else:
        logging.warning(f"[LOG_EXEC_FAIL] Unknown payload structure: {payload}")
        log_generic_csv("logs/malformed_failure_logs.csv", {"timestamp": ts, "payload": json.dumps(payload)})
        err_msg = "Unknown payload. Required: (ticket, reason) or (ticket, score, method)."
        return jsonify({"error": err_msg, "received_payload": payload}), 400
    log_generic_csv(log_file_target, log_data)
    return jsonify({"status": "ok", "message": "Log received."}), 200


@app.route("/log_exit_decision", methods=["POST"])
def log_exit_decision_route_specific():
    raw_bytes = request.get_data()
    logging.info(f"[LOG_EXIT_DECISION_SPECIFIC] Raw bytes: {raw_bytes!r}")
    s = ""
    try:
        s = raw_bytes.rstrip(b'\x00').decode('utf-8')
        payload = json.loads(s)
    except Exception as e:
        logging.error(f"[LOG_EXIT_DECISION_SPECIFIC] JSON parse error: {e}, raw: {s!r}", exc_info=True)
        return jsonify({"error": f"Invalid JSON: {e}", "raw_string_content": s}), 400
    if not all(k in payload for k in ["ticket", "score", "method"]):
        return jsonify({"error": "Missing fields: ticket, score, method", "payload": payload}), 400
    log_data = {
        "timestamp": datetime.now().isoformat(), "ticket": payload.get("ticket", ""),
        "score": payload.get("score", 0.0), "method": payload.get("method", "")
    }
    log_generic_csv(EXIT_DECISIONS_CSV, log_data)
    logging.info(f"[LOG_EXIT_DECISION_SPECIFIC] Logged: {log_data}")
    return jsonify({"status": "ok", "message": "Exit decision logged."}), 200


@app.route("/log_fail_entry", methods=["POST"])
def log_fail_entry_route():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Empty payload"}), 400
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "score": payload.get("score", 0.0),
            "features": str(payload.get("features", "")),
            "reason": payload.get("reason", "Unknown reason")
        }
        log_generic_csv(AI_FAIL_ENTRY_CSV, log_data)
        logging.info(f"[FAIL_ENTRY_LOG] Logged AI entry failure: {log_data['reason']}")
        return jsonify({"status": "logged"}), 200
    except Exception as e:
        logging.error(f"[FAIL_ENTRY_LOG] Error logging failed entry: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# --- HTML View Routes for Logs ---
def _generate_html_table_page(csv_path, page_title, error_message_not_found):
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(error_message_not_found)
        df = pd.read_csv(csv_path)
        html_table = df.to_html(classes="table table-striped table-hover", index=False, border=0, escape=True)
        return f"""
        <html><head><title>{page_title}</title>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
            <style> body {{ padding: 20px; font-family: Arial, sans-serif; }} h1 {{ margin-bottom: 20px; }} </style>
        </head><body><h1>{page_title}</h1>
            {html_table if not df.empty else '<p>No data logged yet.</p>'}
            <br><p><a href="/dashboard" class="btn btn-primary">⬅ Back to Dashboard</a></p>
        </body></html>"""
    except FileNotFoundError:
        logging.warning(f"[HTML_TABLE_VIEW] File not found: {csv_path}")
        return f"<p style='color:orange;'>⚠️ {error_message_not_found}</p><p><a href='/dashboard'>Back</a></p>", 404
    except Exception as e:
        logging.error(f"[HTML_TABLE_VIEW] Error loading log for {csv_path}: {e}", exc_info=True)
        return f"<p style='color:red;'>❌ Error loading log: {e}</p><p><a href='/dashboard'>Back</a></p>", 500


@app.route("/fail_log", methods=["GET"])
def fail_log_view_route():
    return _generate_html_table_page(EXECUTION_FAILURES_CSV, "❌ Execution Failure Log", "Exec failure log not found.")


@app.route("/exit_decision_log", methods=["GET"])
def exit_decision_log_view_route():
    return _generate_html_table_page(EXIT_DECISIONS_CSV, "🚪 Exit Decision Log", "Exit decision log not found.")


@app.route("/fail_entry_log", methods=["GET"])
def fail_entry_log_view_route():
    return _generate_html_table_page(AI_FAIL_ENTRY_CSV, "🚫 AI Entry Fail Log", f"{AI_FAIL_ENTRY_CSV} not found.")


# --- Walking Forward Test (WFT) ---
def _run_walking_forward_test_core(
    data_path=TRADE_DATA_LOG_FILE, output_path=WFT_RESULTS_CSV, train_window=300, test_window=50
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
        parsed_data = []
        for _idx, row in df.iterrows():
            features = parse_feature_string(row['features'])
            timestamp = pd.to_datetime(row['timestamp'], errors='coerce')
            if features and pd.notna(timestamp):
                parsed_data.append({'X': features, 'y': row['result'], 'timestamp': timestamp})
        if not parsed_data:
            logging.error("[WFT_CORE] No valid feature/result pairs after parsing for WFT.")
            return {"status": "error", "message": "No valid feature/result data for WFT."}
        expected_len = len(parsed_data[0]['X'])
        df_wft_data = pd.DataFrame([d for d in parsed_data if len(d['X']) == expected_len])
        if len(df_wft_data) < train_window + test_window:
            msg = (f"Not enough consistent data for WFT (req: {train_window + test_window}, "
                   f"found: {len(df_wft_data)})")
            logging.error(f"[WFT_CORE] {msg}")
            return {"status": "error", "message": msg}
        df_wft_data = df_wft_data.sort_values("timestamp").reset_index(drop=True)
        wft_results_list = []
        total_rounds = (len(df_wft_data) - train_window) // test_window
        if total_rounds <= 0:
            msg = (f"Not enough data for one WFT round with train_window={train_window}, test_window={test_window}.")
            logging.error(f"[WFT_CORE] {msg}")
            return {"status": "error", "message": msg}
        for i in range(total_rounds):
            train_start_idx, train_end_idx = i * test_window, i * test_window + train_window
            test_start_idx, test_end_idx = train_end_idx, train_end_idx + test_window
            if test_end_idx > len(df_wft_data):  # Corrected E701
                break
            train_set = df_wft_data.iloc[train_start_idx:train_end_idx]
            test_set = df_wft_data.iloc[test_start_idx:test_end_idx]
            if train_set.empty or test_set.empty:
                logging.warning(f"[WFT_CORE] Skipping round {i+1} due to empty train/test set.")
                continue
            X_train, y_train = np.array(train_set['X'].tolist()), np.array(train_set['y'].tolist())
            X_test = np.array(test_set['X'].tolist())
            wft_model_obj = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=i)
            wft_model_obj.fit(X_train, y_train)
            preds = wft_model_obj.predict(X_test)
            avg_pred, std_pred = np.mean(preds), np.std(preds)
            win_pred = np.mean(preds > 0) * 100 if preds.size > 0 else 0
            mdd_pred = np.min(np.cumsum(preds)) if preds.size > 0 else 0
            res_data = {
                "round": i + 1, "train_start_ts": train_set['timestamp'].iloc[0].isoformat(),
                "test_start_ts": test_set['timestamp'].iloc[0].isoformat(),
                "avg_predicted_metric": round(avg_pred, 4), "std_predicted_metric": round(std_pred, 4),
                "winrate_on_predicted_metric_percent": round(win_pred, 2),
                "max_drawdown_on_predicted_metric": round(mdd_pred, 4),
                "samples_train": len(X_train), "samples_test": len(X_test)
            }
            wft_results_list.append(res_data)
            logging.info(f"[WFT_CORE] R {i+1}/{total_rounds}: AvgPred={avg_pred:.2f}, WinRatePred={win_pred:.1f}%") # Shortened
        if not wft_results_list:
            logging.warning("[WFT_CORE] No WFT rounds were completed.")
            return {"status": "warning", "message": "WFT completed but no rounds generated results."}
        results_df = pd.DataFrame(wft_results_list)
        results_df.to_csv(output_path, index=False)
        logging.info(f"[WFT_CORE] ✅ WFT finished. Results: {output_path}")
        return {"status": "success", "message": f"WFT finished. Results: {output_path}", "rounds": len(wft_results_list)}
    except Exception as e:
        logging.error(f"[WFT_CORE] ❌ Error in WFT: {e}", exc_info=True)
        return {"status": "error", "message": f"WFT failed: {str(e)}"}


@app.route("/wft", methods=["GET"])
def wft_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    logging.info("[WFT_ROUTE] WFT initiated via route.")
    train_win = request.args.get('train_window', 300, type=int)
    test_win = request.args.get('test_window', 50, type=int)
    thread = threading.Thread(
        target=_run_walking_forward_test_core,
        args=(TRADE_DATA_LOG_FILE, WFT_RESULTS_CSV, train_win, test_win)
    )
    thread.daemon = True
    thread.start()
    msg = (f"WFT initiated (train={train_win}, test={test_win}). Check logs/WFT summary.")
    return jsonify({"status": "triggered", "message": msg}), 202


def _plot_wft_summary(ax, df_wft):
    ax.plot(df_wft['round'], df_wft['avg_predicted_metric'], label='Avg Predicted Metric', marker='o')
    ax.plot(df_wft['round'], df_wft['winrate_on_predicted_metric_percent'],
            label='Winrate on Predicted Metric (%)', marker='x')
    ax.set_title("📊 Walking Forward Test (WFT) Summary")
    ax.set_xlabel("Test Round Number")
    ax.set_ylabel("Performance Metric Value")
    ax.grid(True)
    ax.legend()


@app.route("/wft_summary")
def wft_summary_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    err_html = """
        <html><head><title>WFT Summary</title></head><body><h1>📈 WFT Summary</h1>
            <p style="color:{color};font-weight:bold;">{message}</p>
            <p><a href="/wft"><button>🔁 Run WFT Now</button></a>
            <a href="/dashboard"><button>🏠 Back to Dashboard</button></a></p>
        </body></html>"""
    try:
        if not os.path.exists(WFT_RESULTS_CSV):
            msg = f"⚠️ WFT results file ('{WFT_RESULTS_CSV}') not found. Please run WFT first."
            return err_html.format(color="orange", message=msg), 404
        df_wft = pd.read_csv(WFT_RESULTS_CSV)
        req_cols = ['round', 'avg_predicted_metric', 'winrate_on_predicted_metric_percent']
        if df_wft.empty or not all(col in df_wft.columns for col in req_cols):
            return err_html.format(color="red", message="❌ WFT results file empty/incorrect columns."), 200
        for col in req_cols:
            df_wft[col] = pd.to_numeric(df_wft[col], errors='coerce')
        df_wft.dropna(subset=req_cols, inplace=True)
        if len(df_wft) < 1:
            return err_html.format(color="orange", message="⚠️ Not enough valid WFT data for chart."), 200
        img64 = _generate_plot_base64(_plot_wft_summary, df_wft)
        return f"""
        <html><head><title>WFT Summary</title></head><body><h1>📈 WFT Summary</h1>
            <img src="data:image/png;base64,{img64}" alt="WFT Summary Chart"/><br/><br/>
            <p><a href="/wft"><button>🔁 Run WFT Again</button></a>
            <a href="/download_csv"><button>⬇ trade_data.csv</button></a>
            <a href="/dashboard"><button>🏠 Back to Dashboard</button></a></p>
        </body></html>""", 200
    except Exception as e:
        logging.error(f"[WFT_SUMMARY_ROUTE] Error generating WFT summary chart: {e}", exc_info=True)
        return err_html.format(color="red", message=f"❌ An error occurred: {e}"), 500


def _train_model_from_best_wft_round(
    wft_results_path=WFT_RESULTS_CSV, trade_data_path=TRADE_DATA_LOG_FILE,
    output_model_path=BEST_MODEL_PATH, optimize_metric='winrate_on_predicted_metric_percent'
):
    try:
        if not os.path.exists(wft_results_path):
            return False, f"WFT results file not found: {wft_results_path}"
        df_wft = pd.read_csv(wft_results_path)
        if df_wft.empty or optimize_metric not in df_wft.columns:
            return False, f"WFT results empty or missing optimizing metric '{optimize_metric}'."
        best_row = df_wft.loc[df_wft[optimize_metric].idxmax()]
        best_train_start_ts = pd.to_datetime(best_row['train_start_ts'])
        num_samples = int(best_row["samples_train"])
        df_all = pd.read_csv(trade_data_path)
        df_all = df_all[df_all['command'].str.upper() == "PREDICT_ENTRY"]
        df_all['result'] = pd.to_numeric(df_all['result'], errors='coerce')
        df_all.dropna(subset=['features', 'result'], inplace=True)
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        df_all.sort_values('timestamp', inplace=True)
        train_data_list = []
        for _idx, row in df_all.iterrows():
            if row['timestamp'] >= best_train_start_ts:
                features = parse_feature_string(row['features'])
                if features:
                    train_data_list.append({'X': features, 'y': row['result']})
                if len(train_data_list) >= num_samples:  # Corrected E701
                    break
        if len(train_data_list) < num_samples:
            msg = (f"Could not gather enough training samples ({len(train_data_list)} found, "
                   f"{num_samples} expected) for the best WFT round.")
            return False, msg
        df_train_best = pd.DataFrame(train_data_list)
        X_best, y_best = np.array(df_train_best['X'].tolist()), np.array(df_train_best['y'].tolist())
        final_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=int(best_row['round']))
        final_model.fit(X_best, y_best)
        joblib.dump(final_model, output_model_path)
        msg = (f"Best model from WFT round {best_row['round']} (Metric {optimize_metric}: "
               f"{best_row[optimize_metric]:.2f}). Saved to {output_model_path}.")
        logging.info(f"[WFT_BEST_MODEL] {msg}")
        return True, msg
    except Exception as e:
        logging.error(f"[WFT_BEST_MODEL] Error training best model from WFT: {e}", exc_info=True)
        return False, str(e)


@app.route("/activate_best_wft_model", methods=["GET"])
def activate_best_wft_model_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    try:
        success, message = _train_model_from_best_wft_round()
        if not success:
            err_msg = f"Failed to train best model from WFT: {message}"
            return jsonify({"status": "error", "message": err_msg}), 500
        if load_model(path=BEST_MODEL_PATH):
            return jsonify({"status": "success", "message": f"Best WFT model activated. {message}"}), 200
        else:
            err_msg = "Best WFT model trained but failed to load as active model."
            return jsonify({"status": "error", "message": err_msg}), 500
    except Exception as e:
        logging.error(f"[ACTIVATE_BEST_WFT_MODEL] Failed to activate best WFT model: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# --- Report Generation ---
def _create_exit_report_placeholder(output_dir="."):
    try:
        fname_base = f"TradingReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        txt_fname = os.path.join(output_dir, f"{fname_base}.txt")
        content = f"""Trading Exit Report
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
        with open(txt_fname, "w") as f:
            f.write(content)
        logging.info(f"[REPORT_GEN] ✅ Placeholder report saved as {txt_fname}")
        return txt_fname
    except Exception as e:
        logging.error(f"[REPORT_GEN] ❌ Error generating placeholder report: {e}", exc_info=True)
        return None


@app.route("/download_report", methods=["GET"])
def download_report_route():
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    try:
        reports_dir = os.path.abspath(".")
        files = [f for f in os.listdir(reports_dir) if f.startswith("TradingReport_") and f.endswith(".txt")]
        if not files:
            return "<p>❌ No reports found.</p><p><a href='/dashboard'>Back</a></p>", 404
        latest_file = sorted(files, key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)), reverse=True)[0]
        return send_file(os.path.join(reports_dir, latest_file), as_attachment=True)
    except Exception as e:
        logging.error(f"[DOWNLOAD_REPORT] Error finding/sending report: {e}", exc_info=True)
        return f"<p style='color:red;'>❌ Error loading report: {e}</p>", 500


# --- Presets ---
@app.route("/preset/<name>")
def get_preset_route(name):
    if not validate_license(request):
        return jsonify({"error": "Invalid license"}), 403
    try:
        safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        if not safe_name:
            return jsonify({"error": "Invalid preset name"}), 400
        preset_file = os.path.join("./presets/", f"{safe_name}.json")
        with open(preset_file, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        logging.warning(f"[PRESET] Preset file not found: presets/{safe_name}.json")
        return jsonify({"error": f"Preset '{safe_name}' not found"}), 404
    except json.JSONDecodeError:
        logging.error(f"[PRESET] Error decoding JSON for preset: presets/{safe_name}.json", exc_info=True)
        return jsonify({"error": f"Preset '{safe_name}' is not valid JSON"}), 500
    except Exception as e:
        logging.error(f"[PRESET] Error loading preset '{name}': {e}", exc_info=True)
        return jsonify({"error": f"Could not load preset: {e}"}), 500


@app.route('/drift', methods=['POST'])
def drift():
    try:
        if baseline_mean is None:
            logging.error("[DRIFT] baseline_mean is not initialized.")
            return "Error: Baseline mean not initialized.", 500, {'Content-Type': 'text/plain'}
        payload = request.get_json()
        if not payload or 'features' not in payload:
            return "Error: Missing 'features' in payload.", 400, {'Content-Type': 'text/plain'}
        feats_input = payload['features']
        if not isinstance(feats_input, list) or not all(isinstance(x, (int, float)) for x in feats_input):
            return "Error: 'features' must be a list of numbers.", 400, {'Content-Type': 'text/plain'}
        feats = np.array(feats_input, dtype=float)
        if feats.shape[0] != baseline_mean.shape[0]:
            msg = f"Feature length mismatch. Expected {baseline_mean.shape[0]}, got {feats.shape[0]}."
            logging.warning(f"[DRIFT] {msg}")
            return f"Error: {msg}", 400, {'Content-Type': 'text/plain'}
        if len(feats) == 0:
            return "Error: Features array cannot be empty.", 400, {'Content-Type': 'text/plain'}
        drift_val = np.linalg.norm(feats - baseline_mean) / len(feats)
        logging.info(f"[DRIFT] Calculated drift: {drift_val}")
        return str(float(drift_val)), 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        logging.error(f"[DRIFT] Error calculating drift: {e}", exc_info=True)
        return f"Error: Could not calculate drift - {str(e)}", 500, {'Content-Type': 'text/plain'}


@app.route('/rl/predict', methods=['POST'])
def rl_predict():
    if rl_model is None:
        return jsonify(error="RL model not loaded"), 500
    try:
        payload = request.get_json()
        if not payload or "features" not in payload:
            return jsonify(error="Missing 'features' in payload"), 400
        data = payload["features"]
        if not isinstance(data, list) or len(data) != feature_dim:
            return jsonify(error=f"Features must be a list of {feature_dim} numbers"), 400
        obs = np.array(data, dtype=np.float32).reshape(1, -1)
        action, _ = rl_model.predict(obs, deterministic=True)
        return jsonify(action=int(action))
    except Exception as e:
        logging.error(f"[RL_PREDICT] Error: {e}", exc_info=True)
        return jsonify(error=str(e)), 500


@app.route('/rl/store', methods=['POST'])
def rl_store():
    global rl_buffer, buffer_lock  # noqa: F824 (rl_buffer is modified by append, buffer_lock used by 'with')
    try:
        payload = request.get_json()
        required_keys = ["obs", "action", "reward", "next_obs", "done"]
        if not payload or not all(key in payload for key in required_keys):
            return jsonify(error=f"Missing one or more required keys: {required_keys}"), 400
        with buffer_lock:
            rl_buffer.append(payload)
        return jsonify(status="stored", buffer_size=len(rl_buffer))
    except Exception as e:
        logging.error(f"[RL_STORE] Error: {e}", exc_info=True)
        return jsonify(error=str(e)), 500


@app.route('/rl/update', methods=['POST'])
def rl_update():
    global rl_model, rl_buffer, buffer_lock  # noqa: F824 (these are modified or used for modification)
    if rl_model is None:
        return jsonify(error="RL model not loaded"), 500
    transitions_to_learn = []
    with buffer_lock:
        if not rl_buffer:
            return jsonify(status="no_new_data", message="No new transitions in buffer to update model."), 200
        transitions_to_learn = rl_buffer.copy()
        rl_buffer.clear()
    logging.info(f"[RL_UPDATE] Starting RL model update with {len(transitions_to_learn)} transitions.")
    try:
        if rl_model.get_env() is None:
            logging.warning("[RL_UPDATE] RL model environment not set. Setting a default one.")
            temp_env_update = PIMMAEnv(feature_dim)
            rl_model.set_env(DummyVecEnv([lambda: temp_env_update]))
        num_timesteps_to_learn = max(128, len(transitions_to_learn))
        logging.info(f"[RL_UPDATE] Calling rl_model.learn() with total_timesteps={num_timesteps_to_learn}")
        rl_model.learn(total_timesteps=num_timesteps_to_learn, reset_num_timesteps=False)
        rl_model.save(rl_model_path)  # rl_model_path is read from global scope
        logging.info(f"[RL_UPDATE] RL model updated and saved to {rl_model_path}")
        return jsonify(status="rl_model_updated", transitions_processed=len(transitions_to_learn))
    except Exception as e:
        logging.error(f"[RL_UPDATE] Error during RL model update: {e}", exc_info=True)
        return jsonify(error=f"Failed to update RL model: {str(e)}"), 500


# --- Dashboard ---
@app.route("/dashboard")
def dashboard_route():
    # Shortened class definitions for buttons to help with line length
    btn_lg = "list-group-item list-group-item-action task-btn btn btn-light"
    btn_pri = "list-group-item list-group-item-action task-btn btn btn-primary"
    btn_inf = "list-group-item list-group-item-action task-btn btn btn-info"
    btn_suc = "list-group-item list-group-item-action task-btn btn btn-success"
    btn_war = "list-group-item list-group-item-action task-btn btn btn-warning"

    html_parts = [
        "<html><head><title>📊 AI Trading Dashboard</title>",
        "<link href='https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css' rel='stylesheet'>",
        "<style>body{padding:20px;font-family:Arial,sans-serif}.task-btn{margin-bottom:10px;min-width:250px}",
        ".container{max-width:800px}h1,h2{margin-bottom:20px}.list-group-item a{text-decoration:none}",
        ".list-group-item button{width:100%;text-align:left}</style><script>",
        "async function runTask(p,bId,m='GET',b=null){const btn=document.getElementById(bId);",
        "const oT=btn.innerText;btn.disabled=true;btn.innerHTML='<span class=\"spinner-border spinner-border-sm\"></span> Processing...';",
        "let msg='';try{const op={method:m};if(m==='POST'&&b){op.headers={'Content-Type':'application/json'};",
        "op.body=JSON.stringify(b)}const r=await fetch(p,op);const d=await r.json();",
        "msg=r.ok?`✅ Success: ${d.message||JSON.stringify(d)}`:`❌ Error ${r.status}: ${d.error||JSON.stringify(d)}`;",
        "}catch(e){msg=`❌ Network/Script Error: ${e}`}alert(msg);btn.disabled=false;btn.innerText=oT}",
        "function viewPage(p){window.location.href=p}</script></head><body><div class='container'>",
        "<h1>📊 AI Trading Dashboard</h1><h2>🚀 Actions</h2><div class='list-group mb-4'>",
        f"<button id='retrain-btn' class='{btn_pri}' onclick=\"runTask('/retrain','retrain-btn')\">🔁 Trigger Model Retrain</button>",
        f"<button id='wft-btn' class='{btn_inf}' onclick=\"runTask('/wft','wft-btn')\">📈 Trigger WFT</button>", # Shortened text
        f"<button id='activate-wft-btn' class='{btn_suc}' onclick=\"runTask('/activate_best_wft_model','activate-wft-btn')\">🌟 Activate Best WFT Model</button>", # Shortened text
        "</div><h2>📈 Status & Visualizations</h2><div class='list-group mb-4'>",
        f"<button class='{btn_lg}' onclick=\"viewPage('/retrain_status')\">🧠 View Retrain Status</button>",
        f"<button class='{btn_lg}' onclick=\"viewPage('/summary')\">💹 View Trade Summary (JSON)</button>",
        f"<button class='{btn_lg}' onclick=\"viewPage('/visualize_summary')\">📊 View Profit Summary Chart</button>",
        f"<button class='{btn_lg}' onclick=\"viewPage('/wft_summary')\">📈 View WFT Summary Chart</button>",
        f"<button class='{btn_lg}' onclick=\"runTask('/monitor','monitor-btn-silent')\">🩺 API Monitor (Alert)</button>",
        f"<button class='{btn_lg}' onclick=\"viewPage('/explain')\">💡 Model Feature Importance (JSON)</button>",
        "</div><h2>📄 Logs & Reports</h2><div class='list-group mb-4'>",
        f"<button class='{btn_lg}' onclick=\"viewPage('/download_csv')\">⬇ Download Trade Data CSV</button>",
        f"<button class='{btn_lg}' onclick=\"viewPage('/download_report')\">📋 Download Latest Report</button>",
        f"<button class='{btn_war}' onclick=\"viewPage('/fail_log')\">❌ View Execution Failure Log</button>",
        f"<button class='{btn_war}' onclick=\"viewPage('/exit_decision_log')\">🚪 View Exit Decision Log</button>",
        f"<button class='{btn_war}' onclick=\"viewPage('/fail_entry_log')\">🚫 View AI Entry Fail Log</button>",
        "</div></div></body></html>"
    ]
    return "\n".join(html_parts)


# === Background Scheduler ===
def scheduled_model_retrain_task():
    logging.info("[SCHEDULER] Initiating scheduled model retrain...")
    with app.app_context():
        success, message, samples, duration = _retrain_model_core()
    if success:
        logging.info(
            f"[SCHEDULER] ✅ Retrain completed. Samples: {samples}, Duration: {duration}s. Msg: {message}"
        )
    else:
        logging.error(f"[SCHEDULER] ❌ Retrain failed. Msg: {message}")


def scheduled_report_generation_task():
    logging.info("[SCHEDULER] Initiating scheduled report generation...")
    with app.app_context():
        report_file = _create_exit_report_placeholder()
    if report_file:
        logging.info(f"[SCHEDULER] ✅ Scheduled report generated: {report_file}")
    else:
        logging.error("[SCHEDULER] ❌ Scheduled report generation failed.")


def run_scheduler():
    schedule.every().day.at("01:00").do(scheduled_model_retrain_task)
    schedule.every().day.at("00:00").do(scheduled_report_generation_task)
    logging.info("[SCHEDULER] Scheduler started with defined tasks.")
    while True:
        schedule.run_pending()
        time.sleep(60)


# === Main Execution ===
if __name__ == '__main__':
    logging.info("[SERVER_INIT] Flask API service is starting...")
    print("[SERVER_INIT] Flask API service is starting...")
    if not (app.debug and os.environ.get('WERKZEUG_RUN_MAIN') == 'true'):
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logging.info("[SERVER_INIT] Background scheduler thread started.")
        print("[SERVER_INIT] Background scheduler thread started.")
    else:
        logging.info("[SERVER_INIT] Scheduler thread already started or skipped in reloader.")
        print("[SERVER_INIT] Scheduler thread already started or skipped in reloader.")
    print("Flask API is attempting to run on http://127.0.0.1:5000")
    logging.info("Flask API initialized, attempting to run on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)