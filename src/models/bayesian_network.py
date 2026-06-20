import json
import math
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
import config
import pickle
from pathlib import Path
import numpy as np

# Live Inference Bridge: Load the Random Forest classifiers at startup
models = {"MCQ": None, "Code": None, "OCR": None}
model_filenames = {
    "MCQ": "telemetry_mcq_model.pkl",
    "Code": "telemetry_code_model.pkl",
    "OCR": "telemetry_ocr_model.pkl"
}

curr_dir = Path(__file__).resolve().parent
models_dir = None
for _ in range(5):
    check_path = curr_dir / "models"
    if check_path.exists() and (check_path / "telemetry_mcq_model.pkl").exists():
        models_dir = check_path
        break
    # Check if we are inside the engine-python or similar peer folders
    peer_path = curr_dir.parent / "models"
    if peer_path.exists() and (peer_path / "telemetry_mcq_model.pkl").exists():
        models_dir = peer_path
        break
    if curr_dir == curr_dir.parent:
        break
    curr_dir = curr_dir.parent

if models_dir:
    for task_name, filename in model_filenames.items():
        path = models_dir / filename
        if path.exists():
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                    if isinstance(payload, dict) and "model" in payload:
                        models[task_name] = payload
                        print(f"[ML Bridge] Successfully loaded {task_name} model payload from {path}")
                    else:
                        models[task_name] = {
                            "model": payload,
                            "scaler": None,
                            "features": None
                        }
                        print(f"[ML Bridge] Loaded raw {task_name} model from {path}")
            except Exception as e:
                print(f"[ML Bridge] Warning: Failed to load {task_name} model from {path} ({e})")
        else:
            print(f"[ML Bridge] Warning: {filename} not found in {models_dir}")
else:
    print("[ML Bridge] Warning: Models directory not found in parent directory tree. Falling back to rule-based simulation.")


def calculate_variance(alpha: float, beta: float) -> float:
    try:
        alpha = float(alpha)
        beta = float(beta)
        if math.isnan(alpha) or math.isnan(beta):
            return 0.0833
        total = alpha + beta
        denom = (total ** 2) * (total + 1.0)
        if denom <= 0 or math.isnan(denom) or math.isinf(denom):
            return 0.0833
        val = (alpha * beta) / denom
        if math.isnan(val) or math.isinf(val):
            return 0.0833
        return val
    except Exception:
        return 0.0833

def apply_ebbinghaus_decay(alpha: float, time_delta_days: float, decay_rate: float) -> float:
    try:
        alpha = float(alpha)
        time_delta_days = float(time_delta_days)
        decay_rate = float(decay_rate)
        if math.isnan(alpha) or math.isnan(time_delta_days) or math.isnan(decay_rate):
            return 2.0
        if time_delta_days <= 0:
            return alpha
        decayed = 1.0 + (alpha - 1.0) * math.exp(-decay_rate * time_delta_days)
        if math.isnan(decayed) or math.isinf(decayed):
            return 1.0
        return max(1.0, decayed)
    except Exception:
        return 1.0

def calculate_expected_mastery(alpha: float, beta: float) -> float:
    try:
        alpha = float(alpha)
        beta = float(beta)
        if math.isnan(alpha) or math.isnan(beta):
            return 0.5
        total = alpha + beta
        if total <= 0:
            return 0.5
        val = alpha / total
        if math.isnan(val) or math.isinf(val):
            return 0.5
        return val
    except Exception:
        return 0.5

def classify_telemetry(interaction_type: str, metrics: dict, is_correct: bool = True) -> str:
    """
    Predicts student behavioral class depending on the interaction_type:
    - MCQ classes: BLIND_GUESSING, ACCIDENTAL_MISCLICK, THOROUGH_COMPREHENSION_BONUS
    - Code classes: COPY_PASTE_DEPENDENCY, SHOTGUN_DEBUGGING, SYSTEMIC_STRUGGLE_REWARD, etc.
    - OCR classes: FOUNDATIONAL_VOID, PROCEDURAL_FATIGUE
    """
    # Normalize interaction type
    itype = "Code"
    if interaction_type:
        val = str(interaction_type).upper()
        if "MCQ" in val:
            itype = "MCQ"
        elif "OCR" in val:
            itype = "OCR"
        elif "CODE" in val or "RUN" in val or "COMPILE" in val:
            itype = "Code"

    model_info = models.get(itype)
    if model_info and model_info.get("model") is not None:
        try:
            clf = model_info["model"]
            scaler = model_info.get("scaler")
            features = model_info.get("features")
            
            input_vector = []
            if features:
                for f in features:
                    if f == "is_correct":
                        input_vector.append(1.0 if is_correct else 0.0)
                    else:
                        input_vector.append(float(metrics.get(f, 0.0)))
            else:
                # Fallbacks if features meta is absent
                if itype == "MCQ":
                    input_vector = [
                        float(metrics.get("question_word_count", 50)),
                        float(metrics.get("time_to_first_action_sec", 10)),
                        float(metrics.get("reading_velocity", 3.0)),
                        float(metrics.get("option_switch_count", 0)),
                        float(metrics.get("minimum_click_interval_ms", 1000)),
                        float(metrics.get("network_drop_duration_sec", 0.0)),
                        float(metrics.get("total_time_spent_sec", 30)),
                        1.0 if is_correct else 0.0
                    ]
                elif itype == "OCR":
                    input_vector = [
                        float(metrics.get("total_steps_detected", 5)),
                        float(metrics.get("logical_break_step_index", 2)),
                        float(metrics.get("erasure_scribble_ratio", 0.05)),
                        float(metrics.get("spatial_density", 0.4)),
                        float(metrics.get("time_since_last_upload_sec", 60)),
                        1.0 if is_correct else 0.0
                    ]
                else: # Code
                    input_vector = [
                        float(metrics.get("time_spent_seconds", metrics.get("time_spent_sec", 30))),
                        float(metrics.get("time_to_first_edit_sec", 5)),
                        float(metrics.get("compile_count", metrics.get("run_count", 0))),
                        float(metrics.get("syntax_error_ratio", 0.1)),
                        float(metrics.get("runtime_error_ratio", 0.0)),
                        float(metrics.get("paste_char_count", 0)),
                        float(metrics.get("backspace_count", 0)),
                        float(metrics.get("structural_grit_ratio", 1.0)),
                        1.0 if is_correct else 0.0
                    ]
            
            X_input = np.array([input_vector])
            if scaler:
                X_input = scaler.transform(X_input)
                
            prediction = clf.predict(X_input)
            return str(prediction[0])
            
        except Exception as e:
            print(f"[ML Bridge] Inference error for {itype}: {e}. Falling back to rules.")
            
    # Rule-based fallbacks
    if itype == "MCQ":
        time_spent = float(metrics.get("total_time_spent_sec", 30))
        option_switches = int(metrics.get("option_switch_count", 0))
        if time_spent < 5:
            return "BLIND_GUESSING"
        if option_switches > 3:
            return "ACCIDENTAL_MISCLICK"
        return "THOROUGH_COMPREHENSION_BONUS" if is_correct else "NORMAL_INCORRECT"
    elif itype == "OCR":
        erasure_ratio = float(metrics.get("erasure_scribble_ratio", 0.0))
        if erasure_ratio > 0.1:
            return "PROCEDURAL_FATIGUE"
        return "FOUNDATIONAL_VOID"
    else:  # Code
        paste_char_count = int(metrics.get("paste_char_count", 0))
        backspace_count = int(metrics.get("backspace_count", 0))
        run_count = int(metrics.get("run_count", metrics.get("compile_count", 0)))
        time_spent_seconds = float(metrics.get("time_spent_seconds", metrics.get("time_spent_sec", 30)))
        
        if paste_char_count > 30 and backspace_count < 2:
            return "COPY_PASTE_DEPENDENCY"
        if run_count > 4 and time_spent_seconds < 15:
            return "SHOTGUN_DEBUGGING"
        return "NORMAL"

def process_cognitive_update(
    prior_alpha: float,
    prior_beta: float,
    last_practiced_days: float,
    decay_rate: float,
    success: bool,
    behavioral_flags: list,
    influence_weight: float = 1.0,
    telemetry_data: dict = None,
    interaction_type: str = "Code"
) -> tuple:
    # 1. Apply Ebbinghaus forgetting decay
    decayed_alpha = apply_ebbinghaus_decay(prior_alpha, last_practiced_days, decay_rate)
    decayed_beta = prior_beta

    # 2. Run ML classifier if telemetry_data is provided and has contents, otherwise default to "NORMAL"
    if telemetry_data and len(telemetry_data) > 0:
        predicted_behavior = classify_telemetry(interaction_type, telemetry_data, is_correct=success)
    else:
        predicted_behavior = "NORMAL"

    # 3. Behavior-driven modifiers and integer mapping for backward compatibility
    alpha_modifier = 0.0
    beta_modifier = 0.0
    learning_rate_modifier = 1.0
    behavior_class_int = 0  # Default: Normal (0)

    # Map behavioral string output to proper metrics modifiers and integer class codes
    if predicted_behavior == "COPY_PASTE_DEPENDENCY" or predicted_behavior == 2:
        learning_rate_modifier = 0.5  # Copy-paste penalty (reduced from 0.35)
        behavior_class_int = 2
        if "COPY_PASTE_PRONE" not in behavioral_flags:
            behavioral_flags.append("COPY_PASTE_PRONE")
    elif predicted_behavior == "BLIND_GUESSING" or predicted_behavior == 3:
        learning_rate_modifier = 0.5  # Guess penalty (reduced from 0.35)
        behavior_class_int = 3
        if "BLIND_GUESSING" not in behavioral_flags:
            behavioral_flags.append("BLIND_GUESSING")
    elif predicted_behavior == "FOUNDATIONAL_VOID" or predicted_behavior == 4:
        learning_rate_modifier = 0.5  # Foundational void penalty (reduced from 0.35)
        behavior_class_int = 4
        if "FOUNDATIONAL_VOID" not in behavioral_flags:
            behavioral_flags.append("FOUNDATIONAL_VOID")
    elif predicted_behavior in ["SHOTGUN_DEBUGGING", "PROCEDURAL_FATIGUE", 1]:
        learning_rate_modifier = 0.8  # Shotgun debugging penalty (reduced from 0.7)
        behavior_class_int = 1
        if predicted_behavior == "SHOTGUN_DEBUGGING" or predicted_behavior == 1:
            if "SHOTGUN_DEBUGGING" not in behavioral_flags:
                behavioral_flags.append("SHOTGUN_DEBUGGING")
        elif predicted_behavior == "PROCEDURAL_FATIGUE":
            if "PROCEDURAL_FATIGUE" not in behavioral_flags:
                behavioral_flags.append("PROCEDURAL_FATIGUE")
    elif predicted_behavior in ["AMBIGUOUS_ANXIOUS_LEARNER", "ANXIOUS_OVERWORKING", "ACCIDENTAL_MISCLICK", 5]:
        learning_rate_modifier = 0.95  # Anxious overworking/misclick penalty (reduced from 0.9)
        behavior_class_int = 5
        if "ANXIOUS_OVERWORKING" not in behavioral_flags:
            behavioral_flags.append("ANXIOUS_OVERWORKING")
    elif predicted_behavior == "THOROUGH_COMPREHENSION_BONUS":
        learning_rate_modifier = 1.3  # Comprehension reward
        behavior_class_int = 0
        if "THOROUGH_COMPREHENSION" not in behavioral_flags:
            behavioral_flags.append("THOROUGH_COMPREHENSION")
    elif predicted_behavior == "SYSTEMIC_STRUGGLE_REWARD":
        learning_rate_modifier = 1.2  # Reward for grit
        behavior_class_int = 0
        if "SYSTEMIC_STRUGGLE_REWARD" not in behavioral_flags:
            behavioral_flags.append("SYSTEMIC_STRUGGLE_REWARD")

    if "SYNTAX_HESITANT" in behavioral_flags:
        beta_modifier += 0.8

    if success:
        alpha_modifier += 1.0 * influence_weight * learning_rate_modifier
    else:
        beta_modifier += 1.0 * influence_weight

    new_alpha = decayed_alpha + alpha_modifier
    new_beta = decayed_beta + beta_modifier
    expected_mastery = calculate_expected_mastery(new_alpha, new_beta)

    return new_alpha, new_beta, expected_mastery, behavior_class_int

def fetch_or_init_state(user_id: str, node_id: str, mongo_db, pg_conn) -> dict:
    col = mongo_db["student_cognitive_distributions"]
    doc = col.find_one({"user_id": user_id, "node_id": node_id})
    
    if doc:
        return doc
        
    # Seed default Gaussian prior
    now_str = datetime.now(timezone.utc).isoformat()
    new_doc = {
      "user_id": user_id,
      "node_id": node_id,
      "distribution": {
        "type": "BETA",
        "alpha": 2.0,
        "beta": 2.0,
        "variance": 0.0833,
        "confidence_interval_95": [0.05, 0.95]
      },
      "temporal_factors": {
        "last_practiced": now_str,
        "forgetting_curve_decay_rate": config.DEFAULT_DECAY_RATE, 
        "current_adjusted_mastery": 0.50
      },
      "behavioral_flags": []
    }
    
    col.insert_one(new_doc)
    
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_cognitive_states (user_id, node_id, alpha, beta, expected_mastery, last_practiced)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, node_id) DO NOTHING;
            """,
            (user_id, node_id, 2.0, 2.0, 0.50)
        )
        pg_conn.commit()
        
    return new_doc

def save_cognitive_state(
    user_id: str,
    node_id: str,
    alpha: float,
    beta: float,
    mastery: float,
    behavioral_flags: list,
    last_practiced_dt: datetime,
    mongo_db,
    pg_conn
):
    col = mongo_db["student_cognitive_distributions"]
    variance = calculate_variance(alpha, beta)
    
    lower_ci = max(0.01, mastery - 1.96 * (variance ** 0.5))
    upper_ci = min(0.99, mastery + 1.96 * (variance ** 0.5))
    
    col.update_one(
        {"user_id": user_id, "node_id": node_id},
        {
            "$set": {
                "distribution.alpha": float(alpha),
                "distribution.beta": float(beta),
                "distribution.variance": float(variance),
                "distribution.confidence_interval_95": [float(lower_ci), float(upper_ci)],
                "temporal_factors.last_practiced": last_practiced_dt.isoformat(),
                "temporal_factors.current_adjusted_mastery": float(mastery)
            },
            "$addToSet": {
                "behavioral_flags": {"$each": behavioral_flags}
            }
        },
        upsert=True
    )
    
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_cognitive_states (user_id, node_id, alpha, beta, expected_mastery, last_practiced, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, node_id) DO UPDATE SET
                alpha = EXCLUDED.alpha,
                beta = EXCLUDED.beta,
                expected_mastery = EXCLUDED.expected_mastery,
                last_practiced = EXCLUDED.last_practiced,
                updated_at = NOW();
            """,
            (user_id, node_id, float(alpha), float(beta), float(mastery), last_practiced_dt)
        )
        pg_conn.commit()

def propagate_updates_up_dag(
    user_id: str,
    target_node: str,
    success: bool,
    event_timestamp: datetime,
    mongo_db,
    pg_conn,
    r_client=None,
    gamma: float = 1.0
):
    edges = []
    if r_client:
        try:
            cached_edges = r_client.hget("global_dag", target_node)
            if cached_edges:
                edges = json.loads(cached_edges)
        except Exception as e:
            print(f"[DAG] Error fetching DAG from Redis: {e}")

    if not edges:
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT source_node, correlation_weight, w_pre, w_diag
                FROM advanced_dag_edges 
                WHERE target_node = %s;
                """,
                (target_node,)
            )
            edges = cur.fetchall()
        
    propagations_logged = []
    for edge in edges:
        parent_node = edge["source_node"]
        w_pre = float(edge.get("w_pre") or edge.get("correlation_weight") or 0.0)
        w_diag = float(edge.get("w_diag") or edge.get("correlation_weight") or 0.0)
        
        parent_state = fetch_or_init_state(user_id, parent_node, mongo_db, pg_conn)
        
        last_practiced_str = parent_state["temporal_factors"]["last_practiced"]
        last_practiced_dt = datetime.fromisoformat(last_practiced_str.replace("Z", "+00:00"))
        time_delta = (event_timestamp - last_practiced_dt).total_seconds() / (24 * 3600.0)
        time_delta_days = max(0.0, time_delta)
        
        decay_rate = parent_state["temporal_factors"].get("forgetting_curve_decay_rate", config.DEFAULT_DECAY_RATE)
        
        decayed_alpha = apply_ebbinghaus_decay(
            parent_state["distribution"]["alpha"], 
            time_delta_days, 
            decay_rate
        )
        decayed_beta = parent_state["distribution"]["beta"]
        
        if success:
            new_alpha = decayed_alpha + (1.0 * w_pre)
            new_beta = decayed_beta
        else:
            new_alpha = decayed_alpha
            new_beta = decayed_beta + (1.0 * w_diag)
            
        new_mastery = calculate_expected_mastery(new_alpha, new_beta)
        
        save_cognitive_state(
            user_id, 
            parent_node, 
            new_alpha, 
            new_beta, 
            new_mastery, 
            [], 
            event_timestamp, 
            mongo_db, 
            pg_conn
        )
        print(f"[DAG] Propagated {target_node} -> prerequisite {parent_node} (Mastery: {new_mastery:.4f})")
        propagations_logged.append({
            "source_node": parent_node,
            "target_node": target_node,
            "w_pre": w_pre,
            "w_diag": w_diag,
            "expected_mastery": float(new_mastery)
        })
    return propagations_logged

def update_bayesian_network(
    user_id: str,
    failed_node_id: str,
    is_correct: bool,
    telemetry_metrics: dict = None,
    primary_node_id: str = None,
    mongo_db = None,
    pg_conn = None,
    r_client = None
) -> dict:
    """
    Exposes a unified interface to update the student cognitive belief states 
    and propagate updates up the DAG. If correct, updates the primary concept node as successful.
    If incorrect, updates the failed_node_id (if provided) or primary_node_id as failure.
    """
    from database.db_connector import db_connector
    
    if pg_conn is None:
        pg_conn = db_connector.connect_postgres()
    if mongo_db is None:
        mongo_db = db_connector.connect_mongo()
    if r_client is None:
        r_client = db_connector.connect_redis()
        
    target_node = primary_node_id
    if not is_correct and failed_node_id:
        target_node = failed_node_id
        
    if not target_node:
        target_node = failed_node_id or primary_node_id
        
    if not target_node:
        raise ValueError("Could not determine cognitive node target for update (both primary_node_id and failed_node_id are empty).")
        
    # Load state
    state = fetch_or_init_state(user_id, target_node, mongo_db, pg_conn)
    prior_alpha = state["distribution"]["alpha"]
    prior_beta = state["distribution"]["beta"]
    
    # Calculate time-decay elapsed days
    last_practiced_str = state["temporal_factors"]["last_practiced"]
    last_practiced_dt = datetime.fromisoformat(last_practiced_str.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    time_delta = (now - last_practiced_dt).total_seconds() / (24 * 3600.0)
    last_practiced_days = max(0.0, time_delta)
    
    decay_rate = state["temporal_factors"].get("forgetting_curve_decay_rate", config.DEFAULT_DECAY_RATE)
    
    # Determine behavioral flags
    behavioral_flags = []
    if not is_correct:
        behavioral_flags.append("LOGICAL_FLAW_DETECTED")
    else:
        behavioral_flags.append("OCR_GRADED_CORRECT")
        
    # Run cognitive updates
    new_alpha, new_beta, expected_mastery, behavior_class = process_cognitive_update(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        last_practiced_days=last_practiced_days,
        decay_rate=decay_rate,
        success=is_correct,
        behavioral_flags=behavioral_flags,
        influence_weight=1.0,
        telemetry_data=telemetry_metrics,
        interaction_type="OCR"
    )
    
    # Save updated cognitive state
    save_cognitive_state(
        user_id=user_id,
        node_id=target_node,
        alpha=new_alpha,
        beta=new_beta,
        mastery=expected_mastery,
        behavioral_flags=behavioral_flags,
        last_practiced_dt=now,
        mongo_db=mongo_db,
        pg_conn=pg_conn
    )
    
    # Propagate DAG updates
    propagations = propagate_updates_up_dag(
        user_id=user_id,
        target_node=target_node,
        success=is_correct,
        event_timestamp=now,
        mongo_db=mongo_db,
        pg_conn=pg_conn,
        r_client=r_client,
        gamma=config.DEFAULT_GAMMA
    )
    
    return {
        "success": True,
        "target_node": target_node,
        "new_alpha": float(new_alpha),
        "new_beta": float(new_beta),
        "expected_mastery": float(expected_mastery),
        "propagations": propagations
    }

