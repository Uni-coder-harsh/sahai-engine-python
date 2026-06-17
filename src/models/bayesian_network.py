import json
import math
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
import config

def calculate_variance(alpha: float, beta: float) -> float:
    total = alpha + beta
    denom = (total ** 2) * (total + 1.0)
    if denom == 0:
        return 0.0
    return (alpha * beta) / denom

def apply_ebbinghaus_decay(alpha: float, time_delta_days: float, decay_rate: float) -> float:
    if time_delta_days <= 0:
        return alpha
    decayed = 1.0 + (alpha - 1.0) * math.exp(-decay_rate * time_delta_days)
    return max(1.0, decayed)

def calculate_expected_mastery(alpha: float, beta: float) -> float:
    total = alpha + beta
    if total == 0:
        return 0.5
    return alpha / total

def classify_telemetry(time_spent_seconds: float, run_count: int, backspace_count: int, paste_char_count: int, syntax_error_count: int) -> int:
    """
    Simulates a Random Forest telemetry classifier predicting:
    0: Normal
    1: Shotgun Debugging (high run count, low time)
    2: Copy-Paste Dependency (high paste count, low backspace)
    """
    if paste_char_count > 30 and backspace_count < 2:
        return 2  # Copy-Paste
    if run_count > 4 and time_spent_seconds < 15:
        return 1  # Shotgun Debugging
    return 0  # Normal

def process_cognitive_update(
    prior_alpha: float,
    prior_beta: float,
    last_practiced_days: float,
    decay_rate: float,
    success: bool,
    behavioral_flags: list,
    influence_weight: float = 1.0,
    telemetry_data: dict = None
) -> tuple:
    # 1. Apply Ebbinghaus forgetting decay
    decayed_alpha = apply_ebbinghaus_decay(prior_alpha, last_practiced_days, decay_rate)
    decayed_beta = prior_beta

    # 2. Extract telemetry features
    telemetry = telemetry_data or {}
    time_spent = float(telemetry.get("time_spent_seconds", 30))
    run_count = int(telemetry.get("run_count", 0))
    backspace_count = int(telemetry.get("backspace_count", 0))
    paste_count = int(telemetry.get("paste_char_count", 0))
    syntax_errors = int(telemetry.get("syntax_error_count", 0))

    # Run ML classifier
    behavior_class = classify_telemetry(time_spent, run_count, backspace_count, paste_count, syntax_errors)

    # 3. Behavior-driven modifiers
    alpha_modifier = 0.0
    beta_modifier = 0.0

    learning_rate_modifier = 1.0
    if behavior_class == 2:
        learning_rate_modifier = 0.1  # Copy-paste penalty
        if "COPY_PASTE_PRONE" not in behavioral_flags:
            behavioral_flags.append("COPY_PASTE_PRONE")
    elif behavior_class == 1:
        learning_rate_modifier = 0.5  # Shotgun debugging penalty
        if "SHOTGUN_DEBUGGING" not in behavioral_flags:
            behavioral_flags.append("SHOTGUN_DEBUGGING")

    if "SYNTAX_HESITANT" in behavioral_flags:
        beta_modifier += 0.8

    if success:
        alpha_modifier += 1.0 * influence_weight * learning_rate_modifier
    else:
        beta_modifier += 1.0 * influence_weight

    new_alpha = decayed_alpha + alpha_modifier
    new_beta = decayed_beta + beta_modifier
    expected_mastery = calculate_expected_mastery(new_alpha, new_beta)

    return new_alpha, new_beta, expected_mastery, behavior_class

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
