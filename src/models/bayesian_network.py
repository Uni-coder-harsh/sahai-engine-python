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

def process_cognitive_update(
    prior_alpha: float,
    prior_beta: float,
    last_practiced_days: float,
    decay_rate: float,
    success: bool,
    behavioral_flags: list
) -> tuple:
    # 1. Apply Ebbinghaus forgetting decay
    decayed_alpha = apply_ebbinghaus_decay(prior_alpha, last_practiced_days, decay_rate)
    decayed_beta = prior_beta

    # 2. Behavior-driven modifiers
    alpha_modifier = 0.0
    beta_modifier = 0.0

    if "COPY_PASTE_PRONE" in behavioral_flags:
        beta_modifier += 1.5
    
    if "SYNTAX_HESITANT" in behavioral_flags:
        beta_modifier += 0.8

    if success:
        alpha_modifier += 1.0
    else:
        beta_modifier += 1.0

    new_alpha = decayed_alpha + alpha_modifier
    new_beta = decayed_beta + beta_modifier
    expected_mastery = calculate_expected_mastery(new_alpha, new_beta)

    return new_alpha, new_beta, expected_mastery

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
        "alpha": 1.0,
        "beta": 1.0,
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
            (user_id, node_id, 1.0, 1.0, 0.50)
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
    gamma: float = 0.5
):
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT source_node, correlation_weight 
            FROM advanced_dag_edges 
            WHERE target_node = %s;
            """,
            (target_node,)
        )
        edges = cur.fetchall()
        
    for edge in edges:
        parent_node = edge["source_node"]
        weight = float(edge["correlation_weight"])
        
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
        
        discounted_delta = weight * gamma
        if success:
            new_alpha = decayed_alpha + discounted_delta
            new_beta = decayed_beta
        else:
            new_alpha = decayed_alpha
            new_beta = decayed_beta + discounted_delta
            
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
