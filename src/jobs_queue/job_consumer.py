import json
from datetime import datetime
import redis
import config
from database.db_connector import db_connector
from models.bayesian_network import (
    fetch_or_init_state, 
    process_cognitive_update, 
    save_cognitive_state,
    propagate_updates_up_dag
)

class TelemetryJobConsumer:
    """
    Consumes JSON telemetry events from Redis and delegates to the
    Bayesian network mathematical updating and DAG propagation pipelines.
    """
    
    def __init__(self):
        self.r_client = db_connector.connect_redis()
        self.pg_conn = db_connector.connect_postgres()
        self.mongo_db = db_connector.connect_mongo()

    def handle_telemetry_event(self, event: dict):
        user_id = event["user_id"]
        node_id = event["node_id"]
        success = event["success"]
        behavioral_flags = event.get("behavioral_flags", [])
        event_time_str = event["timestamp"]
        
        if isinstance(event_time_str, str):
            event_timestamp = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
        else:
            event_timestamp = event_time_str
            
        print(f"\n[Consumer] Processing telemetry: User {user_id}, Node {node_id}, Success: {success}")
        
        # 1. Load belief parameters
        state = fetch_or_init_state(user_id, node_id, self.mongo_db, self.pg_conn)
        prior_alpha = state["distribution"]["alpha"]
        prior_beta = state["distribution"]["beta"]
        
        # Calculate time-decay elapsed days
        last_practiced_str = state["temporal_factors"]["last_practiced"]
        last_practiced_dt = datetime.fromisoformat(last_practiced_str.replace("Z", "+00:00"))
        time_delta = (event_timestamp - last_practiced_dt).total_seconds() / (24 * 3600.0)
        last_practiced_days = max(0.0, time_delta)
        
        decay_rate = state["temporal_factors"].get("forgetting_curve_decay_rate", config.DEFAULT_DECAY_RATE)
        
        # 2. Compute Bayesian updates
        new_alpha, new_beta, expected_mastery = process_cognitive_update(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            last_practiced_days=last_practiced_days,
            decay_rate=decay_rate,
            success=success,
            behavioral_flags=behavioral_flags
        )
        
        # 3. Commit updated distribution to Postgres and Mongo
        save_cognitive_state(
            user_id=user_id,
            node_id=node_id,
            alpha=new_alpha,
            beta=new_beta,
            mastery=expected_mastery,
            behavioral_flags=behavioral_flags,
            last_practiced_dt=event_timestamp,
            mongo_db=self.mongo_db,
            pg_conn=self.pg_conn
        )
        print(f"[Consumer] Saved Node: {node_id} (Mastery: {expected_mastery:.4f})")
        
        # 4. Propagate up the Curriculum DAG
        propagate_updates_up_dag(
            user_id=user_id,
            target_node=node_id,
            success=success,
            event_timestamp=event_timestamp,
            mongo_db=self.mongo_db,
            pg_conn=self.pg_conn,
            gamma=config.DEFAULT_GAMMA
        )

    def listen(self):
        print(f"[Consumer] Listening on Redis queue: '{config.TELEMETRY_QUEUE}'...")
        try:
            while True:
                try:
                    # Polling Redis queue with a safe 5s timeout to prevent socket hanging
                    packed = self.r_client.blpop(config.TELEMETRY_QUEUE, timeout=5)
                    if packed:
                        _, message_json = packed
                        event = json.loads(message_json)
                        self.handle_telemetry_event(event)
                except redis.exceptions.TimeoutError:
                    continue
                except Exception as ex:
                    print(f"[Consumer] Loop error occurred: {ex}")
        except KeyboardInterrupt:
            print("[Consumer] Shutting down listener.")
        finally:
            db_connector.close_all()
