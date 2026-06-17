import json
import time
from datetime import datetime
import redis
import config
from database.db_connector import db_connector
from utils.logger import logger
from models.bayesian_network import (
    fetch_or_init_state, 
    process_cognitive_update, 
    save_cognitive_state,
    propagate_updates_up_dag,
    apply_ebbinghaus_decay,
    calculate_expected_mastery
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
        self.cache_global_dag()

    def cache_global_dag(self):
        logger.info("Caching global DAG edges to Redis...")
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source_node, target_node, correlation_weight, w_pre, w_diag
                    FROM advanced_dag_edges;
                    """
                )
                rows = cur.fetchall()
            
            # Group by target_node
            dag = {}
            for row in rows:
                src, tgt, weight, w_pre, w_diag = row
                if tgt not in dag:
                    dag[tgt] = []
                dag[tgt].append({
                    "source_node": src,
                    "correlation_weight": float(weight) if weight is not None else 0.0,
                    "w_pre": float(w_pre) if w_pre is not None else (float(weight) if weight is not None else 0.0),
                    "w_diag": float(w_diag) if w_diag is not None else (float(weight) if weight is not None else 0.0)
                })
            
            # Pipeline writes to Redis
            pipe = self.r_client.pipeline()
            pipe.delete("global_dag")
            for tgt, edges in dag.items():
                pipe.hset("global_dag", tgt, json.dumps(edges))
            pipe.execute()
            logger.info(f"Successfully cached {len(dag)} nodes' prerequisite relationships to Redis.")
        except Exception as e:
            logger.error(f"Failed to cache global DAG to Redis: {e}")

    def handle_telemetry_event(self, event: dict):
        user_id = event["user_id"]
        node_id = event["node_id"]
        success = event["success"]
        behavioral_flags = event.get("behavioral_flags", [])
        event_time_str = event["timestamp"]
        
        # Pull MCQ-specific parameters
        influence_weight = event.get("influence_weight", 1.0)
        misconceptions = event.get("misconceptions", [])
        
        if isinstance(event_time_str, str):
            event_timestamp = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
        else:
            event_timestamp = event_time_str
        logger.info(f"Processing telemetry: User {user_id}, Node {node_id}, Success: {success}, Weight: {influence_weight}")
        
        # 1. Load belief parameters for primary concept
        state = fetch_or_init_state(user_id, node_id, self.mongo_db, self.pg_conn)
        prior_alpha = state["distribution"]["alpha"]
        prior_beta = state["distribution"]["beta"]
        
        # Calculate time-decay elapsed days
        last_practiced_str = state["temporal_factors"]["last_practiced"]
        last_practiced_dt = datetime.fromisoformat(last_practiced_str.replace("Z", "+00:00"))
        time_delta = (event_timestamp - last_practiced_dt).total_seconds() / (24 * 3600.0)
        last_practiced_days = max(0.0, time_delta)
        
        decay_rate = state["temporal_factors"].get("forgetting_curve_decay_rate", config.DEFAULT_DECAY_RATE)
        
        # 2. Compute Bayesian updates for primary concept
        new_alpha, new_beta, expected_mastery = process_cognitive_update(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            last_practiced_days=last_practiced_days,
            decay_rate=decay_rate,
            success=success,
            behavioral_flags=behavioral_flags,
            influence_weight=influence_weight
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
        logger.info(f"Saved Node: {node_id} (Mastery: {expected_mastery:.4f})")
        
        # 4. Handle Option Misconceptions (if incorrect)
        for misc in misconceptions:
            m_node_id = misc["node_id"]
            m_weight = misc["weight"]
            
            # Load current state of the misconception concept node
            m_state = fetch_or_init_state(user_id, m_node_id, self.mongo_db, self.pg_conn)
            m_prior_alpha = m_state["distribution"]["alpha"]
            m_prior_beta = m_state["distribution"]["beta"]
            
            # Apply time decay
            m_last_practiced_str = m_state["temporal_factors"]["last_practiced"]
            m_last_practiced_dt = datetime.fromisoformat(m_last_practiced_str.replace("Z", "+00:00"))
            m_time_delta = (event_timestamp - m_last_practiced_dt).total_seconds() / (24 * 3600.0)
            m_last_practiced_days = max(0.0, m_time_delta)
            
            m_decay_rate = m_state["temporal_factors"].get("forgetting_curve_decay_rate", config.DEFAULT_DECAY_RATE)
            m_decayed_alpha = apply_ebbinghaus_decay(m_prior_alpha, m_last_practiced_days, m_decay_rate)
            
            # Increase beta by misconception weight (indicating error evidence on that concept)
            m_new_alpha = m_decayed_alpha
            m_new_beta = m_prior_beta + m_weight
            m_expected_mastery = calculate_expected_mastery(m_new_alpha, m_new_beta)
            
            save_cognitive_state(
                user_id=user_id,
                node_id=m_node_id,
                alpha=m_new_alpha,
                beta=m_new_beta,
                mastery=m_expected_mastery,
                behavioral_flags=['MISCONCEPTION_TRIGGERED'],
                last_practiced_dt=event_timestamp,
                mongo_db=self.mongo_db,
                pg_conn=self.pg_conn
            )
            logger.info(f"Misconception Updated: {m_node_id} (Mastery: {m_expected_mastery:.4f})")
        
        # 5. Propagate up the Curriculum DAG
        propagate_updates_up_dag(
            user_id=user_id,
            target_node=node_id,
            success=success,
            event_timestamp=event_timestamp,
            mongo_db=self.mongo_db,
            pg_conn=self.pg_conn,
            r_client=self.r_client,
            gamma=config.DEFAULT_GAMMA
        )

    def listen(self):
        logger.info(f"Listening on Redis queue: '{config.TELEMETRY_QUEUE}'...")
        try:
            while True:
                try:
                    # Polling Redis queue with a safe 30s timeout to prevent socket hanging
                    packed = self.r_client.blpop(config.TELEMETRY_QUEUE, timeout=30)
                    if packed:
                        _, message_json = packed
                        event = json.loads(message_json)
                        self.handle_telemetry_event(event)
                except redis.exceptions.TimeoutError as te:
                    logger.warning(f"Redis timeout/socket error: {te}. Sleeping 5s before retrying...")
                    time.sleep(5)
                except (redis.exceptions.ConnectionError, redis.exceptions.RedisError) as re:
                    logger.error(f"Redis connection error: {re}. Sleeping 5s before retrying...")
                    time.sleep(5)
                except Exception as ex:
                    logger.error(f"Loop error occurred: {ex}. Sleeping 5s before retrying...")
                    time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Shutting down listener.")
        finally:
            db_connector.close_all()
