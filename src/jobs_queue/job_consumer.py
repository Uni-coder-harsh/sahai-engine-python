import sys
import io
import json
import time
import threading
from datetime import datetime
import redis
import config
from database.db_connector import db_connector
from utils.logger import logger

class DualStream:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

class StdoutCapturer:
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.buffer = io.StringIO()
        sys.stdout = DualStream(self.old_stdout, self.buffer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

    def get_lines(self):
        content = self.buffer.getvalue()
        return [line for line in content.split('\n') if line.strip()]
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
        
        # Initialize RAG vector store and BM25 index on startup
        try:
            from rag.vector_store import vector_store
            from rag.hybrid_searcher import hybrid_searcher
            vector_store.populate_database_embeddings(self.pg_conn)
            hybrid_searcher.initialize_bm25(self.pg_conn)
        except Exception as startup_err:
            logger.error(f"Error executing RAG startup indexing sequence: {startup_err}")
            
        # Threading lock and status flag for on-demand queue processing
        self.lock = threading.Lock()
        self.is_processing = False

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
        # 1. Log the raw JSON payload permanently to MongoDB
        try:
            self.mongo_db["raw_telemetry_logs"].insert_one(event.copy())
            logger.info("Raw telemetry payload logged to MongoDB raw_telemetry_logs.")
        except Exception as mongo_err:
            logger.error(f"Failed to log raw telemetry payload to MongoDB: {mongo_err}")

        interaction_type = event.get("interaction_type", event.get("event_type", "Code"))
        if interaction_type == "OCR_HANDWRITING":
            return self.handle_ocr_handwriting_event(event)

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
        
        # Determine interaction type and extract metrics
        interaction_type = event.get("interaction_type", event.get("event_type", "Code"))
        telemetry_data = event.get("metrics", event)

        logger.info(f"Processing telemetry: User {user_id}, Node {node_id}, Success: {success}, Interaction: {interaction_type}")
        
        # Load belief parameters for primary concept
        state = fetch_or_init_state(user_id, node_id, self.mongo_db, self.pg_conn)
        prior_alpha = state["distribution"]["alpha"]
        prior_beta = state["distribution"]["beta"]
        
        # Calculate time-decay elapsed days
        last_practiced_str = state["temporal_factors"]["last_practiced"]
        last_practiced_dt = datetime.fromisoformat(last_practiced_str.replace("Z", "+00:00"))
        time_delta = (event_timestamp - last_practiced_dt).total_seconds() / (24 * 3600.0)
        last_practiced_days = max(0.0, time_delta)
        
        decay_rate = state["temporal_factors"].get("forgetting_curve_decay_rate", config.DEFAULT_DECAY_RATE)
        
        # Compute Bayesian updates for primary concept with ML behavioral modifier
        new_alpha, new_beta, expected_mastery, behavior_class = process_cognitive_update(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            last_practiced_days=last_practiced_days,
            decay_rate=decay_rate,
            success=success,
            behavioral_flags=behavioral_flags,
            influence_weight=influence_weight,
            telemetry_data=telemetry_data,
            interaction_type=interaction_type
        )
        
        # Print structured log showing behavioral class and updates
        logger.info(f"""
==================================================
           ML INFERENCE & COGNITIVE UPDATE
==================================================
Event ID:          {event.get('event_id', 'N/A')}
User ID:          {user_id}
Node ID:          {node_id}
Interaction Type: {interaction_type}
Predicted Class:  {behavior_class}
Success:          {success}
--------------------------------------------------
Bayesian Updates:
Prior Alpha:      {prior_alpha:.4f} -> New Alpha: {new_alpha:.4f}
Prior Beta:       {prior_beta:.4f} -> New Beta:  {new_beta:.4f}
New Mastery Mean: {expected_mastery:.4f}
==================================================
""")
        
        # Commit updated distribution to Postgres and Mongo
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
        misconceptions_updated = []
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
            misconceptions_updated.append({
                "node_id": m_node_id,
                "weight": m_weight,
                "expected_mastery": float(m_expected_mastery)
            })
        
        # 5. Propagate up the Curriculum DAG
        propagations = propagate_updates_up_dag(
            user_id=user_id,
            target_node=node_id,
            success=success,
            event_timestamp=event_timestamp,
            mongo_db=self.mongo_db,
            pg_conn=self.pg_conn,
            r_client=self.r_client,
            gamma=config.DEFAULT_GAMMA
        )

        return {
            "success": True,
            "user_id": user_id,
            "node_id": node_id,
            "behavior_class": behavior_class,
            "alpha": float(new_alpha),
            "beta": float(new_beta),
            "expected_mastery": float(expected_mastery),
            "misconceptions_updated": misconceptions_updated,
            "propagations": propagations
        }

    def handle_ocr_handwriting_event(self, event: dict) -> dict:
        """
        Extracts handwritten student code, retrieves context, performs LLM logical evaluation,
        logs data to MongoDB, and triggers Bayesian cognitive state updates.
        Captures all standard output prints during the pipeline execution and returns them in the response.
        """
        with StdoutCapturer() as capturer:
            try:
                result = self._internal_handle_ocr_handwriting_event(event)
            except Exception as ex:
                result = {"success": False, "error": str(ex)}
            
            result["developer_debug_logs"] = capturer.get_lines()
            return result

    def _internal_handle_ocr_handwriting_event(self, event: dict) -> dict:
        """
        Internal implementation of handwritten code extraction and logical grading.
        """
        print("\n" + "="*80)
        print("[DEVELOPER DEBUG] Received OCR_HANDWRITING telemetry payload.")
        print(f"[DEVELOPER DEBUG] Event ID: {event.get('event_id')}")

        print(f"[DEVELOPER DEBUG] User ID: {event.get('user_id')}")
        print(f"[DEVELOPER DEBUG] Question ID: {event.get('question_id')}")
        print(f"[DEVELOPER DEBUG] Input Node ID: {event.get('node_id')}")
        image_base64 = event.get("image_base64")
        print(f"[DEVELOPER DEBUG] Image uploaded: {'YES' if image_base64 else 'NO'} (Base64 length: {len(image_base64) if image_base64 else 0} chars)")
        print("="*80)

        logger.info(f"Processing OCR Handwriting telemetry payload for user: {event.get('user_id')}")
        user_id = event["user_id"]
        node_id = event.get("node_id")
        question_id = event.get("question_id")
        
        # Normalize and extract metrics
        telemetry_metrics = event.get("metrics", {})
        if not telemetry_metrics:
            # Extract standard metrics from root keys if they are at the top level
            telemetry_metrics = {
                "time_spent_sec": event.get("time_spent_sec", event.get("time_spent_seconds", 30)),
                "run_count": event.get("run_count", 0),
                "backspace_count": event.get("backspace_count", 0),
                "paste_char_count": event.get("paste_char_count", 0),
                "syntax_error_count": event.get("syntax_error_count", 0),
                "label": event.get("label", "NORMAL")
            }

        # 1. Decode base64 and extract raw text code
        from models.ocr_handler import ocr_handler
        try:
            print("[DEVELOPER DEBUG] Running OCR extraction pipeline...")
            extracted_text = ocr_handler.extract_code_from_image(image_base64)
            if extracted_text:
                print(f"[DEVELOPER DEBUG] OCR extraction status: SUCCESS (Length: {len(extracted_text)} characters)")
            else:
                print("[DEVELOPER DEBUG] OCR extraction status: FAILED (Empty string returned)")
        except Exception as ocr_err:
            error_msg = f"OCR raw extraction failed: {ocr_err}"
            print(f"[DEVELOPER DEBUG] OCR extraction status: FAILED. Reason: {ocr_err}")
            logger.error(error_msg)
            # Log failure details to MongoDB
            self.mongo_db["ocr_handwriting_evaluations"].insert_one({
                "user_id": user_id,
                "interaction_type": "OCR_HANDWRITING",
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "status": "FAILED_OCR",
                "error": error_msg,
                "telemetry_metrics": telemetry_metrics
            })
            return {"success": False, "error": error_msg}

        # 2. Retrieve Allowed node_ids list
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT node_id FROM concept_nodes;")
            allowed_node_ids = [row[0] for row in cur.fetchall()]

        # 3. Retrieve curriculum and question context
        question_context = {}
        if node_id:
            with self.pg_conn.cursor() as cur:
                cur.execute(
                    "SELECT node_id, concept_name, difficulty_baseline FROM concept_nodes WHERE node_id = %s", 
                    (node_id,)
                )
                row = cur.fetchone()
                if row:
                    question_context["node_id"] = row[0]
                    question_context["concept_name"] = row[1]
                    question_context["difficulty"] = float(row[2])

        if question_id:
            with self.pg_conn.cursor() as cur:
                cur.execute(
                    "SELECT id, question_text, difficulty_level FROM questions WHERE id = %s", 
                    (question_id,)
                )
                row = cur.fetchone()
                if row:
                    question_context["question_id"] = row[0]
                    question_context["question_text"] = row[1]
                    question_context["question_difficulty"] = float(row[2])

        # If no target node_id was explicitly provided, execute Hybrid Retrieval to find the most relevant concept node
        if not node_id and extracted_text:
            print("[DEVELOPER DEBUG] RAG system: No explicit Node ID provided. Running Hybrid RAG search...")
            print(f"[DEVELOPER DEBUG] RAG Query text (extracted from OCR):\n{extracted_text}")
            
            from rag.hybrid_searcher import hybrid_searcher
            search_results = hybrid_searcher.search(self.pg_conn, extracted_text, limit=1)
            if search_results:
                node_id = search_results[0]["node_id"]
                question_context["node_id"] = node_id
                question_context["concept_name"] = search_results[0]["concept_name"]
                print(f"[DEVELOPER DEBUG] RAG search resolved target concept: '{node_id}' (Concept: '{search_results[0]['concept_name']}')")
            else:
                node_id = "PY_SYNTAX_01"
                question_context["node_id"] = node_id
                question_context["concept_name"] = "Basic Indentation"
                print(f"[DEVELOPER DEBUG] RAG search yielded no matches. Falling back to default node ID: '{node_id}'")
        else:
            if not extracted_text:
                print("[DEVELOPER DEBUG] RAG system: Skipped RAG search (Extracted text is empty)")
            else:
                print(f"[DEVELOPER DEBUG] RAG system: Node ID already explicitly specified: '{node_id}'. Skipped hybrid search.")

        # 4. Perform LLM logical grading evaluation
        print("[DEVELOPER DEBUG] Invoking LLM logic grader...")
        language = event.get("language", "en")
        grade_result = ocr_handler.evaluate_logic_via_llm(
            extracted_text=extracted_text,
            question_context=question_context,
            telemetry_metrics=telemetry_metrics,
            allowed_node_ids=allowed_node_ids,
            language=language
        )

        is_correct = grade_result.get("is_correct", False)
        failed_node_id = grade_result.get("failed_node_id")
        explanation = grade_result.get("logical_flaw_explanation")

        # 5. Permanent Audit Log write to MongoDB
        audit_record = {
            "user_id": user_id,
            "interaction_type": "OCR_HANDWRITING",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "extracted_text": extracted_text,
            "question_context": question_context,
            "telemetry_metrics": telemetry_metrics,
            "is_correct": is_correct,
            "failed_node_id": failed_node_id,
            "logical_flaw_explanation": explanation
        }
        
        try:
            print("[DEVELOPER DEBUG] Logging audit record to MongoDB...")
            self.mongo_db["ocr_handwriting_evaluations"].insert_one(audit_record)
            print("[DEVELOPER DEBUG] MongoDB audit logging: SUCCESS")
            logger.info("OCR Handwriting logical grading audited to MongoDB ocr_handwriting_evaluations.")
        except Exception as audit_err:
            print(f"[DEVELOPER DEBUG] MongoDB audit logging: FAILED. Reason: {audit_err}")
            logger.error(f"Failed to write OCR audit record to MongoDB: {audit_err}")

        # 6. Trigger cognitive belief update inside Bayesian knowledge network
        from models.bayesian_network import update_bayesian_network
        try:
            print("[DEVELOPER DEBUG] Triggering Bayesian knowledge network belief update...")
            bn_result = update_bayesian_network(
                user_id=user_id,
                failed_node_id=failed_node_id,
                is_correct=is_correct,
                telemetry_metrics=telemetry_metrics,
                primary_node_id=node_id,
                mongo_db=self.mongo_db,
                pg_conn=self.pg_conn,
                r_client=self.r_client
            )
            print(f"[DEVELOPER DEBUG] Bayesian network update: SUCCESS. Result: {bn_result}")
            logger.info(f"Bayesian Network belief updated: {bn_result}")
        except Exception as bn_err:
            print(f"[DEVELOPER DEBUG] Bayesian network update: FAILED. Reason: {bn_err}")
            logger.error(f"Bayesian Network update failure: {bn_err}")
            bn_result = {"success": False, "error": str(bn_err)}

        print("\n" + "="*80)
        print("[DEVELOPER DEBUG] OCR Handwriting evaluation payload processing COMPLETE.")
        print("="*80 + "\n")

        return {
            "success": True,
            "extracted_text": extracted_text,
            "grade_result": grade_result,
            "bayesian_update": bn_result
        }

    def trigger_processing(self):
        """Triggers the on-demand queue processor thread if not already running."""
        with self.lock:
            if not self.is_processing:
                self.is_processing = True
                threading.Thread(target=self._process_queue_loop, daemon=True).start()
                logger.info("Triggered on-demand Redis queue processing thread.")

    def _process_queue_loop(self):
        """Pulls and processes events from Redis until the queue is completely empty."""
        logger.info(f"On-demand queue processing thread started. Reading queue '{config.TELEMETRY_QUEUE}'...")
        try:
            while True:
                # Non-blocking pop from Redis list
                packed = self.r_client.lpop(config.TELEMETRY_QUEUE)
                if not packed:
                    logger.info("Redis queue is empty. Terminating on-demand processor thread.")
                    break
                
                try:
                    event = json.loads(packed)
                    self.handle_telemetry_event(event)
                except Exception as e:
                    logger.error(f"Error handling telemetry event from queue: {e}")
        except Exception as e:
            logger.error(f"Error in on-demand queue processing loop: {e}")
        finally:
            with self.lock:
                self.is_processing = False

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
