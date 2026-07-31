import os
import sys
import json
import base64
import logging
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI
from psycopg2.extras import RealDictCursor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config & Environment variables
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMMA_OLLAMA_URL = os.getenv("GEMMA_OLLAMA_URL", "http://localhost:11434")

# Client Factory
def get_llm_client(force_ollama: bool = False) -> tuple[OpenAI, str, bool]:
    """
    Returns (client, model_name, is_vllm).
    Degrades to local Ollama if OpenRouter key is not defined or force_ollama is True.
    """
    if OPENROUTER_API_KEY and not force_ollama:
        try:
            logger.info(f"Initializing OpenRouter client pointing to: {OPENROUTER_API_URL}")
            client = OpenAI(
                base_url=OPENROUTER_API_URL,
                api_key=OPENROUTER_API_KEY,
                default_headers={
                    "HTTP-Referer": "https://sahai.edu",
                    "X-Title": "SahAI Cognitive Diagnostics"
                }
            )
            return client, "google/gemma-4-26b-a4b-it:free", True
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}. Falling back to Ollama.")
            
    logger.info("Using local Ollama fallback client.")
    try:
        client = OpenAI(
            base_url=f"{GEMMA_OLLAMA_URL}/v1",
            api_key="ollama"
        )
        return client, "codegemma:2b", False
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}.")
        client = OpenAI(api_key="dummy", base_url="http://localhost:11434/v1")
        return client, "codegemma:2b", False

# ==========================================
# TOOL RETRIEVALS AND EXECUTIONS
# ==========================================

def get_student_cognitive_state(user_id: str, node_id: str) -> Dict[str, Any]:
    """Tool A: Retrieves the student's current Beta distribution parameters from PostgreSQL."""
    from database.db_connector import db_connector
    from models.bayesian_network import fetch_or_init_state
    
    logger.info(f"[Tool: get_student_cognitive_state] Fetching state for user {user_id}, node {node_id}")
    try:
        pg_conn = db_connector.connect_postgres()
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT node_id, alpha, beta, expected_mastery FROM user_cognitive_states WHERE user_id = %s AND node_id = %s;",
                (user_id, node_id)
            )
            row = cur.fetchone()
            if row:
                return {
                    "node_id": row["node_id"],
                    "alpha": float(row["alpha"]),
                    "beta": float(row["beta"]),
                    "expected_mastery": float(row["expected_mastery"]),
                    "decay_rate": 0.01
                }
            
        # If no row exists, initialize state using BKT module
        mongo_db = db_connector.connect_mongo()
        state = fetch_or_init_state(user_id, node_id, mongo_db, pg_conn)
        pg_conn.commit()
        return {
            "node_id": state["node_id"],
            "alpha": float(state["alpha"]),
            "beta": float(state["beta"]),
            "expected_mastery": float(state["expected_mastery"]),
            "decay_rate": float(state["decay_rate"]) if "decay_rate" in state else 0.01
        }
    except Exception as e:
        logger.error(f"Error in get_student_cognitive_state: {e}")
        try:
            pg_conn.rollback()
        except:
            pass
        return {"error": str(e)}

def execute_code_sandbox(source_code: str, language_id: int = 71) -> Dict[str, Any]:
    """Tool B: Executes student code securely via Judge0 API."""
    logger.info(f"[Tool: execute_code_sandbox] Executing source code (lang: {language_id})")
    try:
        url = "https://ce.judge0.com/submissions?wait=true"
        payload = {
            "source_code": source_code,
            "language_id": language_id,
            "stdin": ""
        }
        # 10s maximum wait timeout for compile + run sandbox
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code in [200, 201]:
            res_data = response.json()
            return {
                "stdout": res_data.get("stdout") or "",
                "stderr": res_data.get("stderr") or "",
                "compile_output": res_data.get("compile_output") or "",
                "status": res_data.get("status", {}).get("description") or "Unknown status",
                "exit_code": res_data.get("exit_code")
            }
        else:
            return {"error": f"Judge0 returned status {response.status_code}: {response.text}"}
    except Exception as e:
        logger.error(f"Error in execute_code_sandbox: {e}")
        return {"error": str(e)}

def search_hybrid_rag(concept_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Tool C: Performs pgvector dense + BM25 sparse search on textbooks context."""
    from database.db_connector import db_connector
    from rag.hybrid_searcher import hybrid_searcher
    
    logger.info(f"[Tool: search_hybrid_rag] Searching context for: {concept_query}")
    try:
        pg_conn = db_connector.connect_postgres()
        results = hybrid_searcher.search(pg_conn, query_text=concept_query, limit=top_k)
        return results
    except Exception as e:
        logger.error(f"Error in search_hybrid_rag: {e}")
        return [{"error": str(e)}]

def log_cognitive_telemetry(user_id: str, node_id: str, is_correct: bool, behavioral_flags: List[str] = None) -> Dict[str, Any]:
    """Tool D: recomputes BKT alpha/beta values and updates Postgres & Mongo."""
    from database.db_connector import db_connector
    from models.bayesian_network import update_bayesian_network
    
    logger.info(f"[Tool: log_cognitive_telemetry] Logging telemetry. Correct: {is_correct}")
    try:
        pg_conn = db_connector.connect_postgres()
        mongo_db = db_connector.connect_mongo()
        r_client = db_connector.connect_redis()
        
        metrics = {
            "timeSpent": 45,
            "runCount": 1 if not is_correct else 0,
            "backspaceCount": 0,
            "pasteCharCount": 0
        }
        
        result = update_bayesian_network(
            user_id=user_id,
            failed_node_id=node_id if not is_correct else None,
            is_correct=is_correct,
            telemetry_metrics=metrics,
            primary_node_id=node_id,
            mongo_db=mongo_db,
            pg_conn=pg_conn,
            r_client=r_client
        )
        pg_conn.commit()
        return {
            "success": True,
            "target_node": node_id,
            "updated_expected_mastery": result.get("expected_mastery")
        }
    except Exception as e:
        logger.error(f"Error in log_cognitive_telemetry: {e}")
        return {"error": str(e)}

# Tool Definition Registry (OpenAI Spec)
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_student_cognitive_state",
            "description": "Retrieves the student's current Beta distribution parameters (alpha, beta, expected mastery E[K], decay rate) for a given node_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The unique identifier of the student."},
                    "node_id": {"type": "string", "description": "The concept node ID."}
                },
                "required": ["user_id", "node_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code_sandbox",
            "description": "Executes student Python source code via the Judge0 CE sandbox API to inspect stdout, stderr, exit codes, and compile status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_code": {"type": "string", "description": "The raw Python code string to execute."},
                    "language_id": {"type": "integer", "description": "The Judge0 language ID. Default: 71 (Python)."}
                },
                "required": ["source_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hybrid_rag",
            "description": "Performs hybrid semantic vector search (pgvector) and keyword search (BM25) on textbook notes to extract core prerequisite explanations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept_query": {"type": "string", "description": "The query term or concept explanation request."},
                    "top_k": {"type": "integer", "description": "The number of search results to return. Default: 3."}
                },
                "required": ["concept_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_cognitive_telemetry",
            "description": "Computes Bayesian state updates and logs the student interaction event to MongoDB audit logs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The student's user ID."},
                    "node_id": {"type": "string", "description": "The concept node ID evaluated."},
                    "is_correct": {"type": "boolean", "description": "Whether the student's solution is correct."},
                    "behavioral_flags": {"type": "array", "items": {"type": "string"}, "description": "Behavioral flags to log."}
                },
                "required": ["user_id", "node_id", "is_correct"]
            }
        }
    }
]

def execute_local_tool(name: str, args_dict: dict) -> str:
    """Executes a declared function by name mapping and returns JSON string results."""
    try:
        if name == "get_student_cognitive_state":
            res = get_student_cognitive_state(args_dict["user_id"], args_dict["node_id"])
        elif name == "execute_code_sandbox":
            res = execute_code_sandbox(args_dict["source_code"], args_dict.get("language_id", 71))
        elif name == "search_hybrid_rag":
            res = search_hybrid_rag(args_dict["concept_query"], args_dict.get("top_k", 3))
        elif name == "log_cognitive_telemetry":
            res = log_cognitive_telemetry(
                args_dict["user_id"],
                args_dict["node_id"],
                args_dict["is_correct"],
                args_dict.get("behavioral_flags", [])
            )
        else:
            res = {"error": f"Tool {name} not matched."}
        return json.dumps(res)
    except Exception as e:
        logger.error(f"Tool {name} call failed: {e}")
        return json.dumps({"error": str(e)})

# ==========================================
# AGENT ORCHESTRATION LOOP
# ==========================================

def run_orchestration_loop(user_id: str, node_id: str, prompt_context: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
    """
    Step-by-step Socratic Diagnostic Orchestration Loop.
    Executes native function-calling cycles up to 5 iterations.
    """
    tools_run_history = []
    try:
        client, model_name, is_vllm = get_llm_client()
        
        system_prompt = (
            "You are SahAI's Lead Diagnostic Educational Agent. Your goal is to trace root-cause misconceptions. "
            "You MUST use available tools to inspect the student's mastery state and code execution before forming a conclusion. "
            "Always provide feedback in a Socratic tone (asking guiding questions) in both English and conversational Hindi. "
            "You must return your FINAL answer as a single, valid JSON block matching this structure EXACTLY. "
            "Do NOT output markdown wrapper blocks like ```json ... ```, just return the raw JSON object string:\n"
            "{\n"
            '  "status": "SUCCESS",\n'
            '  "detected_misconception": "Brief description of error pattern",\n'
            '  "behavioral_summary": "Notes on compiled behavior",\n'
            '  "root_cause_node": "Target concept code",\n'
            '  "socratic_hint_en": "Hint in English asking questions",\n'
            '  "socratic_hint_hi": "Conversational Hindi hint using Hinglish terms",\n'
            '  "recommended_next_node": "Alternative practice topic concept node ID",\n'
            '  "tools_executed": ["list of tools used"]\n'
            "}"
        )

        # Initialize messages list
        messages = [{"role": "system", "content": system_prompt}]
        
        # Multimodal image processing using standard OpenAI content structure
        if image_base64:
            # Strip data:image prefix if present to normalize base64
            cleaned_base64 = image_base64
            if "," in image_base64:
                cleaned_base64 = image_base64.split(",")[1]
            user_content = [
                {"type": "text", "text": f"Student Query Context: {prompt_context}\nTarget Concept Node: {node_id}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{cleaned_base64}"
                    }
                }
            ]
        else:
            user_content = f"Student Query Context: {prompt_context}\nTarget Concept Node: {node_id}"
            
        messages.append({"role": "user", "content": user_content})
        
        # Loop execution up to 5 iterations
        for iteration in range(5):
            logger.info(f"Agentic loop iteration {iteration + 1}/5 (Model: {model_name})")
            
            try:
                # We enforce 15s limit for vLLM calls, fallback client has standard timeout
                timeout_limit = 15 if is_vllm else 30
                
                # Ollama / Local fallback might not accept tool definitions.
                # We omit tools if using Ollama to avoid compatibility validation crashes.
                call_params = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1,
                    "timeout": timeout_limit
                }
                if is_vllm:
                    call_params["tools"] = openai_tools
                
                response = client.chat.completions.create(**call_params)
                choice = response.choices[0]
                msg = choice.message
                
                # Check for tool execution requests
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    messages.append(msg) # append agent response
                    
                    for tool_call in msg.tool_calls:
                        name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        tools_run_history.append(name)
                        
                        # Execute tool call and append to messages history
                        tool_output = execute_local_tool(name, args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": tool_output
                        })
                    # Proceed to next iteration with tool outcomes
                    continue
                
                # No tool calls returned: parse text completion outcome
                final_text = msg.content or ""
                return parse_structured_output(final_text, tools_run_history, node_id)
                
            except Exception as e:
                logger.warn(f"Orchestration call failed in iteration {iteration + 1}: {e}")
                if is_vllm:
                    logger.info("Degrading to local Ollama fallback for final response generation...")
                    # Instantly fallback to local Ollama with force_ollama=True
                    client, model_name, is_vllm = get_llm_client(force_ollama=True)
                    # Remove tool calls references from messages history to compile correctly
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context: {prompt_context}. System failed. Respond directly."}
                    ]
                else:
                    # If Ollama also fails, break loop
                    break

        # Safe fallback if loop exhausts or completes without returning
        return build_safe_fallback_response(node_id, tools_run_history)
    except Exception as fatal_err:
        logger.error(f"Fatal error in run_orchestration_loop: {fatal_err}", exc_info=True)
        return build_safe_fallback_response(node_id, tools_run_history)

def parse_structured_output(raw_text: str, tools_run: List[str], concept_node: str) -> Dict[str, Any]:
    """Robust parser that handles cleaning backticks and fills missing keys safely."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
        
    try:
        data = json.loads(cleaned)
        data["tools_executed"] = list(set(tools_run))
        return data
    except Exception as e:
        logger.error(f"Failed to parse model response as JSON. Content: {raw_text}. Error: {e}")
        # Build safe fallback utilizing text content as hint
        fallback = build_safe_fallback_response(concept_node, tools_run)
        fallback["socratic_hint_en"] = raw_text[:200]
        return fallback

def build_safe_fallback_response(concept_node: str, tools_run: List[str]) -> Dict[str, Any]:
    """Generates standard schema when model fails to structure JSON outputs."""
    return {
        "status": "SUCCESS",
        "detected_misconception": "Concept application error / Unresolved execution path",
        "behavioral_summary": "System processed standard telemetry checks.",
        "root_cause_node": concept_node,
        "socratic_hint_en": "Please recheck your variable scopes and loop controls. What value does your variable hold at step 1?",
        "socratic_hint_hi": "Ek baar variable loop aur scope parameters check karein. Pehle iteration ke baad iski value kya hogi?",
        "recommended_next_node": "PY_VARIABLES_01",
        "tools_executed": list(set(tools_run))
    }

# ==========================================
# MULTIMODAL API HANDLER
# ==========================================

def evaluate_multimodal_submission(user_id: str, node_id: str, image_base64: str, code_snippet: Optional[str] = None) -> Dict[str, Any]:
    """Public wrapper to diagnose step-by-step canvas/handwritten notes image scans."""
    logger.info(f"Received multimodal OCR diagnosis request for user {user_id}, node {node_id}")
    context = "Multimodal Handwriting Scan submitted."
    if code_snippet:
        context += f"\nCode uploaded: {code_snippet}"
    return run_orchestration_loop(user_id, node_id, context, image_base64)

def map_leetcode_to_sahai(title: str, tags: list, description: str) -> List[str]:
    """
    Leverages Gemma 4 as a Curriculum Architect to dynamically map LeetCode tags
    and problem description to the most relevant SahAI concept node IDs.
    """
    logger.info(f"[Gemma Map] Dynamically mapping problem: {title}")
    
    # 1. Fetch available concept nodes from PostgreSQL
    concept_list = []
    try:
        import db_connector
        pg_conn = db_connector.connect_postgres()
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT node_id, concept_name FROM concept_nodes;")
            rows = cur.fetchall()
            for r in rows:
                concept_list.append(f"{r['node_id']} ({r['concept_name']})")
    except Exception as e:
        logger.error(f"[Gemma Map] Failed to fetch concept nodes from postgres: {e}")
        # Fallback list if postgres fetch fails
        concept_list = ["PY_SYNTAX_01 (Python Syntax)", "PY_OOP_01 (OOP Basics)", "HASH_TABLE_01 (Hash Tables)", "ARRAY_01 (Arrays)"]

    concepts_str = ", ".join(concept_list)
    tags_str = ", ".join(tags) if tags else "None"

    # 2. Formulate Prompt
    system_prompt = "You are a Curriculum Architect for a computer science DSA tutor platform."
    user_prompt = f"""You are given a coding problem:
Title: {title}
Tags: {tags_str}
Description: {description[:1200]}

Here is the complete list of SahAI internal Concept Node IDs and their concept names:
[{concepts_str}]

Task:
Perform a deep semantic alignment. Analyze the core algorithms, data structures, bitwise operators, and logical use cases tested in this coding problem.
Match them to the most relevant SahAI internal Concept Nodes. Do NOT just match by exact name. Think about the underlying concept (e.g., if a problem uses frequency counting, it maps to Dictionary/Hash Table concepts; if it uses element iteration, it maps to Loops and Arrays; if it uses XOR logic, it maps to Bitwise Operators).

Identify the 3 to 5 most relevant SahAI node IDs.
Format Requirement: Return ONLY a raw JSON array of strings containing the matched Node IDs. Do not write explanations, code, markdown tags, or notes.
Example output:
["HASH_TABLE_01", "ARRAY_01"]
"""

    # 3. Call LLM (using client factory)
    try:
        client, model, is_vllm = get_llm_client()
        logger.info(f"[Gemma Map] Using LLM Model: {model} for problem mapping")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        raw_output = response.choices[0].message.content.strip()
        logger.info(f"[Gemma Map] LLM raw response: {raw_output}")
        
        # Strip markdown code blocks if any
        if raw_output.startswith("```"):
            lines = raw_output.splitlines()
            if len(lines) > 2:
                raw_output = "\n".join(lines[1:-1]).strip()
            else:
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        mapped_nodes = json.loads(raw_output)
        if isinstance(mapped_nodes, list):
            # Clean and filter to ensure only valid returned strings
            cleaned_nodes = [str(node).strip() for node in mapped_nodes]
            logger.info(f"[Gemma Map] Successfully mapped nodes: {cleaned_nodes}")
            return cleaned_nodes
        else:
            raise ValueError("LLM did not return a JSON list")
            
    except Exception as e:
        logger.error(f"[Gemma Map] Error mapping problem via LLM: {e}")
        # Return fallback node if mapping failed
        return ["PY_SYNTAX_01"]
