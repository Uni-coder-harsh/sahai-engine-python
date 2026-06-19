import os
import io
import base64
import re
from typing import List, Dict, Tuple
from PIL import Image
import pytesseract
from utils.logger import logger
from utils.llm_client import llm_client
from rag.normalizer import code_normalizer

class OCRHandwritingHandler:
    """
    Orchestrates the conversion of base64 handwritten images into code,
    performs logical evaluation using LLMs, maps results to curriculum node IDs,
    and returns grading metrics.
    """
    
    def __init__(self):
        pass

    def extract_code_from_image(self, base64_string: str) -> str:
        """
        Decodes a base64 image and extracts text using Tesseract OCR.
        Removes base64 prefixes if present.
        """
        logger.info("Executing extract_code_from_image OCR routine...")
        if not base64_string:
            raise ValueError("Empty base64 string provided.")
            
        try:
            # 1. Strip data URL scheme prefix if present (e.g. "data:image/png;base64,")
            if "," in base64_string:
                base64_string = base64_string.split(",", 1)[1]
                
            # 2. Decode base64 bytes
            image_bytes = base64.b64decode(base64_string)
            
            # 3. Load into PIL image
            image = Image.open(io.BytesIO(image_bytes))
            
            # 4. Perform OCR text extraction
            # Hinting Tesseract for generic english code parsing
            raw_text = pytesseract.image_to_string(image, lang='eng')
            
            # 5. Clean OCR noise and normalize layout
            cleaned_text = code_normalizer.clean_extracted_ocr(raw_text)
            logger.info(f"Successfully extracted {len(cleaned_text)} characters of text from image.")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"OCR Extraction failure: {e}")
            raise RuntimeError(f"OCR extraction failed: {str(e)}")

    def evaluate_logic_via_llm(
        self, 
        extracted_text: str, 
        question_context: dict, 
        telemetry_metrics: dict = None, 
        allowed_node_ids: List[str] = None
    ) -> dict:
        """
        Grades extracted student code against a question curriculum context using LLM reasoning.
        Injects allowed node IDs and student IDE telemetry factors for precise root-cause analysis.
        """
        logger.info("Evaluating student code logic via LLM grader...")
        if allowed_node_ids is None:
            allowed_node_ids = []
            
        metrics = telemetry_metrics or {}
        time_spent = metrics.get("time_spent_sec", metrics.get("time_spent_seconds", 30))
        run_count = metrics.get("run_count", 0)
        backspace_count = metrics.get("backspace_count", 0)
        paste_char_count = metrics.get("paste_char_count", 0)
        syntax_error_count = metrics.get("syntax_error_count", 0)
        label = metrics.get("label", "NORMAL")

        # Format node list for prompt context
        allowed_nodes_formatted = "\n".join([f"- {nid}" for nid in allowed_node_ids])

        system_prompt = f"""You are a strict, top-tier Computer Science professor grading student handwritten Python code submissions.
You will evaluate the student's code logic against the provided question description, difficulty level, and topic.

In addition to the code itself, you MUST take into account the student's IDE telemetry indicators to identify their cognitive difficulties:
- Time Spent on Question: {time_spent} seconds. (Very low time indicates guessing or copying; very high time indicates systemic struggling).
- Run Count: {run_count} runs. (High run count with short time suggests blind trial-and-error shotgun debugging).
- Backspace Count: {backspace_count} keystrokes.
- Paste Character Count: {paste_char_count} characters. (High paste character count with low backspace suggests copy-paste dependency).
- Syntax Error Count: {syntax_error_count} errors. (High syntax error count suggests syntactic hesitation or procedural fatigue).
- Mode Label: {label}.

Determine if the student's code is logically correct. If incorrect, pinpoint the SINGLE root-cause concept node_id (from the provided list of allowed node_ids) that represents the failure.
If the failure is syntactic/indents, choose a syntax-related node (e.g. PY_SYNTAX_01). If it's variable initialization, choose PY_SYNTAX_05. If it's a loop failure, choose PY_CONTROL_05 or PY_CONTROL_06.

You MUST respond with ONLY a valid JSON object matching this exact schema:
{{
  "is_correct": boolean,
  "logical_flaw_explanation": "A 1-sentence explanation of where the logic broke down (or null if correct).",
  "failed_node_id": "The exact node_id from the provided list that represents the root cause of the failure (or null if correct)."
}}
Do not include any Markdown wrappers like ```json or any introductory text. Just output raw JSON.
"""

        user_prompt = f"""--- GRADED CURRICULUM NODE IDS ---
{allowed_nodes_formatted}

--- QUESTION CONTEXT ---
{question_context}

--- EXTRACTED STUDENT CODE ---
{extracted_text}

Evaluate and return the grading JSON:"""

        try:
            result = llm_client.request_json(system_prompt, user_prompt)
            
            # Post-validate the result
            is_correct = result.get("is_correct", True)
            failed_node_id = result.get("failed_node_id")
            explanation = result.get("logical_flaw_explanation")
            
            if is_correct:
                result["failed_node_id"] = None
                result["logical_flaw_explanation"] = None
            else:
                # Resolve hallucinated node_id (if not in the allowed list)
                if failed_node_id and failed_node_id not in allowed_node_ids:
                    logger.warn(f"LLM hallucinated node_id: '{failed_node_id}' which is not in curriculum. Resolving fallback...")
                    # Try fuzzy matching or fallback to primary node_id
                    fallback_node = question_context.get("node_id") or "PY_SYNTAX_01"
                    # If the hallucinated node shares prefix, see if we can find it
                    resolved = False
                    for nid in allowed_node_ids:
                        if nid.lower() == failed_node_id.lower():
                            result["failed_node_id"] = nid
                            resolved = True
                            break
                    if not resolved:
                        result["failed_node_id"] = fallback_node
                        
            return result
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Safe grading fallback in case of LLM outage
            return {
                "is_correct": False,
                "logical_flaw_explanation": f"Evaluation system fallback: could not grade code due to error ({str(e)}).",
                "failed_node_id": question_context.get("node_id") or "PY_SYNTAX_01"
            }

ocr_handler = OCRHandwritingHandler()
