import os
import io
import base64
import re
import json
from typing import List, Dict, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import requests
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

    def _score_extracted_text(self, text: str) -> int:
        """
        Calculates a Python-centric score for extracted text.
        Higher scores indicate a higher likelihood of valid Python code.
        """
        if not text:
            return 0
        score = 0
        keyword_count = 0
        
        PYTHON_MARKERS = [
            r'\bdef\b', r'\bclass\b', r'\bself\b', r'\breturn\b', r'\bimport\b',
            r'\bprint\b', r'\bfor\b', r'\bin\b', r'\bif\b', r'\belse\b', r'\binit\b',
            r'__init__', r'\(self\b', r'self\.\w+'
        ]
        
        for marker in PYTHON_MARKERS:
            matches = re.findall(marker, text, re.IGNORECASE)
            keyword_count += len(matches)
            
        if keyword_count == 0:
            return 0  # No python markers found, likely complete gibberish or noise
            
        score += keyword_count * 50
        
        # Add points for alphanumeric character count (normalized)
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        score += alphanumeric_count
        
        # Add points for specific python/code symbols
        score += text.count('=') * 5
        score += text.count(':') * 5
        score += text.count('(') * 3
        score += text.count(')') * 3
        
        return score

    def _clean_llm_ocr_output(self, text: str) -> str:
        """
        Removes reasoning blocks and markdown code block quotes from the LLM output.
        """
        if not text:
            return ""
        # Remove thinking/reasoning tags if any
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        text = text.strip()
        # Clean markdown code blocks
        if "```" in text:
            match = re.search(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)
            else:
                text = text.replace("```", "")
        return text.strip()

    def extract_code_from_image(self, base64_string: str) -> str:
        """
        Decodes a base64 image and extracts text.
        Primary: Groq Multimodal Vision OCR (qwen/qwen3.6-27b) for high-fidelity handwriting transcription.
        Fallback: Local Tesseract OCR auto-rotation engine for offline resiliency.
        """
        print("\n" + "="*80)
        print("[DEVELOPER DEBUG] Starting OCR Handwriting Text Extraction Pipeline")
        print("="*80)
        
        if not base64_string:
            print("[DEVELOPER DEBUG] Image uploaded: NO (Empty or missing payload)")
            print("[DEVELOPER DEBUG] OCR Pipeline Status: FAILED (Reason: Empty base64 string)")
            raise ValueError("Empty base64 string provided.")
            
        print(f"[DEVELOPER DEBUG] Image uploaded: YES (Base64 payload length: {len(base64_string)} chars)")
        
        # Strip data URL scheme prefix if present
        raw_base64 = base64_string
        if "," in raw_base64:
            raw_base64 = raw_base64.split(",", 1)[1]
            
        # Try Groq Vision OCR first
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            print("[DEVELOPER DEBUG] Stage 0: Attempting high-fidelity Vision OCR via Groq (qwen/qwen3.6-27b)...")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}"
            }
            payload = {
                "model": "qwen/qwen3.6-27b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Transcribe the handwritten Python code in this image. Return ONLY the raw Python code. Do not wrap it in markdown block quotes (no ```), do not include any introductions or explanations. Just return the pure code."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{raw_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            try:
                # 20 second timeout for API response
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=20
                )
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    cleaned_code = self._clean_llm_ocr_output(content)
                    if cleaned_code:
                        print("[DEVELOPER DEBUG] Groq Vision OCR Status: SUCCESS")
                        print(f"[DEVELOPER DEBUG] Cleaned Code Snippet Preview:\n{cleaned_code[:500]}\n" + "-"*30)
                        print("="*80)
                        return cleaned_code
                    else:
                        print("[DEVELOPER DEBUG] Groq Vision OCR returned empty text content.")
                else:
                    print(f"[DEVELOPER DEBUG] Groq Vision API returned error status {response.status_code}: {response.text}")
            except Exception as vision_ex:
                print(f"[DEVELOPER DEBUG] Groq Vision API call threw exception: {vision_ex}")
            
            print("[DEVELOPER DEBUG] Groq Vision OCR failed/unavailable. Falling back to local Tesseract OCR engine...")
        else:
            print("[DEVELOPER DEBUG] GROQ_API_KEY not found in environment. Skipping Vision OCR and running local Tesseract OCR...")

        # Fallback Local Tesseract OCR Pipeline
        try:
            # Decode base64 bytes
            image_bytes = base64.b64decode(raw_base64)
            print(f"[DEVELOPER DEBUG] Decoded image size: {len(image_bytes)} bytes")
            
            # Load into PIL image
            image = Image.open(io.BytesIO(image_bytes))
            print(f"[DEVELOPER DEBUG] PIL Image loaded successfully. Size: {image.size}")
            
            # Check if it's a tiny/mock image (e.g., from unit tests)
            w, h = image.size
            if w < 50 or h < 50:
                print("[DEVELOPER DEBUG] Detected tiny/mock image. Skipping auto-rotation detection pass.")
                custom_config = r'--oem 3 --psm 3'
                raw_text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
                cleaned_text = code_normalizer.clean_extracted_ocr(raw_text)
                print(f"[DEVELOPER DEBUG] OCR Pipeline Status: SUCCESS (Mock/Tiny image text extracted)")
                return cleaned_text

            print("[DEVELOPER DEBUG] Starting auto-rotation check (testing angles: 0, 90, 180, 270 degrees)...")
            best_angle = 0
            best_score = -1
            best_raw_text = ""
            
            for angle in [0, 90, 180, 270]:
                print(f"[DEVELOPER DEBUG] Testing image rotation: {angle} degrees...")
                rotated = image.rotate(angle, expand=True)
                gray = rotated.convert('L')
                w_r, h_r = gray.size
                resized = gray.resize((int(w_r * 1.5), int(h_r * 1.5)), Image.Resampling.LANCZOS)
                
                config_str = '--oem 3 --psm 3 --dpi 300'
                try:
                    txt = pytesseract.image_to_string(resized, config=config_str, lang='eng')
                    score = self._score_extracted_text(txt)
                    print(f"  -> Extracted text length: {len(txt.strip())} chars. Python score: {score}")
                    if score > best_score:
                        best_score = score
                        best_angle = angle
                        best_raw_text = txt
                except Exception as rot_ex:
                    print(f"  -> OCR at angle {angle} degrees threw exception: {rot_ex}")
                    
            print(f"[DEVELOPER DEBUG] Auto-rotation evaluation complete. Selected best angle: {best_angle} degrees (Score: {best_score})")
            
            final_text = ""
            if best_score == 0:
                print("[DEVELOPER DEBUG] Warning: Auto-rotation score is 0. Running fallback passes...")
                gray = image.convert('L')
                resized = gray.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
                contrast = ImageEnhance.Contrast(resized).enhance(3.0)
                sharpness = ImageEnhance.Sharpness(contrast).enhance(2.0)
                binarized = sharpness.point(lambda p: 255 if p > 130 else 0)
                try:
                    fallback_text_a = pytesseract.image_to_string(binarized, config='--oem 3 --psm 6 --dpi 300', lang='eng')
                    if fallback_text_a.strip():
                        final_text = fallback_text_a
                except Exception as e_a:
                    print(f"  -> Fallback Stage A failed: {e_a}")
                if not final_text.strip() and best_raw_text.strip():
                    final_text = best_raw_text
            else:
                final_text = best_raw_text
                
            cleaned_text = code_normalizer.clean_extracted_ocr(final_text)
            
            if cleaned_text.strip():
                print(f"[DEVELOPER DEBUG] OCR Pipeline SUCCESS! Extracted text length: {len(cleaned_text)} chars.")
                print(f"[DEVELOPER DEBUG] Cleaned Code Snippet Preview:\n{cleaned_text[:500]}\n" + "-"*30)
                print("="*80)
            else:
                print("[DEVELOPER DEBUG] OCR Pipeline FAILED: Text extraction yielded no valid characters.")
                print("="*80)
            return cleaned_text
            
        except Exception as e:
            print(f"[DEVELOPER DEBUG] OCR Pipeline EXCEPTION: {e}")
            print("="*80)
            logger.error(f"OCR Extraction failure: {e}")
            raise RuntimeError(f"OCR extraction failed: {str(e)}")

    def evaluate_logic_via_llm(
        self, 
        extracted_text: str, 
        question_context: dict, 
        telemetry_metrics: dict = None, 
        allowed_node_ids: List[str] = None,
        language: str = "en"
    ) -> dict:
        """
        Grades extracted student code against a question curriculum context using LLM reasoning.
        Injects allowed node IDs and student IDE telemetry factors for precise root-cause analysis.
        """
        print("\n" + "="*80)
        print("[DEVELOPER DEBUG] Starting LLM Logical Evaluation Grader")
        print("="*80)
        print(f"[DEVELOPER DEBUG] Extracted text length: {len(extracted_text) if extracted_text else 0} chars")
        print(f"[DEVELOPER DEBUG] Question Context: {question_context}")
        print(f"[DEVELOPER DEBUG] Telemetry Metrics: {telemetry_metrics}")
        
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

Determine if the student's code is logically correct. 

IMPORTANT NOTE ON OCR TRANSCRIPTION NOISE:
The code was extracted from a handwritten photo using a local OCR engine. Due to the nature of handwriting and OCR, minor typos, spelling substitutions, or syntax errors may exist that are artifacts of OCR rather than the student's work.
For example:
- 'de!' or 'del' instead of 'def'
- 'init' or 'ink' or 'tart' or '__init_' instead of '__init__'
- 'Céel}}', 'Self ree', or 'node)' instead of 'self'
- 'reKe' or 'make,' instead of 'make'
- 'moclal', 'moclel', or 'model ie' instead of 'model'
- 'col' or 'cat' instead of 'def'
- 'NOtuAN', 'KOtUAY', or 'AOTUAN' instead of 'return'
- 'Get descrsotian' or 'Gok descrsottanw' instead of 'get_description'

When evaluating:
1. Focus on the underlying code logic and structure. Reconstruct the student's likely intended code by correcting obvious OCR noise/transcription errors.
2. If the logic, structure, and intended code are correct, mark "is_correct": true.
3. Only mark "is_correct": false if there is a genuine logical, algorithmic, or structural flaw in their code (e.g., completely missing attributes, incorrect return logic, wrong function logic) that cannot be explained by OCR character recognition substitution.

If incorrect, pinpoint the SINGLE root-cause concept node_id (from the provided list of allowed node_ids) that represents the failure.
If the failure is syntactic/indents, choose a syntax-related node (e.g. PY_SYNTAX_01). If it's variable initialization, choose PY_SYNTAX_05. If it's a loop failure, choose PY_CONTROL_05 or PY_CONTROL_06.

You MUST respond with ONLY a valid JSON object matching this exact schema:
{{
  "is_correct": boolean,
  "logical_flaw_explanation": "A 1-sentence explanation of where the logic broke down (or null if correct).",
  "failed_node_id": "The exact node_id from the provided list that represents the root cause of the failure (or null if correct)."
}}
Do not include any Markdown wrappers like ```json or any introductory text. Just output raw JSON.
"""

        if language == "hi":
            system_prompt += "\nCRITICAL INSTRUCTION: Translate your entire explanation and feedback into natural, conversational Hindi (using Devanagari script). Keep technical terms (like 'Variable Scope' or 'For Loop') in English, but explain the concepts in Hindi.\n"

        user_prompt = f"""--- GRADED CURRICULUM NODE IDS ---
{allowed_nodes_formatted}

--- QUESTION CONTEXT ---
{question_context}

--- EXTRACTED STUDENT CODE ---
{extracted_text}

Evaluate and return the grading JSON:"""

        print("[DEVELOPER DEBUG] Connecting to LLM client...")
        try:
            print("[DEVELOPER DEBUG] Sending prompts to LLM (primary: Groq Llama 3.3, fallback: OpenAI gpt-4o-mini)...")
            result = llm_client.request_json(system_prompt, user_prompt)
            print("[DEVELOPER DEBUG] LLM Connection Status: SUCCESS")
            print("[DEVELOPER DEBUG] LLM Response received: YES")
            print(f"[DEVELOPER DEBUG] Raw LLM Response: {result}")
            
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
                    print(f"[DEVELOPER DEBUG] WARNING: Grader returned hallucinated node_id: '{failed_node_id}' which is not in curriculum. Resolving fallback...")
                    fallback_node = question_context.get("node_id") or "PY_SYNTAX_01"
                    # If the hallucinated node shares prefix, see if we can find it
                    resolved = False
                    for nid in allowed_node_ids:
                        if nid.lower() == failed_node_id.lower():
                            result["failed_node_id"] = nid
                            print(f"[DEVELOPER DEBUG] Resolved fuzzy match for hallucinated node_id: '{failed_node_id}' -> '{nid}'")
                            resolved = True
                            break
                    if not resolved:
                        result["failed_node_id"] = fallback_node
                        print(f"[DEVELOPER DEBUG] Fallback to primary node_id for hallucinated node: '{fallback_node}'")
                        
            print(f"[DEVELOPER DEBUG] Grader Output: is_correct={result['is_correct']}, failed_node_id={result['failed_node_id']}, explanation='{result['logical_flaw_explanation']}'")
            print("="*80)
            return result
        except Exception as e:
            print("[DEVELOPER DEBUG] LLM Connection Status: FAILED")
            print(f"[DEVELOPER DEBUG] LLM Grader failed: {e}")
            print("="*80)
            logger.error(f"LLM evaluation failed: {e}")
            # Safe grading fallback in case of LLM outage
            return {
                "is_correct": False,
                "logical_flaw_explanation": f"Evaluation system fallback: could not grade code due to error ({str(e)}).",
                "failed_node_id": question_context.get("node_id") or "PY_SYNTAX_01"
            }

ocr_handler = OCRHandwritingHandler()
