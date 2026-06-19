import os
import json
import time
import requests
from utils.logger import logger
import config

class MultiProviderLLMClient:
    """
    MNC-grade client wrapper for Groq and OpenAI APIs.
    Provides failover orchestration, automatic retries with exponential backoff,
    and strict JSON response format enforcement.
    """
    
    def __init__(self):
        # API Keys loaded via dotenv
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        
        # Primary models
        self.groq_model = "llama-3.3-70b-versatile"
        self.openai_model = "gpt-4o-mini"
        
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.openai_url = "https://api.openai.com/v1/chat/completions"

    def _clean_json_string(self, content: str) -> str:
        """
        Cleans markdown JSON code blocks from LLM output (e.g. ```json ... ```).
        """
        if not content:
            return ""
        content = content.strip()
        # Remove markdown code fence if present
        if content.startswith("```"):
            # Strip first line which might be ```json or ```
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return content

    def request_json(self, system_prompt: str, user_prompt: str, retries: int = 3, backoff_factor: float = 1.5) -> dict:
        """
        Submits request to Groq (primary) or OpenAI (fallback).
        Enforces JSON output format and handles connections gracefully.
        """
        # Formulate payload
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Attempt Groq first if available
        if self.groq_key:
            logger.info("Attempting logic evaluation via Groq API...")
            headers = {
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.groq_model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
            
            for attempt in range(retries):
                try:
                    response = requests.post(self.groq_url, headers=headers, json=payload, timeout=20)
                    if response.status_code == 200:
                        data = response.json()
                        raw_content = data["choices"][0]["message"]["content"]
                        cleaned_content = self._clean_json_string(raw_content)
                        try:
                            parsed_json = json.loads(cleaned_content)
                            logger.info("Successfully fetched and parsed response from Groq.")
                            return parsed_json
                        except json.JSONDecodeError as jde:
                            logger.error(f"Groq output was not valid JSON: {cleaned_content}. Error: {jde}")
                    else:
                        logger.warn(f"Groq API returned status {response.status_code}: {response.text}")
                except Exception as exc:
                    logger.error(f"Groq connection attempt {attempt + 1} failed: {exc}")
                
                # Sleep before retrying
                if attempt < retries - 1:
                    sleep_time = backoff_factor ** attempt
                    time.sleep(sleep_time)
                    
        # Fallback to OpenAI if Groq fails or is not configured
        if self.openai_key:
            logger.warn("Primary Groq provider failed or unconfigured. Falling back to OpenAI API...")
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.openai_model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
            
            for attempt in range(retries):
                try:
                    response = requests.post(self.openai_url, headers=headers, json=payload, timeout=20)
                    if response.status_code == 200:
                        data = response.json()
                        raw_content = data["choices"][0]["message"]["content"]
                        cleaned_content = self._clean_json_string(raw_content)
                        try:
                            parsed_json = json.loads(cleaned_content)
                            logger.info("Successfully fetched and parsed response from OpenAI.")
                            return parsed_json
                        except json.JSONDecodeError as jde:
                            logger.error(f"OpenAI output was not valid JSON: {cleaned_content}. Error: {jde}")
                    else:
                        logger.warn(f"OpenAI API returned status {response.status_code}: {response.text}")
                except Exception as exc:
                    logger.error(f"OpenAI connection attempt {attempt + 1} failed: {exc}")
                
                # Sleep before retrying
                if attempt < retries - 1:
                    sleep_time = backoff_factor ** attempt
                    time.sleep(sleep_time)
                    
        raise RuntimeError("Both primary (Groq) and fallback (OpenAI) LLM services failed to produce a valid response.")

llm_client = MultiProviderLLMClient()
