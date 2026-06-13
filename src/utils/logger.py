import os
import re
import json
from datetime import datetime

# Local logging paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(os.path.dirname(base_dir), 'logs')
log_file_path = os.path.join(logs_dir, 'app.log')

# Ensure local logs directory exists if running locally
if os.environ.get("ENV_MODE") != "production":
    os.makedirs(logs_dir, exist_ok=True)

# Regex to find sensitive credentials in strings
SENSITIVE_PATTERNS = [
    r"mongodb\+srv://[^@]+@[^\s/]+",
    r"redis://([^@]+)@[^\s/]+",
    r"pass(word)?\s*=\s*[^\s;&]+",
    r"\"password\"\s*:\s*\"[^\"]*\""
]

def sanitize(message):
    message_str = message if isinstance(message, str) else json.dumps(message)
    sanitized = message_str
    
    for pattern in SENSITIVE_PATTERNS:
        def replacement(match):
            m = match.group(0)
            if m.startswith('mongodb+srv:'):
                return 'mongodb+srv://[REDACTED_CREDENTIALS]@host'
            if m.startswith('redis:'):
                return 'redis://default:[REDACTED_TOKEN]@host'
            if 'password' in m.lower():
                return 'password=[REDACTED]'
            return '[REDACTED]'
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized

def format_log(level, message, context=None):
    timestamp = datetime.utcnow().isoformat() + 'Z'
    sanitized_msg = sanitize(message)
    context_str = f" | Context: {sanitize(json.dumps(context))}" if context else ""
    return f"[{timestamp}] [{level.upper()}] {sanitized_msg}{context_str}"

class Logger:
    def info(self, message, context=None):
        formatted = format_log("INFO", message, context)
        print(formatted, flush=True)
        self._write_to_local_file(formatted)

    def warn(self, message, context=None):
        formatted = format_log("WARN", message, context)
        print(formatted, flush=True)
        self._write_to_local_file(formatted)

    def error(self, message, context=None):
        formatted = format_log("ERROR", message, context)
        print(formatted, flush=True)
        self._write_to_local_file(formatted)

    def _write_to_local_file(self, formatted_line):
        if os.environ.get("ENV_MODE") != "production":
            try:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(formatted_line + '\n')
            except Exception as e:
                # Fallback print to prevent app crash if disk is not writable
                print(f"[Logger] Failed to write to local log file: {e}", flush=True)

logger = Logger()
