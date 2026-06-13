import os
from dotenv import load_dotenv
from pathlib import Path
root_dir = Path(__file__).resolve().parents[3]
env_path = root_dir / "ENV" / ".env"
load_dotenv(dotenv_path=env_path)

# Queue Settings
REDIS_URL = os.environ.get("REDIS_URL")
TELEMETRY_QUEUE = os.environ.get("TELEMETRY_QUEUE", "telemetry_queue")

# Relational Database Settings
PG_HOST = os.environ.get("PG_HOST")
PG_PORT = int(os.environ.get("PG_PORT", 5432)) if os.environ.get("PG_PORT") else None
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD")
PG_DATABASE = os.environ.get("PG_DATABASE")
PG_SSL = os.environ.get("PG_SSL", "false").lower() == "true"

# Document Event Sourcing Database Settings
MONGO_URI = os.environ.get("MONGO_URI")

# Cognitive Engine Settings
DEFAULT_DECAY_RATE = 0.02
DEFAULT_GAMMA = 0.5
