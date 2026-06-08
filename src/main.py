import time
from database.db_connector import db_connector
from jobs_queue.job_consumer import TelemetryJobConsumer

def wait_for_connections(retries=10, delay=5):
    """Verifies all datastores are ready before starting the worker thread."""
    print("[Main] Initializing datastore handshakes...")
    for i in range(retries):
        try:
            db_connector.connect_redis()
            db_connector.connect_postgres()
            db_connector.connect_mongo()
            print("[Main] Connection handshakes successful for Postgres, MongoDB, and Redis.")
            return True
        except Exception as e:
            print(f"[Main] Connection attempt {i+1}/{retries} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return False

def main():
    if not wait_for_connections():
        print("[Main] Fatal: Could not establish datastore connections. Shutting down.")
        return
        
    consumer = TelemetryJobConsumer()
    consumer.listen()

if __name__ == "__main__":
    main()
