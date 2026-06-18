import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from database.db_connector import db_connector
from jobs_queue.job_consumer import TelemetryJobConsumer
from utils.logger import logger

# Global consumer instance
consumer = None

class TelemetryHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to suppress default HTTP access logs on stdout (keeps console clean)
        logger.info(f"{self.address_string()} - - {format % args}")

    def do_POST(self):
        global consumer
        if self.path == "/process-telemetry":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                event = json.loads(post_data.decode('utf-8'))
                
                # Relocate/Process the event using the consumer logic
                result = {"success": True}
                if consumer:
                    result = consumer.handle_telemetry_event(event) or {"success": True}
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                logger.error(f"Error processing HTTP telemetry event: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        elif self.path == "/trigger-process-queue":
            try:
                if consumer:
                    consumer.trigger_processing()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "message": "Queue processing triggered"}).encode('utf-8'))
            except Exception as e:
                logger.error(f"Error triggering queue processing: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def wait_for_connections(retries=10, delay=5):
    """Verifies all datastores are ready before starting the worker thread."""
    logger.info("Initializing datastore handshakes...")
    for i in range(retries):
        try:
            db_connector.connect_redis()
            db_connector.connect_postgres()
            db_connector.connect_mongo()
            logger.info("Connection handshakes successful for Postgres, MongoDB, and Redis.")
            return True
        except Exception as e:
            logger.warn(f"Connection attempt {i+1}/{retries} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return False

def main():
    global consumer
    if not wait_for_connections():
        logger.error("Fatal: Could not establish datastore connections. Shutting down.")
        return
        
    consumer = TelemetryJobConsumer()
    
    port = 5000
    server_address = ('', port)
    httpd = HTTPServer(server_address, TelemetryHTTPHandler)
    logger.info(f"Python Math Engine HTTP Server listening on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down HTTP server.")
    finally:
        db_connector.close_all()

if __name__ == "__main__":
    main()
