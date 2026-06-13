import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient
import config

class DatabaseConnector:
    """Manages connections to Redis, PostgreSQL, and MongoDB."""
    
    def __init__(self):
        self.redis_client = None
        self.pg_connection = None
        self.mongo_client = None
        self.mongo_db = None

    def connect_redis(self):
        if not self.redis_client:
            url = config.REDIS_URL
            kwargs = {}
            if url and url.startswith("redis://"):
                url = url.replace("redis://", "rediss://", 1)
                kwargs["ssl_cert_reqs"] = None
            elif url and url.startswith("rediss://"):
                kwargs["ssl_cert_reqs"] = None
            self.redis_client = redis.from_url(url, **kwargs)
        return self.redis_client

    def connect_postgres(self):
        if not self.pg_connection or self.pg_connection.closed:
            kwargs = {
                "host": config.PG_HOST,
                "port": config.PG_PORT,
                "user": config.PG_USER,
                "password": config.PG_PASSWORD,
                "dbname": config.PG_DATABASE
            }
            if config.PG_SSL:
                kwargs["sslmode"] = "require"
            self.pg_connection = psycopg2.connect(**kwargs)
        return self.pg_connection

    def connect_mongo(self):
        if not self.mongo_client:
            self.mongo_client = MongoClient(config.MONGO_URI)
            self.mongo_db = self.mongo_client.get_default_database()
        return self.mongo_db

    def close_all(self):
        if self.pg_connection and not self.pg_connection.closed:
            self.pg_connection.close()
        if self.mongo_client:
            self.mongo_client.close()
            
db_connector = DatabaseConnector()
