import psycopg2
from pgvector.psycopg2 import register_vector
from src.config import settings
from src.utils.logger import setup_logger
from contextlib import contextmanager

# Initialize Enterprise Logger
logger = setup_logger("Database")

class DatabaseClient:
    """
    Enterprise Database Client for BlackBox AI.
    Handles connections, schema initialization, and vector storage.
    """
    
    def __init__(self):
        self.connection_params = {
            "user": settings.DB_USER,
            "password": settings.DB_PASSWORD,
            "host": settings.DB_HOST,
            "port": settings.DB_PORT,
            "dbname": settings.DB_NAME
        }

    @contextmanager
    def get_connection(self):
        """
        Context manager to ensure connections are returned to the pool 
        or closed correctly after use.
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.autocommit = True
            register_vector(conn) # Enable pgvector support for this session
            yield conn
        except Exception as e:
            logger.error(f"Database Connection Error: {e}")
            raise e
        finally:
            if conn:
                conn.close()

    def init_db(self):
        """
        Initializes the pgvector extension and creates the users table.
        Uses IF NOT EXISTS to ensure it is idempotent.
        """
        logger.info("Initializing database schema...")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 1. Enable Vector Extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # 2. Create Users Table (Enrollment)
                    # Vector dimension comes from Enterprise Settings
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS users (
                            id SERIAL PRIMARY KEY,
                            employee_id VARCHAR(50) UNIQUE NOT NULL,
                            full_name VARCHAR(100) NOT NULL,
                            voice_embedding vector({settings.AI_VECTOR_DIMENSION}),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # 3. Create Transcripts Table (Live Ingestion)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS transcripts (
                            id SERIAL PRIMARY KEY,
                            speaker_id INT REFERENCES users(id),
                            text_content TEXT NOT NULL,
                            embedding vector({settings.AI_VECTOR_DIMENSION}),
                            created_at TIMESTAMP NOT NULL
                        );
                    """)
            logger.info("Database schema initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize database: {e}")

    def add_user(self, name: str, emp_id: str, vector: list) -> bool:
        """
        Inserts a newly enrolled user into the database.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO users (full_name, employee_id, voice_embedding)
                        VALUES (%s, %s, %s)
                        RETURNING id;
                    """
                    cur.execute(query, (name, emp_id, vector))
                    user_id = cur.fetchone()[0]
                    logger.info(f"Successfully enrolled user: {name} | ID: {user_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to save user {name}: {e}")
            return False

# Export a singleton instance for use across the app
db_client = DatabaseClient()