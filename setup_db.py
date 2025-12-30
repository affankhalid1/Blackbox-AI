from src.database import db_client # Use the singleton instance
from src.utils.logger import setup_logger
import sys

logger = setup_logger("DB_Setup")

def run_setup():
    logger.info("Starting Enterprise Database Setup...")
    
    try:
        # 1. This uses the internal logic to create tables
        db_client.init_db()
        
        # 2. Manual verification using the Context Manager
        logger.info("Verifying pgvector extension...")
        with db_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                result = cur.fetchone()
                if result:
                    logger.info("Extension 'pgvector' is ACTIVE.")
                else:
                    logger.error("Extension 'pgvector' failed to activate.")
        
        logger.info("Database Setup Successfully Completed!")

    except Exception as e:
        logger.critical(f"Setup Failed! Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_setup()