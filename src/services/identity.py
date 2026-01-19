import os
from src.engine.speaker import speaker_engine
from src.database import db_client
from src.utils.logger import setup_logger

logger = setup_logger("Identity_Service")

# Global Security Threshold
AUTH_THRESHOLD = 0.35 

def identify_user(wav_path: str):
    """
    Takes an audio file path.
    Returns a dictionary: {"id": int, "name": str, "confidence": float} 
    or None if no match is found.
    """
    try:
        # 1. Extract Vector (The "Who are you?" part)
        vector = speaker_engine.get_voice_embedding(wav_path)
        
        # 2. Query Database (The "Do we know you?" part)
        # Using the cosine distance operator (<=>)
        query = """
            SELECT id, full_name, voice_embedding <=> %s::vector AS distance
            FROM users
            ORDER BY distance ASC
            LIMIT 1;
        """
        
        with db_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (vector,))
                match = cur.fetchone()
                
                if match:
                    user_id, name, distance = match
                    confidence = 1 - distance
                    
                    if distance < AUTH_THRESHOLD:
                        logger.info(f"Identified {name} (Conf: {confidence:.1%})")
                        return {
                            "id": user_id, 
                            "name": name, 
                            "confidence": confidence
                        }
                    else:
                        logger.warning(f"Rejecting user. Closest: {name} ({confidence:.1%})")
                        return None
                        
                return None

    except Exception as e:
        logger.error(f"Identity Verification Failed: {e}")
        return None