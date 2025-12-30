import streamlit as st
import time
import os
import pathlib
from src.audio.capture import record_audio
from src.engine.speaker import speaker_engine
from src.database import db_client  # Importing your actual client
from src.utils.logger import setup_logger

# --- Configuration ---
# 0.25 Distance = ~0.75 Similarity (Strict Security)
# 0.40 Distance = ~0.60 Similarity (Loose/Testing)
AUTH_THRESHOLD = 0.35 

logger = setup_logger("Auth_UI")

def find_closest_match(new_vector):
    """
    Queries the database using pgvector's <=> operator (Cosine Distance).
    Returns the closest user and the distance using the existing DatabaseClient.
    """
    # pgvector operator '<=>' calculates cosine distance.
    # Lower distance = Higher similarity.
    query = """
        SELECT id, full_name, voice_embedding <=> %s::vector AS distance
        FROM users
        ORDER BY distance ASC
        LIMIT 1;
    """

    try:
        # Use your existing db_client context manager
        with db_client.get_connection() as conn:
            with conn.cursor() as cur:
                # Pass the vector as a list; psycopg2+pgvector handles the rest
                cur.execute(query, (new_vector,))
                result = cur.fetchone()
                return result
    except Exception as e:
        logger.error(f"Auth Query Failed: {e}")
        st.error(f"Database Error: {e}")
        return None

def run_auth_ui():
    st.set_page_config(page_title="Blackbox AI | Voice Gate", page_icon="🛡️")
    
    st.title("🛡️ Secure Voice Entry")
    st.markdown("### Identify yourself.")

    # Status Container
    status_box = st.empty()

    if st.button("🎙️ Authenticate", type="primary"):
        temp_file = os.path.join("data", "temp", "auth_attempt.wav")
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)

        # 1. Capture Audio
        status_box.info("Listening... Speak your phrase.")
        try:
            # We assume record_audio handles the duration internally
            audio_data = record_audio(temp_file)
            status_box.success("Audio captured. Processing...")
        except Exception as e:
            st.error(f"Capture failed: {e}")
            return

        # 2. Extract Vector
        try:
            new_vector = speaker_engine.get_voice_embedding(temp_file)
        except Exception as e:
            st.error(f"AI Engine Error: {e}")
            return

        # 3. Database Search
        match = find_closest_match(new_vector)

        if match:
            user_id, name, distance = match
            
            # Convert distance to % similarity (approximate)
            # Distance 0 = 100% Match
            # Distance 1 = 0% Match
            similarity = 1 - distance
            
            logger.info(f"Auth Attempt: Closest match {name} with distance {distance:.4f}")

            # 4. Decision Logic
            if distance < AUTH_THRESHOLD:
                # SUCCESS
                st.balloons()
                st.success(f"# ACCESS GRANTED")
                st.markdown(f"### Welcome back, **{name}**!")
                st.caption(f"Confidence Score: {similarity:.2%}")
            else:
                # FAILURE
                st.error("# ACCESS DENIED")
                st.warning("Voice fingerprint does not match authorized personnel.")
                st.caption(f"Closest match was {name} ({similarity:.2%}), but it wasn't close enough.")
        else:
            st.error("System Error: No users found in database.")

if __name__ == "__main__":
    run_auth_ui()