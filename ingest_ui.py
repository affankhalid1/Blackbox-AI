import streamlit as st
import os
import time
from audiorecorder import audiorecorder  # <--- NEW IMPORT
from src.engine.speaker import speaker_engine
from src.engine.transcriber import transcriber
from src.database import db_client
from src.utils.logger import setup_logger

logger = setup_logger("Ingest_UI")

AUTH_THRESHOLD = 0.35 

def identify_speaker(vector):
    """(Same logic as before)"""
    query = """
        SELECT id, full_name, voice_embedding <=> %s::vector AS distance
        FROM users
        ORDER BY distance ASC
        LIMIT 1;
    """
    try:
        with db_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (vector,))
                return cur.fetchone()
    except Exception as e:
        logger.error(f"DB Search Failed: {e}")
        return None

def run_ingestion_ui():
    st.set_page_config(page_title="Blackbox AI | Neural Ingestion", page_icon="🧠", layout="wide")
    
    st.title("🧠 Ingestion Layer")
    st.markdown("### Agentic Command Center")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Status: System Online")
        st.markdown("**Start Recording your Voice:**")
        
        # --- NEW RECORDING INTERFACE ---
        # This replaces the simple button. 
        # It gives you a UI with Record/Stop/Play buttons.
        audio = audiorecorder("🔴 Click to Record", "⏹️ Click to Stop")

        if len(audio) > 0:
            # The component returns an AudioSegment. We need to save it as WAV.
            temp_file = os.path.join("data", "temp", "live_ingest.wav")
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            
            # Save the recorded bytes to disk
            audio.export(temp_file, format="wav")
            
            st.success(f"Audio Captured ({audio.duration_seconds:.1f}s)")

            # --- PROCESS FLOW ---
            user_id = None
            

            # 1. Identification
            with st.status("🔍 Verifying Identity...", expanded=True) as status:
                try:
                    vector = speaker_engine.get_voice_embedding(temp_file)
                    match = identify_speaker(vector)
                    
                    if match:
                        uid, name, distance = match
                        if distance < AUTH_THRESHOLD:
                            user_id = uid
                           
                            status.update(label=f"✅ Identity Verified: {name}", state="complete", expanded=False)
                        else:
                            status.update(label="❌ Identity Mismatch", state="error")
                            st.error(f"Access Denied. Closest: {name}")
                    else:
                        status.update(label="⚠️ Unknown Voice", state="error")
                except Exception as e:
                    st.error(f"Engine Error: {e}")

            # 2. Transcription
            if user_id:
                with st.status("📝 Transcribing...", expanded=True) as status:
                    transcript_text = transcriber.transcribe(temp_file)
                    status.update(label="✅ Complete", state="complete", expanded=False)

                st.markdown("### 🗣️ Transcript")
                st.info(f"> {transcript_text}")
                
                # 3. Save
                if transcript_text:
                    if db_client.save_transcript(user_id, transcript_text):
                        st.toast("Saved to Memory", icon="💾")

    with col2:
        st.subheader("Live Feed")
        st.write("Listening for input...")

if __name__ == "__main__":
    run_ingestion_ui()