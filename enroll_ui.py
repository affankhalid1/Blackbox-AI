
import streamlit as st
import os
import time
from src.config import settings
from src.database import db_client
from src.audio.capture import record_audio
from src.audio.vad import check_quality
from src.audio.normalization import normalize_audio
from src.engine.speaker import speaker_engine
from src.utils.logger import setup_logger

# Initialize Logger
logger = setup_logger("Enrollment_UI")

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="BlackBox AI | Enrollment",
    page_icon="🎙️",
    layout="centered"
)

def run_ui():
    # --- HEADER ---
    st.title("🎙️ BlackBox AI")
    st.subheader("Team Member Voice Enrollment")
    st.markdown("---")

    # --- SIDEBAR (Database Status) ---
    with st.sidebar:
        st.header("System Status")
        try:
            # Quick connection check
            with db_client.get_connection() as conn:
                st.success("✅ Database Connected")
                st.info(f"Port: {settings.DB_PORT}")
        except Exception as e:
            st.error("❌ Database Error")
            st.error(f"{e}")
            st.stop() # Stop execution if DB is down

    # --- STEP 1: USER DETAILS ---
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name", placeholder="e.g., Ali Khan")
    with col2:
        emp_id = st.text_input("Employee ID", placeholder="e.g., EMP-001")

    # --- STEP 2: INSTRUCTIONS ---
    st.info(f"📝 **Instructions:** Click 'Start Recording' and read the script below for **{settings.MIN_SPEECH_DURATION} seconds**.")
    
    script_text = """
    "The quick brown fox jumps over the lazy dog. 
    Main aaj BlackBox AI system test kar raha hoon. 
    This system is designed to identify my voice securely.
    Machine learning aur Artificial Intelligence ka daur hai.
    Please verify my voice identity for the security clearance."
    """
    st.code(script_text, language="text")

    # --- STEP 3: RECORDING ACTION ---
    # We use a button to trigger the whole flow
    if st.button("🔴 Start Recording & Enroll", type="primary", use_container_width=True):
        
        # Validation 1: Check Empty Inputs
        if not full_name or not emp_id:
            st.error("⚠️ Please enter both Name and Employee ID first.")
            return

        # Define temp file path
        temp_file = os.path.join(settings.TEMP_DIR, f"enroll_{emp_id}.wav")

        # UI Visuals for Recording
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # --- PHASE A: RECORDING ---
            status_text.write("🎙️ Recording in progress... Please speak clearly.")
            logger.info(f"Starting enrollment recording for {full_name}")
            
            # Record Audio (Blocking call - runs for 45s)
            # We animate the progress bar roughly since record_audio blocks
            # (For a true non-blocking UI, we'd need threading, but this is fine for MVP)
            raw_audio = record_audio(temp_file)
            progress_bar.progress(100)
            
            # --- PHASE B: QUALITY CHECK ---
            status_text.write("🔍 Analyzing Audio Quality...")
            
            # 1. Normalize
            clean_audio = normalize_audio(raw_audio)
            
            # 2. VAD Check
            is_valid, message = check_quality(clean_audio)
            
            if not is_valid:
                st.error(f"❌ Quality Check Failed: {message}")
                logger.warning(f"Enrollment rejected: {message}")
                return # Stop here, let user try again

            st.success("✅ Audio Quality Passed!")
            
            # --- PHASE C: AI PROCESSING ---
            status_text.write("🧠 Generating Voice Fingerprint (192-dim vector)...")
            vector = speaker_engine.get_voice_embedding(temp_file)
            
            # --- PHASE D: SAVE TO DB ---
            status_text.write("💾 Saving to Database...")
            success = db_client.add_user(full_name, emp_id, vector)
            
            if success:
                st.balloons()
                st.success(f" SUCCESS: {full_name} has been successfully enrolled!")
                logger.info(f"User {full_name} enrolled successfully.")
                
                # Cleanup
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                st.error("❌ Database Error: Could not save user. Check logs.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"UI Error: {e}")

if __name__ == "__main__":
    run_ui()