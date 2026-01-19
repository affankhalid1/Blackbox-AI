import streamlit as st
import os
import time
from audiorecorder import audiorecorder
from src.database import db_client
from src.utils.logger import setup_logger

# --- NEW: Import the Smart Processor ---
# This assumes you saved the processor code in src/audio/processor.py
from src.audio.processor import audio_processor

logger = setup_logger("Ingest_UI")

def run_ingestion_ui():
    st.set_page_config(page_title="Blackbox AI | Neural Ingestion", page_icon="🧠", layout="wide")
    
    st.title("🧠 Neural Ingestion Layer")
    st.markdown("### Agentic Command Center")
    st.caption("Advanced Mode: VAD + Diarization + Overlap Detection Enabled")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Status: System Online")
        st.markdown("**Start Recording:**")
        
        # 1. RECORDING INTERFACE
        audio = audiorecorder("🔴 Click to Record", "⏹️ Click to Stop")

        if len(audio) > 0:
            # Save the recorded bytes to disk
            temp_file = os.path.join("data", "temp", "live_ingest.wav")
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            audio.export(temp_file, format="wav")
            
            st.success(f"Audio Captured ({audio.duration_seconds:.1f}s)")

            # --- SMART PROCESS FLOW ---
            with st.status("🧠 Processing Neural Audio...", expanded=True) as status:
                st.write("1. Slicing Audio (VAD)...")
                st.write("2. Checking Overlaps...")
                
                # The Processor does ALL the heavy lifting here
                # It returns a list of dictionaries: [{'user_name': 'Affan', 'text': 'Hello'}, ...]
                try:
                    results = audio_processor.process_long_audio(temp_file)
                    status.update(label="✅ Processing Complete", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Processing Failed", state="error")
                    st.error(f"Processor Error: {e}")
                    results = []

            # --- DISPLAY & SAVE RESULTS ---
            if not results:
                st.warning("No speech detected or valid speakers found.")
            
            for i, res in enumerate(results):
                user_name = res.get("user_name", "Unknown")
                user_id = res.get("user_id")
                text = res.get("text", "")
                
                # Create a chat-like visual for each segment
                with st.chat_message("user", avatar="👤" if user_name == "Unknown" else "🧑‍💻"):
                    st.markdown(f"**{user_name}**")
                    st.write(text)
                    
                    if user_id:
                        # Save to Database
                        if db_client.save_transcript(user_id, text):
                            st.caption(f"💾 Saved to Memory (ID: {user_id})")
                    else:
                        st.caption("⚠️ Unverified User - Not Saved")

    with col2:
        st.subheader("Live Feed")
        st.write("Waiting for audio input...")
        
        # Optional: You can visualize the "Chunks" here later
        if 'results' in locals() and results:
            st.json(results, expanded=False)

if __name__ == "__main__":
    run_ingestion_ui()