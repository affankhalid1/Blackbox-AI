import os
import time
import sys
from src.audio.capture import record_audio
from src.engine.speaker import speaker_engine
from src.engine.transcriber import transcriber
from src.database import db_client
from src.utils.logger import setup_logger

# --- Configuration ---
AUTH_THRESHOLD = 0.35
TEMP_FILE = os.path.join("data", "temp", "cli_ingest.wav")

logger = setup_logger("CLI_Ingest")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def identify_user(wav_path):
    """
    Returns (user_id, name, confidence_score) or None
    """
    try:
        # 1. Get Vector
        vector = speaker_engine.get_voice_embedding(wav_path)
        
        # 2. Query DB
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
                    uid, name, distance = match
                    if distance < AUTH_THRESHOLD:
                        return uid, name, (1 - distance)
                    else:
                        logger.warning(f"Auth Failed. Closest: {name} ({1-distance:.1%})")
                        return None
                return None
    except Exception as e:
        logger.error(f"Identification Error: {e}")
        return None

def process_command():
    print("\n" + "="*40)
    print("🔴 RECORDING... (Speak Now)")
    print("="*40)
    
    # 1. Capture
    try:
        record_audio(TEMP_FILE)
    except Exception as e:
        logger.error(f"Mic Error: {e}")
        return

    print("\n⏳ Processing...")

    # 2. Identify
    user_data = identify_user(TEMP_FILE)
    
    if user_data:
        uid, name, conf = user_data
        print(f"✅ VERIFIED USER: {name} (Confidence: {conf:.1%})")
        
        # 3. Transcribe
        print("📝 Transcribing...")
        text = transcriber.transcribe(TEMP_FILE)
        print(f"\n> COMMAND: \"{text}\"\n")
        
        # 4. Save
        if text:
            if db_client.save_transcript(uid, text):
                print("💾 Saved to Memory.")
            else:
                print("❌ Database Error.")
        else:
            print("⚠️ No speech detected.")
    else:
        print("⛔ ACCESS DENIED: Voice not recognized.")

def run_cli_loop():
    # Ensure temp dir exists
    os.makedirs(os.path.dirname(TEMP_FILE), exist_ok=True)
    
    clear_screen()
    print("🤖 BLACKBOX AI | AGENTIC CLI MODE")
    print("-----------------------------------")
    print(f"Auth Threshold: {AUTH_THRESHOLD}")
    print("Model: Faster-Whisper (Medium)")
    print("-----------------------------------")

    while True:
        try:
            # Simple Trigger Loop
            input("\n👉 Press [ENTER] to broadcast a command (or Ctrl+C to quit)...")
            process_command()
            
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down Agent...")
            sys.exit(0)
        except Exception as e:
            logger.critical(f"Critical Loop Error: {e}")

if __name__ == "__main__":
    run_cli_loop()