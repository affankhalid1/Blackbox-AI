import os
import sys
from src.config import settings
from src.database import db_client
from src.utils.logger import setup_logger
# From the Hardware module
from src.audio.capture import record_audio
# From the Intelligence module
from src.audio.vad import check_quality
from src.audio.normalization import normalize_audio
from src.engine.speaker import get_voice_embedding

# Initialize Enterprise Logger for the Enrollment module
logger = setup_logger("Enrollment_UI")

def run_enrollment():
    """Main Orchestration for User Enrollment."""
    
    logger.info("Starting Enterprise Enrollment Workflow.")
    
    # 1. Database Readiness Check
    try:
        db_client.init_db()
    except Exception as e:
        logger.critical(f"Database not ready. Aborting: {e}")
        return

    # 2. Collect User Metadata
    print("\n" + "="*40)
    print("   BLACKBOX AI: TEAM MEMBER ENROLLMENT")
    print("="*40)
    
    full_name = input("Enter Full Name: ").strip()
    emp_id = input("Enter Employee ID: ").strip()

    if not full_name or not emp_id:
        logger.error("Name and ID cannot be empty.")
        return

    # Define temporary path for the enrollment wave file
    temp_wav = os.path.join(settings.TEMP_DIR, f"enroll_{emp_id}.wav")

    # 3. Audio Quality Firewall (The Loop)
    while True:
        print(f"\n📝 INSTRUCTION: Please speak for {settings.MIN_SPEECH_DURATION} seconds.")
        input("Press [ENTER] to start recording...")
        
        try:
            # Record using our improved audio module
            raw_audio = record_audio(temp_wav)

            clean_audio = normalize_audio(raw_audio) # New step!
            
            # Quality Verification Gate
            is_valid, message = check_quality(clean_audio)
            
            if is_valid:
                logger.info("Audio Quality Gate: PASSED.")
                break
            else:
                logger.warning(f"Audio Quality Gate: REJECTED - {message}")
                print(f"❌ REJECTED: {message}. Please try again.")
        
        except Exception as e:
            logger.error(f"Hardware/Recording Error: {e}")
            input("Check your microphone and press Enter to retry...")

    # 4. AI Feature Extraction
    try:
        logger.info("Extracting 192-dimension voice fingerprint...")
        vector = get_voice_embedding(temp_wav)
    except Exception as e:
        logger.error(f"AI Engine Error: {e}")
        return

    # 5. Persistent Storage
    success = db_client.add_user(full_name, emp_id, vector)
    
    # Clean up sensitive audio data after processing
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        logger.debug("Temporary enrollment file deleted.")

    if success:
        print("\n" + "*"*40)
        print(f"SUCCESS: {full_name} is now in the BlackBox System.")
        print("*"*40 + "\n")
        logger.info(f"Enrollment completed for {emp_id}.")
    else:
        logger.error("Enrollment failed during database insertion.")

if __name__ == "__main__":
    try:
        run_enrollment()
    except KeyboardInterrupt:
        print("\n\nEnrollment cancelled by user.")
        sys.exit(0)