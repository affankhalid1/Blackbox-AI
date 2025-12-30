import os
import torch
from faster_whisper import WhisperModel
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("AI_Transcriber")

class Transcriber:
    def __init__(self, model_size="medium", compute_type="float16"):
        """
        Initializes Faster-Whisper.
        model_size options: "tiny", "base", "small", "medium", "large-v3"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # If CPU, we must use float32 or int8, float16 is GPU only
        if self.device == "cpu":
            compute_type = "int8"
            
        logger.info(f"Loading Whisper Model ({model_size}) on {self.device}...")
        
        model_path = os.path.join(settings.BASE_DIR, "data", "models", "whisper")
        os.makedirs(model_path, exist_ok=True)

        try:
            self.model = WhisperModel(
                model_size, 
                device=self.device, 
                compute_type=compute_type, 
                download_root=model_path
            )
            logger.info("Whisper Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Whisper: {e}")
            raise e

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes a .wav file to text.
        """
        try:
            logger.debug(f"Transcribing: {audio_path}")
            
            # segments is a generator, so we must iterate over it
            segments, info = self.model.transcribe(
                audio_path, 
                beam_size=5, 
                language="en",
                condition_on_previous_text=False
            )
            
            # Combine all segments into one string
            full_text = " ".join([segment.text for segment in segments]).strip()
            
            logger.info(f"Transcription finished ({info.duration:.2f}s): {full_text}")
            return full_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

# Export Singleton
# 'medium' is a good balance for GPU. Use 'small' if you want it faster.
transcriber = Transcriber(model_size="medium")