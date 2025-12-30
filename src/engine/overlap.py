import torch
from pyannote.audio import Pipeline
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("AI_Overlap_Detector")

class OverlapDetector:
    """
    Fast 'Traffic Cop' that detects if multiple people are speaking at once.
    Uses Pyannote's segmentation pipeline.
    """
    def __init__(self):
        # Requires HF_TOKEN in .env
        auth_token = settings.HF_TOKEN
        if not auth_token:
            logger.warning("HF_TOKEN missing! Overlap detection may fail.")

        logger.info("Initializing Pyannote Overlap Detector...")
        try:
            # We use the segmentation model which detects 'overlap' classes
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/overlapped-speech-detection",
                use_auth_token=auth_token
            )
            
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                logger.info("Overlap Detector running on GPU (CUDA).")
            else:
                logger.info("Overlap Detector running on CPU.")
                
        except Exception as e:
            logger.critical(f"Failed to load Overlap Detector: {e}")

    def has_overlap(self, wav_path: str) -> bool:
        """
        Returns TRUE if significant overlapping speech is detected.
        """
        try:
            # Run the pipeline on the audio file
            output = self.pipeline(wav_path)
            
            # The output contains timelines of overlaps. 
            # If we find any segment > 0.5 seconds, we flag it.
            for speech in output.get_timeline().support():
                if speech.duration > 0.5:
                    logger.warning(f"Overlap detected! ({speech.duration:.2f}s)")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Overlap check error: {e}")
            return False # Fail safe: assume no overlap to keep moving

# Singleton Instance
overlap_detector = OverlapDetector()