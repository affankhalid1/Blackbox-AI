import torch
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from src.config import settings
from src.utils.logger import setup_logger
import os
import pathlib
import time

logger = setup_logger("AI_Speaker_Engine")

class SpeakerEngine:
    """
    Enterprise AI Engine for Voice Fingerprinting.
    Uses SpeechBrain's ECAPA-TDNN model to create 192-dim vectors.
    """
    
    def __init__(self):
        # Define the local path for the model weights
        model_dir = os.path.join(settings.BASE_DIR, "data", "models", "speaker_encoder")
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info("Initializing SpeechBrain Encoder (ECAPA-TDNN)...")
        
        # Load the model. It will download to data/models on the first run.
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_dir,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        logger.info(f"Speaker Engine loaded on: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

    def get_voice_embedding(self, wav_path: str) -> list:
        """
        Takes a .wav file and returns a 192-dimension Python list.
        """
        try:
            # FIX: Sanitize the path for Windows
            wav_path = str(pathlib.Path(wav_path).resolve()).replace("\\", "/")
            
            logger.debug(f"Extracting embedding for: {wav_path}")

            time.sleep(1.0)
            
            # 1. Load audio file
            signal = self.classifier.load_audio(wav_path)
            
            # 2. Encode the voice into a vector
            # .encode_batch() returns a tensor of shape [batch, 1, 192]
            embeddings = self.classifier.encode_batch(signal)
            
            # 3. Clean and flatten the vector
            # We convert to a standard Python list for PostgreSQL pgvector compatibility
            vector = embeddings.squeeze().cpu().numpy().tolist()
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to extract voice fingerprint: {e}")
            raise e

# Export a singleton instance
speaker_engine = SpeakerEngine()