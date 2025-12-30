import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("Audio_Normalization")

def normalize_audio(audio_bytes: bytes) -> bytes:
    """
    Standardizes audio volume to a target peak level.
    Ensures quiet voices are boosted for better AI accuracy.
    """
    # Convert raw bytes to float array for precise math
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    
    # 1. Find the current peak amplitude
    max_val = np.max(np.abs(audio_np))
    
    if max_val == 0:
        logger.warning("Attempted to normalize a completely silent chunk.")
        return audio_bytes

    # 2. Calculate the gain needed to reach target (approx 90% of max range)
    # 32767 is the limit for 16-bit audio
    target_peak = 30000 
    gain = target_peak / max_val
    
    # Limit gain to prevent boosting extreme background noise (e.g., max 5x boost)
    gain = min(gain, 5.0) 
    
    logger.debug(f"Normalization: Applying a gain of {gain:.2f}x to signal.")
    
    # 3. Apply gain and convert back to 16-bit integers
    normalized_audio = (audio_np * gain).clip(-32768, 32767).astype(np.int16)
    
    return normalized_audio.tobytes()