import numpy as np
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("Audio_VAD")

def calculate_speech_duration(audio_np: np.ndarray, rate: int = 16000) -> float:
    """
    Calculates net speech duration by ignoring silence.
    """
    # Analyze in 100ms chunks
    chunk_size = int(rate * 0.1) 
    num_chunks = len(audio_np) // chunk_size
    speech_chunks = 0
    
    # Silence threshold (Lower than volume threshold for sensitivity)
    threshold = 300 
    
    for i in range(num_chunks):
        chunk = audio_np[i*chunk_size : (i+1)*chunk_size]
        if np.max(np.abs(chunk)) > threshold:
            speech_chunks += 1
            
    return speech_chunks * 0.1

def check_quality(raw_audio: bytes) -> tuple[bool, str]:
    """
    The 'Quality Firewall' from your Enrollment Workflow.
    Returns (Passed, Message).
    """
    # Convert raw bytes to a numerical array for analysis
    audio_np = np.frombuffer(raw_audio, dtype=np.int16)
    
    # 1. Volume Check (Must be > MIN_VOLUME_THRESHOLD from .env)
    max_vol = np.max(np.abs(audio_np))
    logger.debug(f"Quality Check: Max Amplitude detected at {max_vol}")
    
    if max_vol < settings.MIN_VOLUME_THRESHOLD:
        return False, f"Too Quiet (Detected: {max_vol}). Please speak closer to the mic."

    # 2. Speech Duration Check (Must be > 30s from your diagram)
    speech_sec = calculate_speech_duration(audio_np)
    logger.info(f"Quality Check: Net Speech Duration is {speech_sec:.2f}s")
    
    if speech_sec < settings.MIN_SPEECH_DURATION:
        return False, f"Not enough speech (Detected: {speech_sec:.1f}s). Please read the full script."

    return True, "Quality Passed"


import torch
import numpy as np
import os
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("Audio_VAD")

# --- Silero Model Initialization ---
# We load this once at the module level so it stays in memory
logger.info("Loading Silero VAD Neural Network...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
logger.info("Silero VAD Loaded Successfully.")

def calculate_speech_duration(audio_np: np.ndarray, rate: int = 16000) -> float:
    """
    Uses Silero Neural Network to calculate exact speech duration.
    Much more accurate than simple amplitude thresholds.
    """
    # Silero expects float32 tensors
    audio_float = audio_np.astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio_float)
    
    # Get speech timestamps (start/end points of actual talking)
    speech_timestamps = get_speech_timestamps(
        audio_tensor, 
        model, 
        sampling_rate=rate,
        threshold=0.5 # Confidence threshold (0.5 = 50% sure it's speech)
    )
    
    # Calculate total duration by adding up all speech segments
    total_samples = sum([(ts['end'] - ts['start']) for ts in speech_timestamps])
    duration_sec = total_samples / rate
    
    return duration_sec

def check_quality(raw_audio: bytes) -> tuple[bool, str]:
    """
    The 'Quality Firewall' using AI-based speech detection.
    """
    audio_np = np.frombuffer(raw_audio, dtype=np.int16)
    
    # 1. Volume Check (Physical Gate)
    max_vol = np.max(np.abs(audio_np))
    if max_vol < settings.MIN_VOLUME_THRESHOLD:
        return False, f"Too Quiet (Detected: {max_vol}). Please speak louder."

    # 2. AI Speech Check (Intelligence Gate)
    speech_sec = calculate_speech_duration(audio_np)
    logger.info(f"AI Analysis: {speech_sec:.2f}s of human speech detected.")
    
    if speech_sec < settings.MIN_SPEECH_DURATION:
        return False, f"Need more speech (AI detected: {speech_sec:.1f}s). Please continue reading."

    return True, "Quality Passed"