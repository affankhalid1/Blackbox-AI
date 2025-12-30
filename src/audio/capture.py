import pyaudio
import wave
import numpy as np
import os
from datetime import datetime
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("Audio_Capture")

def record_audio(output_path: str) -> bytes:
    """
    Captures audio from hardware and saves it to a file.
    Includes logic for the Ingestion Workflow's timestamping.
    """
    p = pyaudio.PyAudio()
    
    # 1. Capture the Start Timestamp (Enterprise Requirement)
    start_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    logger.info(f"Capture started at {start_time}")

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)

    logger.info(f"Recording... target: {output_path}")
    frames = []

    # Record for the duration set in config (e.g., 45s)
    # 16000/1024 = ~15.6 chunks per second
    duration = 45
    for _ in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save to disk
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    
    return b''.join(frames)