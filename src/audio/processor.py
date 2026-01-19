import os
import torch
import torchaudio
from src.engine.separator import audio_separator
from src.engine.overlap import overlap_detector
from src.services.identity import identify_user
from src.engine.speaker import speaker_engine
from src.engine.transcriber import transcriber
from src.utils.logger import setup_logger

logger = setup_logger("Audio_Processor_Pipeline")

class AudioProcessor:
    def __init__(self):
        # Load VAD model for slicing the 2-min file
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.get_speech_timestamps = utils[0]
        self.save_audio = utils[1]
        self.read_audio = utils[2]

    def process_long_audio(self, main_wav_path: str):
        """
        The Master Loop:
        1. Slices long audio into chunks.
        2. Checks each chunk for overlap.
        3. Separates if needed.
        4. Identifies & Transcribes.
        """
        logger.info(f"🚀 Starting robust processing for: {main_wav_path}")
        
        # 1. SLICE (VAD)
        # Get timestamps of speech bubbles (e.g., 0-10s, 12-15s)
        wav = self.read_audio(main_wav_path)

        # --- NEW: PRE-NORMALIZATION (Smart 5x Limit) ---
        # This helps Overlap Detector hear better
        max_val = wav.abs().max()
        if max_val > 0:
            target_peak = 0.9
            gain = target_peak / max_val
            if gain > 5.0: gain = 5.0 
            wav = wav * gain
            logger.debug(f"Processor: Boosted main audio by {gain:.2f}x")

        speech_timestamps = self.get_speech_timestamps(wav, self.vad_model)
        
        results = []

        # Process each "speech bubble" individually
        for i, segment in enumerate(speech_timestamps):
            start = segment['start'] / 16000  # assuming 16k sample rate
            end = segment['end'] / 16000
            duration = end - start

            # Create a temp file for this specific chunk
            chunk_name = f"chunk_{i}_{start:.1f}_{end:.1f}.wav"
            chunk_path = os.path.join("data", "temp", "chunks", chunk_name)
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
            
            # Save the chunk to disk
            self.save_audio(chunk_path, wav[segment['start']:segment['end']], 16000)

            logger.info(f"--- Processing Chunk {i} ({duration:.1f}s) ---")

            # 2. CHECK OVERLAP
            if overlap_detector.has_overlap(chunk_path):
                logger.warning(f"⚡ Overlap detected in Chunk {i}! Engaging Separator.")
                
                # 3a. SEPARATE
                # This returns list: ['chunk_spk1.wav', 'chunk_spk2.wav', 'chunk_spk3.wav']
                clean_files = audio_separator.separate_audio(chunk_path)
                
                # Process each separated track as a unique speaker
                for clean_file in clean_files:
                    result = self._analyze_single_track(clean_file)
                    if result:
                        results.append(result)
            else:
                # 3b. NO OVERLAP (Fast Path)
                logger.info(f"✅ Clean audio in Chunk {i}. Standard processing.")
                result = self._analyze_single_track(chunk_path)
                if result:
                    results.append(result)

        return results

    def _analyze_single_track(self, wav_path):
        """
        Helper: Identifies and Transcribes a single clean WAV file.
        """
        # A. IDENTIFY
        user_identity = identify_user(wav_path) # <--- Clean 1-line call
        
        user_name = "Unknown"
        user_id = None
        
        if user_identity:
            user_name = user_identity["name"]
            user_id = user_identity["id"]
        
        # B. TRANSCRIBE
        text = transcriber.transcribe(wav_path)
        
        if text:
            return {
                "user_id": user_id,
                "user_name": user_name, 
                "text": text, 
                "file": wav_path
            }
        return None

# Export
audio_processor = AudioProcessor()