import torch
import torchaudio
import torchaudio.transforms as T
from speechbrain.inference.separation import SepformerSeparation
from src.config import settings
from src.utils.logger import setup_logger
import os

logger = setup_logger("AI_Separator")

class AudioSeparator:
    """
    The 'Un-Mixer'. Splits complex audio into 3 distinct clean tracks.
    Uses SpeechBrain SepFormer (WSJ03Mix).
    """
    def __init__(self):
        model_dir = os.path.join(settings.BASE_DIR, "data", "models", "sepformer")
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info("Initializing SepFormer (3-Speaker Model)...")
        self.model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj03mix",
            savedir=model_dir,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )

        # Define Resampler (8k -> 16k) for compatibility with Whisper/SpeakerId
        self.resampler = T.Resample(orig_freq=8000, new_freq=16000)

    def separate_audio(self, wav_path: str) -> list[str]:
        """
        Separates audio and saves 3 clean .wav files.
        Returns a list of paths to the new files.
        """
        logger.info(f"Separating sources for: {wav_path}")
        
        # 1. Run the AI Separation
        est_sources = self.model.separate_file(path=wav_path)
        
        # est_sources shape: [batch, time, 3_speakers]
        # We need to save each speaker as a separate file
        
        output_paths = []
        base_name = os.path.splitext(wav_path)[0]
        
        # Loop through the 3 potential speakers
        for i in range(est_sources.shape[2]):
            # Extract single speaker track
            source = est_sources[:, :, i]

            # --- STEP 1: RESAMPLE (Fix Chipmunk Bug) ---
            # Move resampler to same device as source
            self.resampler = self.resampler.to(source.device)
            source_16k = self.resampler(source)
            
            # # Check if this track is just silence (empty channel)
            # # If energy is too low, skip it to save processing time
            # if source.abs().max() < 0.01:
            #     continue

            # --- ROBUST GHOST FILTER ---
            # 1. Calculate Root Mean Square (RMS) Amplitude (Average Energy)
            rms_energy = torch.sqrt(torch.mean(source**2))
            
            # 2. Define a Threshold
            # 0.005 is a good baseline. Adjust higher if you get too much static.
            energy_threshold = 0.005 
            
            if rms_energy < energy_threshold:
                logger.info(f"Skipping Track {i+1}: Detected silence/noise (Energy: {rms_energy:.4f})")
                continue

            # Save clean file
            out_path = f"{base_name}_spk{i+1}.wav"
            torchaudio.save(out_path, source.cpu(), 16000) # Model native rate
            output_paths.append(out_path)
            
        logger.info(f"Separation complete. Generated {len(output_paths)} clean tracks.")
        return output_paths

# Singleton Instance
audio_separator = AudioSeparator()



import torch
import torchaudio
# import torchaudio.transforms as T
from speechbrain.inference.separation import SepformerSeparation
from src.config import settings
from src.utils.logger import setup_logger
import os

logger = setup_logger("AI_Separator")

class AudioSeparator:
    def __init__(self):
        model_dir = os.path.join(settings.BASE_DIR, "data", "models", "sepformer")
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info("Initializing SepFormer (3-Speaker Model)...")
        self.model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj03mix",
            savedir=model_dir,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Define Resampler (8k -> 16k) for compatibility with Whisper/SpeakerId
        self.resampler = T.Resample(orig_freq=8000, new_freq=16000)

    def separate_audio(self, wav_path: str) -> list:
        logger.info(f"Separating sources for: {wav_path}")
        
        # 1. Run AI Separation (Outputs 8000Hz tensor)
        est_sources = self.model.separate_file(path=wav_path)
        
        output_paths = []
        base_name = os.path.splitext(wav_path)[0]
        
        for i in range(est_sources.shape[2]):
            source = est_sources[:, :, i]

            # --- STEP 1: RESAMPLE (Fix Chipmunk Bug) ---
            # Move resampler to same device as source
            self.resampler = self.resampler.to(source.device)
            source_16k = self.resampler(source)

            # --- SMART NORMALIZATION (Your Logic Adapted for GPU) ---
            # Standard tensors are float values between -1.0 and 1.0
            max_val = source_16k.abs().max()
            
            if max_val > 0:
                # Calculate gain to reach approx 90% volume (0.9)
                target_peak = 0.9
                gain = target_peak / max_val
                
                # Apply your "5x Limit" safety cap
                # If gain > 5.0, we clamp it to 5.0
                if gain > 5.0:
                    gain = 5.0
                
                # Apply the gain
                source_16k = source_16k * gain

            # --- STEP 3: GHOST FILTER (RMS) ---
            # Now that it's normalized, this threshold is safe and robust.
            rms_energy = torch.sqrt(torch.mean(source_16k**2))
            energy_threshold = 0.005 

            if rms_energy < energy_threshold:
                logger.info(f"Skipping Track {i+1}: Silence/Noise (Energy: {rms_energy:.4f})")
                continue

            # Save clean file at correct 16k rate
            out_path = f"{base_name}_spk{i+1}.wav"
            # Ensure tensor is (Channels, Time) for saving
            if source_16k.dim() == 1:
                source_16k = source_16k.unsqueeze(0)
                
            torchaudio.save(out_path, source_16k.cpu(), 16000)
            output_paths.append(out_path)
            
        logger.info(f"Separation complete. Generated {len(output_paths)} clean tracks.")
        return output_paths

audio_separator = AudioSeparator()