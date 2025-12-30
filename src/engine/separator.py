import torch
import torchaudio
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
            
            # Check if this track is just silence (empty channel)
            # If energy is too low, skip it to save processing time
            if source.abs().max() < 0.01:
                continue

            # Save clean file
            out_path = f"{base_name}_spk{i+1}.wav"
            torchaudio.save(out_path, source.cpu(), 8000) # Model native rate
            output_paths.append(out_path)
            
        logger.info(f"Separation complete. Generated {len(output_paths)} clean tracks.")
        return output_paths

# Singleton Instance
audio_separator = AudioSeparator()