import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Database Configuration ---
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int  # Pydantic will auto-convert string "5432" to int 5432
    DB_NAME: str

    # Secrets
    HF_TOKEN:str

    # --- AI Configuration ---
    AI_VECTOR_DIMENSION: int = 192
    
    # --- Audio Quality Gate ---
    MIN_SPEECH_DURATION: int = 30
    MIN_VOLUME_THRESHOLD: int = 500
    
    # --- Paths (Computed automatically) ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @property
    def DATA_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data")
    
    @property
    def TEMP_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "temp")
        
    @property
    def MODELS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "models")

    # Load from .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Initialize Singleton
settings = Settings()

# Auto-create directories on startup
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)