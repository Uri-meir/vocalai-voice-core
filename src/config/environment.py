import os
import sys
import yaml
from dotenv import load_dotenv

# Load .env file for secrets
# Load .env file for secrets (Override ensures local .env takes precedence over shell vars)
load_dotenv(override=True)

class ConfigManager:
    _config = None

    @classmethod
    def _load_config(cls):
        if cls._config is None:
            base_path = os.path.dirname(__file__)
            config_path = os.path.join(base_path, "config.yml")
            try:
                with open(config_path, "r") as f:
                    content = f.read()
                    # Interpolate environment variables manually
                    # This is a simple implementation, for advanced use replace with a regex or library
                    for key, value in os.environ.items():
                        content = content.replace(f"${{{key}}}", value.strip())
                    cls._config = yaml.safe_load(content)
            except FileNotFoundError:
                print("❌ Error: config.yml not found in src/config/")
                sys.exit(1)
        return cls._config

    @classmethod
    def get(cls, path, default=None):
        """Retrieves a value from the config using dot notation (e.g. 'gemini.voice_name')."""
        config = cls._load_config()
        keys = path.split(".")
        value = config
        for key in keys:
            value = value.get(key)
            if value is None:
                return default
        return value

    # --- Secrets (from .env) ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER")
    GEMINI_VOICE_NAME = os.getenv("GEMINI_VOICE_NAME", "Puck")
    N8N_WEBHOOK_URL_DTC = os.getenv("N8N_WEBHOOK_URL_DTC")
    WHATSAPP_WEBHOOK_URL = os.getenv("WHATSAPP_WEBHOOK_URL")

    @property
    def GEMINI_MODEL_ID(self):
        # Allow env var override, otherwise fallback to config.yml
        env_model = os.getenv("GEMINI_MODEL_ID")
        if env_model: return env_model
        
        # Read from config.yml (which points to 2.5 flash native audio)
        return self.get("gemini.model_id", "models/gemini-2.0-flash-exp")

    # --- Computed Properties ---
    @property
    def CHUNK_SIZE(self):
        vad_enabled = self.get("vad.enabled", False)
        return self.get("audio.chunk_size_vad") if vad_enabled else self.get("audio.chunk_size_default")

    @classmethod
    def get_system_instruction(cls):
        """Loads the system instruction from file."""
        try:
            base_path = os.path.dirname(__file__)
            file_path = os.path.join(base_path, "system_instruction.txt")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None

    @classmethod
    def validate(cls):
        """Validates that essential environment variables are set."""
        if not cls.GEMINI_API_KEY:
            print("❌ Error: Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env file.")
            sys.exit(1)

# Singleton Instance for easy import
config = ConfigManager()
