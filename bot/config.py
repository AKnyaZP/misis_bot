from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class TgBotSettings(BaseSettings):
    bot_token: str

    hf_model : str

    hf_token: str

    model_config = SettingsConfigDict(
        env_file = Path(__file__).parent.joinpath(".env"), env_file_encoding="utf-8"
    )

settings = TgBotSettings()