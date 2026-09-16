"""
Centralized configuration loader.
Reads environment variables once so the rest of the app
never touches os.environ directly — keeps secrets contained
and makes testing/mocking easier.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables


class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    APP_ENV: str = os.getenv("APP_ENV", "development")

    def validate(self) -> None:
        """Fail fast and loudly if required secrets are missing."""
        if not self.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )


settings = Settings()