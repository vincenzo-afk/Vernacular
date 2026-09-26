"""
Settings loaded from environment variables. Never hardcode API keys —
see CLAUDE.md / AGENTS.md constraints. Populate a local .env from
.env.example.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    assemblyai_api_key: str
    elevenlabs_api_key: str

    # Translation / register-detection LLM — provider-agnostic naming
    # since either OpenAI, Anthropic, Groq, or Google may be used (see
    # README.md tech stack table).
    llm_api_key: str
    llm_provider: str = "openai"  # openai | anthropic | groq | google
    llm_model: str = "gpt-4o"

    # Fallback TTS, used per ARCHITECTURE.md §5 if ElevenLabs fails
    fallback_tts_provider: str = "openai"
    fallback_tts_api_key: str | None = None

    register_confidence_threshold: float = 0.5

    class Config:
        env_file = ".env"


settings = Settings()  # type: ignore[call-arg]
