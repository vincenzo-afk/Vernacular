"""
Settings loaded from environment variables. Never hardcode API keys —
see CLAUDE.md / AGENTS.md constraints. Populate a local .env from
.env.example.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

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

    # Ask the register detector to also explain its reading (powers the
    # explainable tone/sarcasm UI). Costs ~60-100 extra output tokens
    # per detection; set REGISTER_EXPLAIN=false if a live latency
    # benchmark shows the detection stage over budget.
    register_explain: bool = True

    # Only used by `python -m app.benchmark --live`. Optional so normal
    # operation needs no extra config; the benchmark fails with a clear
    # message if it is unset (see app/benchmark.py).
    benchmark_voice_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    """
    Lazily constructs and caches the Settings singleton on first
    access, rather than at import time. This matters concretely: every
    pipeline module reaches app.config transitively via
    app/ws/session.py, so if `Settings()` were instantiated eagerly at
    module load (as it originally was), simply *importing* session.py
    — including from a future test for it, or from any tool that
    inspects the module — would crash with a pydantic ValidationError
    in any environment without a populated .env, CI included. Deferring
    construction to first real use means import always succeeds, and
    the clear validation error only surfaces when settings are actually
    needed (i.e. when a session is handled), which is also a much
    easier moment to see and diagnose the error from.
    """
    return Settings()  # type: ignore[call-arg]
