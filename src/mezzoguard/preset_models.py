def view_available_models() -> dict[str, list[str]]:
    """List the models with built-in guard configurations."""
    from .content_guard.config import MODELS_CONFIG as CONTENT_MODELS
    from .prompt_guard.config import MODELS_CONFIG as PROMPT_MODELS

    return {"prompt_guard": list(PROMPT_MODELS), "content_guard": list(CONTENT_MODELS)}


__all__ = ["view_available_models"]
