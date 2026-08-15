from typing import Literal


class PROMPTGUARD:
    """Mezzo Prompt Guard Models"""

    # Mezzo Prompt Guard v2 Models
    MEZZO_PROMPT_GUARD_V2_LARGE = "RyanStudio/Mezzo-Prompt-Guard-v2-Large"
    MEZZO_PROMPT_GUARD_V2_BASE = "RyanStudio/Mezzo-Prompt-Guard-v2-Base"
    MEZZO_PROMPT_GUARD_V2_SMALL = "RyanStudio/Mezzo-Prompt-Guard-v2-Small"


class CONTENTGUARD:
    """Mezzo Content Guard Models"""

    MEZZO_CONTENT_GUARD_V1_5_LARGE = "RyanStudio/Mezzo-Content-Guard-v1.5-Large"
    MEZZO_CONTENT_GUARD_V1_5_BASE = "RyanStudio/Mezzo-Content-Guard-v1.5-Base"
    MEZZO_CONTENT_GUARD_V1_5_SMALL = "RyanStudio/Mezzo-Content-Guard-v1.5-Small"
    MEZZO_CONTENT_GUARD_V1_5_TINY = "RyanStudio/Mezzo-Content-Guard-v1.5-Tiny"
    MEZZO_CONTENT_GUARD_V1_5_NANO = "RyanStudio/Mezzo-Content-Guard-v1.5-Nano"

    MEZZO_CONTENT_GUARD_V1_5_NANO_PREVIEW = "RyanStudio/Mezzo-Content-Guard-v1.5-Nano-Preview"

    MEZZO_CONTENT_GUARD_LARGE = "RyanStudio/Mezzo-Content-Guard-Large"
    MEZZO_CONTENT_GUARD_BASE = "RyanStudio/Mezzo-Content-Guard-Base"
    MEZZO_CONTENT_GUARD_SMALL = "RyanStudio/Mezzo-Content-Guard-Small"

    MEZZO_CONTENT_GUARD_LARGE_PREVIEW = "RyanStudio/Mezzo-Content-Guard-Large-Preview"


_RECOMMENDED_MODELS = {
    "prompt_guard": {
        "quality": PROMPTGUARD.MEZZO_PROMPT_GUARD_V2_LARGE,
        "speed": PROMPTGUARD.MEZZO_PROMPT_GUARD_V2_SMALL,
        "balance": PROMPTGUARD.MEZZO_PROMPT_GUARD_V2_BASE,
    },
    "content_guard": {
        "quality": CONTENTGUARD.MEZZO_CONTENT_GUARD_V1_5_LARGE,
        "speed": CONTENTGUARD.MEZZO_CONTENT_GUARD_V1_5_SMALL,
        "balance": CONTENTGUARD.MEZZO_CONTENT_GUARD_V1_5_BASE,
    },
}


def get_recommended_model(
    task: Literal["prompt_guard", "content_guard"], priority: Literal["quality", "speed", "balance"]
) -> str:
    """Get a recommended model based on task and priority"""
    if task not in _RECOMMENDED_MODELS:
        raise ValueError(f"Invalid task: {task}. Must be 'prompt_guard' or 'content_guard'.")
    if priority not in _RECOMMENDED_MODELS[task]:
        raise ValueError(f"Unknown priority: {priority}. Must be 'quality', 'speed', or 'balance'.")
    return _RECOMMENDED_MODELS[task][priority]


def view_available_models() -> dict[str, list[str]]:
    """View a list of available models"""
    prompt_guard_models = [value for key, value in vars(PROMPTGUARD).items() if not key.startswith("_")]
    content_guard_models = [value for key, value in vars(CONTENTGUARD).items() if not key.startswith("_")]

    return {"prompt_guard": prompt_guard_models, "content_guard": content_guard_models}
