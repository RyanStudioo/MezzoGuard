from typing import Literal


class PROMPTGUARD:
    """Mezzo Prompt Guard Models"""

    # Mezzo Prompt Guard v2 Models
    MEZZO_PROMPT_GUARD_V2_LARGE = "RyanStudio/Mezzo-Prompt-Guard-v2-Large"
    MEZZO_PROMPT_GUARD_V2_BASE = "RyanStudio/Mezzo-Prompt-Guard-v2-Base"
    MEZZO_PROMPT_GUARD_V2_SMALL = "RyanStudio/Mezzo-Prompt-Guard-v2-Small"

class CONTENTGUARD:
    """Mezzo Content Guard Models"""

    MEZZO_CONTENT_GUARD_LARGE = "RyanStudio/Mezzo-Content-Guard-Large"
    MEZZO_CONTENT_GUARD_BASE = "RyanStudio/Mezzo-Content-Guard-Base"
    MEZZO_CONTENT_GUARD_SMALL = "RyanStudio/Mezzo-Content-Guard-Small"

    MEZZO_CONTENT_GUARD_LARGE_PREVIEW = "RyanStudio/Mezzo-Content-Guard-Large-Preview"
    
def get_recommended_model(task: Literal["prompt_guard", "content_guard"], priority: Literal["quality", "speed", "balance"]):
    """Get a recommended model based on task and priority"""
    if task == "prompt_guard":
        if priority == "quality":
            return PROMPTGUARD.MEZZO_PROMPT_GUARD_V2_LARGE
        elif priority == "speed":
            return PROMPTGUARD.MEZZO_PROMPT_GUARD_V2_SMALL
        elif priority == "balance":
            return PROMPTGUARD.MEZZO_PROMPT_GUARD_V2_BASE
        return None
    elif task == "content_guard":
        if priority == "quality":
            return CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE
        elif priority == "speed":
            return CONTENTGUARD.MEZZO_CONTENT_GUARD_SMALL
        elif priority == "balance":
            return CONTENTGUARD.MEZZO_CONTENT_GUARD_BASE
        return None
    else:
        raise ValueError(f"Invalid task: {task}")

def view_available_models() -> dict[str, list[str]]:
    """View a list of available models"""
    prompt_guard_models = [value for key, value in vars(PROMPTGUARD).items() if not key.startswith("_")]
    content_guard_models = [value for key, value in vars(CONTENTGUARD).items() if not key.startswith("_")]
    
    return {
        "prompt_guard": prompt_guard_models,
        "content_guard": content_guard_models
    }