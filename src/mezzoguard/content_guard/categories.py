from .._types import Category


class Category(Category):
    DIVISIVE = "divisive"
    HATE_SPEECH = "hate-speech"
    SELF_HARM = "self-harm"
    SEXUAL = "sexual"
    TOXIC = "toxic"
    VIOLENCE = "violence"


__all__ = ["Category"]