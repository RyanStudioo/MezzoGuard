from __future__ import annotations

import argparse
from typing import Any

MODEL_TYPE_CHOICES = ("auto", "prompt", "content")
_MODEL_TYPE_NAMES = {"prompt": "prompt_guard", "content": "content_guard"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mezzo-guard",
        description="Mezzo Guard command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Serve a guard model over HTTP")
    serve.add_argument("model", help="Hugging Face model id or local model directory")
    serve.add_argument(
        "--type",
        choices=MODEL_TYPE_CHOICES,
        default="auto",
        dest="model_type",
        help="Guard type. 'auto' detects from presets or the model's .mezzoguard config.",
    )
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve.add_argument(
        "--dtype",
        default="auto",
        help="Model precision, e.g. auto, float32, float16, bfloat16 (default: auto)",
    )
    backend = serve.add_mutually_exclusive_group()
    backend.add_argument("--onnx", action="store_true", help="Use the ONNX Runtime backend")
    backend.add_argument("--torch-compile", action="store_true", help="Compile the model with torch.compile")
    serve.add_argument("--compile-mode", default="default", help="torch.compile mode (default: default)")
    serve.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda, cuda:N or mps (default: auto)",
    )
    serve.add_argument(
        "--max-batch-texts",
        type=int,
        default=64,
        help="Maximum number of texts per request (default: 64)",
    )
    serve.add_argument(
        "--max-text-chars",
        type=int,
        default=50_000,
        help="Maximum characters per text (default: 50000)",
    )
    serve.add_argument("--log-level", default="info", help="uvicorn log level (default: info)")
    return parser


def detect_model_type(model: str) -> str | None:
    """Detect 'prompt_guard' or 'content_guard' from presets or the model config."""
    from .content_guard.config import MODELS_CONFIG as CONTENT_MODELS
    from .prompt_guard.config import MODELS_CONFIG as PROMPT_MODELS

    if model in PROMPT_MODELS:
        return "prompt_guard"
    if model in CONTENT_MODELS:
        return "content_guard"

    from .base_classes import ModelConfig

    config = ModelConfig.from_model_name(model)
    if config is not None and config.model_type in ("prompt_guard", "content_guard"):
        return config.model_type
    return None


def create_guard(model_type: str, model: str, **kwargs: Any):
    if model_type == "prompt_guard":
        from .prompt_guard import Guard
    elif model_type == "content_guard":
        from .content_guard import Guard
    else:
        raise ValueError(f"Invalid model type '{model_type}'. Must be 'prompt_guard' or 'content_guard'.")
    return Guard(model, **kwargs)


def run_server(app: Any, host: str, port: int, log_level: str) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            "Serving requires FastAPI and uvicorn, which are not installed. "
            'Install them with `pip install "mezzo-guard[server]"`.'
        ) from e
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def _serve(args: argparse.Namespace) -> int:
    try:
        from .server import create_app
    except ImportError:
        print(
            "Serving requires FastAPI and uvicorn, which are not installed. "
            'Install them with `pip install "mezzo-guard[server]"`.'
        )
        return 1

    model_type = args.model_type
    if model_type == "auto":
        detected = detect_model_type(args.model)
        if detected is None:
            print(
                f"Could not detect the guard type for '{args.model}'. Pass --type prompt or --type content explicitly."
            )
            return 2
        model_type = detected
    else:
        model_type = _MODEL_TYPE_NAMES[model_type]

    guard = create_guard(
        model_type,
        args.model,
        dtype=args.dtype,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        use_onnx=args.onnx,
        device=args.device,
    )
    app = create_app(
        guard,
        model_type=model_type,
        max_batch_texts=args.max_batch_texts,
        max_text_chars=args.max_text_chars,
    )
    print(f"Serving {args.model} ({model_type}) on http://{args.host}:{args.port}")
    run_server(app, args.host, args.port, args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


__all__ = ["main", "build_parser", "create_guard", "detect_model_type", "run_server"]
