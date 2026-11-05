import pytest

from mezzoguard import cli
from mezzoguard.base_classes import ModelConfig


def test_parser_defaults():
    args = cli.build_parser().parse_args(["serve", "some/model"])
    assert args.command == "serve"
    assert args.model == "some/model"
    assert args.model_type == "auto"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.dtype == "auto"
    assert args.onnx is False
    assert args.torch_compile is False
    assert args.compile_mode == "default"
    assert args.device == "auto"
    assert args.max_batch_texts == 64
    assert args.max_text_chars == 50_000
    assert args.log_level == "info"


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_detect_model_type_presets():
    assert cli.detect_model_type("RyanStudio/Mezzo-Prompt-Guard-v2-Small") == "prompt_guard"
    assert cli.detect_model_type("RyanStudio/Mezzo-Content-Guard-Small") == "content_guard"


def test_detect_model_type_from_config(monkeypatch):
    config = ModelConfig(model_type="content_guard", categories=[])
    monkeypatch.setattr(ModelConfig, "from_model_name", classmethod(lambda cls, name: config))
    assert cli.detect_model_type("some/unknown") == "content_guard"


def test_detect_model_type_unknown(monkeypatch):
    monkeypatch.setattr(ModelConfig, "from_model_name", classmethod(lambda cls, name: None))
    assert cli.detect_model_type("some/unknown") is None


def test_serve_builds_guard_and_runs_server(monkeypatch, prompt_guard):
    captured = {}

    def fake_create_guard(model_type, model, **kwargs):
        captured["model_type"] = model_type
        captured["model"] = model
        captured["kwargs"] = kwargs
        return prompt_guard

    def fake_run_server(app, host, port, log_level):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(cli, "create_guard", fake_create_guard)
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    exit_code = cli.main(
        [
            "serve",
            "RyanStudio/Mezzo-Prompt-Guard-v2-Small",
            "--dtype",
            "float16",
            "--onnx",
            "--compile-mode",
            "reduce-overhead",
            "--device",
            "cpu",
            "--port",
            "9000",
        ]
    )

    assert exit_code == 0
    assert captured["model_type"] == "prompt_guard"
    assert captured["model"] == "RyanStudio/Mezzo-Prompt-Guard-v2-Small"
    assert captured["kwargs"] == {
        "dtype": "float16",
        "torch_compile": False,
        "compile_mode": "reduce-overhead",
        "use_onnx": True,
        "device": "cpu",
    }
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9000
    assert captured["app"] is not None


def test_serve_explicit_type(monkeypatch, prompt_guard):
    captured = {}

    def fake_create_guard(model_type, model, **kwargs):
        captured["model_type"] = model_type
        return prompt_guard

    monkeypatch.setattr(cli, "create_guard", fake_create_guard)
    monkeypatch.setattr(cli, "run_server", lambda *args, **kwargs: None)

    assert cli.main(["serve", "some/model", "--type", "content"]) == 0
    assert captured["model_type"] == "content_guard"


def test_serve_unknown_model_type(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_model_type", lambda model: None)
    assert cli.main(["serve", "some/unknown"]) == 2
    assert "Could not detect" in capsys.readouterr().out
