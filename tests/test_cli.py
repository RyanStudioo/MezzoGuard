import pytest

from mezzoguard import cli


def test_parser_defaults():
    args = cli.build_parser().parse_args(["serve", "some/model"])
    assert args.command == "serve"
    assert args.model == "some/model"
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


def test_serve_builds_guard_and_runs_server(monkeypatch, prompt_guard):
    captured = {}

    def fake_create_guard(model, **kwargs):
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


def test_create_guard_dispatches_unified(monkeypatch):
    import mezzoguard.guard as guard_module

    captured = {}

    class FakeGuard:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setattr(guard_module, "Guard", FakeGuard)

    guard = cli.create_guard("some/model", dtype="float16")
    assert isinstance(guard, FakeGuard)
    assert captured == {"model": "some/model", "kwargs": {"dtype": "float16"}}
