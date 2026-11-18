from pathlib import Path

from mezzoguard.model import Model


class DummyModel(Model):
    """Model instance without loading a pipeline, for cache-path unit tests."""

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def _from_prediction(cls, results: list):
        return results


def test_onnx_dir_for_local_model_is_not_the_model_dir(tmp_path):
    model_dir = tmp_path / "my-guard"
    model_dir.mkdir()

    export_dir = Path(DummyModel(str(model_dir))._resolve_onnx_dir())

    assert export_dir != model_dir
    assert export_dir.parent.name == "local"
    assert export_dir.name.startswith("my-guard-")


def test_onnx_dir_is_stable_for_the_same_local_model(tmp_path):
    model_dir = tmp_path / "my-guard"
    model_dir.mkdir()

    assert DummyModel(str(model_dir))._resolve_onnx_dir() == DummyModel(str(model_dir))._resolve_onnx_dir()


def test_onnx_dir_distinguishes_local_models_with_the_same_basename(tmp_path):
    first = tmp_path / "first" / "my-guard"
    second = tmp_path / "second" / "my-guard"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    assert DummyModel(str(first))._resolve_onnx_dir() != DummyModel(str(second))._resolve_onnx_dir()


def test_onnx_dir_for_hub_model_keeps_org_and_repo():
    export_dir = Path(DummyModel("RyanStudio/Mezzo-Content-Guard-Small")._resolve_onnx_dir())

    assert export_dir.parts[-2:] == ("RyanStudio", "Mezzo-Content-Guard-Small")


def test_onnx_dir_sanitizes_repo_ids():
    export_dir = Path(DummyModel("../../evil")._resolve_onnx_dir())

    assert ".." not in export_dir.parts
    assert export_dir.parts[-1] == "evil"


def test_onnx_export_complete_requires_an_onnx_file(tmp_path):
    assert not DummyModel._onnx_export_complete(str(tmp_path))

    (tmp_path / "config.json").write_text("{}")
    assert not DummyModel._onnx_export_complete(str(tmp_path))

    (tmp_path / "model.onnx").write_bytes(b"")
    assert DummyModel._onnx_export_complete(str(tmp_path))


def test_onnx_export_complete_is_false_for_missing_dir(tmp_path):
    assert not DummyModel._onnx_export_complete(str(tmp_path / "missing"))
