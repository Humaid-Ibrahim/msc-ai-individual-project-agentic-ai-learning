"""Offline regression tests for hint-generator credential handling."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATHS = (
    REPOSITORY_ROOT / "alfworld_runs" / "hints_creation.py",
    REPOSITORY_ROOT / "webshop_runs" / "hints_creation.py",
)


def load_module(path: Path):
    module_name = f"credential_test_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HintCredentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = tuple(load_module(path) for path in MODULE_PATHS)

    def test_missing_key_fails_before_openai_client_construction(self):
        class UnexpectedOpenAI:
            def __init__(self, **_kwargs):
                raise AssertionError("OpenAI client must not be constructed")

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = UnexpectedOpenAI

        for module in self.modules:
            for key_value in (None, "", "   "):
                with self.subTest(module=module.__name__, key_value=key_value):
                    with mock.patch.dict(os.environ, {}, clear=False):
                        if key_value is None:
                            os.environ.pop("OPENAI_API_KEY", None)
                        else:
                            os.environ["OPENAI_API_KEY"] = key_value
                        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                                module.get_agent_and_model("GPTChat")

    def test_environment_key_is_forwarded_without_network_access(self):
        received = []

        class RecordingOpenAI:
            def __init__(self, **kwargs):
                received.append(kwargs)

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = RecordingOpenAI

        for module in self.modules:
            with self.subTest(module=module.__name__):
                received.clear()
                with mock.patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "environment-test-value"},
                    clear=False,
                ):
                    with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                        module.get_agent_and_model("GPTChat")
                self.assertEqual(received[0]["api_key"], "environment-test-value")

    def test_programmatic_secret_injection_takes_precedence(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                with mock.patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "environment-test-value"},
                    clear=False,
                ):
                    self.assertEqual(
                        module._resolve_openai_api_key("injected-test-value"),
                        "injected-test-value",
                    )

    def test_local_model_selection_does_not_require_openai_key(self):
        class FakeVLLMChat:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        for module in self.modules:
            with self.subTest(module=module.__name__):
                with mock.patch.object(module, "VLLMChat", FakeVLLMChat, create=True):
                    with mock.patch.dict(os.environ, {}, clear=False):
                        os.environ.pop("OPENAI_API_KEY", None)
                        agent, model = module.get_agent_and_model("VLLMChat")
                self.assertIsInstance(agent, FakeVLLMChat)
                self.assertEqual(model, module.AGENT_MODEL_MAPPING["VLLMChat"][0])

    def test_cli_does_not_accept_api_keys(self):
        for path in MODULE_PATHS:
            with self.subTest(path=path):
                self.assertNotIn("--openai_api_key", path.read_text())


if __name__ == "__main__":
    unittest.main()
