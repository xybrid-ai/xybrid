"""Init/example regressions using the real generated Python API without a cdylib.

Only the compiled extension is stubbed. Loading the package under an isolated
name exercises its actual generated signatures, records, and typed errors.
The existing pytest suite also collects these tests; native artifacts are optional.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


SDK_ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE = "_xybrid_init_test"
# Records boltffi lays out as fixed-size C structs (every field a scalar) are
# types of the compiled bridge, aliased by the generated package as
# `Name = _native.Name`; it then attaches the wire codec to them.
NATIVE_STRUCTS = frozenset(
    re.findall(
        r"^(\w+) = _native\.\1$",
        (SDK_ROOT / "xybrid" / "_bolt" / "__init__.py").read_text(),
        re.MULTILINE,
    )
)


def _native_struct(name: str) -> type:
    """A positional record standing in for one compiled-bridge struct type."""

    def __init__(self, *args, **kwargs):
        fields = type(self).__match_args__
        values = dict(zip(fields, args), **kwargs)
        if len(args) > len(fields) or set(values) != set(fields):
            raise TypeError(f"{name} takes exactly {fields}")
        for field in fields:
            object.__setattr__(self, field, values[field])

    def __eq__(self, other):
        return type(other) is type(self) and all(
            getattr(self, field) == getattr(other, field) for field in self.__match_args__
        )

    return type(name, (), {"__init__": __init__, "__eq__": __eq__, "__hash__": None})


class _NativeStub(ModuleType):
    def __init__(self) -> None:
        super().__init__(f"{TEST_PACKAGE}._bolt._native")
        self.calls: list[tuple] = []
        self.init_error: Exception | None = None

    def __getattr__(self, name):
        if name.startswith("_register_"):
            return lambda *args: None
        if name in NATIVE_STRUCTS:
            struct_type = _native_struct(name)
            setattr(self, name, struct_type)
            return struct_type
        raise AttributeError(name)

    def _initialize_loader(self, path):
        pass

    def set_binding(self, binding):
        self.binding = binding

    def configure_runtime(self, api_key, gateway_url):
        self.calls.append(("runtime", api_key, gateway_url))

    def _boltffi_xybrid_observability_config_xybrid(self, api_key):
        self.calls.append(("export_key", api_key))
        return 1

    def _boltffi_xybrid_observability_config_set_traces_endpoint(self, handle, endpoint):
        self.calls.append(("endpoint", endpoint))

    def _boltffi_xybrid_observability_config_set_metadata_only(self, handle, metadata_only):
        self.calls.append(("metadata_only", metadata_only))

    def _boltffi_xybrid_observability_config_init(self, handle):
        self.calls.append(("export_init",))
        if self.init_error is not None:
            raise self.init_error

    def _boltffi_xybrid_observability_config_release(self, handle):
        pass


class InitTests(unittest.TestCase):
    def setUp(self):
        self.native = _NativeStub()
        self.modules = patch.dict(sys.modules, {self.native.__name__: self.native})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(
            TEST_PACKAGE,
            SDK_ROOT / "xybrid" / "__init__.py",
            submodule_search_locations=[str(SDK_ROOT / "xybrid")],
        )
        self.sdk = importlib.util.module_from_spec(spec)
        sys.modules[TEST_PACKAGE] = self.sdk
        spec.loader.exec_module(self.sdk)

    def _runtime(self, api_key=None, gateway_url=None):
        def encode(value):
            return self.sdk._bolt._boltffi_wire_optional(
                value, self.sdk._bolt._boltffi_wire_string
            )

        return ("runtime", encode(api_key), encode(gateway_url))

    def test_anonymous_init_uses_current_signature_and_is_idempotent(self):
        self.sdk.init()
        self.sdk.init(api_key="ignored", ingest_url="https://ignored.example")
        self.assertTrue(self.sdk.is_initialized())
        self.assertEqual(self.native.binding, "python")
        self.assertEqual(self.native.calls, [self._runtime()])

    def test_auth_and_gateway_alone_do_not_enable_or_replace_export(self):
        self.sdk.init(api_key="cloud-key", gateway_url="https://gateway.example/v1")
        self.assertEqual(
            self.native.calls,
            [self._runtime("cloud-key", "https://gateway.example/v1")],
        )

    def test_blank_ingest_url_does_not_enable_export(self):
        self.sdk.init(api_key="cloud-key", ingest_url="  ")
        self.assertEqual(self.native.calls, [self._runtime("cloud-key")])

    def test_explicit_legacy_ingest_maps_to_metadata_only_otlp_traces(self):
        self.sdk.init(
            api_key=" export-key ",
            gateway_url="https://gateway.example/v1",
            ingest_url=" https://collector.example/ ",
        )
        self.assertTrue(self.sdk.is_initialized())
        self.assertEqual(
            self.native.calls,
            [
                ("export_key", "export-key"),
                ("endpoint", "https://collector.example/v1/traces"),
                ("metadata_only", True),
                ("export_init",),
                self._runtime(" export-key ", "https://gateway.example/v1"),
            ],
        )
        self.sdk.init(ingest_url="invalid-ignored-after-success")
        self.assertEqual(len(self.native.calls), 5)

    def test_trace_endpoint_mapping_keeps_full_paths_and_query(self):
        for endpoint, expected in [
            ("http://localhost:4318", "http://localhost:4318/v1/traces"),
            (
                "https://collector.example/custom/traces?tenant=one",
                "https://collector.example/custom/traces?tenant=one",
            ),
            (
                "https://collector.example/?tenant=one",
                "https://collector.example/v1/traces?tenant=one",
            ),
        ]:
            with self.subTest(endpoint=endpoint):
                self.sdk._INITIALIZED = False
                self.native.calls.clear()
                self.sdk.init(api_key="key", ingest_url=endpoint)
                self.assertIn(("endpoint", expected), self.native.calls)

    def test_invalid_ingest_url_has_no_native_side_effects_and_can_retry(self):
        for endpoint in [
            "ftp://collector.example",
            "/relative",
            "https://",
            "http://host:bad",
            "http://host:99999",
            "http://[invalid",
        ]:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(self.sdk.ConfigError):
                    self.sdk.init(api_key="key", ingest_url=endpoint)
                self.assertFalse(self.sdk.is_initialized())
                self.assertEqual(self.native.calls, [])
        self.sdk.init(api_key="key", ingest_url="https://collector.example")
        self.assertTrue(self.sdk.is_initialized())

    def test_explicit_ingest_requires_explicit_key(self):
        for key in [None, "", "  "]:
            with self.subTest(key=key), self.assertRaises(self.sdk.ConfigError):
                self.sdk.init(api_key=key, ingest_url="https://collector.example")
            self.assertFalse(self.sdk.is_initialized())
            self.assertEqual(self.native.calls, [])

    def test_native_export_failure_propagates_typed_error_and_allows_retry(self):
        payload = self.sdk._bolt.XybridErrorConfigError(message="export failed")
        self.native.init_error = RuntimeError(payload._boltffi_wire())
        with self.assertRaisesRegex(self.sdk.ConfigError, "export failed"):
            self.sdk.init(api_key="key", ingest_url="https://collector.example")
        self.assertFalse(self.sdk.is_initialized())
        self.assertFalse(any(call[0] == "runtime" for call in self.native.calls))
        self.native.init_error = None
        self.sdk.init(api_key="key", ingest_url="https://collector.example")
        self.assertTrue(self.sdk.is_initialized())

    def test_native_struct_aliases_are_the_only_bridge_types_the_package_imports(self):
        # Guards the stub above: if the generator aliases another bridge type,
        # NATIVE_STRUCTS must pick it up rather than every test failing on import.
        self.assertIn("XybridSignalStats", NATIVE_STRUCTS)
        self.assertIs(self.sdk.XybridSignalStats, self.native.XybridSignalStats)

    def test_per_signal_stats_round_trip_through_the_generated_codec(self):
        stats = self.sdk.XybridObservabilityStats(
            enabled=True,
            traces=self.sdk.XybridSignalStats(1, 2, 3, 4, 5),
            logs=self.sdk.XybridSignalStats(0, 0, 0, 0, 0),
            metrics=self.sdk.XybridSignalStats(6, 7, 8, 9, 2**64 - 1),
        )
        wire = stats._boltffi_wire()
        self.assertEqual(len(wire), 1 + 3 * 5 * 8)
        decoded = self.sdk.XybridObservabilityStats._boltffi_from_wire(wire)
        self.assertEqual(decoded, stats)
        self.assertEqual(decoded.metrics.pending, 2**64 - 1)

    def test_documented_run_options_match_generated_constructor(self):
        readme = (SDK_ROOT / "README.md").read_text()
        sources = [
            ("quickstart", (SDK_ROOT / "examples" / "quickstart.py").read_text()),
            *[
                (f"README block {index}", block)
                for index, block in enumerate(
                    re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
                )
            ],
        ]
        count = 0
        for name, source in sources:
            for node in ast.walk(ast.parse(source)):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "XybridRunOptions"
                ):
                    continue
                with self.subTest(source=name):
                    # The config variable is provided by preceding example code;
                    # all other arguments must be real explicit constructor inputs.
                    values = {
                        keyword.arg: (
                            None if isinstance(keyword.value, ast.Name)
                            else ast.literal_eval(keyword.value)
                        )
                        for keyword in node.keywords
                    }
                    options = self.sdk.XybridRunOptions(**values)
                    self.assertIsNone(options.traceparent)
                    self.assertIsNone(options.tracestate)
                    self.assertIsNone(options.conversation_id)
                    self.assertFalse(options.suppress_trace)
                    options._boltffi_wire()
                count += 1
        self.assertEqual(count, 2, "must exercise both quickstart and README options")


if __name__ == "__main__":
    unittest.main()
