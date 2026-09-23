import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "verify_android_jni", Path(__file__).resolve().parents[1] / "verify_android_jni.py"
)
jni = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jni)


def symbol(name, kind="FUNC", binding="GLOBAL", visibility="DEFAULT", index="12"):
    return f"  1: 0000000120 32 {kind} {binding} {visibility} {index} {name}\n"


class VerifyAndroidJniTest(unittest.TestCase):
    def setUp(self):
        self.name = jni.JNI_PREFIX + "run"
        self.glue = f"JNIEXPORT void JNICALL {self.name}(JNIEnv *env) {{}}"
        self.base = symbol(jni.BOLT_SYMBOL)

    def test_matching_defined_symbols(self):
        self.assertEqual(jni.verify(self.glue, self.base + symbol(self.name)), 1)

    def test_same_count_with_wrong_name_fails(self):
        with self.assertRaisesRegex(ValueError, "missing JNI exports"):
            jni.verify(self.glue, self.base + symbol(jni.JNI_PREFIX + "wrong"))

    def test_undefined_hidden_local_and_non_function_do_not_count(self):
        for kwargs in [dict(index="UND"), dict(visibility="HIDDEN"),
                       dict(binding="LOCAL"), dict(kind="OBJECT")]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "missing JNI"):
                jni.verify(self.glue, self.base + symbol(self.name, **kwargs))

    def test_unexpected_extra_export_fails(self):
        with self.assertRaisesRegex(ValueError, "unexpected JNI"):
            jni.verify(self.glue, self.base + symbol(self.name) + symbol(jni.JNI_PREFIX + "stale"))

    def test_no_generated_symbols_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no entry points"):
            jni.verify("", self.base)

    def test_missing_or_undefined_bolt_abi_fails(self):
        for base in ["", symbol(jni.BOLT_SYMBOL, index="UND")]:
            with self.subTest(base=base), self.assertRaisesRegex(ValueError, "Bolt C ABI"):
                jni.verify(self.glue, base + symbol(self.name))

    def test_symbol_versions_and_protected_exports(self):
        self.assertEqual(jni.verify(self.glue, self.base + symbol(self.name + "@@XYBRID", visibility="PROTECTED")), 1)


if __name__ == "__main__":
    unittest.main()
