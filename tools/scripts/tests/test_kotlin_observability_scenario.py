"""Check the Kotlin scenario's device-to-receiver setup without an emulator."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "bindings/kotlin/consumer-smoke/observability-scenario.sh"
)


class KotlinObservabilityScenarioTests(unittest.TestCase):
    def run_scenario(self, host="127.0.0.1", failures=0, reverse_status=0):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            for name in ("debug/app-debug.apk", "androidTest/debug/app-debug-androidTest.apk"):
                apk = root / "app/build/outputs/apk" / name
                apk.parent.mkdir(parents=True, exist_ok=True)
                apk.touch()
            adb = root / "adb"
            adb.write_text(
                "#!/usr/bin/env bash\n"
                'printf "%s\\n" "$*" >> "$ADB_LOG"\n'
                'case "$*" in\n'
                '  "reverse --no-rebind "*) exit "$REVERSE_STATUS" ;;\n'
                '  "shell -T "*)\n'
                '    [[ "$*" == *"sleep 3; } | nc -w 2 -W 2 127.0.0.1 4318"* ]] || exit 1\n'
                '    count=$(grep -c "^shell -T " "$ADB_LOG")\n'
                '    if ((count <= FAILURES)); then echo "HTTP/1.0 404 Not Found"; '
                "else printf '%s\\n' 'HTTP/1.0 200 OK' '{\"status\":\"ok\"}'; fi ;;\n"
                '  "shell am instrument "*) echo "OK (1 test)" ;;\n'
                '  "exec-out "*) echo "{}" ;;\n'
                '  "logcat -d") echo "device logs" ;;\n'
                "esac\n",
                encoding="utf-8",
            )
            adb.chmod(0o755)
            sleep = root / "sleep"
            sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            sleep.chmod(0o755)
            log = root / "adb.log"
            result = subprocess.run(
                ["bash", str(script), f"http://{host}:4318", str(root / "expectations.json")],
                env=dict(os.environ, ADB=str(adb), ADB_LOG=str(log), FAILURES=str(failures),
                         REVERSE_STATUS=str(reverse_status), PATH=f"{root}{os.pathsep}{os.environ['PATH']}"),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result, log.read_text().splitlines(), (root / "logcat.txt").read_text()

    def test_forward_probe_instrument_and_remove_in_order(self):
        result, calls, _ = self.run_scenario(failures=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("The device reaches the receiver", result.stdout)
        forward = calls.index("reverse --no-rebind tcp:4318 tcp:4318")
        probes = [index for index, call in enumerate(calls) if call.startswith("shell -T ")]
        self.assertEqual(len(probes), 2)
        self.assertIn("GET /healthz HTTP/1.0", calls[probes[0]])
        instrument = next(index for index, call in enumerate(calls) if call.startswith("shell am instrument "))
        self.assertIn("http://127.0.0.1:4318", calls[instrument])
        self.assertLess(forward, min(probes))
        self.assertLess(max(probes), instrument)
        self.assertEqual(calls[-1], "reverse --remove tcp:4318")

    def test_unreachable_receiver_dumps_logs_without_instrumenting(self):
        result, calls, logcat = self.run_scenario(failures=10)
        self.assertEqual(result.returncode, 3)
        self.assertIn("unreachable from the device", result.stderr)
        self.assertEqual(sum(call.startswith("shell -T ") for call in calls), 10)
        self.assertFalse(any(call.startswith("shell am instrument ") for call in calls))
        self.assertEqual(calls[-1], "reverse --remove tcp:4318")
        self.assertIn("device logs", logcat)

    def test_forward_conflict_does_not_remove_an_existing_mapping(self):
        result, calls, logcat = self.run_scenario(reverse_status=1)
        self.assertEqual(result.returncode, 3)
        self.assertIn("could not forward", result.stderr)
        self.assertFalse(any(call.startswith("reverse --remove ") for call in calls))
        self.assertIn("device logs", logcat)

    def test_non_loopback_leaves_connection_untouched(self):
        result, calls, _ = self.run_scenario(host="10.0.2.2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(call.startswith(("reverse ", "shell -T ")) for call in calls))
        self.assertTrue(any("http://10.0.2.2:4318" in call for call in calls))


if __name__ == "__main__":
    unittest.main()
