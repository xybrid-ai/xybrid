"""The agent skills in agents/skills/ and their mirror in .agents/skills/.

agents/skills/ is canonical: .claude/skills links to it, and
tools/scripts/install-skills.sh downloads from it. .agents/skills/ is a copy
for agents that read that directory. Every file in a mirrored skill is
byte-identical to the canonical one, except that a canonical SKILL.md without
YAML frontmatter is mirrored with frontmatter (`name`, `description`) put in
front of an identical body. Copies drifted before this test (the xybrid-init
body did), so a skill edit that forgets its mirror now fails.

xybrid-connect also carries a contract with the CLI and the platform: its
exit-code table must match `xybrid connect`'s (crates/xybrid-cli/src/platform/
report.rs), its proof-of-work vectors must hold under an independent
implementation, the SHA-256 it prints must be its reference script's, and the
script must run the connect contract end to end against a loopback platform:
one workspace, a gitignored file, private recovery state, no key on stdout.

CI's Agent skills job runs this file on every pull request, docs-only ones
included, because skills are Markdown and edited in docs-only PRs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "agents" / "skills"
MIRROR = ROOT / ".agents" / "skills"
CONNECT = CANONICAL / "xybrid-connect"
SCRIPT = CONNECT / "scripts" / "xybrid_agent_connect.py"
REPORT = ROOT / "crates" / "xybrid-cli" / "src" / "platform" / "report.rs"
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def split_frontmatter(text: str) -> tuple[str | None, str]:
    match = FRONTMATTER.match(text)
    if not match:
        return None, text
    return match.group(1), text[match.end():]


def skill_files(skill: Path) -> list[str]:
    """Files a skill ships: bytecode caches and editor droppings are not part of it."""
    return sorted(
        p.relative_to(skill).as_posix()
        for p in skill.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".DS_Store"
    )


class MirrorTests(unittest.TestCase):
    def test_claude_skills_link_to_the_canonical_directory(self):
        link = ROOT / ".claude" / "skills"
        self.assertTrue(link.is_symlink(), ".claude/skills must stay a symlink")
        self.assertEqual(os.readlink(link), "../agents/skills")

    def test_every_skill_is_mirrored_with_the_same_files(self):
        canonical = sorted(p.name for p in CANONICAL.iterdir() if p.is_dir())
        mirrored = sorted(p.name for p in MIRROR.iterdir() if p.is_dir())
        self.assertEqual(mirrored, canonical)
        for name in canonical:
            with self.subTest(skill=name):
                self.assertEqual(skill_files(MIRROR / name), skill_files(CANONICAL / name))

    def test_mirrored_files_match_byte_for_byte(self):
        for skill in sorted(p for p in CANONICAL.iterdir() if p.is_dir()):
            for rel in skill_files(skill):
                with self.subTest(file=f"{skill.name}/{rel}"):
                    source, copy = (skill / rel).read_bytes(), (MIRROR / skill.name / rel).read_bytes()
                    if rel != "SKILL.md":
                        self.assertEqual(copy, source)
                        continue
                    front, body = split_frontmatter(source.decode("utf-8"))
                    mirror_front, mirror_body = split_frontmatter(copy.decode("utf-8"))
                    if front is not None:
                        self.assertEqual(copy, source, "a SKILL.md with frontmatter is mirrored as-is")
                    else:
                        self.assertIsNotNone(mirror_front, "the mirror carries the frontmatter")
                        self.assertEqual(mirror_body.lstrip("\n"), body, "the mirror's body is the canonical file")

    def test_mirror_frontmatter_names_the_skill(self):
        for skill in sorted(p for p in MIRROR.iterdir() if p.is_dir()):
            with self.subTest(skill=skill.name):
                front, _ = split_frontmatter((skill / "SKILL.md").read_text(encoding="utf-8"))
                self.assertIsNotNone(front)
                fields = dict(line.split(": ", 1) for line in front.splitlines() if ": " in line)
                name = fields.get("name", "").strip('"')
                self.assertEqual(name, skill.name)
                self.assertTrue(fields.get("description", "").strip('" '), "a description is required")

    def test_canonical_frontmatter_is_json_quoted(self):
        # xybrid-platform publishes skills with frontmatter by JSON-parsing
        # each value (web/src/lib/server/agent-skills.ts), so a canonical
        # skill that has frontmatter keeps them double-quoted JSON strings.
        for skill in sorted(p for p in CANONICAL.iterdir() if p.is_dir()):
            front, _ = split_frontmatter((skill / "SKILL.md").read_text(encoding="utf-8"))
            if front is None:
                continue
            with self.subTest(skill=skill.name):
                for key in ("name", "description"):
                    match = re.search(rf"^{key}: (.+)$", front, re.M)
                    self.assertIsNotNone(match, key)
                    self.assertIsInstance(json.loads(match.group(1)), str)


def load_script():
    """The reference script as a module, without leaving a __pycache__ in the skill."""
    spec = importlib.util.spec_from_file_location("xybrid_agent_connect", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def independent_bits(challenge: str, nonce: int) -> int:
    digest = hashlib.sha256(f"{challenge}:{nonce}".encode("utf-8")).digest()
    value = int.from_bytes(digest, "big")
    return 256 - value.bit_length()


class ConnectSkillTests(unittest.TestCase):
    skill = (CONNECT / "SKILL.md").read_text(encoding="utf-8")

    def test_exit_codes_match_the_cli(self):
        # report.rs documents its contract as `//! | <code> | `status`, ... |`.
        cli = {}
        for code, statuses in re.findall(r"^//! \| (\d) \| (.+?) \|$", REPORT.read_text(encoding="utf-8"), re.M):
            # A parenthesis explains the row; statuses named there belong to other rows.
            for status in re.findall(r"`([a-z_]+)`", re.sub(r"\(.*?\)", "", statuses)):
                cli.setdefault(status, int(code))
        table = {}
        for code, statuses in re.findall(r"^\| (\d) \| (`[a-z_]+`(?:, `[a-z_]+`)*) \|", self.skill, re.M):
            for status in re.findall(r"`([a-z_]+)`", statuses):
                table[status] = int(code)
        self.assertGreaterEqual(len(table), 9, table)
        for status, code in table.items():
            with self.subTest(status=status):
                self.assertEqual(cli.get(status), code, f"the CLI exits {cli.get(status)} for {status}")
        script = load_script()
        for status, code in table.items():
            with self.subTest(script_status=status):
                self.assertEqual(script.EXIT.get(status), code)

    def test_proof_of_work_vectors_hold(self):
        rows = re.findall(r"^\| `([^`]+)` \| (\d+) \| (\d+) \| (\d+) \| `([0-9a-f]{64})` \|$", self.skill, re.M)
        self.assertGreaterEqual(len(rows), 8)
        self.assertTrue(any(int(d) % 8 for _, d, _, _, _ in rows), "some difficulty is not byte-aligned")
        script = load_script()
        for challenge, difficulty, nonce, bits, digest in rows:
            difficulty, nonce, bits = int(difficulty), int(nonce), int(bits)
            with self.subTest(challenge=challenge, difficulty=difficulty):
                self.assertEqual(hashlib.sha256(f"{challenge}:{nonce}".encode()).hexdigest(), digest)
                self.assertEqual(independent_bits(challenge, nonce), bits)
                self.assertGreaterEqual(bits, difficulty)
                if difficulty <= 17:  # smallest: every smaller nonce falls short
                    self.assertTrue(all(independent_bits(challenge, n) < difficulty for n in range(nonce)))
                    self.assertEqual(script.solve(challenge, difficulty, float("inf")), nonce)

    def test_printed_digest_is_the_scripts(self):
        printed = re.findall(r"# sha256: ([0-9a-f]{64})", self.skill)
        self.assertEqual(printed, [hashlib.sha256(SCRIPT.read_bytes()).hexdigest()])


# --- the reference script against a loopback platform ------------------------

KEY = "sk_live_" + "FAKE_TEST_KEY"  # obviously fake, and split so no scanner reads it as a key
TOKEN = "TestClaimTokenTestClaimTokenTestClaimToken1"


class Platform(BaseHTTPRequestHandler):
    """The /v1/agent/* answers of xybrid-platform's backend/src/api/agent.rs."""

    closed = False
    requests: list = []

    def log_message(self, *args):
        pass

    def reply(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        type(self).requests.append(("GET", self.path, None, dict(self.headers)))
        if self.closed:
            return self.reply(403, {"error": "agent_signup_closed", "message": "closed", "reason": "disabled"})
        if self.path == "/v1/agent/challenge":
            return self.reply(200, {"challenge": "xybrid-pow-test-vector", "difficulty": 8,
                                    "expires_at": "2999-01-01T00:00:00Z", "algorithm": "sha256-leading-zero-bits"})
        if self.path == "/v1/agent/workspace" and self.headers.get("Authorization") == f"Bearer {KEY}":
            return self.reply(200, {"organization": {"slug": "app", "name": "app"}, "project": {"slug": "default",
                                    "name": "Default"}, "scope": "client", "provisional": True, "claimed": False,
                                    "expired": False, "key_expires_at": "2999-01-01T00:00:00Z",
                                    "claim_expires_at": "2999-01-08T00:00:00Z",
                                    "verification": {"state": "received", "traces_received": True, "spans": 2,
                                                     "traces": 1, "first_ingested_at": None, "last_ingested_at": None}})
        self.reply(401, {"error": "invalid_api_key", "message": "no"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        type(self).requests.append(("POST", self.path, body, dict(self.headers)))
        if self.closed:
            return self.reply(403, {"error": "agent_signup_closed", "message": "closed", "reason": "disabled"})
        if not body["nonce"].isdigit() or independent_bits(body["challenge"], int(body["nonce"])) < 8:
            return self.reply(400, {"error": "proof_of_work_invalid", "message": "no"})
        self.reply(201, {"api_key": KEY, "key_prefix": KEY[8:16], "scope": "client",
                         "organization": {"slug": "app", "name": "app"}, "project": {"slug": "default", "name": "Default"},
                         # chrono writes nanoseconds, which Python < 3.11 can't parse as-is.
                         "key_expires_at": "2999-01-01T00:00:00.123456789Z", "claim_expires_at": "2999-01-08T00:00:00Z",
                         "claim_token": TOKEN, "limits": {"gateway_daily_requests": 25, "max_projects": 5},
                         "replayed": False})


@unittest.skipUnless(shutil.which("git"), "git is required")
class ScriptContractTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.home = self.root / "home"
        self.home.mkdir()
        (self.home / "gitconfig").write_text("")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Platform)
        Platform.closed, Platform.requests = False, []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.dir.cleanup()

    def repo(self, name):
        path = self.root / name
        path.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=self.env())
        return path

    def env(self, **extra):
        keep = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT") if k in os.environ}
        keep.update(HOME=str(self.home), USERPROFILE=str(self.home), GIT_CONFIG_NOSYSTEM="1",
                    GIT_CONFIG_GLOBAL=str(self.home / "gitconfig"))
        keep.update(extra)
        return keep

    def run_script(self, cwd, *args, **env):
        result = subprocess.run([sys.executable, "-I", str(SCRIPT), *args, "--platform-url", self.url,
                                 "--allow-custom-host"], cwd=cwd, env=self.env(**env), capture_output=True, text=True)
        self.assertNotIn(KEY, result.stdout + result.stderr, "the key never reaches the output")
        return result.returncode, json.loads(result.stdout)

    def test_connect_writes_an_ignored_file_and_private_recovery(self):
        app = self.repo("app")
        code, report = self.run_script(app, "connect", "--write-env", ".env", "--agent", "claude-code",
                                       "--binding", "python", XYBRID_CONSOLE_URL="http://localhost:5173")
        self.assertEqual((code, report["status"]), (0, "created"), report)
        self.assertEqual(report["claim_url"], f"http://localhost:5173/claim/{TOKEN}")
        self.assertTrue(report["env_file"]["written"] and report["env_file"]["gitignored"])
        self.assertEqual(report["key"]["preview"], "sk_live_FAKE…")
        text = (app / ".env").read_text()
        self.assertIn(f"\nXYBRID_API_KEY={KEY}\n", "\n" + text)
        self.assertNotIn(TOKEN, text, "the claim token never goes into the project file")
        ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=app, env=self.env())
        self.assertEqual(ignored.returncode, 0)
        store = self.home / ".xybrid" / "credentials.json"
        saved = json.loads(store.read_text())
        self.assertEqual(saved["version"], 1)
        self.assertEqual(saved["pending"], [])
        (entry,) = saved["credentials"]
        self.assertEqual((entry["api_key"], entry["claim_token"]), (KEY, TOKEN))
        self.assertEqual(entry["repository_root"], str(app.resolve()))
        if os.name == "posix":
            self.assertEqual(store.stat().st_mode & 0o777, 0o600)
            self.assertEqual(store.parent.stat().st_mode & 0o777, 0o700)
        posts = [r for r in Platform.requests if r[0] == "POST"]
        self.assertEqual(len(posts), 1)
        _, _, body, headers = posts[0]
        self.assertRegex(headers["Idempotency-Key"], r"^[A-Za-z0-9_-]{32,128}$")
        self.assertLessEqual(set(body), {"source", "project_name", "install_id", "client_info", "challenge", "nonce"})
        self.assertEqual(body["nonce"], "47")

        code, again = self.run_script(app, "connect", "--write-env", ".env")
        self.assertEqual((code, again["status"]), (0, "existing"))
        self.assertEqual(len([r for r in Platform.requests if r[0] == "POST"]), 1, "no second workspace")
        code, status = self.run_script(app, "status")
        self.assertEqual((code, status["status"], status["verification"]["traces_received"]), (0, "connected", True))

    def test_closed_tracked_ci_and_opt_out_create_nothing(self):
        tracked = self.repo("tracked")
        (tracked / ".env").write_text("OTHER=1\n")
        subprocess.run(["git", "add", ".env"], cwd=tracked, check=True, env=self.env())
        code, report = self.run_script(tracked, "connect", "--write-env", ".env")
        self.assertEqual((code, report["status"], report["env_file"]["suggestions"]), (6, "env_file_tracked", [".env.local"]))
        self.assertEqual(self.run_script(tracked, "connect", "--write-env", ".env.local", CI="true")[0], 5)
        self.assertEqual(self.run_script(tracked, "connect", "--write-env", ".env.local", DO_NOT_TRACK="1")[0], 5)
        Platform.closed = True
        code, report = self.run_script(tracked, "connect", "--write-env", ".env.local")
        self.assertEqual((code, report["status"], report["error"]["code"]), (3, "signup_closed", "agent_signup_closed"))
        self.assertFalse((tracked / ".env.local").exists())
        self.assertFalse([r for r in Platform.requests if r[0] == "POST"])


if __name__ == "__main__":
    unittest.main()
