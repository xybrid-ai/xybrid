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
import shlex
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

    def section(self, heading: str) -> str:
        start = self.skill.index(f"\n## {heading}")
        end = self.skill.find("\n## ", start + 1)
        return self.skill[start:end if end >= 0 else len(self.skill)]

    def test_unity_builds_never_pack_a_stale_key(self):
        # A failed player build skips OnPostprocessBuild, so the resource a
        # keyed build generated stays behind. A later build without the key
        # must remove it before deciding there is nothing to copy, or Unity
        # packs the old key into the player and it exports.
        recipe = self.section("Step 6: Wire explicit, metadata-only trace export")
        (step,) = [b for b in re.findall(r"```csharp\n(.*?)```", recipe, re.S) if "XybridKeyBuildStep" in b]

        def method(name: str) -> str:
            match = re.search(rf"\b{name}\([^)]*\)\s*(=>[^;]*;|\{{.*?\n    \}})", step, re.S)
            self.assertIsNotNone(match, name)
            return match.group(1)

        pre, post, delete = method("OnPreprocessBuild"), method("OnPostprocessBuild"), method("DeleteKeyResource")
        self.assertIn("DeleteKeyResource()", pre)
        self.assertLess(pre.index("DeleteKeyResource()"), pre.index("return"), "removed before the no-key return")
        self.assertIn("DeleteKeyResource()", post)
        self.assertIn("AssetDatabase.DeleteAsset(AssetPath)", delete)
        self.assertIn("File.Delete(AssetPath)", delete, "a file the AssetDatabase never imported goes too")
        self.assertNotIn("the next build replaces it", recipe)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_every_verification_command_uses_the_file_key(self):
        # With key.source "env_file", connect saved no key, and the CLI never
        # reads project files: each xybrid command of Step 7 (both auth status
        # checks and the fallback run) needs that file's key, read without
        # printing it, in a shell of its own (agent shells keep no state).
        step = self.section("Step 7: Verify one real trace")
        lines = [line.strip() for b in re.findall(r"```bash\n(.*?)```", step, re.S) for line in b.splitlines()]
        setup = [line for line in lines if re.match(r"^(FILE=|[a-z_]+\(\) \{)", line)]
        self.assertTrue(setup, "Step 7 says how to read the file's key")
        commands = [line for line in lines if re.search(r"(^|\s)xybrid\s", line) and not line.startswith("#")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            log = tmp / "calls.log"
            stub = tmp / "bin" / "xybrid"
            stub.parent.mkdir()
            stub.write_text('#!/bin/sh\nprintf "%s|%s\\n" "${XYBRID_API_KEY:-}" "$*" >> "$CALLS"\n')
            stub.chmod(0o755)
            project = tmp / "app"
            project.mkdir()
            (project / ".env.local").write_text(f"OTHER=1\nEXPO_PUBLIC_XYBRID_API_KEY={KEY}\n")
            seen = {}
            for command in commands:
                result = subprocess.run(["bash", "-c", "\n".join(setup + [command])], cwd=project, capture_output=True,
                                        text=True, env={"PATH": f"{stub.parent}{os.pathsep}{os.environ['PATH']}",
                                                        "CALLS": str(log), "TMPDIR": str(tmp)})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn(KEY, result.stdout + result.stderr, "the key is never printed")
            for line in log.read_text().splitlines():
                key, args = line.split("|", 1)
                kind = "run" if "--observability run" in args else "status" if args.startswith("auth status") else args
                seen.setdefault(kind, set()).add(key)
        self.assertEqual(set(seen), {"status", "run"}, seen)
        for kind, keys in seen.items():
            with self.subTest(command=kind):
                self.assertIn(KEY, keys, f"no `{kind}` command in Step 7 passes the file's key")


# --- the reference script against a loopback platform ------------------------

KEY = "sk_live_" + "FAKE_TEST_KEY"  # obviously fake, and split so no scanner reads it as a key
OTHER_KEY = "sk_live_" + "FAKE_OTHER_KEY"  # a key the user configured themselves
REVOKED_KEY = "sk_live_" + "FAKE_REVOKED_KEY"
TOKEN = "TestClaimTokenTestClaimTokenTestClaimToken1"


class Platform(BaseHTTPRequestHandler):
    """The /v1/agent/* answers of xybrid-platform's backend/src/api/agent.rs."""

    closed = False
    requests: list = []
    workspace: dict = {}  # fields that override the GET /v1/agent/workspace answer
    workspace_failure = None  # (status, body, headers) instead of that answer
    provision: dict = {}  # fields that override the POST /v1/agent/connect answer

    def log_message(self, *args):
        pass

    def reply(self, status, body, headers=None):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
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
        bearer = self.headers.get("Authorization")
        if self.path == "/v1/agent/workspace" and bearer in (f"Bearer {KEY}", f"Bearer {OTHER_KEY}"):
            if type(self).workspace_failure:
                return self.reply(*type(self).workspace_failure)
            # Each key names its own workspace, so a test can tell which one was checked.
            name = "app" if bearer == f"Bearer {KEY}" else "other"
            answer = {"organization": {"slug": name, "name": name}, "project": {"slug": "default",
                      "name": "Default"}, "scope": "client", "provisional": True, "claimed": False,
                      "expired": False, "key_expires_at": "2999-01-01T00:00:00Z",
                      "claim_expires_at": "2999-01-08T00:00:00Z",
                      "verification": {"state": "received", "traces_received": True, "spans": 2,
                                       "traces": 1, "first_ingested_at": None, "last_ingested_at": None}}
            answer.update(type(self).workspace)
            return self.reply(200, answer)
        self.reply(401, {"error": "invalid_api_key", "message": "no"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        type(self).requests.append(("POST", self.path, body, dict(self.headers)))
        if self.closed:
            return self.reply(403, {"error": "agent_signup_closed", "message": "closed", "reason": "disabled"})
        if not body["nonce"].isdigit() or independent_bits(body["challenge"], int(body["nonce"])) < 8:
            return self.reply(400, {"error": "proof_of_work_invalid", "message": "no"})
        answer = {"api_key": KEY, "key_prefix": KEY[8:16], "scope": "client",
                  "organization": {"slug": "app", "name": "app"}, "project": {"slug": "default", "name": "Default"},
                  # chrono writes nanoseconds, which Python < 3.11 can't parse as-is.
                  "key_expires_at": "2999-01-01T00:00:00.123456789Z", "claim_expires_at": "2999-01-08T00:00:00Z",
                  "claim_token": TOKEN, "limits": {"gateway_daily_requests": 25, "max_projects": 5}, "replayed": False}
        answer.update(type(self).provision)
        self.reply(201, answer)


@unittest.skipUnless(shutil.which("git"), "git is required")
class ScriptContractTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.home = self.root / "home"
        self.home.mkdir()
        (self.home / "gitconfig").write_text("")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Platform)
        Platform.closed, Platform.requests, Platform.workspace, Platform.workspace_failure = False, [], {}, None
        Platform.provision = {}
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

    def run_script(self, cwd, *args, flags=(), **env):
        result = subprocess.run([sys.executable, *flags, "-I", str(SCRIPT), *args, "--platform-url", self.url,
                                 "--allow-custom-host"], cwd=cwd, env=self.env(**env), capture_output=True)
        out, err = result.stdout.decode("utf-8"), result.stderr.decode("utf-8", "replace")
        for key in (KEY, OTHER_KEY, REVOKED_KEY):
            self.assertNotIn(key, out + err, "a key never reaches the output")
        self.assertEqual(err, "")
        return result.returncode, json.loads(out)

    @property
    def store(self):
        return self.home / ".xybrid" / "credentials.json"

    def seed(self, app, **fields):
        """Save a credential for `app`, as an earlier connect would have."""
        credential = {"platform_origin": self.url, "repository_root": str(app.resolve()), "api_key": KEY,
                      "scope": "client", "organization": {"slug": "app", "name": "app"},
                      "project": {"slug": "default", "name": "Default"}, "key_expires_at": "2999-01-01T00:00:00Z",
                      "claim_expires_at": "2999-01-08T00:00:00Z", "claim_token": TOKEN, "claimed": False,
                      "created_at": "2026-01-01T00:00:00Z"}
        credential.update(fields)
        self.store.parent.mkdir(mode=0o700, exist_ok=True)
        self.store.write_text(json.dumps({"version": 1, "credentials": [credential], "pending": [], "retired": []}))
        self.store.chmod(0o600)

    def saved(self):
        return json.loads(self.store.read_text())

    def workspace_checks(self):
        return [r[3]["Authorization"] for r in Platform.requests if r[1] == "/v1/agent/workspace"]

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
            self.assertEqual((app / ".env").stat().st_mode & 0o777, 0o600, "a new key file is private, as the CLI's")
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

    def test_a_tracked_default_alternative_is_not_suggested(self):
        app = self.repo("expo")
        (app / ".env.local").write_text("OTHER=1\n")
        subprocess.run(["git", "add", ".env.local"], cwd=app, check=True, env=self.env())
        code, report = self.run_script(app, "connect", "--write-env", ".env.local")
        self.assertEqual((code, report["status"]), (6, "env_file_tracked"))
        self.assertEqual(report["env_file"]["suggestions"], [".env.development.local"])
        self.assertIn(".env.development.local", report["tell_the_user"])

    def test_a_properties_key_with_a_blank_separator_is_kept(self):
        # java.util.Properties reads `xybrid.apiKey value` as an assignment, as the CLI does.
        app = self.repo("android")
        own = "sk_live_" + "OWN"
        (app / "local.properties").write_text(f"sdk.dir=/sdk\nxybrid.apiKey {own}\n")
        code, report = self.run_script(app, "connect", "--write-env", "local.properties")
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "existing", "env_file"))
        self.assertEqual((app / "local.properties").read_text(), f"sdk.dir=/sdk\nxybrid.apiKey {own}\n")
        self.assertFalse([r for r in Platform.requests if r[0] == "POST"])

    def test_a_store_in_another_repository_is_ignored_there_or_refused(self):
        app, other = self.repo("app"), self.repo("dotfiles")
        store = other / ".xybrid" / "credentials.json"
        code, report = self.run_script(app, "connect", "--write-env", ".env", XYBRID_CREDENTIALS_FILE=str(store))
        self.assertEqual((code, report["status"]), (0, "created"), report)
        ignored = subprocess.run(["git", "check-ignore", "-q", ".xybrid/credentials.json"], cwd=other, env=self.env())
        self.assertEqual(ignored.returncode, 0, "the store's directory is gitignored in the repository holding it")
        code, report = self.run_script(self.repo("app2"), "connect", "--write-env", ".env",
                                       XYBRID_CREDENTIALS_FILE=str(other / "credentials.json"))
        self.assertEqual((code, report["error"]["code"]), (1, "credentials_in_repository"))
        self.assertFalse((other / "credentials.json").exists())

    # --- the saved credential follows the platform, as `xybrid connect` does ---

    def test_a_claim_made_in_the_console_restores_an_expired_saved_key(self):
        app = self.repo("claimed")
        self.seed(app, key_expires_at="2000-01-01T00:00:00Z")
        Platform.workspace = {"provisional": False, "claimed": True, "key_expires_at": None, "claim_expires_at": None}
        code, report = self.run_script(app, "connect", "--write-env", ".env")
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "existing", "credentials"), report)
        self.assertEqual(self.workspace_checks(), [f"Bearer {KEY}"], "the platform is asked before the saved deadline")
        self.assertIsNone(report["claim_url"])
        self.assertEqual((report["workspace"]["claimed"], report["workspace"]["lifecycle_source"]), (True, "server"))
        self.assertIn(f"\nXYBRID_API_KEY={KEY}\n", "\n" + (app / ".env").read_text())
        (entry,) = self.saved()["credentials"]
        self.assertTrue(entry["claimed"], "the claim is saved")
        self.assertEqual(entry["env_file"], {"path": ".env", "variable": "XYBRID_API_KEY"})

    def test_the_saved_deadline_decides_only_when_the_platform_cant_answer(self):
        app = self.repo("unanswered")
        self.seed(app, key_expires_at="2000-01-01T00:00:00Z")
        Platform.workspace_failure = (503, {"error": "unavailable", "message": "later"})
        code, report = self.run_script(app, "connect", "--write-env", ".env")
        self.assertEqual((code, report["status"]), (7, "workspace_expired"), report)
        self.assertEqual(report["claim_url"], None, "a custom host without XYBRID_CONSOLE_URL has no console")
        self.assertFalse((app / ".env").exists())
        Platform.workspace_failure = None
        Platform.workspace = {"expired": True, "key_expires_at": "2000-01-01T00:00:00Z"}
        code, report = self.run_script(app, "connect", "--write-env", ".env", XYBRID_CONSOLE_URL="http://localhost:5173")
        self.assertEqual((code, report["status"]), (7, "workspace_expired"), report)
        self.assertEqual(report["claim_url"], f"http://localhost:5173/claim/{TOKEN}")
        Platform.workspace = {"expired": True, "key_expires_at": "2000-01-01T00:00:00Z",
                              "claim_expires_at": "2000-01-08T00:00:00Z"}
        code, report = self.run_script(app, "connect", "--write-env", ".env")
        self.assertEqual((code, report["error"]["code"]), (1, "claim_expired"), report)
        self.assertEqual(self.saved()["credentials"][0]["claim_expires_at"], "2000-01-08T00:00:00Z")

    def test_a_key_the_platform_rejects_is_set_aside(self):
        app = self.repo("revoked")
        self.seed(app, api_key=REVOKED_KEY)
        code, report = self.run_script(app, "connect", "--write-env", ".env")
        self.assertEqual((code, report["error"]["code"]), (1, "stored_credential_rejected"), report)
        self.assertFalse((app / ".env").exists(), "a rejected key is never written")
        saved = self.saved()
        self.assertEqual(saved["credentials"], [])
        self.assertEqual([(r["reason"], r["credential"]["api_key"]) for r in saved["retired"]],
                         [("rejected by the platform", REVOKED_KEY)])
        code, report = self.run_script(app, "claim", XYBRID_CONSOLE_URL="http://localhost:5173")
        self.assertEqual((code, report["status"]), (0, "claimable"), "its claim link is still recoverable")

    def test_status_and_claim_save_what_the_platform_reports(self):
        Platform.workspace = {"provisional": False, "claimed": True, "key_expires_at": None}
        for command in ("status", "claim"):
            with self.subTest(command=command):
                app = self.repo(command)
                self.seed(app)
                code, report = self.run_script(app, command, XYBRID_CONSOLE_URL="http://localhost:5173")
                self.assertEqual((code, report["status"]), (0, {"status": "connected", "claim": "claimed"}[command]))
                self.assertIsNone(report["claim_url"], "a claimed workspace has no claim link")
                self.assertTrue(self.saved()["credentials"][0]["claimed"])

    # --- status checks the key connect reported ---------------------------------

    def test_status_checks_the_key_connect_reported(self):
        app = self.repo("explicit")
        self.seed(app)  # a saved key that is not the one in use
        code, report = self.run_script(app, "connect", "--write-env", ".env", XYBRID_API_KEY=OTHER_KEY)
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "existing", "env"), report)
        self.assertIn("status", report["next_steps"][1]["command"])
        code, report = self.run_script(app, "status", XYBRID_API_KEY=OTHER_KEY)
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "connected", "env"), report)
        self.assertEqual(report["workspace"]["organization"]["name"], "other")
        self.assertIsNone(report["stored"], "the saved credential describes another key")

        (app / ".env.local").write_text(f"OTHER=1\nEXPO_PUBLIC_XYBRID_API_KEY={OTHER_KEY}\n")
        flags = ("--write-env", ".env.local", "--env-name", "EXPO_PUBLIC_XYBRID_API_KEY")
        code, report = self.run_script(app, "connect", *flags)
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "existing", "env_file"), report)
        verify = report["next_steps"][1]["command"]
        self.assertIn("--write-env .env.local --env-name EXPO_PUBLIC_XYBRID_API_KEY", verify)
        code, report = self.run_script(app, "status", *flags)
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "connected", "env_file"), report)
        self.assertEqual(report["workspace"]["organization"]["name"], "other")
        self.assertEqual(self.workspace_checks(), [f"Bearer {OTHER_KEY}"] * 2, "the saved key was never checked")
        code, report = self.run_script(app, "status")
        self.assertEqual((code, report["key"]["source"], report["workspace"]["organization"]["name"]),
                         (0, "credentials", "app"))

    def test_the_suggested_check_runs_from_where_connect_ran(self):
        # Run from a subdirectory, the verify command names the file as it was given, not repository-relative.
        app = self.repo("monorepo")
        sub = app / "mobile"
        sub.mkdir()
        (sub / ".env.local").write_text(f"EXPO_PUBLIC_XYBRID_API_KEY={OTHER_KEY}\n")
        flags = ("--write-env", ".env.local", "--env-name", "EXPO_PUBLIC_XYBRID_API_KEY")
        code, report = self.run_script(sub, "connect", *flags)
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "existing", "env_file"), report)
        self.assertEqual(report["env_file"]["path"], "mobile/.env.local")
        words = shlex.split(report["next_steps"][1]["command"])
        self.assertEqual(words[3:8], ["status", *flags], words)
        code, report = self.run_script(sub, *words[3:8])
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "connected", "env_file"), report)
        # The suggested command names the script by absolute path, so it runs as printed from the
        # project directory even though the script lives elsewhere.
        self.assertTrue(os.path.isabs(words[0]) or words[0] == "python3", words)
        self.assertEqual((words[1], os.path.realpath(words[2])), ("-I", os.path.realpath(str(SCRIPT))), words)

    def test_a_file_with_conflicting_values_gets_no_check_that_must_fail(self):
        # status refuses a file that gives the variable two values, so connect suggests no command reading it.
        app = self.repo("conflict")
        third = "sk_live_" + "FAKE_THIRD_KEY"
        (app / ".env").write_text(f"XYBRID_API_KEY={OTHER_KEY}\nXYBRID_API_KEY={third}\n")
        code, report = self.run_script(app, "connect", "--write-env", ".env")
        self.assertEqual((code, report["status"], report["key"]["source"]), (0, "existing", "env_file"), report)
        verify = report["next_steps"][1]
        self.assertIsNone(verify["command"], verify)
        self.assertIn("keep a single XYBRID_API_KEY line", verify["summary"])
        self.assertIn("different values on several lines", report["tell_the_user"])
        self.assertNotIn(third, json.dumps(report))
        code, report = self.run_script(app, "status", "--write-env", ".env")
        self.assertEqual((code, report["error"]["code"]), (1, "env_file_ambiguous"), report)
        self.assertEqual(Platform.requests, [], "neither run sends a key")

    def test_a_workspace_answer_off_the_contract_is_never_saved(self):
        # A deadline without a UTC offset can't be compared, and the CLI refuses a store holding one: the
        # answer is a platform_error (as `xybrid auth status` reports it) and the saved deadlines stay.
        app = self.repo("naive")
        self.seed(app)

        def lifecycle():
            (entry,) = self.saved()["credentials"]
            return {k: entry.get(k) for k in ("key_expires_at", "claim_expires_at", "claimed", "scope")}

        before = lifecycle()
        for answer in ({"key_expires_at": "2999-01-01T00:00:00"}, {"claim_expires_at": "2999-01-08"},
                       {"key_expires_at": "2999-01-01T00:00:00+00:60"}):
            with self.subTest(answer=answer):
                Platform.workspace = answer
                code, report = self.run_script(app, "status")
                self.assertEqual((code, report["status"], report["error"]["code"]), (1, "error", "platform_error"),
                                 report)
                self.assertEqual(lifecycle(), before)
                code, report = self.run_script(app, "claim", XYBRID_CONSOLE_URL="http://localhost:5173")
                self.assertEqual((code, report["status"]), (0, "claimable"), report)
                code, report = self.run_script(app, "connect", "--write-env", ".env")
                self.assertEqual((code, report["status"], report["workspace"]["lifecycle_source"]),
                                 (0, "existing", "stored"), report)
                self.assertEqual(lifecycle(), before)
        Platform.workspace_failure = (200, {"hello": 1})
        code, report = self.run_script(app, "status")
        self.assertEqual((code, report["error"]["code"]), (1, "platform_error"), report)

    def test_a_provisioning_answer_off_the_contract_is_retried_not_saved(self):
        app = self.repo("naive-provision")
        Platform.provision = {"key_expires_at": "2999-01-01T00:00:00"}
        code, report = self.run_script(app, "connect", "--write-env", ".env")
        self.assertEqual((code, report["error"]["code"]), (1, "invalid_platform_response"), report)
        self.assertFalse((app / ".env").exists())
        saved = self.saved()
        self.assertEqual(saved["credentials"], [])
        (pending,) = saved["pending"]
        Platform.provision = {}
        code, report = self.run_script(app, "connect", "--write-env", ".env")
        self.assertEqual((code, report["status"]), (0, "created"), report)
        keys = [r[3]["Idempotency-Key"] for r in Platform.requests if r[0] == "POST"]
        self.assertEqual(keys, [pending["idempotency_key"]] * 2, "the retry repeats the same request")
        self.assertEqual(self.saved()["credentials"][0]["key_expires_at"], "2999-01-01T00:00:00.123456789Z")

    # --- the project file, read and written as the CLI does ----------------------

    def test_quoted_values_spanning_lines_are_refused(self):
        app = self.repo("multiline")
        for text in ('CERT="-----BEGIN\nXYBRID_API_KEY=example\n-----END"\n',  # text inside CERT, not a key
                     'CERT="-----BEGIN\n'):  # an appended key would land inside CERT
            with self.subTest(text=text):
                (app / ".env").write_bytes(text.encode())
                code, report = self.run_script(app, "connect", "--write-env", ".env")
                self.assertEqual((code, report["status"], report["error"]["code"]), (1, "error", "env_file_ambiguous"),
                                 report)
                self.assertEqual((app / ".env").read_bytes(), text.encode())
        self.assertEqual(Platform.requests, [], "refused before any request")

    def test_existing_text_is_kept_byte_for_byte(self):
        # CRLF endings and non-ASCII text survive, whatever the locale's encoding
        # (`-X utf8=0` with LC_ALL=C makes it ASCII here, as cp1252 is on Windows).
        app = self.repo("crlf")
        original = "A=1\r\nNAME=caf\u00e9 \u2603\r\n".encode("utf-8")
        (app / ".env").write_bytes(original)
        code, report = self.run_script(app, "connect", "--write-env", ".env", flags=("-X", "utf8=0"), LC_ALL="C")
        self.assertEqual((code, report["status"]), (0, "created"), report)
        out = (app / ".env").read_bytes()
        self.assertTrue(out.startswith(original + b"\r\n# "), out)
        self.assertTrue(out.endswith(f"\r\nXYBRID_API_KEY={KEY}\r\n".encode()), out)
        self.assertNotIn(b"\n", out.replace(b"\r\n", b""), "every line keeps CRLF")
        self.assertIn("\u201c", report["tell_the_user"], "the report itself is UTF-8")
        (app / "latin1.env").write_bytes("NAME=caf\u00e9\n".encode("latin-1"))
        code, report = self.run_script(app, "connect", "--write-env", "latin1.env")
        self.assertEqual((code, report["error"]["code"]), (1, "env_file_unwritable"), report)

    def test_an_invalid_variable_name_changes_nothing(self):
        app = self.repo("names")
        for target, name in ((".env", "XYBRID-API-KEY"), (".env", "1KEY"), (".env", "xybrid.apiKey"),
                             ("local.properties", "xybrid apiKey"), ("Secrets.xcconfig", "API-KEY")):
            with self.subTest(target=target, name=name):
                code, report = self.run_script(app, "connect", "--write-env", target, "--env-name", name)
                self.assertEqual((code, report["error"]["code"]), (1, "invalid_env_name"), report)
                self.assertFalse((app / target).exists())
        self.assertFalse((app / ".gitignore").exists())
        self.assertFalse(self.store.parent.exists())
        self.assertEqual(Platform.requests, [])

    def test_a_single_empty_assignment_is_filled_in_place(self):
        app = self.repo("ios")
        (app / "Secrets.xcconfig").write_bytes(b"PRODUCT_NAME = MyApp\r\nXYBRID_API_KEY[config=Debug] =\r\nB = 2\r\n")
        code, report = self.run_script(app, "connect", "--write-env", "Secrets.xcconfig")
        self.assertEqual((code, report["status"], report["env_file"]["written"]), (0, "created", True), report)
        text = (app / "Secrets.xcconfig").read_bytes().decode()
        self.assertTrue(text.startswith("PRODUCT_NAME = MyApp\r\n// "), text)
        self.assertTrue(text.endswith(f"\r\nXYBRID_API_KEY[config=Debug] = {KEY}\r\nB = 2\r\n"), text)


class ProjectFileTests(unittest.TestCase):
    """The script reads project files as crates/xybrid-cli/src/platform/env_file.rs does (its test cases)."""

    script = load_script()

    def test_lookups_match_the_cli(self):
        cases = {
            "dotenv": {
                "present": ["XYBRID_API_KEY=abc", "export XYBRID_API_KEY=abc", '  XYBRID_API_KEY = "abc"',
                            "XYBRID_API_KEY=\x1c", "XYBRID_API_KEY= \x1f # c",
                            "XYBRID_API_KEY='abc' # comment", "XYBRID_API_KEY=\nXYBRID_API_KEY=abc\nXYBRID_API_KEY=\n"],
                "empty": ["XYBRID_API_KEY=", 'XYBRID_API_KEY=""', "XYBRID_API_KEY= # later", "XYBRID_API_KEY=\u2003\xa0",
                          'A="one line" # "quoted"\nXYBRID_API_KEY=\n', "A='it''s'\nXYBRID_API_KEY=\n",
                          '# CERT="open\nXYBRID_API_KEY=\n', 'CERT="-----BEGIN\nMIIE==\n-----END"\nXYBRID_API_KEY=\n'],
                "absent": ["# XYBRID_API_KEY=abc", "XYBRID_API_KEY_OLD=abc", "MY_XYBRID_API_KEY=abc", "",
                           # Rust's trim keeps \x1c-\x1f, which Python's bare strip() takes.
                           "\x1cXYBRID_API_KEY=abc", "XYBRID_API_KEY\x1f=abc"],
                "several_empty": ['XYBRID_API_KEY=\nA=1\nexport XYBRID_API_KEY=""\n'],
                "quoted_lines": ['CERT="-----BEGIN\nXYBRID_API_KEY=\n-----END"\n',
                                 "CERT='-----BEGIN\nexport XYBRID_API_KEY=abc\n-----END'\n", "CERT=`one\nXYBRID_API_KEY=\n`\n",
                                 'A="x\\"\nXYBRID_API_KEY=\n"\n', 'XYBRID_API_KEY="\nabc"\n', 'CERT="open\nXYBRID_API_KEY="x"\n',
                                 'XYBRID_API_KEY=abc\nCERT="\nXYBRID_API_KEY=\n"\n', 'CERT="-----BEGIN\n'],
            },
            "properties": {
                "present": ["xybrid.apiKey=abc", "xybrid.apiKey : abc", "xybrid.apiKey abc"],
                "empty": ["xybrid.apiKey=", "xybrid.apiKey", "  xybrid.apiKey", "xybrid.apiKey\t", "\x0cxybrid.apiKey",
                          "sdk.dir = C:\\\\\nxybrid.apiKey=\n", "# note \\\nxybrid.apiKey=\n",
                          "other = foo \\\n\nxybrid.apiKey=\n"],
                "absent": ["! xybrid.apiKey=abc", "", "   ", "\n\t\n", "xybrid.apiKeyOld", "xybrid", "\x1cxybrid.apiKey",
                           "sdk.dir=/opt/android/sdk\\\n", "sdk.dir=/opt/android/sdk\\"],
                "several_empty": ["xybrid.apiKey=\nsdk.dir=/sdk\nxybrid.apiKey :\n", "xybrid.apiKey=\nsdk.dir=/sdk\nxybrid.apiKey\n"],
                "continuation": ["other = foo \\\nxybrid.apiKey=\n", "other = foo \\\n  xybrid.apiKey=abc\n",
                                 "other = a \\\\\\\nxybrid.apiKey=\n", "xybrid.apiKey = \\\n  abc\n",
                                 "xybrid.apiKey = abc\nother = \\\nxybrid.apiKey\n"],
            },
            "xcconfig": {
                "present": ["XYBRID_API_KEY[config=Debug] = abc // note",
                            "XYBRID_API_KEY[config=Debug][sdk=iphoneos*] = abc // note"],
                "empty": ["XYBRID_API_KEY = // filled in later", "XYBRID_API_KEY[config=Debug] =",
                          "/* a */ /* b */\n/*\nc\n*/ /* d */\nXYBRID_API_KEY =\n", "/*\nPRODUCT_NAME = Old\n*/\nXYBRID_API_KEY =\n",
                          "HEADER_SEARCH_PATHS = $(SRCROOT)/**\nXYBRID_API_KEY =\n", "OTHER_FLAGS = -a \\\n  -b\nXYBRID_API_KEY =\n",
                          "OTHER_FLAGS = -a \\\n\nXYBRID_API_KEY =\n", "OTHER_FLAGS = -a \\\n// end\nXYBRID_API_KEY =\n"],
                "absent": ["// XYBRID_API_KEY = abc", "/* XYBRID_API_KEY = old */\nPRODUCT_NAME = MyApp\n",
                           "HEADER_SEARCH_PATHS = $(SRCROOT)/**\n", "OTHER_LDFLAGS = -ObjC \\\n", "OTHER_LDFLAGS = -ObjC \\"],
                "several_empty": ["XYBRID_API_KEY =\nPRODUCT_NAME = MyApp\nXYBRID_API_KEY[config=Release] = // later\n"],
                "block_comment": ["/*\nXYBRID_API_KEY =\n*/\n", "/*\nXYBRID_API_KEY = old\n*/\n",
                                  "PRODUCT_NAME = MyApp\n  /* disabled:\n  XYBRID_API_KEY[config=Debug] =\n  */\n",
                                  "/*\n * notes\n */ XYBRID_API_KEY =\n", "/* note */ XYBRID_API_KEY = old\n",
                                  "XYBRID_API_KEY = /* set me */\n", "XYBRID_API_KEY = abc\n/*\nXYBRID_API_KEY =\n*/\n",
                                  "PRODUCT_NAME = MyApp\n/* notes\n",
                                  "/* Secrets */ /* disabled until launch:\nXYBRID_API_KEY = old\n*/\n",
                                  "/*\nheader\n*/ /* a */ /* second block\nXYBRID_API_KEY =\n*/\n",
                                  "/* a */ /* b */ XYBRID_API_KEY = old\n", "/* a */ /* b\n", "/*\nheader\n*/ /* b\n",
                                  "FOO = a \\\n/* start\nXYBRID_API_KEY =\n*/\n", "FOO = a \\\n/* start\n"],
                "continuation": ["OTHER_FLAGS = -a \\\nXYBRID_API_KEY =\n", "OTHER_FLAGS = -a \\ // note\n  XYBRID_API_KEY = abc\n",
                                 "OTHER_FLAGS = -a \\;\nXYBRID_API_KEY =\n", "OTHER_FLAGS = -a \\\n  -b \\\n  XYBRID_API_KEY =\n",
                                 "XYBRID_API_KEY = \\\n  $(SECRET)\n", "FOO = a \\\n/* a */ XYBRID_API_KEY =\n",
                                 "FOO = a \\\n/* a */ \\\nXYBRID_API_KEY =\n"],
            },
        }
        names = {"dotenv": "XYBRID_API_KEY", "properties": "xybrid.apiKey", "xcconfig": "XYBRID_API_KEY"}
        for fmt, expected in cases.items():
            for outcome, texts in expected.items():
                for text in texts:
                    with self.subTest(format=fmt, text=text):
                        self.assertEqual(self.script.lookup(fmt, text, names[fmt]), outcome)

    def test_writes_match_the_cli(self):
        upsert = self.script.upsert
        cases = [
            ("dotenv", "A=1\r\nexport XYBRID_API_KEY=\r\nB=2\r\n", ["c"], f"A=1\r\n# c\r\nexport XYBRID_API_KEY={KEY}\r\nB=2\r\n"),
            ("dotenv", "# app\nexport OTHER=1", ["c"], f"# app\nexport OTHER=1\n\n# c\nXYBRID_API_KEY={KEY}\n"),
            ("dotenv", "", [], f"XYBRID_API_KEY={KEY}\n"),
            ("dotenv", "", ["name\nEVIL=1"], f"# name EVIL=1\nXYBRID_API_KEY={KEY}\n"),
            ("xcconfig", "A = 1\n  XYBRID_API_KEY[config=Debug][sdk=iphoneos*] = // later\n", [],
             f"A = 1\nXYBRID_API_KEY[config=Debug][sdk=iphoneos*] = {KEY}\n"),
            ("xcconfig", "OTHER_LDFLAGS = -ObjC \\", ["c"], f"OTHER_LDFLAGS = -ObjC \\\n\n// c\nXYBRID_API_KEY = {KEY}\n"),
            ("properties", "sdk.dir=/sdk\n  xybrid.apiKey\nother=1\n", [], f"sdk.dir=/sdk\nxybrid.apiKey={KEY}\nother=1\n"),
        ]
        for fmt, text, comment, expected in cases:
            name = "xybrid.apiKey" if fmt == "properties" else "XYBRID_API_KEY"
            with self.subTest(format=fmt, text=text):
                self.assertEqual(upsert(fmt, text, name, KEY, comment), expected)
                self.assertTrue(self.script.assigns(fmt, expected, name, KEY))
        self.assertFalse(self.script.assigns("dotenv", f'CERT="\nXYBRID_API_KEY={KEY}\n"\n', "XYBRID_API_KEY", KEY))

    def test_variable_names_match_the_cli(self):
        valid = self.script.valid_variable
        self.assertTrue(valid("dotenv", "EXPO_PUBLIC_XYBRID_API_KEY") and valid("properties", "xybrid.apiKey"))
        self.assertTrue(valid("properties", "xybrid-api.key_1") and valid("xcconfig", "_KEY"))
        for bad in ("", "1ABC", "A B", "A=B", "A\nB", "xybrid.apiKey", "A-B", "\u00c9T\u00c9", "A" * 129):
            self.assertFalse(valid("dotenv", bad), bad)
        for bad in ("a=b", "a:b", "a b", ""):
            self.assertFalse(valid("properties", bad), bad)


    def test_only_times_with_an_offset_are_taken(self):
        # What the CLI's chrono parser reads back from the shared store, and nothing that can't be compared.
        timestamp = self.script.timestamp
        for raw in ("2999-01-01T00:00:00Z", "2999-01-01t00:00:00z", "2999-01-01T00:00:00.123456789Z",
                    "2999-01-01T00:00:00-05:00", "2999-01-01T00:00:00+23:59"):
            self.assertEqual(timestamp(raw), raw, raw)
            self.script.passed(raw)  # comparable with now
        for raw in ("2999-01-01T00:00:00", "2999-01-01", "2999-01-01 00:00:00Z", "2999-01-01T00:00:00+0000",
                    "2999-01-01T00:00:00+24:00", "2999-01-01T00:00:00+00:60", "2999-02-30T00:00:00Z",
                    "2999-01-01T00:00Z", "\u0662999-01-01T00:00:00Z", "2999-01-01T00:00:00Z\n", None, 2999):
            self.assertIsNone(timestamp(raw), raw)


if __name__ == "__main__":
    unittest.main()
