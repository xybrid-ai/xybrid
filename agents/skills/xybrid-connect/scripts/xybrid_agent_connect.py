#!/usr/bin/env python3
"""Connect a project to Xybrid without the CLI: the `xybrid connect` contract.

  python3 -I xybrid_agent_connect.py connect --write-env .env --agent claude-code --binding python
  python3 -I xybrid_agent_connect.py status   # trace verification for this project's key
  python3 -I xybrid_agent_connect.py claim    # the private claim link, from the private store

Prints one JSON object (`status`, `tell_the_user`, ...) with the CLI's exit
codes. Never prints the key. Saves recovery state in the CLI's private store
(~/.xybrid/credentials.json, or XYBRID_CREDENTIALS_FILE), so a later
`xybrid auth claim` / `xybrid auth status` finds it. Python 3.8+, stdlib only.
"""
import argparse
import base64
import hashlib
import json
import os
import platform
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PINNED = {  # API origin -> (console origin, OTLP trace endpoint)
    "https://api.xybrid.dev": ("https://console.xybrid.dev", "https://ingest.xybrid.dev/v1/traces"),
    "https://staging.api.xybrid.dev": ("https://staging.xybrid.dev", "https://staging.ingest.xybrid.dev/v1/traces"),
}
CI_MARKERS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "CIRCLECI", "JENKINS_URL", "TF_BUILD",
              "TEAMCITY_VERSION", "BITBUCKET_BUILD_NUMBER", "CODEBUILD_BUILD_ID", "TRAVIS", "DRONE")
EXIT = {"created": 0, "existing": 0, "connected": 0, "claimable": 0, "signup_closed": 3, "offline": 4,
        "ci_skipped": 5, "opted_out": 5, "env_file_tracked": 6, "workspace_expired": 7,
        "not_connected": 1, "error": 1}
MAX_DIFFICULTY = 28          # the platform clamps difficulty to 18..=28
RETRY_WINDOW = 9 * 60        # Idempotency-Key retries are honored for 10 minutes
KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,256}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
LOCAL = " On-device inference works without it."


class Stop(Exception):
    """End with `status` (and `error.code` when status is error)."""

    def __init__(self, status, tell, code=None, message=None):
        super().__init__(tell)
        self.status, self.tell, self.code, self.message = status, tell, code, message


def now():
    return datetime.now(timezone.utc)


def stamp(at):
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(raw):
    """RFC 3339 as the platform and the CLI write it (fractions up to nanoseconds, `Z`)."""
    raw = re.sub(r"\.(\d+)", lambda m: "." + (m.group(1) + "00000")[:6], raw.strip()).replace("Z", "+00:00")
    return datetime.fromisoformat(raw)


def env(name):
    value = os.environ.get(name, "").strip()
    return value or None


def preview(key):
    prefix = next((p for p in ("sk_live_", "sk_test_") if key.startswith(p)), "")
    rest = key[len(prefix):]
    return prefix + rest[:min(8, len(rest) // 3)] + "\u2026"


# --- proof of work ---------------------------------------------------------

def zero_bits(digest):
    bits = 0
    for byte in digest:
        if byte:
            return bits + 8 - byte.bit_length()
        bits += 8
    return bits


def solve(challenge, difficulty, deadline):
    """Smallest canonical decimal nonce with SHA256(challenge ":" nonce) >= difficulty zero bits."""
    prefix = hashlib.sha256((challenge + ":").encode("utf-8"))
    nonce = 0
    while True:
        h = prefix.copy()
        h.update(str(nonce).encode("ascii"))
        if zero_bits(h.digest()) >= difficulty:
            return nonce
        nonce += 1
        if nonce % 65536 == 0 and time.monotonic() > deadline:
            return None


# --- hosts and HTTP --------------------------------------------------------

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # never follow: credentials must not move hosts
        return None


def host_for(raw, allow_custom):
    from urllib.parse import urlsplit
    url = urlsplit(raw.strip())
    if url.scheme not in ("http", "https") or not url.hostname or url.path not in ("", "/"):
        raise Stop("error", "Couldn't connect: the platform URL is not an http(s) origin." + LOCAL, "invalid_platform_url")
    origin = "%s://%s" % (url.scheme, url.netloc.lower())
    if origin in PINNED:
        console, traces = PINNED[origin]
        return {"origin": origin, "console": console, "traces": traces, "custom": False, "loopback": False}
    loopback = url.hostname in ("localhost", "127.0.0.1", "::1")
    if not allow_custom:
        raise Stop("error", "Couldn't connect: %s is not a Xybrid platform host." % origin + LOCAL, "custom_host_not_allowed")
    if url.scheme != "https" and not loopback:
        raise Stop("error", "Couldn't connect: %s must use https." % origin + LOCAL, "insecure_platform_url")
    console = env("XYBRID_CONSOLE_URL")
    return {"origin": origin, "console": console.rstrip("/") if console else None, "traces": None, "custom": True,
            "loopback": loopback}


def call(host, method, path, body=None, headers=None):
    """(status, json body, response headers); status None when the host can't be reached."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(host["origin"] + path, data=data, method=method)
    request.add_header("Accept", "application/json")
    request.add_header("User-Agent", "xybrid-agent-connect/1")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    handlers = [NoRedirect()]
    if host["loopback"]:
        handlers.append(urllib.request.ProxyHandler({}))  # never send a loopback request through a proxy
    opener = urllib.request.build_opener(*handlers)
    try:
        with opener.open(request, timeout=30) as response:
            raw, status, got = response.read(65537), response.status, response.headers
    except urllib.error.HTTPError as err:
        raw, status, got = err.read(65537), err.code, err.headers
    except (urllib.error.URLError, OSError) as err:
        return None, {"detail": str(getattr(err, "reason", err))}, {}
    try:
        parsed = json.loads(raw) if raw and len(raw) <= 65536 else {}
    except ValueError:
        parsed = {}
    return status, parsed if isinstance(parsed, dict) else {}, got


# --- Git and the project file ----------------------------------------------

def git(root, *args):
    clean = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    return subprocess.run(["git", "-C", str(root)] + list(args), capture_output=True, env=clean)


def repository_root(start):
    for directory in [start] + list(start.parents):
        if (directory / ".git").exists():
            return directory
    return None


def is_tracked(root, rel):
    out = git(root, "ls-files", "-z", "--cached", "--", ":(literal)" + rel)
    if out.returncode != 0:
        raise Stop("error", "Couldn't connect: git ls-files failed." + LOCAL, "git_failed")
    return bool(out.stdout)


def is_ignored(root, rel):
    code = git(root, "check-ignore", "-q", "--", rel).returncode
    if code not in (0, 1):
        raise Stop("error", "Couldn't connect: git check-ignore failed." + LOCAL, "git_failed")
    return code == 0


def ensure_ignored(root, rel, what="Xybrid project key"):
    if is_ignored(root, rel):
        return False
    if any(ord(c) < 32 for c in rel):
        raise Stop("error", "Couldn't connect: %r can't be gitignored." % rel + LOCAL, "env_file_not_ignorable")
    pattern = "/" + re.sub(r"([\\*?\[])", r"\\\1", rel.rstrip(" ")) + "\\ " * (len(rel) - len(rel.rstrip(" ")))
    path = root / ".gitignore"
    existing = path.read_bytes() if path.exists() else b""
    nl = "\r\n" if b"\r\n" in existing else "\n"
    lead = nl if existing and not existing.endswith(b"\n") else ""
    with open(path, "a", newline="") as handle:
        handle.write("%s# %s written by xybrid_agent_connect.py; keep it out of Git%s%s%s" % (lead, what, nl, pattern, nl))
    if not is_ignored(root, rel):
        raise Stop("error", "Couldn't connect: %s is still not ignored after adding %s to .gitignore." % (rel, pattern) + LOCAL,
                   "env_file_not_ignorable")
    return True


def file_format(path):
    name = path.name.lower()
    if name.endswith(".xcconfig"):
        return "xcconfig", "XYBRID_API_KEY", "//"
    if name.endswith(".properties"):
        return "properties", "xybrid.apiKey", "#"
    if name == ".env" or name.startswith(".env.") or name.endswith(".env"):
        return "dotenv", "XYBRID_API_KEY", "#"
    raise Stop("error", "Couldn't connect: --write-env must be a dotenv, .properties or .xcconfig file." + LOCAL,
               "unsupported_env_file")


def assigned(fmt, text, name):
    """None if `name` is unassigned, else its value as a loader would read it."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", "!")):
            continue
        if fmt == "properties":  # as java.util.Properties reads it: the key ends at the first `=`, `:` or blank
            match = re.match(r"^([^=:\s]+)\s*[=:]?\s*(.*)$", line)
        elif fmt == "xcconfig":  # NAME[config=Debug] = value
            match = re.match(r"^([A-Za-z0-9_]+)(?:\[[^\]]*\])*\s*=\s*(.*)$", line)
        else:
            if line.startswith("export "):
                line = line[7:].lstrip()
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if match and match.group(1) == name:
            value = match.group(2).split("//")[0] if fmt == "xcconfig" else match.group(2)
            return value.strip().strip("'\"").strip()
    return None


def write_atomic(path, text, private=False):
    fd, temp = tempfile.mkstemp(prefix=".xybrid-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if not private and path.exists():  # a new file keeps mkstemp's 0600, as the CLI's does
            os.chmod(temp, path.stat().st_mode & 0o777)
        os.replace(temp, str(path))
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def write_key(target, credential):
    text = target["path"].read_text(encoding="utf-8") if target["path"].exists() else ""
    nl = "\r\n" if "\r\n" in text else "\n"
    org = (credential.get("organization") or {}).get("name", "this project")
    day = lambda raw: raw[:10] if raw else "?"
    lines = ['Xybrid provisional workspace "%s": restricted client key written by xybrid_agent_connect.py.' % org,
             "Claim it with GitHub by %s to keep it: run `xybrid_agent_connect.py claim` (or `xybrid auth claim`)."
             % day(credential.get("key_expires_at")),
             "Unclaimed, the key stops on %s (UTC) and the workspace is deleted on %s." % (
                 day(credential.get("key_expires_at")), day(credential.get("claim_expires_at"))),
             "Never commit this file or share the key."]
    out = text + (nl if text and not text.endswith("\n") else "") + (nl if text.strip() else "")
    out += "".join("%s %s%s" % (target["comment"], line, nl) for line in lines)
    sep = " = " if target["format"] == "xcconfig" else "="
    out += "%s%s%s%s" % (target["variable"], sep, credential["api_key"], nl)
    write_atomic(target["path"], out)


# --- the private store (same file and format as the CLI) -------------------

def guard_store(store):
    """Keep a store inside any Git working tree (not only this project's) out of commits, as the CLI does."""
    directory = store.parent.resolve()
    existing = next((d for d in [directory] + list(directory.parents) if d.exists()), None)
    root = repository_root(existing) if existing is not None else None
    if root is None:
        return
    if directory == root:
        raise Stop("error", "Couldn't connect: %s would keep the credentials directly in the Git repository %s; point "
                   "XYBRID_CREDENTIALS_FILE to a directory of its own." % (store, root) + LOCAL, "credentials_in_repository")
    rel = directory.relative_to(root).as_posix()
    if is_tracked(root, rel):
        raise Stop("error", "Couldn't connect: the credentials directory %s holds files tracked by Git; point "
                   "XYBRID_CREDENTIALS_FILE to a directory of its own." % rel + LOCAL, "credentials_tracked")
    ensure_ignored(root, rel, "Xybrid credentials store")


def store_path():
    raw = os.environ.get("XYBRID_CREDENTIALS_FILE")
    return Path(raw).absolute() if raw else Path.home() / ".xybrid" / "credentials.json"


class Store:
    def __init__(self, path, console=None):
        self.path, self.lock_file, self.console = path, None, console

    def __enter__(self):
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == "posix":
            info, uid = directory.stat(), os.geteuid()
            shared = info.st_mode & 0o020 and not (info.st_gid == os.getegid() == uid)
            if info.st_uid != uid or shared or (info.st_mode & 0o002 and not info.st_mode & 0o1000):
                raise Stop("error", "Couldn't connect: %s is not private to this account (chmod 700 it)." % directory + LOCAL,
                           "credentials_unsafe")
            import fcntl
            self.lock_file = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            deadline = time.monotonic() + 90
            while True:
                try:
                    fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise Stop("error", "Couldn't connect: another xybrid process is updating %s." % self.path + LOCAL,
                                   "credentials_busy")
                    time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        if self.lock_file is not None:
            os.close(self.lock_file)

    def load(self):
        if not self.path.exists():
            return {"version": 1, "credentials": [], "pending": [], "retired": []}
        if os.name == "posix":
            info = self.path.stat()
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise Stop("error", "Couldn't connect: %s is not private (chmod 600 it)." % self.path + LOCAL, "credentials_unsafe")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError:
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("version"), int):
            raise Stop("error", "Couldn't connect: %s is not a valid xybrid credentials file; it was left untouched." % self.path
                       + LOCAL, "credentials_corrupt")
        if data["version"] > 1:
            raise Stop("error", "Couldn't connect: %s was written by a newer xybrid." % self.path + LOCAL, "credentials_unsupported")
        for field in ("credentials", "pending", "retired"):
            data.setdefault(field, [])
        return data

    def save(self, data):
        write_atomic(self.path, json.dumps(data, indent=2) + "\n", private=True)


def find(entries, origin, root, kind=None):
    for entry in entries:
        if entry.get("platform_origin") == origin and entry.get("repository_root") == root and \
                (kind is None or entry.get("kind") == kind):
            return entry
    return None


def claim_url(host, credential):
    token, console = credential.get("claim_token"), credential.get("console_origin") or host["console"]
    deadline = credential.get("claim_expires_at")
    if credential.get("claimed") or not token or not console or not TOKEN_RE.match(token):
        return None
    if deadline and now() >= parse_time(deadline):
        return None
    return "%s/claim/%s" % (console.rstrip("/"), token)


# --- commands ---------------------------------------------------------------

def base_report(command, host):
    return {"schema_version": 1, "command": command, "client": "xybrid_agent_connect.py", "status": "error",
            "error": None, "platform": {"api_origin": host["origin"], "console_origin": host["console"],
                                        "traces_endpoint": host["traces"], "custom_host": host["custom"]},
            "key": None, "env_file": None, "credentials": None, "workspace": None, "claim_url": None,
            "verification": None, "next_steps": [], "tell_the_user": ""}


def project(cwd):
    repo = repository_root(cwd)
    return repo, str(repo or cwd)


def connect(args, host, report):
    if not args.allow_ci:
        marker = next((m for m in CI_MARKERS if (env(m) or "false").lower() not in ("false", "0", "no")), None)
        if marker:
            raise Stop("ci_skipped", "Skipped connecting to the Xybrid console because this looks like a CI run "
                       "(%s is set). On-device inference works without a key." % marker)
    for name in ("XYBRID_TELEMETRY_OPTOUT", "DO_NOT_TRACK"):
        if (env(name) or "").lower() in ("1", "true", "yes"):
            raise Stop("opted_out", "Skipped connecting to the Xybrid console because %s=%s opts out of telemetry. "
                       "On-device inference works without a key." % (name, env(name)))
    if (env("OTEL_SDK_DISABLED") or "").lower() == "true":
        raise Stop("opted_out", "Skipped connecting to the Xybrid console because OTEL_SDK_DISABLED=true opts out "
                   "of telemetry. On-device inference works without a key.")

    cwd = Path.cwd().resolve()
    repo, root = project(cwd)
    path = (cwd / args.write_env).parent.resolve() / Path(args.write_env).name
    if not path.parent.is_dir() or path.is_symlink():
        raise Stop("error", "Couldn't connect: %s is not a usable file path." % path + LOCAL, "env_file_unwritable")
    fmt, default_var, comment = file_format(path)
    variable = args.env_name or default_var
    if repo is not None and repository_root(path.parent) != repo:
        raise Stop("error", "Couldn't connect: %s is outside this Git repository." % path + LOCAL, "env_file_outside_repository")
    rel = path.relative_to(repo).as_posix() if repo else str(path)
    target = {"path": path, "format": fmt, "variable": variable, "comment": comment}
    report["env_file"] = {"path": rel, "variable": variable, "format": fmt, "written": False,
                          "already_configured": False, "tracked": False, "gitignored": False,
                          "gitignore_updated": False, "suggestions": []}
    if repo is not None and is_tracked(repo, rel):
        candidates = {"dotenv": (".env.local", ".env.development.local"),
                      "properties": ("local.properties", "xybrid.local.properties"),
                      "xcconfig": ("Secrets.xcconfig", "Secrets.local.xcconfig")}[fmt]
        prefix = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
        suggestions = [prefix + c for c in candidates if c != path.name and not is_tracked(repo, prefix + c)][:1]
        report["env_file"].update(tracked=True, suggestions=suggestions)
        raise Stop("env_file_tracked", "Didn't connect: %s is tracked by Git, so a key written there would be committed. "
                   "Use an untracked file%s. On-device inference works without a key." % (
                       rel, " such as %s" % suggestions[0] if suggestions else ""))
    if env("XYBRID_API_KEY"):
        report["key"] = {"source": "env", "preview": preview(env("XYBRID_API_KEY")), "scope": None}
        raise Stop("existing", "This environment already provides a Xybrid key (XYBRID_API_KEY); nothing was changed.")
    current = assigned(fmt, path.read_text(encoding="utf-8"), variable) if path.exists() else None

    store = store_path()
    guard_store(store)
    with Store(store, host["console"]) as lock:
        data = lock.load()
        report["credentials"] = {"path": str(store), "saved": False, "recovery": False, "moved_corrupt_file": None}
        saved = find(data["credentials"], host["origin"], root)
        if repo is not None:
            report["env_file"]["gitignore_updated"] = ensure_ignored(repo, rel)
            report["env_file"]["gitignored"] = True
        if current is not None and (not saved or current != saved.get("api_key")):
            report["env_file"]["already_configured"] = bool(current)
            if not current:
                raise Stop("error", "Couldn't connect: %s has an empty %s line; remove it or choose another file." % (rel, variable)
                           + LOCAL, "env_file_ambiguous")
            report["key"] = {"source": "env_file", "preview": None, "scope": None}
            raise Stop("existing", "%s already sets %s; nothing was changed." % (rel, variable))
        if saved and not saved.get("claimed") and saved.get("key_expires_at") and \
                now() >= parse_time(saved["key_expires_at"]):
            url = claim_url(host, saved)
            report["claim_url"] = url
            raise Stop("workspace_expired", "The provisional Xybrid workspace saved for this project expired: its key no "
                       "longer reaches the platform.%s On-device inference keeps working." % (
                           " Claim it with GitHub to restore the key: %s (private link)." % url if url else
                           " Its claim period ended or its link is not saved here."))
        if saved:
            credential, status = saved, "existing"
        else:
            credential, status = provision(args, host, lock, data, root), "created"
        report["credentials"].update(saved=True, recovery=bool(credential.get("claim_token")))
        if current is None:
            if repo is not None and (is_tracked(repo, rel) or not is_ignored(repo, rel)):
                raise Stop("error", "Couldn't write the key: %s became tracked or stopped being ignored; it is saved "
                           "privately in %s." % (rel, store) + LOCAL, "env_file_not_ignored")
            write_key(target, credential)
            report["env_file"]["written"] = True
            credential["env_file"] = {"path": rel, "variable": variable}
            lock.save(data)
        else:
            report["env_file"]["already_configured"] = True
    report["key"] = {"source": "created" if status == "created" else "credentials",
                     "preview": preview(credential["api_key"]), "scope": credential.get("scope")}
    report["workspace"] = {k: credential.get(k) for k in ("organization", "project", "key_expires_at", "claim_expires_at",
                                                          "terms_url", "privacy_url")}
    report["workspace"].update(provisional=not credential.get("claimed"), claimed=bool(credential.get("claimed")))
    report["claim_url"] = claim_url(host, credential)
    report["verification"] = {"state": "not_checked", "traces_received": None}
    report["next_steps"] = [
        {"kind": "wire_init", "summary": "Enable metadata-only trace export in the app: read %s from %s and pass it to "
         "the binding's Xybrid observability preset. The key alone exports nothing." % (variable, rel), "command": None},
        {"kind": "verify", "summary": "After one real inference has flushed, run this script's `status` command: "
         "verification.traces_received is true only then.", "command": "python3 -I xybrid_agent_connect.py status"},
        {"kind": "relay", "summary": "Give the user tell_the_user in chat, including the claim link. Keep the link and the "
         "key out of commits, PRs, issues and logs.", "command": None}]
    org = (credential.get("organization") or {}).get("name", "this project")
    tell = ("Created a provisional Xybrid workspace \u201c%s\u201d and saved its restricted key to %s%s." if status == "created"
            else "This project is already connected to the Xybrid workspace \u201c%s\u201d; its key is in %s%s.") % (
        org, rel, " (gitignored)" if repo else " (not in a Git repository, so not gitignored)")
    tell += (" Trace export is not configured by connecting: the key alone sends nothing until the app's observability "
             "setup enables it.")
    if report["claim_url"]:
        key_day, claim_day = credential["key_expires_at"][:10], credential["claim_expires_at"][:10]
        tell += (" Claim it with GitHub by %s: %s (this link is private: don't put it in commits, PRs or issues). Until it "
                 "is claimed, platform access stops on %s and the workspace is scheduled for deletion on %s. On-device "
                 "inference keeps working. The link is saved privately in %s." % (
                     key_day, report["claim_url"], key_day, claim_day, store))
    for label, url in (("Terms", credential.get("terms_url")), ("Privacy", credential.get("privacy_url"))):
        if url:
            tell += " %s: %s." % (label, url)
    report["tell_the_user"] = tell
    report["status"] = status


def provision(args, host, lock, data, root):
    """GET a challenge, solve it, save the retry state, POST /v1/agent/connect."""
    pending = find(data["pending"], host["origin"], root, "connect")
    if pending and not (0 <= (now() - parse_time(pending["started_at"])).total_seconds() < RETRY_WINDOW):
        data["pending"].remove(pending)
        pending = None

    def fresh():
        return {"platform_origin": host["origin"], "repository_root": root, "kind": "connect",
                "idempotency_key": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("="),
                "project_name": (args.project_name or Path(root).name)[:64],
                "source": (args.agent or "")[:64] or None, "started_at": stamp(now())}

    def solve_into(record):
        status, body, _ = call(host, "GET", "/v1/agent/challenge")
        check_failure(status, body, record, lock, data)
        challenge, difficulty = body.get("challenge"), body.get("difficulty")
        if body.get("algorithm") not in (None, "sha256-leading-zero-bits") or not isinstance(challenge, str) or \
                not challenge or len(challenge) > 512 or any(ord(c) < 32 for c in challenge) or \
                not isinstance(difficulty, int) or not 0 <= difficulty <= MAX_DIFFICULTY:
            raise Stop("error", "Couldn't connect: the platform sent a challenge this script can't solve." + LOCAL,
                       "unsupported_challenge")
        expires = parse_time(body["expires_at"]) if body.get("expires_at") else now() + timedelta(minutes=5)
        nonce = solve(challenge, difficulty, time.monotonic() + max(5, (expires - now()).total_seconds() - 15))
        if nonce is None:
            raise Stop("error", "Couldn't connect: the proof of work did not finish before the challenge expired." + LOCAL,
                       "challenge_too_hard")
        record.update(challenge=challenge, nonce=str(nonce))
        if record not in data["pending"]:
            data["pending"].append(record)
        lock.save(data)  # recovery state first: a lost answer is retried with the same key

    if pending is None:
        pending = fresh()
        solve_into(pending)
    refreshed = restarted = False
    busy = 0
    while True:
        info = {"os": platform.system().lower()[:64], "arch": platform.machine().lower()[:64],
                "binding": (args.binding or "agent-script")[:64]}
        body = {"client_info": info, "challenge": pending["challenge"], "nonce": pending["nonce"]}
        for field in ("source", "project_name"):
            if pending.get(field):
                body[field] = pending[field]
        status, answer, headers = call(host, "POST", "/v1/agent/connect", body, {"Idempotency-Key": pending["idempotency_key"]})
        if status in (200, 201) and isinstance(answer.get("api_key"), str) and KEY_RE.match(answer["api_key"]):
            break
        if status in (200, 201):
            status, answer = 502, {"error": "invalid_platform_response", "message": "unexpected platform response"}
        code = answer.get("error")
        if code in ("challenge_expired", "challenge_invalid", "proof_of_work_invalid") and not refreshed:
            refreshed = True
            solve_into(pending)
        elif code in ("idempotency_key_expired", "idempotency_key_conflict", "workspace_claimed",
                      "workspace_unavailable") and not restarted:
            restarted = True
            data["pending"].remove(pending)
            pending = fresh()
            solve_into(pending)
        elif code == "agent_signup_busy" and busy < 3:
            busy += 1
            wait = headers.get("Retry-After", "2")
            time.sleep(min(5, max(1, int(wait) if wait.isdigit() else 2)))
        else:
            check_failure(status, answer, pending, lock, data, applied=True)
            raise Stop("error", "Couldn't connect: unexpected platform answer." + LOCAL, "platform_error")
    token = answer.get("claim_token") if TOKEN_RE.match(answer.get("claim_token") or "") else None
    credential = {"platform_origin": host["origin"], "repository_root": root, "api_key": answer["api_key"],
                  "scope": answer.get("scope"), "organization": answer.get("organization"),
                  "project": answer.get("project"), "key_expires_at": answer.get("key_expires_at"),
                  "claim_expires_at": answer.get("claim_expires_at"), "claim_token": token,
                  "source": pending.get("source"), "console_origin": host["console"] if host["custom"] else None,
                  "claimed": False, "terms_url": answer.get("terms_url"), "privacy_url": answer.get("privacy_url"),
                  "created_at": stamp(now())}
    credential = {k: v for k, v in credential.items() if v is not None or k == "claimed"}
    data["pending"].remove(pending)
    data["credentials"] = [c for c in data["credentials"]
                           if not (c.get("platform_origin") == host["origin"] and c.get("repository_root") == root)]
    data["credentials"].append(credential)
    lock.save(data)
    return credential


def check_failure(status, body, pending, lock, data, applied=False):
    """Raise the outcome for a failed call; keep retry state when the outcome is unknown."""
    if status is not None and 200 <= status < 300:
        return
    code, message = body.get("error"), body.get("message") or body.get("detail") or "no answer"
    unknown = applied and (status is None or status >= 500)
    if not unknown and pending in data["pending"]:
        data["pending"].remove(pending)
        lock.save(data)
    retry = (" A workspace may already have been created: run the same command again within the next few minutes; it "
             "repeats the same request instead of creating another." if unknown else "")
    if code == "agent_signup_closed":
        raise Stop("signup_closed", "Xybrid isn't creating workspaces for coding agents right now, so this project "
                   "wasn't connected. On-device inference works without a key; to see traces, create a workspace and "
                   "API key in the console%s." % (" at " + lock.console if lock.console else ""), code, message)
    if status is None:
        raise Stop("offline", "Couldn't reach Xybrid from this environment." + (retry or " No workspace was created.")
                   + LOCAL, "offline", message)
    raise Stop("error", "Couldn't connect this project to the Xybrid console: %s (HTTP %s).%s" % (
        message.rstrip("."), status, retry) + LOCAL, code or "platform_error", message)


def status_or_claim(args, host, report):
    _, root = project(Path.cwd().resolve())
    store = store_path()
    data = Store(store).load() if store.exists() else {"credentials": []}
    credential = find(data["credentials"], host["origin"], root)
    report["credentials"] = {"path": str(store), "saved": bool(credential),
                             "recovery": bool(credential and credential.get("claim_token")), "moved_corrupt_file": None}
    if not credential:
        raise Stop("not_connected", "This project has no saved Xybrid key; connect it first.", "not_connected")
    report["key"] = {"source": "credentials", "preview": preview(credential["api_key"]), "scope": credential.get("scope")}
    if args.command == "claim":
        report["claim_url"] = claim_url(host, credential)
        if not report["claim_url"]:
            raise Stop("error", "No claim link is saved for this project, or its claim period ended.", "claim_unavailable")
        report["status"] = "claimable"
        report["tell_the_user"] = "Claim the provisional Xybrid workspace with GitHub: %s (private link)." % report["claim_url"]
        return
    status, body, headers = call(host, "GET", "/v1/agent/workspace",
                                 headers={"Authorization": "Bearer " + credential["api_key"]})
    if status is None:
        report["verification"] = {"state": "unavailable", "traces_received": None}
        raise Stop("offline", "Couldn't reach Xybrid to check the key; that says nothing about whether it still works.",
                   "offline")
    if status in (401, 403) and (body.get("error") == "workspace_expired" or headers.get("X-Xybrid-Workspace") == "expired"):
        body = {"expired": True}
    elif status != 200:
        raise Stop("error", "The platform couldn't report on the key right now (HTTP %s)." % status,
                   body.get("error") or "platform_error", body.get("message"))
    verification = body.get("verification") or {"state": "unavailable", "traces_received": None}
    if verification.get("state") not in ("received", "none", "unavailable"):
        verification = {"state": "unavailable", "traces_received": None}
    report["verification"] = verification
    report["workspace"] = {k: body.get(k) for k in ("organization", "project", "provisional", "claimed", "expired",
                                                    "key_expires_at", "claim_expires_at")}
    if body.get("expired"):
        url = claim_url(host, credential)
        raise Stop("workspace_expired", "The provisional Xybrid workspace expired: its key no longer reaches the "
                   "platform.%s On-device inference keeps working." % (
                       " Claim it with GitHub to restore the key: %s (private link)." % url if url else ""))
    report["status"] = "connected"
    report["tell_the_user"] = {
        "received": "Traces from this project have reached the console (%s spans so far)." % verification.get("spans"),
        "none": "No traces from this project have been stored yet: the key alone exports nothing, so check that "
                "observability is enabled and flushed.",
    }.get(verification["state"], "The platform can't confirm traces right now (verification unavailable); that is not "
                                 "a confirmation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("connect", "status", "claim"))
    parser.add_argument("--write-env", help="untracked project file to write the key to (connect)")
    parser.add_argument("--env-name", help="variable to set (default XYBRID_API_KEY; xybrid.apiKey for .properties)")
    parser.add_argument("--agent", help="self-reported agent name, e.g. claude-code")
    parser.add_argument("--binding", help="the app's SDK: python, flutter, react-native, ...")
    parser.add_argument("--project-name")
    parser.add_argument("--platform-url", default=env("XYBRID_PLATFORM_URL") or "https://api.xybrid.dev")
    parser.add_argument("--allow-custom-host", action="store_true")
    parser.add_argument("--allow-ci", action="store_true")
    args = parser.parse_args()
    report = {"schema_version": 1, "command": args.command, "status": "error", "tell_the_user": ""}
    try:
        host = host_for(args.platform_url, args.allow_custom_host)
        report = base_report(args.command, host)
        if args.command == "connect":
            if not args.write_env:
                raise Stop("error", "Pass --write-env with the project file for the key.", "invalid_env_file")
            connect(args, host, report)
        else:
            status_or_claim(args, host, report)
    except (OSError, ValueError) as err:  # local I/O, or data that doesn't parse
        report.update(status="error", tell_the_user="Couldn't connect: %s." % err + LOCAL,
                      error={"code": "io_error" if isinstance(err, OSError) else "invalid_data", "message": str(err)})
    except Stop as stop:
        report["status"] = stop.status
        report["tell_the_user"] = stop.tell
        if stop.code:
            report["error"] = {"code": stop.code, "message": stop.message or stop.tell}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return EXIT[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
