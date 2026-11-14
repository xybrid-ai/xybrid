#!/usr/bin/env python3
"""Connect a project to Xybrid without the CLI: the `xybrid connect` contract.

  python3 -I xybrid_agent_connect.py connect --write-env .env --agent claude-code --binding python
  python3 -I xybrid_agent_connect.py status   # trace verification for the key connect reported
  python3 -I xybrid_agent_connect.py claim    # the private claim link, from the private store

Prints one JSON object (`status`, `tell_the_user`, ...) with the CLI's exit
codes. Never prints the key. Saves recovery state in the CLI's private store
(~/.xybrid/credentials.json, or XYBRID_CREDENTIALS_FILE), so a later
`xybrid auth claim` / `xybrid auth status` finds it, and asks the platform
about a saved key before trusting its deadlines. `status` checks the key
`connect` reported: XYBRID_API_KEY, else the key in --write-env when that is
passed (with the same --env-name), else the saved one. Project files are read
and written as the CLI's env_file.rs does them. Python 3.8+, stdlib only.
"""
import argparse
import base64
import collections
import hashlib
import json
import os
import platform
import re
import secrets
import shlex
import string
import subprocess
import sys
import tempfile
import time
import unicodedata
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
EXIT = {"created": 0, "existing": 0, "connected": 0, "claimable": 0, "claimed": 0, "signup_closed": 3,
        "offline": 4, "ci_skipped": 5, "opted_out": 5, "env_file_tracked": 6, "workspace_expired": 7,
        "not_connected": 1, "error": 1}
MAX_DIFFICULTY = 28          # the platform clamps difficulty to 18..=28
RETRY_WINDOW = 9 * 60        # Idempotency-Key retries are honored for 10 minutes
MAX_RETIRED = 20             # set-aside credentials kept for manual recovery, as the CLI keeps them
KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,256}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
WORD = re.compile(r"^[A-Za-z0-9_@%+=:,./-]+$")  # a shell word that needs no quoting
LOCAL = " On-device inference works without it."
KEEPS = " On-device inference keeps working."
def self_command():
    """The command that re-runs this script: its interpreter and absolute path.

    Suggested commands must work from the project directory, while the script
    lives next to the installed skill or in a temporary directory, so they name
    it by absolute path, each part as one shell word."""
    words = [sys.executable or "python3", "-I", os.path.abspath(__file__)]
    if os.name == "nt":
        return " ".join(w if WORD.match(w) else '"%s"' % w for w in words)
    return " ".join(shlex.quote(w) for w in words)


SELF = self_command()
CLAIM = "`%s claim` (or `xybrid auth claim`)" % SELF
NEW = " To start over with a new workspace instead, run `xybrid connect --new` (with the CLI)."
EXPORT_NOTE = (" Trace export is not configured by connecting: the key alone sends nothing until the app's "
               "observability setup enables it.")
FAILED = {"connect": "Couldn't connect", "status": "Couldn't check the key", "claim": "Couldn't show the claim link"}
ACTION = ["connect"]  # the running command, for error wording


class Stop(Exception):
    """End with `status` (and `error.code` when status is error)."""

    def __init__(self, status, tell, code=None, message=None):
        super().__init__(tell)
        self.status, self.tell, self.code, self.message = status, tell, code, message


def fail(code, message):
    """An error (`status: error`) that ends the run, `message` saying why."""
    return Stop("error", "%s: %s." % (FAILED[ACTION[0]], message.rstrip(".")) + LOCAL, code, message)


def now():
    return datetime.now(timezone.utc)


def stamp(at):
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(raw):
    """RFC 3339 as the platform and the CLI write it (fractions up to nanoseconds, `Z`)."""
    raw = re.sub(r"\.([0-9]+)", lambda m: "." + (m.group(1) + "00000")[:6], raw.strip().upper())
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


# A date, `T`, a time and its UTC offset: a subset of what the CLI's parser (chrono) reads, so a deadline saved
# here always loads there, and one without an offset (which can't be compared) is never taken.
RFC3339 = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?"
                     r"(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])")


def timestamp(raw):
    """`raw` if it is an RFC 3339 time with its UTC offset, else None."""
    if not isinstance(raw, str) or not RFC3339.fullmatch(raw):
        return None
    try:
        parse_time(raw)
    except ValueError:  # out of range: month 13, February 30, ...
        return None
    return raw


def passed(raw):
    return bool(raw) and now() >= parse_time(raw)


def day(raw):
    return raw[:10] if raw else "?"


def env(name):
    value = os.environ.get(name, "").strip()
    return value or None


def preview(key):
    prefix = next((p for p in ("sk_live_", "sk_test_") if key.startswith(p)), "")
    rest = key[len(prefix):]
    return prefix + rest[:min(8, len(rest) // 3)] + "\u2026"


def clean_name(raw):
    """A platform-supplied name with control characters and runs of blanks squeezed, as the CLI cleans it."""
    text = "".join(" " if unicodedata.category(c) == "Cc" else c for c in raw)
    return " ".join(text.split())[:64]


def named(raw):
    """`{"slug", "name"}` from the platform, cleaned, or None."""
    if not isinstance(raw, dict) or not all(isinstance(raw.get(k), str) for k in ("slug", "name")):
        return None
    return {"slug": clean_name(raw["slug"]), "name": clean_name(raw["name"])}


def safe_link(raw):
    return raw.strip() if isinstance(raw, str) and re.match(r"^https?://[^\s\x00-\x1f]+$", raw.strip()) else None


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
        raise fail("invalid_platform_url", "the platform URL is not an http(s) origin")
    origin = "%s://%s" % (url.scheme, url.netloc.lower())
    if origin in PINNED:
        console, traces = PINNED[origin]
        return {"origin": origin, "console": console, "traces": traces, "custom": False, "loopback": False}
    loopback = url.hostname in ("localhost", "127.0.0.1", "::1")
    if not allow_custom:
        raise fail("custom_host_not_allowed", "%s is not a Xybrid platform host" % origin)
    if url.scheme != "https" and not loopback:
        raise fail("insecure_platform_url", "%s must use https" % origin)
    console = env("XYBRID_CONSOLE_URL")
    return {"origin": origin, "console": console.rstrip("/") if console else None, "traces": None, "custom": True,
            "loopback": loopback}


def host_flags(host):
    """The flags that select `host`, for commands the report suggests."""
    if host["origin"] == "https://api.xybrid.dev":
        return []
    return ["--platform-url", host["origin"]] + (["--allow-custom-host"] if host["custom"] else [])


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


def deadlines_ok(body, required=False):
    """Whether `body`'s deadlines are RFC 3339 times with an offset (or, unless `required`, absent or null)."""
    return all(timestamp(body.get(k)) or (not required and body.get(k) is None)
               for k in ("key_expires_at", "claim_expires_at"))


def check_key(host, key):
    """Ask GET /v1/agent/workspace about `key`: ("ok", answer), or ("expired" | "rejected" | "redirect" |
    "offline" | "failed" | "malformed", details). Only the platform's own `401 invalid_api_key` rejects a key:
    a 401 from a proxy in front of it says nothing about the key. A 2xx answer off the contract (no organization
    or project, a deadline that isn't an RFC 3339 time) is a failure without a platform error code, as the CLI
    takes it: nothing in it is saved."""
    if not KEY_RE.match(key):
        return "malformed", {"error": "invalid_api_key_format", "message": "the key is not in the Xybrid key format"}
    status, body, headers = call(host, "GET", "/v1/agent/workspace", headers={"Authorization": "Bearer " + key})
    if status is None:
        return "offline", body
    if 200 <= status < 300:
        if named(body.get("organization")) and named(body.get("project")) and deadlines_ok(body):
            return "ok", body
        return "failed", {"status": status, "message": "unexpected platform response"}
    if 300 <= status < 400:
        return "redirect", {"status": status}
    code = body.get("error")
    if status in (401, 403) and (code == "workspace_expired" or
                                 (headers.get("X-Xybrid-Workspace") or "").strip() == "expired"):
        return "expired", body
    if status == 401 and code == "invalid_api_key":
        return "rejected", body
    return "failed", dict(body, status=status)


# --- Git ---------------------------------------------------------------------

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
        raise fail("git_failed", "git ls-files failed")
    return bool(out.stdout)


def is_ignored(root, rel):
    code = git(root, "check-ignore", "-q", "--", rel).returncode
    if code not in (0, 1):
        raise fail("git_failed", "git check-ignore failed")
    return code == 0


def ensure_ignored(root, rel, what="Xybrid project key"):
    if is_ignored(root, rel):
        return False
    if any(unicodedata.category(c) == "Cc" for c in rel):
        raise fail("env_file_not_ignorable", "%r can't be gitignored" % rel)
    pattern = "/" + re.sub(r"([\\*?\[])", r"\\\1", rel.rstrip(" ")) + "\\ " * (len(rel) - len(rel.rstrip(" ")))
    path = root / ".gitignore"
    existing = path.read_bytes() if path.exists() else b""
    nl = "\r\n" if b"\r\n" in existing else "\n"
    lead = nl if existing and not existing.endswith(b"\n") else ""
    with open(path, "a", encoding="utf-8", newline="") as handle:
        handle.write("%s# %s written by xybrid_agent_connect.py; keep it out of Git%s%s%s" % (lead, what, nl, pattern, nl))
    if not is_ignored(root, rel):
        raise fail("env_file_not_ignorable", "%s is still not ignored after adding %s to .gitignore" % (rel, pattern))
    return True


# --- the project file, read and written as crates/xybrid-cli/src/platform/env_file.rs does ---
#
# Unrelated text is kept byte for byte (UTF-8, line endings untouched), a nonempty value is never replaced,
# and a single empty assignment is filled in place. A line that loaders read differently is neither filled
# nor trusted: when the variable is set in or at the edge of a multi-line construct (a dotenv quoted value
# left open at the end of its line, a .properties line ending in an odd number of backslashes, an xcconfig
# value ending in a backslash or a /* */ block, which Xcode doesn't know), or the file ends inside a quoted
# value or block an appended key would land in, the file is refused as ambiguous.

Assignment = collections.namedtuple("Assignment", "name condition value")
Target = collections.namedtuple("Target", "path rel given format variable text lookup")  # given: --write-env as typed
AMBIGUITIES = ("several_empty", "quoted_lines", "continuation", "block_comment")
NAME_CHARS = frozenset(string.ascii_letters + string.digits + "_")
# Unicode White_Space, which Rust's str::trim strips. Python's bare strip() also takes \x1c-\x1f, which Rust
# keeps, so every strip in this port names this set.
WS = "\t\n\x0b\x0c\r \x85\xa0\u1680" + "".join(map(chr, range(0x2000, 0x200b))) + "\u2028\u2029\u202f\u205f\u3000"


def file_format(path):
    name = path.name.lower()
    if name.endswith(".xcconfig"):
        return "xcconfig"
    if name.endswith(".properties"):
        return "properties"
    if name == ".env" or name.startswith(".env.") or name.endswith(".env"):
        return "dotenv"
    raise fail("unsupported_env_file", "--write-env must be a dotenv, .properties or .xcconfig file")


def default_variable(fmt):
    return "xybrid.apiKey" if fmt == "properties" else "XYBRID_API_KEY"


def valid_variable(fmt, name):
    """Whether `fmt` can hold `name` safely: a C identifier, or `[A-Za-z0-9_.-]+` for .properties."""
    if not name or len(name) > 128:
        return False
    if fmt == "properties":
        return all(c in NAME_CHARS or c in ".-" for c in name)
    return (name[0] in string.ascii_letters or name[0] == "_") and all(c in NAME_CHARS for c in name)


def strip_export(line):
    """The rest of a dotenv line after `export `, or None."""
    if line.startswith("export") and line[6:7] in (" ", "\t"):
        return line[6:].lstrip(WS)
    return None


def parse_assignment(fmt, line):
    """The assignment on `line`, its raw value not yet unquoted, or None."""
    line = line.lstrip(WS)
    if fmt == "dotenv":
        if line.startswith("#"):
            return None
        rest = strip_export(line)
        line = line if rest is None else rest
        if "=" not in line:
            return None
        name, value = line.split("=", 1)
        return Assignment(name.strip(WS), "", value)
    if fmt == "properties":  # as java.util.Properties reads it: the key ends at the first `=`, `:` or blank
        if not line or line[0] in "#!":
            return None
        end = next((i for i, c in enumerate(line) if c in "=: \t\x0c"), len(line))
        rest = line[end:].lstrip(WS)
        if rest[:1] in ("=", ":"):
            rest = rest[1:].lstrip(WS)
        return Assignment(line[:end], "", rest)
    if line.startswith("//") or line.startswith("#"):
        return None
    start = 0  # NAME[config=Debug][sdk=iphoneos*] = value: each condition may itself hold `=`
    while True:
        eq = line.find("=", start)
        if eq < 0:
            return None
        bracket = line.find("[", start, eq)
        if bracket < 0:
            break
        close = line.find("]", bracket)
        if close < 0:
            return None
        start = close + 1
    left, value = line[:eq], line[eq + 1:]
    bracket = left.find("[")
    name, condition = (left[:bracket].strip(WS), left[bracket:].strip(WS)) if bracket >= 0 else (left.strip(WS), "")
    return Assignment(name, condition, value.split("//")[0])


def value_of(fmt, raw):
    """A value as an application reads it: quotes, inline comments and surrounding blanks removed."""
    value = raw.strip(WS)
    if fmt != "dotenv":
        return value
    for quote in "\"'":
        if value.startswith(quote):
            return value[1:].split(quote)[0].strip(WS)
    at = value.find(" #")
    if at >= 0:
        return value[:at].strip(WS)
    return "" if value.startswith("#") else value


def closing_quote(text, quote):
    """Where the first `quote` no backslash escapes is in `text`, or None."""
    i = 0
    while i < len(text):
        if text[i] == quote:
            return i
        if text[i] == "\\" and text[i + 1:i + 2] == quote:
            i += 1
        i += 1
    return None


def continues_xcconfig(value):
    """Whether an xcconfig value (its `//` comment removed) runs on: it ends in `\\`, then blanks and one `;`."""
    value = value.rstrip(WS)
    if value.endswith(";"):
        value = value[:-1].rstrip(WS)
    return value.endswith("\\")


def pieces(text):
    """`text` split after every "\\n" and nothing else (not at the other breaks str.splitlines knows)."""
    out, start = [], 0
    while start < len(text):
        end = text.find("\n", start) + 1 or len(text)
        out.append(text[start:end])
        start = end
    return out


def text_lines(text):
    return [piece.rstrip("\r\n") for piece in pieces(text)]


class Scanner:
    """Reads a file's lines in order, following its format's multi-line constructs.

    `read` returns (None, assignment or None) for a line every loader reads the same way, or (why,
    [readings]) for a line in or at the edge of a construct, with the assignments it could be.
    """

    def __init__(self, fmt):
        self.fmt, self.state, self.quote = fmt, "plain", None

    def open(self):
        """The construct the lines so far end inside that a blank line doesn't close, if any."""
        return {"quoted": "quoted_lines", "block": "block_comment"}.get(self.state)

    def read(self, line):
        return getattr(self, "read_" + self.fmt)(line)

    def read_dotenv(self, line):
        # Node's and Python's dotenv run a quoted value to its closing quote; line-based loaders stop at the line.
        reading = parse_assignment("dotenv", line)
        if self.state == "quoted":
            if closing_quote(line, self.quote) is not None:
                self.state = "plain"
            return "quoted_lines", [reading]
        if reading is None:
            return None, None
        value = reading.value.lstrip(WS)
        if value[:1] in ('"', "'", "`") and closing_quote(value[1:], value[0]) is None:
            self.state, self.quote = "quoted", value[0]
            return "quoted_lines", [reading]
        return None, reading

    def read_properties(self, line):
        # java.util.Properties joins a line ending in an odd number of backslashes to the next one.
        reading = parse_assignment("properties", line)
        rest = line.lstrip(" \t\x0c")
        continues = (len(line) - len(line.rstrip("\\"))) % 2 == 1
        if self.state == "continued":
            if not rest:
                self.state = "plain"
                return None, None
            if not continues:
                self.state = "plain"
            return "continuation", [reading]
        if not rest or rest[0] in "#!":
            return None, None
        if continues:
            self.state = "continued"
            return "continuation", [reading]
        return None, reading

    def read_xcconfig(self, line):
        # Xcode joins a value ending in `\` to the next line, and has no block comments: it reads the lines
        # inside /* */ as live, which their author meant as commented out.
        reading = parse_assignment("xcconfig", line)
        if self.state == "block":
            meant, at = None, line.find("*/")
            if at >= 0:
                self.state = "plain"
                rest = line[at + 2:]
                after = self.after_blocks(rest)
                meant = parse_assignment("xcconfig", rest if after is None else after)
            return "block_comment", [reading, meant]
        if self.state == "continued":
            rest = line.split("//")[0]
            if not rest.strip(WS):
                self.state = "plain"
                return None, None
            if not continues_xcconfig(rest):
                self.state = "plain"
            after = self.after_blocks(line)
            return "continuation", [reading, None if after is None else parse_assignment("xcconfig", after)]
        after = self.after_blocks(line)
        if after is not None:
            return "block_comment", [parse_assignment("xcconfig", after)]
        if reading is None:
            return None, None
        if continues_xcconfig(reading.value):
            self.state = "continued"
            return "continuation", [reading]
        if "/*" in reading.value or "*/" in reading.value:
            return "block_comment", [reading]
        return None, reading

    def after_blocks(self, text):
        """For text starting with `/*`: what follows every /* */ block in a row ("" when one stays open,
        which opens the block state). None if `text` doesn't start with a block."""
        rest = text.lstrip(WS)
        if not rest.startswith("/*"):
            return None
        rest = rest[2:]
        while True:
            at = rest.find("*/")
            if at < 0:
                self.state = "block"
                return ""
            rest = rest[at + 2:]
            if not rest.lstrip(WS).startswith("/*"):
                return rest
            rest = rest.lstrip(WS)[2:]


def lookup(fmt, text, name):
    """How `text` assigns `name`: "absent", "empty" (on one line, which may be filled in), "present" (never
    replaced), or why no line can be filled nor the file appended to (one of AMBIGUITIES)."""
    scanner, empty, present, unclear = Scanner(fmt), 0, False, None
    for line in text_lines(text):
        why, reading = scanner.read(line)
        if why is None:
            if reading is not None and reading.name == name:
                if value_of(fmt, reading.value):
                    present = True
                else:
                    empty += 1
        elif any(r is not None and r.name == name for r in reading):
            unclear = unclear or why
    if unclear:
        return unclear
    if present:
        return "present"
    if empty == 0:
        return scanner.open() or "absent"  # an appended key goes after a blank line, which ends a join
    return "empty" if empty == 1 else "several_empty"


def values(fmt, text, name):
    """The nonempty values `text` gives `name` on lines every loader reads the same way."""
    scanner, found = Scanner(fmt), []
    for line in text_lines(text):
        why, reading = scanner.read(line)
        if why is None and reading is not None and reading.name == name and value_of(fmt, reading.value):
            found.append(value_of(fmt, reading.value))
    return found


def assigns(fmt, text, name, value):
    """Whether `text` sets `name` to exactly `value` on a line every loader reads the same way."""
    return value in values(fmt, text, name)


def conflicting(target):
    """Whether the file gives its variable different nonempty values: which one an app reads depends on its
    loader, so `status` refuses to pick one."""
    return len(set(values(target.format, target.text or "", target.variable))) > 1


def ambiguous(file, name, why):
    return fail("env_file_ambiguous", {
        "several_empty": "%s assigns %s an empty value on more than one line, and which one the app reads depends on "
                         "its loader; keep a single %s line or choose another --write-env file" % (file, name, name),
        "quoted_lines": "%s has a quoted value spanning several lines that holds or sets %s, or that never closes, and "
                        "dotenv loaders disagree on where such a value ends; set %s on a line of its own outside any "
                        "quoted value, or choose another --write-env file" % (file, name, name),
        "continuation": "%s sets %s on a line that a trailing backslash joins to another, and loaders disagree on "
                        "which lines that joins; set %s on a line of its own, or choose another --write-env file"
                        % (file, name, name),
        "block_comment": "%s sets %s in or next to a /* */ block, or leaves one open. Xcode has no block comments: it "
                         "reads an assignment inside one as live although the block marks it as commented out. "
                         "Comment lines out with // instead, or choose another --write-env file" % (file, name),
    }[why])


def assignment(fmt, name, condition, key):
    return "%s%s = %s" % (name, condition, key) if fmt == "xcconfig" else "%s=%s" % (name, key)


def upsert(fmt, text, name, key, comment):
    """`text` with `name` set to `key` and `comment` above it, everything else unchanged: a single empty
    assignment is filled in place (keeping an xcconfig condition), otherwise the block is appended."""
    if not valid_variable(fmt, name):
        raise fail("invalid_env_name", "%r is not a valid variable name for a %s file" % (name, fmt))
    if not KEY_RE.match(key):
        raise fail("invalid_api_key_format", "the platform returned a key with unexpected characters; nothing was written")
    nl = "\r\n" if "\r\n" in text else "\n"
    prefix = "//" if fmt == "xcconfig" else "#"
    block = "".join("%s %s%s" % (prefix, "".join(" " if unicodedata.category(c) == "Cc" else c for c in line).strip(WS),
                                 nl) for line in comment)
    found = lookup(fmt, text, name)
    if found == "present":
        raise fail("env_file_has_key", "%s already has a value; it is never replaced" % name)
    if found in AMBIGUITIES:
        raise ambiguous("the file", name, found)
    if found == "absent":
        out = text
        if out and not out.endswith("\n"):
            out += nl
        if out.strip(WS):
            out += nl  # the blank line also ends a join by a trailing backslash on the last line
        return out + block + assignment(fmt, name, "", key) + nl
    scanner, out = Scanner(fmt), []
    for piece in pieces(text):
        content = piece.rstrip("\r\n")
        ending = piece[len(content):]
        why, reading = scanner.read(content)
        if why is None and reading is not None and reading.name == name and not value_of(fmt, reading.value):
            export = "export " if fmt == "dotenv" and strip_export(content.lstrip(WS)) is not None else ""
            out.append(block + export + assignment(fmt, name, reading.condition, key) + (ending or nl))
        else:
            out.append(piece)
    return "".join(out)


def read_project_file(path):
    """The file's text, or None if it doesn't exist. Symlinks are refused: replacing one would detach it."""
    if path.is_symlink():
        raise fail("env_file_symlink", "%s is a symbolic link; point --write-env at the real file" % path)
    if not path.exists():
        return None
    if not path.is_file():
        raise fail("env_file_unwritable", "%s is not a regular file" % path)
    try:
        raw = path.read_bytes()
    except OSError as err:
        raise fail("env_file_unwritable", "could not read %s: %s" % (path, err))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise fail("env_file_unwritable", "%s is not UTF-8 text" % path)


def prepare_target(cwd, repo, raw, env_name):
    """The --write-env file, checked before anything is requested or changed."""
    joined = cwd / raw
    if joined.name in ("", ".", ".."):
        raise fail("invalid_env_file", "--write-env %s must name a file" % raw)
    try:
        parent = joined.parent.resolve(strict=True)
    except (OSError, RuntimeError) as err:
        raise fail("env_file_unwritable", "the directory %s is not usable: %s" % (joined.parent, err))
    if not parent.is_dir():
        raise fail("env_file_unwritable", "%s is not a directory" % parent)
    path = parent / joined.name
    fmt = file_format(path)
    variable = default_variable(fmt) if env_name is None else env_name
    if not valid_variable(fmt, variable):
        raise fail("invalid_env_name", "%r is not a valid variable name for a %s file" % (variable, fmt))
    if repository_root(parent) != repo:
        raise fail("env_file_outside_repository", "%s is outside this Git repository, so it can't be checked or "
                   "gitignored from here; choose a file inside it" % path)
    text = read_project_file(path)
    return Target(path, path.relative_to(repo).as_posix() if repo else str(path), raw, fmt, variable, text,
                  "absent" if text is None else lookup(fmt, text, variable))


def check_writable(path):
    """The file's directory accepts new files (nothing is left behind)."""
    try:
        with tempfile.NamedTemporaryFile(prefix=".xybrid-write-check-", dir=str(path.parent)):
            pass
    except OSError as err:
        raise fail("env_file_unwritable", "%s can't be written: %s" % (path, err))


def write_atomic(path, text, private=False):
    fd, temp = tempfile.mkstemp(prefix=".xybrid-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:  # no newline translation
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


def file_comment(credential):
    """The comment above the key: the claim command and dates, never the claim token or link. The workspace can be
    claimed until the claim deadline; its key stops at the earlier key deadline."""
    org = org_name(credential)
    if not is_provisional(credential):
        return ['Xybrid key for the workspace "%s", written by xybrid_agent_connect.py.' % org,
                "Never commit this file or share the key."]
    return ['Xybrid provisional workspace "%s": restricted client key written by xybrid_agent_connect.py.' % org,
            "Claim it with GitHub by %s to keep it: run `xybrid_agent_connect.py claim` (or `xybrid auth claim`)."
            % day(credential.get("claim_expires_at")),
            "Unclaimed, the key stops on %s (UTC) and the workspace is deleted on %s." % (
                day(credential.get("key_expires_at")), day(credential.get("claim_expires_at"))),
            "Never commit this file or share the key."]


# --- the private store (same file and format as the CLI) -------------------

def guard_store(store):
    """Keep a store inside any Git working tree (not only this project's) out of commits, as the CLI does."""
    directory = store.parent.resolve()
    existing = next((d for d in [directory] + list(directory.parents) if d.exists()), None)
    root = repository_root(existing) if existing is not None else None
    if root is None:
        return
    if directory == root:
        raise fail("credentials_in_repository", "%s would keep the credentials directly in the Git repository %s; "
                   "point XYBRID_CREDENTIALS_FILE to a directory of its own" % (store, root))
    rel = directory.relative_to(root).as_posix()
    if is_tracked(root, rel):
        raise fail("credentials_tracked", "the credentials directory %s holds files tracked by Git; point "
                   "XYBRID_CREDENTIALS_FILE to a directory of its own" % rel)
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
                raise fail("credentials_unsafe", "%s is not private to this account (chmod 700 it)" % directory)
            import fcntl
            self.lock_file = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            deadline = time.monotonic() + 90
            while True:
                try:
                    fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise fail("credentials_busy", "another xybrid process is updating %s" % self.path)
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
                raise fail("credentials_unsafe", "%s is not private (chmod 600 it)" % self.path)
        try:
            data = json.loads(self.path.read_bytes().decode("utf-8"))
        except ValueError:
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("version"), int):
            raise fail("credentials_corrupt", "%s is not a valid xybrid credentials file; it was left untouched"
                       % self.path)
        if data["version"] > 1:
            raise fail("credentials_unsupported", "%s was written by a newer xybrid" % self.path)
        for field in ("credentials", "pending", "retired"):
            data.setdefault(field, [])
        return data

    def save(self, data):
        write_atomic(self.path, json.dumps(data, indent=2) + "\n", private=True)


def read_store(path):
    """The store without its lock, for reading: (data or None, the Stop it raised or None)."""
    try:
        return (Store(path).load() if path.exists() else None), None
    except Stop as stop:
        return None, stop


def update_stored(path, host, root, checked, update):
    """Apply `update` to the store under its lock, only while this project's entry still holds `checked`, the
    key the platform just answered about: another process may have saved another key meanwhile. Best effort."""
    try:
        with Store(path) as lock:
            data = lock.load()
            entry = find(data["credentials"], host["origin"], root)
            if entry is not None and entry.get("api_key") == checked:
                update(data)
                lock.save(data)
    except (Stop, OSError, ValueError):
        pass


def find(entries, origin, root, kind=None):
    for entry in entries:
        if entry.get("platform_origin") == origin and entry.get("repository_root") == root and \
                (kind is None or entry.get("kind") == kind):
            return entry
    return None


def push_retired(data, credential, reason):
    data["retired"].append({"reason": reason, "retired_at": stamp(now()), "credential": credential})
    del data["retired"][:-MAX_RETIRED]


def put(data, credential):
    """Save `credential`, retiring a different key saved for the same host and repository."""
    same = [c for c in data["credentials"] if c.get("platform_origin") == credential["platform_origin"] and
            c.get("repository_root") == credential["repository_root"]]
    data["credentials"] = [c for c in data["credentials"] if all(c is not s for s in same)]
    for previous in same:
        if previous.get("api_key") != credential.get("api_key"):
            push_retired(data, previous, "replaced")
    data["credentials"].append(credential)


def retire(data, origin, root, reason):
    entry = find(data["credentials"], origin, root)
    if entry is not None:
        data["credentials"] = [c for c in data["credentials"] if c is not entry]
        push_retired(data, entry, reason)


def mark_claimed(data, token):
    """Record a claim for every repository in the workspace behind `token`."""
    for credential in data["credentials"]:
        if not credential.get("claimed") and credential.get("claim_token") == token:
            credential["claimed"] = True


def refresh(credential, body):
    """`credential` with the platform's lifecycle answer folded in, as the CLI's refresh_credential."""
    credential = dict(credential, organization=named(body["organization"]), project=named(body["project"]))
    if isinstance(body.get("scope"), str):
        credential["scope"] = body["scope"]
    if body.get("claimed") is True:
        credential["claimed"] = True
    for field in ("key_expires_at", "claim_expires_at"):
        if timestamp(body.get(field)):
            credential[field] = body[field]
    return credential


def store_refresh(data, host, root, body):
    refreshed = refresh(find(data["credentials"], host["origin"], root), body)
    put(data, refreshed)
    if body.get("claimed") is True and refreshed.get("claim_token"):
        mark_claimed(data, refreshed["claim_token"])


def is_provisional(credential):
    return not credential.get("claimed") and bool(credential.get("key_expires_at"))


def key_expired(credential):
    return is_provisional(credential) and passed(credential["key_expires_at"])


def claimable(credential):
    return not credential.get("claimed") and bool(credential.get("claim_token")) and \
        bool(credential.get("claim_expires_at")) and not passed(credential["claim_expires_at"])


def retired_claimable(data, origin, root):
    """The newest set-aside credential of this project whose claim link still works."""
    for retired in reversed(data.get("retired") or []):
        credential = retired.get("credential") if isinstance(retired, dict) else None
        if isinstance(credential, dict) and credential.get("platform_origin") == origin and \
                credential.get("repository_root") == root and claimable(credential):
            return credential
    return None


def claim_url(host, credential):
    token = credential.get("claim_token")
    if not claimable(credential) or not TOKEN_RE.match(token):
        return None
    console = (credential.get("console_origin") or host["console"]) if host["custom"] else host["console"]
    return "%s/claim/%s" % (console.rstrip("/"), token) if console else None


def org_name(credential):
    return (credential.get("organization") or {}).get("name") or "this project"


# --- report pieces ------------------------------------------------------------

def base_report(command, host):
    return {"schema_version": 1, "command": command, "client": "xybrid_agent_connect.py", "status": "error",
            "error": None, "platform": {"api_origin": host["origin"], "console_origin": host["console"],
                                        "traces_endpoint": host["traces"], "custom_host": host["custom"]},
            "key": None, "env_file": None, "credentials": None, "stored": None, "workspace": None, "claim_url": None,
            "verification": None, "next_steps": [], "tell_the_user": ""}


def key_report(source, credential):
    return {"source": source, "preview": preview(credential["api_key"]), "scope": credential.get("scope")}


def stored_report(credential):
    if not credential:
        return None
    return {"key_expires_at": credential.get("key_expires_at"), "claim_expires_at": credential.get("claim_expires_at"),
            "claimed": bool(credential.get("claimed")), "recovery": bool(credential.get("claim_token"))}


def workspace_from_server(body):
    report = {k: body.get(k) if timestamp(body.get(k)) else None for k in ("key_expires_at", "claim_expires_at")}
    report.update(organization=named(body["organization"]), project=named(body["project"]),
                  provisional=body.get("provisional") is True, claimed=body.get("claimed") is True,
                  expired=body.get("expired") is True, lifecycle_source="server")
    return report


def workspace_from_credential(credential, source):
    report = {k: credential.get(k) for k in ("organization", "project", "key_expires_at", "claim_expires_at",
                                             "terms_url", "privacy_url")}
    report.update(provisional=is_provisional(credential), claimed=bool(credential.get("claimed")),
                  expired=key_expired(credential), lifecycle_source=source)
    return report


def verification_from(body):
    verification = body.get("verification")
    if not isinstance(verification, dict) or verification.get("state") not in ("received", "none", "unavailable"):
        return {"state": "unavailable", "traces_received": None}
    return verification


UNAVAILABLE = {"state": "unavailable", "traces_received": None}


def expired_outcome(lead, claim_deadline, has_token, url):
    """Raise the outcome for an expired provisional workspace, as the CLI's expired_outcome: `workspace_expired`
    while it can still be claimed, else an error (`claim_expired`), since nothing restores it then."""
    if claim_deadline and passed(claim_deadline):
        ended = day(claim_deadline)
        raise Stop("error", "%s Its claim period ended on %s, so it can no longer be claimed and is scheduled for "
                   "deletion.%s%s" % (lead, ended, NEW, KEEPS), "claim_expired",
                   "The workspace expired and its claim period ended on %s; it can no longer be claimed" % ended)
    by = " by %s" % day(claim_deadline) if claim_deadline else ""
    if url:
        restore = " Claim it with GitHub%s to restore the key: %s (private link; also %s)." % (by, url, CLAIM)
    elif has_token:
        restore = " Claim it with GitHub%s to restore the key: run %s." % (by, CLAIM)
    elif claim_deadline:
        restore = " It can be claimed until %s with its claim link, which is not saved on this machine." % day(
            claim_deadline)
    else:
        restore = " Its claim link is not saved on this machine."
    raise Stop("workspace_expired", lead + restore + NEW + KEEPS)


def expired_lead(name, key_deadline):
    return "The provisional Xybrid workspace \u201c%s\u201d expired%s: its key no longer reaches the platform." % (
        name, " on %s" % day(key_deadline) if key_deadline else "")


def next_steps(host, target, home):
    """wire_init / verify / relay for a key in the project file ("file"), a key the file already held that this
    script didn't save ("env_file"), or XYBRID_API_KEY ("env"). The command runs from the directory this one ran
    in, so it names the --write-env file as it was given."""
    check, unshown = ["status"], ""
    if home == "env":
        source, note = "read XYBRID_API_KEY from the environment the app already uses", " with the same XYBRID_API_KEY"
    else:
        source, note = "read %s from %s" % (target.variable, target.rel), ""
    if home == "env_file":
        check += ["--write-env", target.given]
        if target.variable != default_variable(target.format):
            check += ["--env-name", target.variable]
        note = " with %s: this script didn't save that key, so `status` reads it from that file" % " ".join(check[1:])
        if conflicting(target):  # `status` refuses such a file, so a command would be certain to fail
            unshown = (" No command is shown because %s gives %s different values on several lines, and `status` "
                       "won't guess which one the app reads: keep a single %s line first."
                       % (target.rel, target.variable, target.variable))
    words = check + host_flags(host)
    command = " ".join([SELF] + words) if not unshown and all(WORD.match(w) for w in words) else None
    if not command and not unshown:
        unshown = " No command is shown because a value can't be written as a plain shell word."
    return [
        {"kind": "wire_init", "summary": "Enable metadata-only trace export in the app: %s and pass it to the "
         "binding's Xybrid observability preset. The key alone exports nothing." % source, "command": None},
        {"kind": "verify", "summary": "Note verification.spans from this script's `status` command%s, run one real "
         "inference, let it flush, and run `status` again: only a rise in verification.spans verifies the run.%s"
         % (note, unshown), "command": command},
        {"kind": "relay", "summary": "Give the user tell_the_user in chat, including the claim link. Keep the link and the "
         "key out of commits, PRs, issues and logs.", "command": None}]


# --- connect -------------------------------------------------------------------

def project(cwd):
    repo = repository_root(cwd)
    return repo, str(repo or cwd)


def holds_saved_key(host, root, target):
    """Whether the project file assigns the key saved for this host and repository (an unreadable store: no)."""
    data, _ = read_store(store_path())
    saved = find(data["credentials"], host["origin"], root) if data else None
    return bool(saved) and assigns(target.format, target.text or "", target.variable, saved.get("api_key"))


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
    target = prepare_target(cwd, repo, args.write_env, args.env_name)
    rel = target.rel
    # Reported where the CLI reports it: not for a file refused before the store is opened.
    file_report = {"path": rel, "variable": target.variable, "format": target.format, "written": False,
                   "already_configured": target.lookup == "present", "tracked": False, "gitignored": False,
                   "gitignore_updated": False, "suggestions": []}
    if repo is not None and is_tracked(repo, rel):
        candidates = {"dotenv": (".env.local", ".env.development.local"),
                      "properties": ("local.properties", "xybrid.local.properties"),
                      "xcconfig": ("Secrets.xcconfig", "Secrets.local.xcconfig")}[target.format]
        prefix = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
        suggestions = [prefix + c for c in candidates if c != target.path.name and not is_tracked(repo, prefix + c)][:1]
        report["env_file"] = dict(file_report, tracked=True, suggestions=suggestions)
        raise Stop("env_file_tracked", "Didn't connect: %s is tracked by Git, so a key written there would be committed. "
                   "Use an untracked file%s. On-device inference works without a key." % (
                       rel, " such as %s" % suggestions[0] if suggestions else ""))
    explicit = env("XYBRID_API_KEY")
    if explicit:
        report["env_file"] = dict(file_report, gitignored=repo is not None and is_ignored(repo, rel))
        report["key"] = {"source": "env", "preview": preview(explicit), "scope": None}
        report["next_steps"] = next_steps(host, target, "env")
        raise Stop("existing", "This environment already provides a Xybrid key (XYBRID_API_KEY); nothing was changed. "
                   "Run this script's `status` command with the same XYBRID_API_KEY to check it.")
    # A key already in the file is never replaced. If it is the saved key, it is checked with the platform below.
    if target.lookup == "present" and not holds_saved_key(host, root, target):
        updated = ensure_ignored(repo, rel) if repo is not None else False
        report["env_file"] = dict(file_report, gitignored=repo is not None, gitignore_updated=updated)
        report["key"] = {"source": "env_file", "preview": None, "scope": None}
        report["next_steps"] = next_steps(host, target, "env_file")
        raise Stop("existing", "%s already sets %s; nothing was changed%s.%s" % (
            rel, target.variable, " except adding it to .gitignore" if updated else "",
            " It sets it to different values on several lines, so which one the app reads depends on its loader: "
            "keep a single %s line." % target.variable if conflicting(target) else ""))
    if target.lookup in AMBIGUITIES:
        raise ambiguous(rel, target.variable, target.lookup)

    store = store_path()
    guard_store(store)
    with Store(store, host["console"]) as lock:
        data = lock.load()
        report["credentials"] = {"path": str(store), "saved": False, "recovery": False, "moved_corrupt_file": None}
        check_writable(target.path)
        report["env_file"] = file_report  # reported even if provisioning fails: .gitignore may change here
        if repo is not None:
            file_report.update(gitignore_updated=ensure_ignored(repo, rel), gitignored=True)
        session = Connect(args, host, report, repo, root, target, store, lock, data)
        saved = find(data["credentials"], host["origin"], root)
        if saved:
            session.reuse(saved)
        else:
            session.created(provision(args, host, lock, data, root))


class Connect:
    """A connect run past preflight, holding the store's lock so concurrent runs serialize, as the CLI's does."""

    def __init__(self, args, host, report, repo, root, target, store, lock, data):
        self.args, self.host, self.report, self.repo, self.root = args, host, report, repo, root
        self.target, self.store, self.lock, self.data = target, store, lock, data

    def reuse(self, credential):
        """Reuse the saved credential after asking the platform about it: a claim made in the console, new
        deadlines or a revocation only show there. Only without an answer do the saved deadlines decide."""
        kind, body = check_key(self.host, credential["api_key"])
        if kind == "ok":
            credential = refresh(credential, body)
            put(self.data, credential)
            if body.get("claimed") is True and credential.get("claim_token"):
                mark_claimed(self.data, credential["claim_token"])
            self.lock.save(self.data)
            self.report["workspace"] = workspace_from_server(body)
            self.report["verification"] = verification_from(body)
            if body.get("expired") is True:
                return self.expired(credential)
            return self.existing(credential)
        if kind == "expired":
            self.report["workspace"] = workspace_from_credential(credential, "stored")
            return self.expired(credential)
        if kind in ("rejected", "malformed"):
            retire(self.data, self.host["origin"], self.root, "rejected by the platform")
            self.lock.save(self.data)
            raise fail("stored_credential_rejected", "the saved key for this project was rejected by %s; it may have "
                       "been revoked or its workspace deleted. It was set aside in %s. If the workspace can still be "
                       "claimed, %s shows its saved link; otherwise `xybrid connect --new` (with the CLI) creates a "
                       "new workspace" % (self.host["origin"], self.store, CLAIM))
        if kind == "redirect":
            raise fail("redirect_refused", "the platform answered with a redirect (HTTP %s), which is never followed"
                       % body["status"])
        self.report["verification"] = UNAVAILABLE
        self.report["workspace"] = workspace_from_credential(credential, "stored")
        if key_expired(credential):
            return self.expired(credential)
        self.existing(credential)

    def mark_saved(self, credential):
        self.report["credentials"].update(saved=True, recovery=bool(credential.get("claim_token")))

    def expired(self, credential):
        self.report["key"] = key_report("credentials", credential)
        self.mark_saved(credential)
        self.report["claim_url"] = claim_url(self.host, credential)
        expired_outcome(expired_lead(org_name(credential), credential.get("key_expires_at")),
                        credential.get("claim_expires_at"), bool(credential.get("claim_token")), self.report["claim_url"])

    def existing(self, credential):
        self.report["key"] = key_report("credentials", credential)
        self.mark_saved(credential)
        self.report["claim_url"] = claim_url(self.host, credential)
        self.report["verification"] = self.report["verification"] or {"state": "not_checked", "traces_received": None}
        written = self.write(credential, created=False)
        if written is None:
            return
        self.report["next_steps"] = next_steps(self.host, self.target, "file")
        note = (" Its key was written to %s%s." % (self.target.rel, self.ignore_note()) if written
                else " %s already had it." % self.target.rel)
        self.report["tell_the_user"] = "This project is already connected to the %sXybrid workspace \u201c%s\u201d.%s" % (
            "provisional " if is_provisional(credential) else "", org_name(credential), note) + EXPORT_NOTE + \
            self.claim_sentences(credential)
        self.report["status"] = "existing"

    def created(self, credential):
        self.report["key"] = key_report("created", credential)
        self.report["workspace"] = workspace_from_credential(credential, "provision")
        self.mark_saved(credential)
        self.report["claim_url"] = claim_url(self.host, credential)
        self.report["verification"] = {"state": "not_checked", "traces_received": None}
        self.report["next_steps"] = next_steps(self.host, self.target, "file")
        written = self.write(credential, created=True)
        if written is None:
            return
        saved_to = (self.target.rel + self.ignore_note() if written else "the private credentials store (%s already "
                    "set %s)" % (self.target.rel, self.target.variable))
        tell = "Created a provisional Xybrid workspace \u201c%s\u201d and saved its restricted key to %s." % (
            org_name(credential), saved_to)
        if self.host["custom"]:
            tell += " It lives on the custom platform host %s, not the Xybrid production service." % self.host["origin"]
        tell += EXPORT_NOTE + self.claim_sentences(credential)
        for label, url in (("Terms", credential.get("terms_url")), ("Privacy", credential.get("privacy_url"))):
            if url:
                tell += " %s: %s." % (label, url)
        self.report["tell_the_user"] = tell
        self.report["status"] = "created"

    def ignore_note(self):
        return " (gitignored)" if self.repo is not None else " (not in a Git repository, so not gitignored)"

    def write(self, credential, created):
        """Write the key into the project file unless it gained a value meanwhile. True if written, False if it
        already had a value, None if writing failed (reported: the credential is saved already)."""
        try:
            written = self.write_project_file(credential)
        except OSError as err:
            written = fail("env_file_unwritable", "could not write %s: %s" % (self.target.path, err))
        except Stop as stop:
            written = stop
        if isinstance(written, Stop):
            self.file_failed(written, credential, created)
            return None
        if written:
            credential["env_file"] = {"path": self.target.rel, "variable": self.target.variable}
            put(self.data, credential)
            self.lock.save(self.data)
        return written

    def write_project_file(self, credential):
        # Proof of work and platform requests take a while: a checkout or a .gitignore edit may have changed
        # whether Git tracks or ignores the file since preflight, so both are checked again right before.
        target, report = self.target, self.report["env_file"]
        text = read_project_file(target.path) or ""
        if lookup(target.format, text, target.variable) == "present":
            report["already_configured"] = True
            return False
        if self.repo is not None:
            if is_tracked(self.repo, target.rel):
                report.update(tracked=True, gitignored=False)
                raise fail("env_file_tracked", "%s became tracked by Git while connecting, so a key written there would "
                           "be committed; nothing was written to it. Untrack it or choose another --write-env file"
                           % target.rel)
            if not is_ignored(self.repo, target.rel):
                report["gitignored"] = False
                raise fail("env_file_not_ignored", "%s stopped being ignored by Git while connecting (its .gitignore "
                           "entry changed); nothing was written to it" % target.rel)
        write_atomic(target.path, upsert(target.format, text, target.variable, credential["api_key"],
                                         file_comment(credential)))
        report["written"] = True
        return True

    def file_failed(self, stop, credential, created):
        """The file couldn't be written after the credential was saved: say what exists and how to finish."""
        done = ("Created the provisional Xybrid workspace \u201c%s\u201d and saved its restricted key privately in %s"
                if created else "This project is connected to the Xybrid workspace \u201c%s\u201d (its key is saved "
                "privately in %s)") % (org_name(credential), self.store)
        reason = (stop.message or stop.tell).rstrip(".")
        self.report.update(status="error", error={"code": stop.code, "message": reason}, next_steps=[])
        self.report["tell_the_user"] = ("%s, but couldn't write it to %s: %s. Fix that and run the same command again; "
                                        "it reuses the saved key instead of creating another workspace.%s" % (
                                            done, self.target.rel, reason, self.claim_sentences(credential)))

    def claim_sentences(self, credential):
        """How to claim the workspace, and both deadlines: it can be claimed until the claim deadline (when it is
        deleted), while its key stops working at the earlier key deadline."""
        key_at, claim_at = credential.get("key_expires_at"), credential.get("claim_expires_at")
        if not is_provisional(credential) or not key_at or not claim_at:
            return ""
        url = claim_url(self.host, credential)
        if url:
            claim = (" Claim it with GitHub by %s: %s (this link is private: don't put it in commits, PRs or issues)."
                     % (day(claim_at), url))
        elif credential.get("claim_token"):
            claim = " Claim it with GitHub by %s with %s." % (day(claim_at), CLAIM)
        else:
            claim = (" Its claim link was not saved on this machine, so it can't be claimed from here; the key works "
                     "until %s." % day(key_at))
        recover = (" The link is saved privately in %s; %s shows it again." % (self.store, CLAIM)
                   if credential.get("claim_token") else "")
        return "%s Until it is claimed, platform access stops on %s and the workspace is scheduled for deletion on %s.%s%s" % (
            claim, day(key_at), day(claim_at), KEEPS, recover)


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
            raise fail("unsupported_challenge", "the platform sent a challenge this script can't solve")
        expires = parse_time(body["expires_at"]) if timestamp(body.get("expires_at")) else now() + timedelta(minutes=5)
        nonce = solve(challenge, difficulty, time.monotonic() + max(5, (expires - now()).total_seconds() - 15))
        if nonce is None:
            raise fail("challenge_too_hard", "the proof of work did not finish before the challenge expired")
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
        if status in (200, 201) and isinstance(answer.get("api_key"), str) and KEY_RE.match(answer["api_key"]) and \
                named(answer.get("organization")) and named(answer.get("project")) and deadlines_ok(answer, True):
            break  # the contract's shape, as the CLI parses it; anything else may have been applied, so it's retried
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
            raise fail("platform_error", "unexpected platform answer")
    token = answer.get("claim_token") if TOKEN_RE.match(answer.get("claim_token") or "") else None
    credential = {"platform_origin": host["origin"], "repository_root": root, "api_key": answer["api_key"],
                  "scope": answer.get("scope") if isinstance(answer.get("scope"), str) else None,
                  "organization": named(answer.get("organization")), "project": named(answer.get("project")),
                  "key_expires_at": answer["key_expires_at"], "claim_expires_at": answer["claim_expires_at"],
                  "claim_token": token,
                  "source": pending.get("source"), "console_origin": host["console"] if host["custom"] else None,
                  "claimed": False, "terms_url": safe_link(answer.get("terms_url")),
                  "privacy_url": safe_link(answer.get("privacy_url")), "created_at": stamp(now())}
    credential = {k: v for k, v in credential.items() if v is not None or k == "claimed"}
    data["pending"].remove(pending)
    put(data, credential)
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


# --- status and claim ---------------------------------------------------------------

def resolve_key(args, cwd, repo, stored):
    """(key, source, label) the way connect reported it, or None: XYBRID_API_KEY, then the --write-env file
    (as connect reads it), then the saved credential. The CLI's resolver has --api-key first; this script
    takes no key on its command line, where process listings and shell history would show it."""
    explicit = env("XYBRID_API_KEY")
    if explicit:
        return explicit, "env", "XYBRID_API_KEY"
    if args.write_env:
        target = prepare_target(cwd, repo, args.write_env, args.env_name)
        if target.lookup in AMBIGUITIES:
            raise ambiguous(target.rel, target.variable, target.lookup)
        if conflicting(target):
            raise fail("env_file_ambiguous", "%s assigns %s different values on several lines, and which one the app "
                       "reads depends on its loader; keep a single %s line" % (target.rel, target.variable,
                                                                               target.variable))
        found = values(target.format, target.text or "", target.variable)
        if found:
            return found[0], "env_file", "%s in %s" % (target.variable, target.rel)
    if stored:
        return stored["api_key"], "credentials", "saved credentials"
    return None


def status(args, host, report):
    cwd = Path.cwd().resolve()
    repo, root = project(cwd)
    path = store_path()
    data, warning = read_store(path)
    stored = find(data["credentials"], host["origin"], root) if data else None
    report["credentials"] = {"path": str(path), "saved": bool(stored),
                             "recovery": bool(stored and stored.get("claim_token")), "moved_corrupt_file": None}
    resolved = resolve_key(args, cwd, repo, stored)
    if resolved is None:
        if warning:
            raise warning
        raise Stop("not_connected", "This project has no Xybrid key: XYBRID_API_KEY is not set%s, and no key is saved "
                   "for it. Connect it first." % (", --write-env sets none" if args.write_env else "") + LOCAL,
                   "not_connected", "No Xybrid key for this project on %s" % host["origin"])
    key, source, label = resolved
    # The saved credential describes the key only when it holds that key, however the key arrived.
    matching = stored if stored and stored.get("api_key") == key else None
    report["stored"] = stored_report(matching)
    report["key"] = {"source": source, "preview": preview(key), "scope": matching.get("scope") if matching else None}
    kind, body = check_key(host, key)
    if kind == "ok":
        if isinstance(body.get("scope"), str):
            report["key"]["scope"] = body["scope"]
        workspace = report["workspace"] = workspace_from_server(body)
        verification = report["verification"] = verification_from(body)
        if matching and refresh(matching, body) != matching:  # save what the platform says (a claim, deadlines)
            update_stored(path, host, root, key, lambda saved: store_refresh(saved, host, root, body))
        name = workspace["organization"]["name"]
        if workspace["expired"]:
            expired_outcome(expired_lead(name, workspace["key_expires_at"]), workspace["claim_expires_at"],
                            bool(matching and matching.get("claim_token")), None)
        if workspace["claimed"]:
            lifecycle = " It has been claimed."
        elif workspace["provisional"] and workspace["key_expires_at"] and workspace["claim_expires_at"]:
            lifecycle = " It is unclaimed: its key stops on %s and it is deleted on %s unless claimed (%s)." % (
                day(workspace["key_expires_at"]), day(workspace["claim_expires_at"]), CLAIM)
        else:
            lifecycle = " It is unclaimed (%s)." % CLAIM if workspace["provisional"] else ""
        traces = {
            "received": " Traces from this project have reached the console (%s spans so far)." % (
                verification.get("spans") or 0),
            "none": " No traces from this project have been stored yet: the key alone exports nothing, so check that "
                    "observability is enabled and flushed.",
        }.get(verification["state"], " The platform can't confirm traces right now (verification unavailable); that "
                                     "is not a confirmation.")
        report["tell_the_user"] = "The Xybrid key (%s) works for the workspace \u201c%s\u201d.%s%s" % (
            label, name, lifecycle, traces)
        report["status"] = "connected"
        return
    if kind == "malformed":
        raise fail("invalid_api_key", "the key from %s is not in the Xybrid key format, so it wasn't sent" % label)
    if kind == "rejected":
        if matching:
            update_stored(path, host, root, key, lambda saved: retire(saved, host["origin"], root,
                                                                       "rejected by the platform"))
            raise fail("stored_credential_rejected", "the saved key for this project was rejected; it may have been "
                       "revoked or its workspace deleted. It was set aside. If the workspace can still be claimed, %s "
                       "shows its saved link" % CLAIM)
        raise fail("invalid_api_key", "the key from %s was rejected by the platform" % label)
    if kind == "redirect":
        raise fail("redirect_refused", "the platform answered with a redirect (HTTP %s), which is never followed"
                   % body["status"])
    report["workspace"] = workspace_from_credential(matching, "stored") if matching else None
    report["verification"] = UNAVAILABLE
    if kind == "expired":
        expired_outcome(expired_lead(org_name(matching) if matching else "this project's workspace",
                                     matching and matching.get("key_expires_at")),
                        matching and matching.get("claim_expires_at"), bool(matching and matching.get("claim_token")),
                        None)
    fallback = (" Saved deadlines (not confirmed by the server): the key stops on %s." % day(matching["key_expires_at"])
                if matching and is_provisional(matching) else "")
    if kind == "offline":
        raise Stop("offline", "Couldn't reach Xybrid to check the key; that says nothing about whether it still works."
                   + fallback, "offline", body.get("detail"))
    raise Stop("error", "The platform couldn't report on the key right now (HTTP %s).%s" % (body["status"], fallback),
               body.get("error") or "platform_error", body.get("message"))


def claim(args, host, report):
    _, root = project(Path.cwd().resolve())
    path = store_path()
    data = Store(path).load() if path.exists() else {"credentials": [], "pending": [], "retired": []}
    stored = find(data["credentials"], host["origin"], root)
    quarantined = None if stored else retired_claimable(data, host["origin"], root)
    report["credentials"] = {"path": str(path), "saved": bool(stored),
                             "recovery": bool(stored and stored.get("claim_token")), "moved_corrupt_file": None}
    credential = stored or quarantined
    if credential is None:
        raise Stop("error", "No claim link is saved for this project (%s) on %s. If it was connected from another "
                   "directory, run this command there." % (root, host["origin"]), "claim_unavailable")
    report["workspace"] = workspace_from_credential(credential, "stored")
    if stored and not stored.get("claimed"):  # a claim made in the console only shows on the platform
        kind, body = check_key(host, stored["api_key"])
        if kind == "ok":
            update_stored(path, host, root, stored["api_key"], lambda saved: store_refresh(saved, host, root, body))
            credential = refresh(stored, body)
            report["workspace"] = workspace_from_server(body)
    report["stored"] = stored_report(credential)
    report["key"] = key_report("credentials", credential)
    name = org_name(credential)
    if credential.get("claimed"):
        report["status"] = "claimed"
        report["tell_the_user"] = "The Xybrid workspace \u201c%s\u201d has already been claimed.%s" % (
            name, " Manage it at %s." % host["console"] if host["console"] else "")
        return
    if not credential.get("claim_token"):
        raise Stop("error", "This project's claim link was not saved, so it can't be recovered here.", "claim_unavailable")
    if passed(credential.get("claim_expires_at")):
        raise Stop("error", "The claim period for \u201c%s\u201d ended on %s; the workspace is scheduled for deletion."
                   % (name, day(credential["claim_expires_at"])), "claim_expired")
    url = claim_url(host, credential)
    if not url:
        raise Stop("error", "No console origin is known for %s; set XYBRID_CONSOLE_URL to the console of this custom "
                   "host." % host["origin"], "claim_unavailable")
    report["status"], report["claim_url"] = "claimable", url
    report["tell_the_user"] = ("Claim the Xybrid workspace \u201c%s\u201d with GitHub by %s: %s (this link is private: "
                               "don't put it in commits, PRs or issues).%s%s" % (
                                   name, day(credential["claim_expires_at"]), url,
                                   " Its key has stopped working; claiming restores it." if key_expired(credential) else "",
                                   " The platform rejected this project's saved key, so it was set aside; this is the "
                                   "last claim link saved for it." if quarantined else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("connect", "status", "claim"))
    parser.add_argument("--write-env", help="connect: untracked project file to write the key to; status: the file "
                        "connect found a key in (key.source env_file)")
    parser.add_argument("--env-name", help="variable to set (default XYBRID_API_KEY; xybrid.apiKey for .properties)")
    parser.add_argument("--agent", help="self-reported agent name, e.g. claude-code")
    parser.add_argument("--binding", help="the app's SDK: python, flutter, react-native, ...")
    parser.add_argument("--project-name")
    parser.add_argument("--platform-url", default=env("XYBRID_PLATFORM_URL") or "https://api.xybrid.dev")
    parser.add_argument("--allow-custom-host", action="store_true")
    parser.add_argument("--allow-ci", action="store_true")
    args = parser.parse_args()
    ACTION[0] = args.command
    report = {"schema_version": 1, "command": args.command, "status": "error", "tell_the_user": ""}
    try:
        host = host_for(args.platform_url, args.allow_custom_host)
        report = base_report(args.command, host)
        if args.command == "connect":
            if not args.write_env:
                raise fail("invalid_env_file", "pass --write-env with the project file for the key")
            connect(args, host, report)
        elif args.command == "status":
            status(args, host, report)
        else:
            claim(args, host, report)
    except (OSError, ValueError) as err:  # local I/O, or data that doesn't parse
        report.update(status="error", tell_the_user="%s: %s." % (FAILED[args.command], err) + LOCAL,
                      error={"code": "io_error" if isinstance(err, OSError) else "invalid_data", "message": str(err)})
    except Stop as stop:
        report["status"] = stop.status
        report["tell_the_user"] = stop.tell
        if stop.code:
            report["error"] = {"code": stop.code, "message": stop.message or stop.tell}
    # UTF-8 whatever the locale, as the CLI writes it.
    sys.stdout.buffer.write((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    sys.stdout.flush()
    return EXIT[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
