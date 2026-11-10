---
name: "xybrid-connect"
description: "Connect this project to the Xybrid console with no sign-up: run `xybrid connect` to create or reuse a provisional workspace, keep its restricted key in a gitignored file, wire explicit metadata-only trace export for Rust, Python, Android/Kotlin, iOS/Swift, Flutter, React Native/Expo or Unity, verify one real trace, and give the user the private claim link. Use when asked to connect a project to Xybrid, see traces or observability in the Xybrid console, or set up a Xybrid API key. Skip for offline-only requests, CI, or opt-outs."
---

Connect this project to the Xybrid console without a sign-up. `xybrid connect` creates (or reuses) a **provisional workspace** with a restricted client key, writes the key to a gitignored project file, and keeps a private claim link. You then wire **explicit, metadata-only** trace export into the app, verify one real trace, and give the user the claim link. On-device inference works with or without any of this.

**Input**: $ARGUMENTS (optional: the app directory or its binding, e.g. `flutter`, `apps/mobile`)

If this skill was installed during the current session, your agent may not list it yet: read this file directly instead of invoking it.

---

## Rules that apply to every step

- **Never print the key.** Don't `cat`, `grep`, `echo` or open the file that holds it in a way that shows the value, and never put it in chat, code, commits, PRs, issues, logs or tool output. Never pass it to `--api-key` on a command line. The CLI's JSON never contains it (only `key.preview`, e.g. `sk_live_AbCd1234…`).
- **The claim link is private.** Whoever opens it first can claim the workspace. Give it to the user in chat only. It must never appear in source, comments, commits, PRs, issues or any public place. The private credentials store is its only other home.
- **Read the key from the gitignored file at build or run time.** Never hard-code it, and never copy it into a tracked file.
- **A key alone exports nothing.** Trace export starts only from the app's explicit observability call (or `xybrid --observability`). Wire it with metadata-only content.
- **Keep observability the app already owns.** Starting a Xybrid observability config replaces the process's active one; never replace an app's own exporter without asking.
- **Report what actually happened.** A saved key is not "export configured", and an export that ran is not "verified". Only `verification.traces_received: true` is verified.
- **One workspace.** Don't pass `--new`, `--allow-ci`, `--allow-custom-host` or `--platform-url` unless the user asked for that.

## Step 1: Decide whether to connect

Skip this skill, and say why in one line, when:

- the user asked for offline-only, no cloud, no account or no telemetry;
- this is CI or another unattended build (`CI`, `GITHUB_ACTIONS`, `GITLAB_CI`, `BUILDKITE`, `CIRCLECI`, `JENKINS_URL`, `TF_BUILD`, `TEAMCITY_VERSION`, `BITBUCKET_BUILD_NUMBER`, `CODEBUILD_BUILD_ID`, `TRAVIS` or `DRONE` set; the CLI refuses there too);
- the user opted out: `XYBRID_TELEMETRY_OPTOUT` or `DO_NOT_TRACK` is `1`/`true`/`yes`, or `OTEL_SDK_DISABLED=true`.

Then look for observability the app already configures (search for `ObservabilityConfig`, `init_observability`, `XybridObservabilityConfig`, `Xybrid.Observability`, `InitializeObservability`, `ingest_url`/`ingestUrl`, or `OTEL_EXPORTER_OTLP_*` in its run configuration):

- Already exporting to Xybrid with a key: don't connect again. Go to Step 7 and verify.
- Exporting somewhere else (a Collector, Langfuse, Phoenix): stop and ask whether to switch. Don't add the Xybrid preset next to it.

Otherwise tell the user what you're about to do, in one line, before running anything. For example:

> Connecting this project to a free provisional Xybrid workspace so its traces (model, timing, usage, device type, errors; no prompts or outputs) show up in the console. No sign-up: you can claim it with GitHub later or ignore it.

## Step 2: Pick the project file for the key

| App | Detect by | `--write-env` | `--env-name` | The app reads |
|---|---|---|---|---|
| Rust | `Cargo.toml` depends on `xybrid-sdk` | `.env` | | `XYBRID_API_KEY` |
| Python | `xybrid` in `pyproject.toml` / `requirements*.txt` | `.env` | | `XYBRID_API_KEY` |
| Android (Kotlin) | `settings.gradle(.kts)`, `ai.xybrid:xybrid-kotlin` | `local.properties` (Gradle root) | | `xybrid.apiKey` |
| iOS / macOS (Swift) | `*.xcodeproj` / `project.yml` / `Project.swift` using `Xybrid` | `Config/Secrets.xcconfig` | | `XYBRID_API_KEY` |
| Flutter | `pubspec.yaml` with `xybrid_flutter` | `.env.xybrid` | | `XYBRID_API_KEY` |
| Expo | `package.json` with `expo` and `@xybrid/react-native` | `.env.local` | `EXPO_PUBLIC_XYBRID_API_KEY` | `EXPO_PUBLIC_XYBRID_API_KEY` |
| React Native (bare) | `package.json` with `@xybrid/react-native`, no `expo` | the env loader's file, usually `.env` | | `XYBRID_API_KEY` |
| Unity | `ProjectSettings/ProjectVersion.txt`, `Packages/manifest.json` with `ai.xybrid` | `.env.local` (project root, next to `Assets/`) | | `XYBRID_API_KEY` |

- The file must be inside the Git repository you run `xybrid connect` from. In a monorepo, run from the repository root and pass the app's path, e.g. `--write-env apps/mobile/.env.local`. The directory must already exist.
- The CLI writes dotenv (`.env`, `.env.*`, `*.env`), Java properties (`*.properties`) and xcconfig (`*.xcconfig`) files only. It adds the file to `.gitignore` and checks `git check-ignore`. It never replaces a value that is already there, and refuses a file Git tracks.
- Credentials are kept per repository: one key per repository and platform host.

## Step 3: Make sure the CLI can connect

```bash
xybrid connect --help >/dev/null 2>&1; echo $?     # PowerShell: xybrid connect --help *> $null; $LASTEXITCODE
```

- `0`: go on.
- `2` (`unrecognized subcommand 'connect'`) or `xybrid: command not found`: this CLI predates `connect`. If the user already approved installing the Xybrid CLI in this session, re-run that same installer once (`curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh` on macOS/Linux, `irm https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.ps1 | iex` on Windows) and check again. Never install anything the user didn't approve.
- Still no `connect`, or the environment only has pip/npm/pub: use the [no-CLI path](#no-cli-path-pip-npm-or-pub-only-environments). If that isn't possible either, say: "This environment can't connect to the Xybrid console yet; on-device inference works without it." and stop.

If `~/.xybrid` isn't writable (some sandboxes), the CLI fails with a `credentials_*` error before creating anything. Then, and only then, prefix this and every later `xybrid` command with `XYBRID_CREDENTIALS_FILE=.xybrid/credentials.json`; the CLI gitignores that directory.

## Step 4: Run `xybrid connect`

From the repository root:

```bash
xybrid connect --agent <claude-code|codex|cursor|...> --write-env <file> [--env-name <VAR>] --json
```

It prints one JSON object on stdout and exits with the status's code. It contacts the Xybrid API, solves a small proof of work (a few seconds), saves the key and claim token to `~/.xybrid/credentials.json` (0600), then writes the key and a comment with the claim command and dates (never the link) to the file. It never starts an exporter. Don't run two at once.

## Step 5: Act on the result

| Exit | `status` | Meaning | Do |
|---|---|---|---|
| 0 | `created` | New workspace and key (`workspace.new_project_in_existing_workspace: true`: a new project in an unclaimed workspace this machine already holds) | Steps 6, 7 and 8 |
| 0 | `existing` | The project already has a key: saved (`key.source: "credentials"`), already in the file (`"env_file"`), or from `XYBRID_API_KEY` / `--api-key` (`"env"` / `"flag"`). Nothing was replaced | Steps 6 (if not wired yet), 7 and 8 |
| 3 | `signup_closed` | Xybrid isn't creating agent workspaces right now | Relay `tell_the_user`. Don't retry, and don't ask the user for a key now |
| 4 | `offline` | Xybrid couldn't be reached | Relay. If `tell_the_user` says a workspace may already exist, re-run the same command once within a few minutes (it repeats the same request) |
| 5 | `ci_skipped`, `opted_out` | CI or an opt-out | Relay. Never add `--allow-ci` on your own |
| 6 | `env_file_tracked` | Git tracks the file | Re-run once with `env_file.suggestions[0]` and read the key from that file in Step 6. Never untrack, rewrite or delete a tracked file |
| 7 | `workspace_expired` | The saved workspace's key expired; it can still be claimed (`claim_url`) | Relay (it includes the claim link). Export can't work until it is claimed. `--new` only if the user asks |
| 1 | `error` | `error.code` says why | Relay `tell_the_user`; fix only the local causes below |

Useful `error.code` values: `claim_expired` (past the claim deadline; `xybrid connect --new` only if the user asks), `stored_credential_rejected` (the saved key was revoked or deleted; `tell_the_user` says how to recover), `unsupported_env_file` / `invalid_env_name` / `env_file_outside_repository` / `env_file_symlink` / `env_file_ambiguous` / `env_file_not_ignorable` / `env_file_unwritable` (choose another file), `credentials_*` (the private store; see Step 3, never move or delete it), `agent_signup_rate_limited` / `agent_signup_busy` / `agent_signup_unavailable` / `client_ip_unavailable` (server side: relay, don't loop). An `error` with `credentials.saved: true` and a `claim_url` means the workspace exists but the file write failed: fix the file and re-run the same command, which reuses the saved key.

Fields you'll use: `env_file.path` / `variable` / `written` / `already_configured` / `gitignored`, `key.source` / `preview`, `credentials.path` / `recovery`, `workspace.organization.name` / `key_expires_at` / `claim_expires_at` / `terms_url` / `privacy_url` / `limits`, `claim_url`, `verification.state`, `next_steps[]` (`wire_init`, `verify`, `relay`, each with a `summary` and an optional `command`) and `tell_the_user`.

## Step 6: Wire explicit, metadata-only trace export

Every recipe reads the key the CLI wrote, starts the binding's **Xybrid preset with metadata-only content**, and flushes before the process stops or is backgrounded. Adapt names (`my-app`, file paths) to the project, keep its structure, and build or type-check after editing.

- Passing the key to the SDK's init (`apiKey`) is cloud authentication, not export. Add it only when the app uses cloud routing or fallback: those calls spend the workspace's small gateway quota, and the platform gateway records them under its own policy, which metadata-only SDK export doesn't change.
- Leave resource sampling off and don't add device IDs, hostnames or installation IDs to telemetry unless the user asks.
- Don't add `run*` variants or edit generated binding code (`xybrid/_bolt`, `XybridBolt.*`, `rust/frb_generated*`).

### Rust

The key: `.env` (`XYBRID_API_KEY`). If the app already loads configuration (`dotenvy`, `figment`, `config`), read it through that; otherwise this std-only helper reads the line the CLI wrote.

```rust
use std::time::Duration;

use xybrid_sdk::observability::{self, ContentPolicy, ObservabilityConfig};

/// `XYBRID_API_KEY` from the environment, else from the gitignored `.env`
/// that `xybrid connect` wrote. Never log the value.
fn xybrid_api_key() -> Option<String> {
    let from_env = std::env::var("XYBRID_API_KEY")
        .ok()
        .filter(|key| !key.trim().is_empty());
    let from_file = || {
        let text = std::fs::read_to_string(".env").ok()?;
        text.lines()
            .map(str::trim)
            .map(|line| line.strip_prefix("export ").unwrap_or(line))
            .find_map(|line| line.strip_prefix("XYBRID_API_KEY="))
            .map(|value| value.trim().trim_matches(['"', '\'']).to_string())
    };
    from_env
        .or_else(from_file)
        .filter(|key| !key.trim().is_empty())
}

/// Starts metadata-only trace export to the Xybrid console. Returns whether it runs.
fn start_xybrid_traces() -> xybrid_sdk::SdkResult<bool> {
    let Some(key) = xybrid_api_key() else {
        return Ok(false); // no key: run locally, export nothing
    };
    observability::init_observability(
        ObservabilityConfig::xybrid(&key)
            .content(ContentPolicy::metadata_only())
            .service(
                env!("CARGO_PKG_NAME"),
                Some(env!("CARGO_PKG_VERSION").into()),
            ),
    )?;
    // Cloud routing/fallback is separate and optional: xybrid_sdk::set_api_key(&key);
    Ok(true)
}

fn main() -> xybrid_sdk::SdkResult<()> {
    let exporting = start_xybrid_traces()?;
    // ... load and run models as the app already does ...
    if exporting {
        // Take the workspace handle before shutdown; read the notice after it.
        let feedback = observability::workspace_feedback();
        observability::shutdown_observability(Duration::from_secs(5))?;
        if feedback.is_some_and(|f| f.is_expired()) {
            eprintln!("Xybrid: this workspace's key expired; claim it (`xybrid auth claim`) to resume export.");
        }
    }
    Ok(())
}
```

`init_observability` returns configuration errors; `xybrid_sdk::init().observability(...).run()` only logs them. In a long-running service, also call `observability::flush_observability(Duration::from_secs(5))` after the verification run. An app that owns OpenTelemetry providers keeps them: `ObservabilityConfig::with_observer(...)` exists for that, so ask before changing it.

### Python

The key: `.env` (`XYBRID_API_KEY`). With `python-dotenv` installed, `dotenv_values(".env").get("XYBRID_API_KEY")` works too (it doesn't touch `os.environ`).

```python
import atexit
import os
from pathlib import Path
from typing import Optional

import xybrid


def xybrid_api_key(env_file: str = ".env") -> Optional[str]:
    """XYBRID_API_KEY from the environment, else the gitignored file `xybrid connect` wrote."""
    key = os.environ.get("XYBRID_API_KEY", "").strip()
    path = Path(env_file)
    if not key and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if line.startswith("XYBRID_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                break
    return key or None


def start_xybrid_traces() -> bool:
    """Metadata-only trace export to the Xybrid console. Returns whether it runs."""
    key = xybrid_api_key()
    if key is None:
        return False  # no key: run locally, export nothing
    config = xybrid.XybridObservabilityConfig.xybrid(key)
    config.set_metadata_only(True)
    config.set_service("my-app", None)
    config.init()  # consumes the config; raises xybrid.XybridError if rejected
    atexit.register(xybrid.observability_shutdown, 5000)
    # Cloud routing/fallback is separate and optional: xybrid.init(api_key=key)
    return True


if start_xybrid_traces():
    # ... run one real inference the way the app does, then:
    xybrid.observability_flush(5000)
    print("traces exported:", xybrid.observability_stats().traces.exported)
```

`xybrid.init(api_key=...)` is cloud authentication only. The legacy `xybrid.init(api_key, ingest_url=...)` form also starts metadata-only export, but new code uses `XybridObservabilityConfig`.

### Android (Kotlin)

The key: `local.properties` at the Gradle root (`xybrid.apiKey`), turned into `BuildConfig.XYBRID_API_KEY` at build time.

```kotlin
// app/build.gradle.kts
import java.util.Properties

// xybrid.apiKey from the gitignored local.properties that `xybrid connect` wrote.
val xybridApiKey: String = Properties().run {
    val file = rootProject.file("local.properties")
    if (file.exists()) file.inputStream().use { load(it) }
    getProperty("xybrid.apiKey").orEmpty()
}

android {
    defaultConfig {
        // Escaped into a Java string literal. Empty when the file has no key.
        buildConfigField(
            "String",
            "XYBRID_API_KEY",
            "\"" + xybridApiKey.replace("\\", "\\\\").replace("\"", "\\\"") + "\"",
        )
    }
    buildFeatures {
        buildConfig = true // off by default since AGP 8
    }
}
```

Groovy `build.gradle`:

```groovy
// app/build.gradle (Groovy)
def localProperties = new Properties()
def localFile = rootProject.file('local.properties')
if (localFile.exists()) localFile.withInputStream { localProperties.load(it) }
def xybridApiKey = localProperties.getProperty('xybrid.apiKey', '')

android {
    defaultConfig {
        buildConfigField 'String', 'XYBRID_API_KEY',
            '"' + xybridApiKey.replace('\\', '\\\\').replace('"', '\\"') + '"'
    }
    buildFeatures {
        buildConfig true
    }
}
```

In the `Application` subclass (registered as `android:name` in the manifest). The chainable setters and `start()` are extension functions, so import them:

```kotlin
package com.example.app

import ai.xybrid.Xybrid
import ai.xybrid.XybridError
import ai.xybrid.displayMessage
import ai.xybrid.metadataOnly
import ai.xybrid.service
import ai.xybrid.start
import android.app.Application
import android.content.ComponentCallbacks2
import android.util.Log

class App : Application() {
    private var exporting = false

    override fun onCreate() {
        super.onCreate()
        // apiKey = key would add cloud auth/fallback; it does not export.
        Xybrid.init(this)
        val key = BuildConfig.XYBRID_API_KEY.ifBlank { null } ?: return // no key: export nothing
        exporting = try {
            Xybrid.Observability.xybrid(key)
                .metadataOnly()
                .service("my-app", BuildConfig.VERSION_NAME)
                .start()
            true
        } catch (e: XybridError) {
            Log.w("Xybrid", "Trace export not started: ${e.displayMessage}")
            false
        }
    }

    // The UI left the screen: deliver queued spans off the main thread.
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (exporting && level >= ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN) {
            Thread { runCatching { Xybrid.Observability.flush(2_000) } }.start()
        }
    }
}
```

`Xybrid.Observability.flush` blocks: call it off the main thread, or use `flushAsync()` from a coroutine. To keep the key out of release builds, set the field in `buildTypes { debug { ... } }` instead of `defaultConfig`.

### iOS / macOS (Swift)

The key: `Config/Secrets.xcconfig` (create the `Config` directory first), which the CLI writes as `XYBRID_API_KEY = sk_live_…`.

1. In the tracked xcconfig of the Debug configuration (e.g. `Config/Debug.xcconfig`), add `#include? "Secrets.xcconfig"`. The `?` keeps builds without the file working. If the target has no base configuration yet, set one: `configFiles` in XcodeGen's `project.yml`, `.debug(name: "Debug", xcconfig: ...)` in Tuist, or, in a plain Xcode project, Project > Info > Configurations. Avoid hand-editing `project.pbxproj`; if you must, build afterwards to prove it.
2. In the target's Info.plist, add `<key>XybridAPIKey</key><string>$(XYBRID_API_KEY)</string>`.
3. Start export once at launch:

```swift
import Foundation
import Xybrid

enum XybridSetup {
    /// The key the build copied from the gitignored Secrets.xcconfig into
    /// Info.plist (`XybridAPIKey`). Never log the value.
    static var apiKey: String? {
        let value = Bundle.main.object(forInfoDictionaryKey: "XybridAPIKey") as? String
        let key = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return key.isEmpty || key.hasPrefix("$(") ? nil : key
    }

    /// Starts metadata-only trace export to the Xybrid console. Returns whether it runs.
    @discardableResult
    static func start() -> Bool {
        // Xybrid.initialize(apiKey: apiKey) would add cloud auth/fallback; it does not export.
        Xybrid.initialize()
        guard let key = apiKey else { return false } // no key: run locally, export nothing
        do {
            try Xybrid.Observability.xybrid(apiKey: key)
                .metadataOnly()
                .service(Bundle.main.bundleIdentifier ?? "app",
                         version: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String)
                .start()
            return true
        } catch {
            print("Xybrid trace export not started")
            return false
        }
    }
}
```

```swift
@main
struct MyApp: App {
    @Environment(\.scenePhase) private var phase
    init() { XybridSetup.start() }
    var body: some Scene {
        WindowGroup { ContentView() }
            .onChange(of: phase) { newPhase in
                if newPhase == .background {
                    Task { try? await Xybrid.Observability.flushAsync(timeoutMs: 2_000) }
                }
            }
    }
}
```

UIKit apps call `XybridSetup.start()` in `application(_:didFinishLaunchingWithOptions:)` and flush in `sceneDidEnterBackground`. A SwiftPM tool without an Info.plist reads `ProcessInfo.processInfo.environment["XYBRID_API_KEY"]` and the dotenv instead.

### Flutter

The key: `.env.xybrid` (a dedicated file, because `--dart-define-from-file` hands every variable in the file to the build). Run and build with `flutter run --dart-define-from-file=.env.xybrid` (and the same flag on `flutter build`). For IDEs, add the flag to `.vscode/launch.json` (`"toolArgs": ["--dart-define-from-file=.env.xybrid"]`) or Android Studio's "Additional run args"; those hold the file name, never the key. Flutter's dotenv reader rejects `export` lines and multi-line values; the CLI writes neither.

```dart
import 'dart:async';
import 'dart:ui' show AppExitResponse;

import 'package:flutter/widgets.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

/// Compiled in from the gitignored file `xybrid connect` wrote:
/// `flutter run --dart-define-from-file=.env.xybrid`. Empty without it.
const xybridApiKey = String.fromEnvironment('XYBRID_API_KEY');

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // apiKey: xybridApiKey would add cloud auth/fallback; it does not export.
  await Xybrid.init();
  if (xybridApiKey.isNotEmpty) {
    try {
      XybridObservability.init(const ObservabilityConfig.xybrid(
        apiKey: xybridApiKey,
        metadataOnly: true,
        serviceName: 'my_app',
      ));
    } catch (error) {
      debugPrint('Xybrid trace export not started: $error');
    }
  }
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  // Deliver queued spans when the app is backgrounded or closed.
  late final AppLifecycleListener _lifecycle = AppLifecycleListener(
    onHide: () => unawaited(XybridObservability.flush(timeoutMs: 2000)),
    onExitRequested: () async {
      await XybridObservability.shutdown(timeoutMs: 5000);
      return AppExitResponse.exit;
    },
  );

  @override
  void dispose() {
    _lifecycle.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => const SizedBox.shrink();
}
```

### React Native / Expo

Expo: `.env.local` with `EXPO_PUBLIC_XYBRID_API_KEY`. Expo inlines `process.env.EXPO_PUBLIC_*` into the JavaScript bundle at build time; restart Metro with `npx expo start --clear` after the file changes. EAS cloud builds don't upload gitignored files, so they build without the key (no export) unless the user adds it as an EAS environment variable. Bare React Native: read `XYBRID_API_KEY` through the loader the project already uses (`react-native-config`: `Config.XYBRID_API_KEY`; `react-native-dotenv`: `import { XYBRID_API_KEY } from '@env'`). If it has none, don't add one silently: tell the user that export needs an env loader.

```ts
import { AppState } from 'react-native';
import { Observability, ObservabilityConfig, Xybrid } from '@xybrid/react-native';

// Expo inlines EXPO_PUBLIC_* from the gitignored .env.local at bundle time.
// Reference it literally (no destructuring) or Expo cannot inline it.
const XYBRID_API_KEY = process.env.EXPO_PUBLIC_XYBRID_API_KEY ?? '';

/** Metadata-only trace export to the Xybrid console. Resolves whether it runs. */
export async function startXybridTraces(): Promise<boolean> {
  // { apiKey: XYBRID_API_KEY } would add cloud auth/fallback; it does not export.
  await Xybrid.initialize();
  if (!XYBRID_API_KEY) return false; // no key: run locally, export nothing
  try {
    await ObservabilityConfig.xybrid(XYBRID_API_KEY)
      .setMetadataOnly(true)
      .setService('my-app')
      .init();
  } catch {
    console.warn('Xybrid trace export not started');
    return false;
  }
  // Deliver queued spans before the OS suspends the app.
  AppState.addEventListener('change', (state) => {
    if (state === 'background') {
      Observability.flush(2000).catch(() => undefined);
    }
  });
  return true;
}
```

Call `startXybridTraces()` once at startup (the root component's first effect, or before `registerRootComponent`).

### Unity

The key: `.env.local` at the Unity project root. The CLI doesn't write raw-key resources, so the Editor reads the dotenv directly and a build step copies the key into a gitignored resource for players, deleting it when the build finishes.

1. Add to the repository's `.gitignore`, with the Unity project's path from the repository root, and confirm with `git check-ignore -v`:

   ```gitignore
   /Assets/Resources/XybridApiKey.txt
   /Assets/Resources/XybridApiKey.txt.meta
   ```

2. Add these three files (adapt the folder; projects with assembly definitions need the bootstrap's assembly to reference `Xybrid.SDK.Runtime`, and the Editor script's assembly to reference the runtime one):

```csharp
// Assets/Xybrid/XybridKey.cs
using System;
using System.IO;
using UnityEngine;

/// <summary>Reads the Xybrid key without hard-coding it. Never log the value.</summary>
public static class XybridKey
{
    /// <summary>The gitignored dotenv `xybrid connect` wrote, at the Unity project root.</summary>
    public const string EnvFile = ".env.local";
    public const string Variable = "XYBRID_API_KEY";
    /// <summary>Assets/Resources/XybridApiKey.txt, generated for player builds only (gitignored).</summary>
    public const string ResourceName = "XybridApiKey";

    public static string Load()
    {
#if UNITY_EDITOR
        // Play mode reads the dotenv directly (the Editor's working directory is the project root).
        return ReadDotenv(EnvFile, Variable);
#else
        TextAsset asset = Resources.Load<TextAsset>(ResourceName);
        string key = asset != null ? asset.text.Trim() : null;
        return string.IsNullOrEmpty(key) ? null : key;
#endif
    }

    /// <summary>The value of <paramref name="name"/> in a dotenv file, or null.</summary>
    public static string ReadDotenv(string path, string name)
    {
        if (!File.Exists(path)) return null;
        foreach (string raw in File.ReadAllLines(path))
        {
            string line = raw.Trim();
            if (line.StartsWith("export ", StringComparison.Ordinal)) line = line.Substring(7).TrimStart();
            if (!line.StartsWith(name + "=", StringComparison.Ordinal)) continue;
            string value = line.Substring(name.Length + 1).Trim().Trim('"', '\'');
            return value.Length == 0 ? null : value;
        }
        return null;
    }
}
```

```csharp
// Assets/Xybrid/XybridBootstrap.cs
using System;
using UnityEngine;
using Xybrid;

/// <summary>Starts metadata-only trace export before the first scene and flushes on pause/quit.</summary>
public sealed class XybridBootstrap : MonoBehaviour
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Boot()
    {
        // apiKey: key would add cloud auth/fallback; it does not export.
        XybridClient.Initialize();
        string key = XybridKey.Load();
        if (string.IsNullOrEmpty(key)) return; // no key: run locally, export nothing
        try
        {
            XybridClient.InitializeObservability(
                ObservabilityConfig.Xybrid(key)
                    .WithMetadataOnly()
                    .WithService(Application.productName, Application.version));
        }
        catch (XybridException)
        {
            Debug.LogWarning("Xybrid trace export not started.");
            return;
        }
        var host = new GameObject("XybridBootstrap");
        host.AddComponent<XybridBootstrap>();
        DontDestroyOnLoad(host);
    }

    // Mobile OSes can kill a paused app without OnApplicationQuit.
    private void OnApplicationPause(bool paused)
    {
        if (paused) Deliver(() => XybridClient.FlushObservability(TimeSpan.FromSeconds(2)));
    }

    private void OnApplicationQuit() =>
        Deliver(() => XybridClient.ShutdownObservability(TimeSpan.FromSeconds(5)));

    private static void Deliver(Action export)
    {
        try { export(); }
        catch (XybridException) { /* best effort: never block pause or quit */ }
    }
}
```

```csharp
// Assets/Xybrid/Editor/XybridKeyBuildStep.cs
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;

/// <summary>
/// Copies the key from the gitignored dotenv into a gitignored raw-key resource
/// for the player build, and deletes the resource when the build finishes.
/// </summary>
public sealed class XybridKeyBuildStep : IPreprocessBuildWithReport, IPostprocessBuildWithReport
{
    private const string AssetPath = "Assets/Resources/" + XybridKey.ResourceName + ".txt";

    public int callbackOrder => 0;

    public void OnPreprocessBuild(BuildReport report)
    {
        string key = XybridKey.ReadDotenv(XybridKey.EnvFile, XybridKey.Variable);
        if (string.IsNullOrEmpty(key)) return; // builds without the file export nothing
        Directory.CreateDirectory("Assets/Resources");
        File.WriteAllText(AssetPath, key);
        AssetDatabase.ImportAsset(AssetPath);
    }

    public void OnPostprocessBuild(BuildReport report) => AssetDatabase.DeleteAsset(AssetPath);
}
```

A failed player build skips the post-build step and leaves `XybridApiKey.txt` behind (still gitignored); the next build replaces it.

## Step 7: Verify one real trace

1. Run one real inference **through the app's wiring** and let it flush: run the Rust or Python entry point, `flutter run` on an available device or desktop, or the mobile or Unity app on an emulator or simulator if one is available.
2. If you can't run the app here, verify the workspace with the CLI instead (it uses the saved key). Use the model the app uses, or the setup prompt's `kokoro-82m`:

   ```bash
   OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT=no_content xybrid --observability run --model kokoro-82m --input-text "hello" -o "${TMPDIR:-/tmp}/xybrid-verify.wav"
   ```

   In PowerShell, set `$env:OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT = "no_content"` first. This proves the key and the workspace's ingestion, not the app's wiring. Say so in the final message.
3. Ask the platform, retrying for about two minutes (ingestion is asynchronous):

   ```bash
   xybrid auth status --json
   ```

   - `verification.traces_received: true` (`state: "received"`): verified.
   - `state: "none"`: nothing stored yet. Check that export started, that the run flushed (or shut down) before the process ended, and that the key is the project's. Not verified.
   - `state: "unavailable"`: the platform can't tell right now. Not verified.
   - exit 7 (`workspace_expired`): relay the claim link; nothing exports until it is claimed.

   `auth status` uses `XYBRID_API_KEY` from the environment when set, so make sure it is unset or is this project's key. If `connect` reported `key.source: "env_file"` (a key it didn't save), pass that key without printing it: `XYBRID_API_KEY="$(sed -n 's/^XYBRID_API_KEY=//p' .env | head -n 1)" xybrid auth status --json`.

The SDKs' delivery counters (`xybrid.observability_stats().traces.exported`, `Xybrid.Observability.stats`, `XybridObservability.stats`, `Observability.stats()`, `XybridClient.ObservabilityStats()`) show what the exporter sent. They don't prove storage.

## Step 8: Tell the user

In chat, never anywhere else, include:

- `tell_the_user` from the connect result, with the claim link and both deadlines (unclaimed, the key stops on `workspace.key_expires_at` and the workspace is deleted on `workspace.claim_expires_at`; claiming any time before the deletion date keeps the workspace and its history and restores the key), and the terms and privacy links when present.
- What you wired (metadata-only Xybrid trace export, in which file and init call, and whether cloud authentication was added), and the exact verification result: verified through the app, verified only through the CLI, or not verified (`none`/`unavailable`), and why.
- For mobile, web, Expo and Unity builds: the key ships inside the app and can be extracted. It is a restricted client key (trace export, a small gateway quota); claiming the workspace lets the user rotate it.
- That `xybrid auth claim` shows the link again later.

In commits and PR descriptions, say only that a Xybrid key is configured in a gitignored file: no key, no link, no claim token.

## What metadata-only sends

Metadata-only omits governed content: prompts, outputs, messages, tool-call arguments and results, and prompt variables. It still sends model and provider names, operation timing, token usage, errors, service and device descriptors (OS, architecture, chip, RAM; never hostnames), tool definitions (schemas), and any attributes the app adds itself. It is not blanket sanitization: never put secrets or personal data into model IDs, attribute values, error messages or tool schemas.

The Xybrid preset exports **traces only**. `record_feedback` / `recordFeedback` / `RecordFeedback` and the evaluation helpers return `false` with it, because no logs signal is configured. Sending feedback to Xybrid's `/v1/logs` needs its own deliberate OTLP logs configuration; never point a generic Collector preset (`OtlpConfig::collector(...)`, `ObservabilityConfig.otlp(...)`) at Xybrid's ingest as an all-signal setup.

## No-CLI path (pip, npm or pub-only environments)

When no CLI with `connect` can be installed, speak the same contract directly. The rules don't change: pinned hosts only, the key goes only into an untracked and gitignored project file, the claim token goes only into the private store and the user's chat, and nothing prints the key.

The reference implementation is [`scripts/xybrid_agent_connect.py`](scripts/xybrid_agent_connect.py) next to this file (Python 3.8+, standard library only). It prints the CLI's JSON shape and exit codes, and saves its recovery state in the CLI's own store, so a later `xybrid auth claim` or `xybrid auth status` finds it. Use the copy installed with this skill, or download it to a directory outside the repository and check its SHA-256 before running it:

```bash
curl -sSL -o "${TMPDIR:-/tmp}/xybrid_agent_connect.py" https://raw.githubusercontent.com/xybrid-ai/xybrid/master/agents/skills/xybrid-connect/scripts/xybrid_agent_connect.py
# sha256: 1f4365c9dbc1b6fe7c27e0db11cedf3f93f4a55d958d3fb83c284be3bdc0608e
python3 -I "${TMPDIR:-/tmp}/xybrid_agent_connect.py" connect --write-env .env --agent claude-code --binding python
python3 -I "${TMPDIR:-/tmp}/xybrid_agent_connect.py" status   # Step 7, instead of `xybrid auth status --json`
python3 -I "${TMPDIR:-/tmp}/xybrid_agent_connect.py" claim    # the private claim link again
```

Its `status` values, exit codes and fields match Steps 5 and 7. It provisions one workspace per repository and doesn't add projects to an existing workspace (`POST /v1/agent/projects`); the CLI does.

### The contract

| Host | API origin | Console (claim links) | OTLP traces |
|---|---|---|---|
| Production | `https://api.xybrid.dev` | `https://console.xybrid.dev` | `https://ingest.xybrid.dev/v1/traces` |
| Staging | `https://staging.api.xybrid.dev` | `https://staging.xybrid.dev` | `https://staging.ingest.xybrid.dev/v1/traces` |

Never follow redirects on these requests, and never take a host from a response.

1. **Preflight**, before any request: the CI markers and opt-outs of Step 1 end with `ci_skipped` / `opted_out`. A nonempty `XYBRID_API_KEY` or an existing value in the target file ends with `existing`, changing nothing. A target that `git ls-files` lists ends with `env_file_tracked`. Then make sure the target is ignored: when `git check-ignore -q -- <path>` fails, append a root-anchored, glob-escaped entry (`/<path>`) to the repository's `.gitignore` and check again. Open the private store (step 5) and stop if it isn't private to this account.
2. **Challenge:** `GET /v1/agent/challenge` answers `200 {"challenge", "difficulty", "expires_at", "algorithm": "sha256-leading-zero-bits"}`. The challenge is opaque, valid for five minutes and single-use; the difficulty is 18 to 28 bits (20 by default). `403 {"error": "agent_signup_closed", "reason", "console_url", "min_cli_version"}` means `signup_closed`; `503 agent_signup_unavailable` means try again later.
3. **Proof of work:** find the smallest `nonce`, a canonical decimal `u64` (ASCII digits, no sign, no leading zeros, `"0"` allowed), such that `SHA256(UTF8(challenge) || ":" || ASCII(nonce))` starts with at least `difficulty` zero bits. Bits count from the most significant bit of the first byte.
4. **Retry state first:** generate an `Idempotency-Key` of 32 to 128 characters from `[A-Za-z0-9_-]` (32 random bytes in unpadded base64url) and save it with the challenge and nonce in the private store **before** posting, so a lost answer is retried instead of creating a second workspace.
5. **Connect:** `POST /v1/agent/connect` with `Content-Type: application/json`, exactly one `Idempotency-Key` header, and a body of at most 16 KiB: `{"challenge", "nonce", "source"?, "project_name"?, "install_id"?, "client_info"?}`. `source` is the self-reported agent (normalized to 32 characters of `[a-z0-9-]`), `project_name` becomes the workspace name (48 characters), `client_info` takes only `cli_version`, `sdk_version`, `os`, `arch` and `binding` (64 characters each), and `install_id` (16 to 128 URL-safe characters) is optional; the script omits it.
   - `201` (new) or `200` with `"replayed": true` (a retry with the same key within ten minutes: same workspace and deadlines, a **new** key and claim token, the previous key revoked): `{"api_key", "key_prefix", "scope": "client", "organization": {"slug", "name"}, "project": {"slug", "name"}, "key_expires_at", "claim_expires_at", "claim_token", "limits": {"gateway_daily_requests", "max_projects"}, "terms_url"?, "privacy_url"?, "replayed"}`.
   - `400 challenge_expired | challenge_invalid | proof_of_work_invalid`: solve a fresh challenge once, same key. `409 idempotency_key_expired | idempotency_key_conflict | workspace_claimed | workspace_unavailable`: start over once with a new key and challenge. `503 agent_signup_busy` (`Retry-After: 2`): wait and retry with the same key, up to three times. `429 agent_signup_rate_limited` (`Retry-After`), `409 challenge_replayed`, `503 client_ip_unavailable`: stop and relay.
   - No answer, a 5xx or a malformed 2xx: the request may have been applied. Keep the retry state and repeat the same request (same key, challenge and nonce) within ten minutes; the platform resolves a retry before it checks the challenge.
6. **Save, then write:** check that the key matches `^[A-Za-z0-9_-]{8,256}$` and the claim token `^[A-Za-z0-9_-]{32,128}$`. Save both, with the deadlines, in the private store first. Check again that Git doesn't track the target and still ignores it, then append a comment (claim command and dates, never the token or link) and the assignment: `XYBRID_API_KEY=<key>` (dotenv), `xybrid.apiKey=<key>` (properties) or `XYBRID_API_KEY = <key>` (xcconfig, `//` comments), keeping the rest of the file byte for byte.
7. **Claim link:** `<pinned console>/claim/<claim_token>`, built locally, shown only in chat.
8. **Verify:** `GET /v1/agent/workspace` with `Authorization: Bearer <key>` answers `{"organization", "project", "scope", "provisional", "claimed", "expired", "key_expires_at", "claim_expires_at", "verification": {"state": "received" | "none" | "unavailable", "traces_received", "spans", "traces", "first_ingested_at", "last_ingested_at"}}`; `401 invalid_api_key` for an unknown, revoked or purged key. Errors everywhere are flat JSON: `{"error": "<code>", "message": "...", ...details}`.

The private store is the CLI's `~/.xybrid/credentials.json` (or `XYBRID_CREDENTIALS_FILE`): directory `0700`, file `0600`, replaced atomically; on Unix, writers hold an exclusive `flock` on `credentials.json.lock`. It is `{"version": 1, "credentials": [...], "pending": [...], "retired": [...]}`. Each credential holds `platform_origin`, `repository_root` (the canonical repository root), `api_key`, `scope`, `organization`, `project`, `key_expires_at`, `claim_expires_at`, `claim_token`, `source`, `env_file` (`{"path", "variable"}`), `claimed`, `terms_url`, `privacy_url` and `created_at`; a custom host's entry also holds `console_origin`. A pending request holds `platform_origin`, `repository_root`, `kind: "connect"`, `idempotency_key`, `challenge`, `nonce`, `project_name`, `source` and `started_at`.

### Proof-of-work test vectors

Computed with Python's `hashlib` from the rule above (and cross-checked with Node's `crypto`). `5`, `11`, `13`, `17` and `9` are not byte-aligned; at difficulty 20 the smallest nonce for the first challenge happens to give 21 bits.

| Challenge | Difficulty | Smallest nonce | Leading zero bits | `SHA256(challenge:nonce)` |
|---|---|---|---|---|
| `xybrid-pow-test-vector` | 0 | 0 | 2 | `3760f3ce0bd03c428c2a0cdd118cc02fefda19b1e7e87db527c361f48983b48a` |
| `xybrid-pow-test-vector` | 5 | 14 | 5 | `045c579e7e46f378bf77d71ff6cad5dda6fce51056d05e588a994f1a6d0c328b` |
| `xybrid-pow-test-vector` | 8 | 47 | 8 | `00a615298c9f024eab0423f486a3edc8ccb77f37e6dabc4316093e522dbb88f6` |
| `xybrid-pow-test-vector` | 11 | 307 | 11 | `0010adb245117b119688cc5c816c9da10bd79796124298e599ff47e9f086011f` |
| `xybrid-pow-test-vector` | 13 | 2900 | 13 | `0006376aed46ca79d07209421ec24ddcac5449903803ee3ab6ad44d7163b6e1a` |
| `xybrid-pow-test-vector` | 17 | 92240 | 17 | `000068272ae3c8eb6ed13e9de215c9cb04e3840eb58315e0001d99d43d419700` |
| `xybrid-pow-test-vector` | 20 | 440423 | 21 | `00000774801b51fb7c2a2eb6adc94e712510afd4be3900f3b533a3ddc96bdc3b` |
| `v1.1767225600.20.AAAAAAAAAAAAAAAAAAAAAA.vector` | 13 | 3285 | 13 | `00070302dfe9b8308c2a75cb92dab86178fe872a45bff43d4c0c3201cba94415` |
| `v1.1767225600.20.AAAAAAAAAAAAAAAAAAAAAA.vector` | 20 | 1184772 | 20 | `00000b8afcdee75b5c67559885bcdc734320fa0d18f81867cf0d2000f5647dc5` |
| `café-☃` | 9 | 3423 | 11 | `001756b5bda09576f3f141df96df07f351afb973b62b40df4b5bc3a6eca45afb` |

Two more checks: `xybrid-pow-test-vector:440422` has 0 leading zero bits, and the largest nonce, `xybrid-pow-test-vector:18446744073709551615`, has 4.

## Later

- `xybrid auth status [--json]`: key source, workspace lifecycle and trace verification. Add `--show-claim-url` for the link.
- `xybrid auth claim [--no-browser]`: the private claim link again; opens a browser only on a terminal.
- `xybrid auth logout`: forgets this project's saved credentials on this machine. It doesn't revoke the key or edit the project file.
- An unclaimed workspace's key stops at `key_expires_at`; claiming before `claim_expires_at` restores it, after which a new process (or a fresh observability config) exports again. Local inference never stops.
