//! `xybrid auth`: inspect, recover or forget this project's Xybrid credentials.
//!
//! | Command | Does |
//! |---|---|
//! | `auth status` | Key source, server lifecycle and trace verification; stored deadlines as a labeled fallback |
//! | `auth claim` | Prints (and on a terminal opens) the private claim link built from the pinned console |
//! | `auth logout` | Removes the saved credentials for this host and repository; never revokes the key |
//! | `auth signup` | Hidden alias of `xybrid connect` |
//!
//! Like `connect`, these run before any observability setup and never export.

use std::io::IsTerminal;
use std::path::Path;
use std::time::Duration;

use chrono::Utc;
use clap::Subcommand;

use super::connect::{
    self, claim_url_for, is_workspace_expired, platform_host, refresh_credential,
    workspace_from_credential, workspace_from_server, ConnectArgs, PlatformArgs,
};
use crate::platform::client::{ApiFailure, PlatformClient};
use crate::platform::credentials::{Credential, Store};
use crate::platform::env_file::{self, EnvFormat, Lookup};
use crate::platform::git;
use crate::platform::host::PlatformHost;
use crate::platform::notices::human_date;
use crate::platform::report::{
    CredentialsReport, KeyReport, PlatformReport, Report, Status, StoredReport, VerificationReport,
};
use crate::platform::resolve::{KeySource, ResolvedKey};
use crate::platform::{ci_marker, key_preview, CliError};

const STORE_LOCK_TIMEOUT: Duration = Duration::from_secs(10);

/// Subcommands of `xybrid auth`.
#[derive(Subcommand, Debug, Clone)]
pub(crate) enum AuthCommand {
    /// Show this project's key source, workspace lifecycle and trace verification
    ///
    /// The key comes from --api-key, then XYBRID_API_KEY, then the credentials
    /// saved by `xybrid connect` for this platform host and repository. Exits 7
    /// when the workspace's key expired but it can still be claimed.
    Status {
        /// Print one JSON object
        #[arg(long)]
        json: bool,
        /// Include the private claim link (omitted by default)
        #[arg(long)]
        show_claim_url: bool,
        /// Allow a --platform-url other than the Xybrid production or staging API
        #[arg(long)]
        allow_custom_host: bool,
    },
    /// Show the private link to claim this project's workspace with GitHub
    ///
    /// Opens it in a browser when run from a terminal. The link is built from
    /// the pinned console origin and the claim token saved by `xybrid connect`.
    Claim {
        /// Print the link without opening a browser
        #[arg(long)]
        no_browser: bool,
        /// Print one JSON object
        #[arg(long)]
        json: bool,
        /// Allow a --platform-url other than the Xybrid production or staging API
        #[arg(long)]
        allow_custom_host: bool,
    },
    /// Remove this project's saved credentials from this machine
    ///
    /// Does not revoke the key on the server, and does not edit project files:
    /// the output says whether the project file still configures the key.
    Logout {
        /// Print one JSON object
        #[arg(long)]
        json: bool,
        /// Allow a --platform-url other than the Xybrid production or staging API
        #[arg(long)]
        allow_custom_host: bool,
    },
    /// Alias of `xybrid connect`
    #[command(hide = true)]
    Signup(ConnectArgs),
}

/// Run an `xybrid auth` subcommand and return the exit code.
pub(crate) fn run(ctx: &PlatformArgs<'_>, command: &AuthCommand) -> i32 {
    match command {
        AuthCommand::Status {
            json,
            show_claim_url,
            allow_custom_host,
        } => finish("auth status", *json, ctx.quiet, |report| {
            status(ctx, *show_claim_url, *allow_custom_host, report)
        }),
        AuthCommand::Claim {
            no_browser,
            json,
            allow_custom_host,
        } => finish("auth claim", *json, ctx.quiet, |report| {
            claim(ctx, *no_browser, *json, *allow_custom_host, report)
        }),
        AuthCommand::Logout {
            json,
            allow_custom_host,
        } => finish("auth logout", *json, ctx.quiet, |report| {
            logout(ctx, *allow_custom_host, report)
        }),
        AuthCommand::Signup(args) => connect::run(ctx, args),
    }
}

fn finish(
    command: &'static str,
    json: bool,
    quiet: bool,
    body: impl FnOnce(&mut Report) -> Result<(), CliError>,
) -> i32 {
    let mut report = Report::new(command);
    if let Err(err) = body(&mut report) {
        report.fail(&err);
    }
    report.emit(json, quiet)
}

/// The host, project root and (if readable) stored credential.
struct Project {
    host: PlatformHost,
    store: Option<Store>,
    root: std::path::PathBuf,
    stored: Option<Credential>,
    /// A rejected (quarantined) credential whose claim link still works.
    quarantined: Option<Credential>,
    store_warning: Option<CliError>,
}

fn project(
    ctx: &PlatformArgs<'_>,
    allow_custom_host: bool,
    report: &mut Report,
) -> Result<Project, CliError> {
    let host = platform_host(ctx.platform_url, allow_custom_host)?;
    report.platform = Some(PlatformReport::from_host(&host));
    let cwd = std::env::current_dir().map_err(|err| {
        CliError::new(
            "project_root_unavailable",
            format!("Could not read the current directory: {err}"),
        )
    })?;
    let root = git::project_root(&cwd)?;
    let (store, stored, quarantined, store_warning) = match Store::locate() {
        Ok(store) => match store.read() {
            Ok(file) => {
                let stored = file.find(host.origin(), &root).cloned();
                let quarantined = file
                    .retired_claimable(host.origin(), &root, Utc::now())
                    .cloned();
                (Some(store), stored, quarantined, None)
            }
            Err(err) => (Some(store), None, None, Some(err)),
        },
        Err(err) => (None, None, None, Some(err)),
    };
    Ok(Project {
        host,
        store,
        root,
        stored,
        quarantined,
        store_warning,
    })
}

fn credentials_report(project: &Project) -> Option<CredentialsReport> {
    let store = project.store.as_ref()?;
    Some(CredentialsReport {
        path: store.path().display().to_string(),
        saved: project.stored.is_some(),
        recovery: project
            .stored
            .as_ref()
            .is_some_and(|c| c.claim_token.is_some()),
        moved_corrupt_file: None,
    })
}

fn stored_report(credential: &Credential) -> StoredReport {
    StoredReport {
        key_expires_at: credential.key_expires_at,
        claim_expires_at: credential.claim_expires_at,
        claimed: credential.claimed,
        recovery: credential.claim_token.is_some(),
    }
}

fn status(
    ctx: &PlatformArgs<'_>,
    show_claim_url: bool,
    allow_custom_host: bool,
    report: &mut Report,
) -> Result<(), CliError> {
    let project = project(ctx, allow_custom_host, report)?;
    report.credentials = credentials_report(&project);
    report.stored = project.stored.as_ref().map(stored_report);
    let resolved = ctx.explicit_key.clone().or_else(|| {
        project.stored.as_ref().map(|credential| ResolvedKey {
            key: credential.api_key.clone(),
            source: KeySource::Credentials,
            credential: Some(credential.clone()),
        })
    });
    let Some(resolved) = resolved else {
        if let Some(err) = project.store_warning {
            return Err(err);
        }
        report.status = Status::NotConnected;
        report.error = Some(crate::platform::report::ErrorReport {
            code: "not_connected".into(),
            message: format!(
                "No Xybrid key for this project on {}",
                project.host.origin()
            ),
        });
        report.tell_the_user = "This project has no Xybrid key: none was passed with --api-key \
            or XYBRID_API_KEY, and `xybrid connect` saved none for it. Run `xybrid connect` to \
            create one. On-device inference works without it."
            .into();
        return Ok(());
    };
    report.key = Some(KeyReport {
        source: KeyReport::source_name(resolved.source),
        preview: Some(key_preview(resolved.key.expose())),
        scope: resolved.credential.as_ref().and_then(|c| c.scope.clone()),
    });
    let stored = resolved.credential.clone();
    if show_claim_url {
        report.claim_url = project
            .stored
            .as_ref()
            .and_then(|c| claim_url_for(&project.host, c, Utc::now()));
    }

    let client = PlatformClient::new(project.host.clone());
    match client.workspace(&resolved.key) {
        Ok(workspace) => {
            if let Some(key) = &mut report.key {
                key.scope = workspace.scope.clone().or(key.scope.take());
            }
            report.workspace = Some(workspace_from_server(&workspace));
            report.verification = Some(VerificationReport::from_server(
                workspace.verification.as_ref(),
            ));
            if let (Some(credential), Some(store)) = (&stored, &project.store) {
                let refreshed = refresh_credential(credential.clone(), &workspace);
                if refreshed != *credential {
                    update_stored(store, &project, |file| {
                        file.put(refreshed.clone(), Utc::now());
                    });
                }
                if show_claim_url {
                    report.claim_url = claim_url_for(&project.host, &refreshed, Utc::now());
                }
            }
            let name = workspace.organization.name.clone();
            if workspace.expired {
                report.status = Status::WorkspaceExpired;
                report.tell_the_user = expired_copy(
                    &name,
                    workspace.key_expires_at,
                    workspace.claim_expires_at,
                    stored.as_ref(),
                );
                return Ok(());
            }
            report.status = Status::Connected;
            let verification = workspace.verification.as_ref();
            let traces = match verification.map(|v| v.state.as_str()) {
                Some("received") => format!(
                    " Traces from this project have reached the console ({} spans so far).",
                    verification.and_then(|v| v.spans).unwrap_or_default()
                ),
                Some("none") => " No traces from this project have been stored yet: the key \
                    alone exports nothing, so check that observability is enabled and flushed."
                    .into(),
                _ => " The platform can't confirm traces right now (verification unavailable); \
                    that is not a confirmation."
                    .into(),
            };
            let lifecycle = if workspace.claimed {
                " It has been claimed.".to_string()
            } else if workspace.provisional {
                match (workspace.key_expires_at, workspace.claim_expires_at) {
                    (Some(key), Some(claim)) => format!(
                        " It is unclaimed: its key stops on {} and it is deleted on {} unless \
                         claimed (`xybrid auth claim`).",
                        human_date(key),
                        human_date(claim)
                    ),
                    _ => " It is unclaimed (`xybrid auth claim`).".into(),
                }
            } else {
                String::new()
            };
            report.tell_the_user = format!(
                "The Xybrid key ({}) works for the workspace \u{201c}{name}\u{201d}.{lifecycle}{traces}",
                resolved.source.describe()
            );
        }
        Err(failure) if is_workspace_expired(&failure) => {
            let fallback = stored.as_ref();
            let name = fallback
                .and_then(|c| c.organization.as_ref())
                .map_or_else(|| "this project's workspace".into(), |o| o.name.clone());
            report.status = Status::WorkspaceExpired;
            report.workspace = fallback.map(|c| workspace_from_credential(c, "stored"));
            report.verification = Some(VerificationReport::state("unavailable"));
            report.tell_the_user = expired_copy(
                &name,
                fallback.and_then(|c| c.key_expires_at),
                fallback.and_then(|c| c.claim_expires_at),
                fallback,
            );
        }
        Err(failure @ ApiFailure::Status { status: 401, .. }) => {
            if resolved.source == KeySource::Credentials {
                if let Some(store) = &project.store {
                    update_stored(store, &project, |file| {
                        file.retire(
                            project.host.origin(),
                            &project.root,
                            "rejected by the platform",
                            Utc::now(),
                        );
                    });
                }
                return Err(CliError::new(
                    "stored_credential_rejected",
                    format!(
                        "The saved key for this project was rejected ({}); it may have been \
                         revoked or its workspace deleted. It was set aside. If the workspace \
                         can still be claimed, `xybrid auth claim` shows its saved link; \
                         otherwise run `xybrid connect --new`",
                        failure.describe()
                    ),
                ));
            }
            return Err(CliError::new(
                "invalid_api_key",
                format!(
                    "The key from {} was rejected ({})",
                    resolved.source.describe(),
                    failure.describe()
                ),
            ));
        }
        Err(failure @ ApiFailure::Redirect { .. }) => {
            return Err(CliError::new("redirect_refused", failure.describe()));
        }
        Err(failure) => {
            report.workspace = stored
                .as_ref()
                .map(|c| workspace_from_credential(c, "stored"));
            report.verification = Some(VerificationReport::state("unavailable"));
            let fallback = stored
                .as_ref()
                .and_then(|c| c.key_expires_at.map(|at| (c, at)))
                .filter(|(c, _)| c.is_provisional())
                .map(|(_, at)| {
                    format!(
                        " Saved deadlines (not confirmed by the server): the key stops on {}.",
                        human_date(at)
                    )
                })
                .unwrap_or_default();
            if let ApiFailure::Offline(_) = failure {
                report.status = Status::Offline;
                report.error = Some(crate::platform::report::ErrorReport {
                    code: "offline".into(),
                    message: failure.describe(),
                });
                report.tell_the_user = format!(
                    "Couldn't reach Xybrid to check the key; that says nothing about whether it \
                     still works.{fallback}"
                );
            } else {
                let code = failure.code().unwrap_or("platform_error").to_string();
                report.status = Status::Error;
                report.error = Some(crate::platform::report::ErrorReport {
                    code,
                    message: failure.describe(),
                });
                report.tell_the_user = format!(
                    "The platform couldn't report on the key right now ({}).{fallback}",
                    failure.describe()
                );
            }
        }
    }
    Ok(())
}

fn expired_copy(
    name: &str,
    key_deadline: Option<chrono::DateTime<Utc>>,
    claim_deadline: Option<chrono::DateTime<Utc>>,
    stored: Option<&Credential>,
) -> String {
    let stopped = key_deadline
        .map(|at| format!(" on {}", human_date(at)))
        .unwrap_or_default();
    let claim = match (
        claim_deadline,
        stored.is_some_and(|c| c.claim_token.is_some()),
    ) {
        (Some(deadline), true) => format!(
            " Claim it with GitHub by {} to restore the key: run `xybrid auth claim`.",
            human_date(deadline)
        ),
        (Some(deadline), false) => format!(
            " It can be claimed until {} with its claim link, which is not saved on this machine.",
            human_date(deadline)
        ),
        _ => String::new(),
    };
    format!(
        "The provisional Xybrid workspace \u{201c}{name}\u{201d} expired{stopped}: its key no longer \
         reaches the platform.{claim} On-device inference keeps working."
    )
}

/// Apply `update` to the store under its lock. Best effort: status output
/// does not depend on it.
fn update_stored(
    store: &Store,
    _project: &Project,
    update: impl FnOnce(&mut crate::platform::credentials::CredentialsFile),
) {
    let Ok(lock) = store.lock(STORE_LOCK_TIMEOUT) else {
        return;
    };
    let Ok(loaded) = lock.load() else {
        return;
    };
    let mut file = loaded.file;
    update(&mut file);
    let _ = lock.save(&file);
}

fn claim(
    ctx: &PlatformArgs<'_>,
    no_browser: bool,
    json: bool,
    allow_custom_host: bool,
    report: &mut Report,
) -> Result<(), CliError> {
    let project = project(ctx, allow_custom_host, report)?;
    if let Some(err) = project.store_warning {
        return Err(err);
    }
    report.credentials = credentials_report(&project);
    let now = Utc::now();
    let quarantined = project.stored.is_none() && project.quarantined.is_some();
    let Some(credential) = project.stored.as_ref().or(project.quarantined.as_ref()) else {
        return Err(CliError::new(
            "claim_unavailable",
            format!(
                "No claim link is saved for this project ({}) on {}. If `xybrid connect` ran in \
                 another directory, run `xybrid auth claim` there. Otherwise the link can't be \
                 recovered: run `xybrid connect --new` for a new workspace",
                project.root.display(),
                project.host.origin()
            ),
        ));
    };
    report.stored = Some(stored_report(credential));
    report.workspace = Some(workspace_from_credential(credential, "stored"));
    report.key = Some(KeyReport {
        source: "credentials",
        preview: Some(key_preview(credential.api_key.expose())),
        scope: credential.scope.clone(),
    });
    let name = credential
        .organization
        .as_ref()
        .map_or_else(|| "this project".into(), |o| o.name.clone());
    if credential.claimed {
        report.status = Status::Claimed;
        let console = project
            .host
            .console_origin()
            .map(|c| format!(" Manage it at {c}."))
            .unwrap_or_default();
        report.tell_the_user = format!(
            "The Xybrid workspace \u{201c}{name}\u{201d} has already been claimed.{console}"
        );
        return Ok(());
    }
    if credential.claim_token.is_none() {
        return Err(CliError::new(
            "claim_unavailable",
            "This project's claim link was not saved, so it can't be recovered here. Run \
             `xybrid connect --new` for a new workspace",
        ));
    }
    if credential.claim_expires_at.is_some_and(|at| now >= at) {
        return Err(CliError::new(
            "claim_expired",
            format!(
                "The claim period for \u{201c}{name}\u{201d} ended{}; the workspace is scheduled for \
                 deletion. Run `xybrid connect --new` for a new workspace",
                credential
                    .claim_expires_at
                    .map(|at| format!(" on {}", human_date(at)))
                    .unwrap_or_default()
            ),
        ));
    }
    let Some(url) = claim_url_for(&project.host, credential, now) else {
        return Err(CliError::new(
            "claim_unavailable",
            format!(
                "No console origin is known for {}; set {} to the console of this custom host",
                project.host.origin(),
                crate::platform::host::CONSOLE_URL_ENV
            ),
        ));
    };
    report.status = Status::Claimable;
    report.claim_url = Some(url.clone());
    let interactive =
        !no_browser && !json && std::io::stdout().is_terminal() && ci_marker().is_none();
    let opened = interactive && open_browser(&url);
    report.opened_browser = Some(opened);
    let deadline = credential
        .claim_expires_at
        .map(|at| format!(" by {}", human_date(at)))
        .unwrap_or_default();
    let expired = if credential.key_expired(now) {
        " Its key has stopped working; claiming restores it."
    } else {
        ""
    };
    let set_aside = if quarantined {
        " The platform rejected this project's saved key, so it was set aside; this is the last \
         claim link saved for it, and claiming works only if the workspace still exists."
    } else {
        ""
    };
    report.tell_the_user = format!(
        "Claim the Xybrid workspace \u{201c}{name}\u{201d} with GitHub{deadline}: {url} (this link \
         is private: don't put it in commits, PRs or issues).{expired}{set_aside}"
    );
    Ok(())
}

/// Open `url` with the platform's default handler. Only called on a terminal.
fn open_browser(url: &str) -> bool {
    use std::process::{Command, Stdio};
    let mut command = if cfg!(target_os = "macos") {
        let mut c = Command::new("open");
        c.arg(url);
        c
    } else if cfg!(windows) {
        // `start` would reinterpret characters; the URL handler does not.
        let mut c = Command::new("rundll32");
        c.arg("url.dll,FileProtocolHandler").arg(url);
        c
    } else {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}

fn logout(
    ctx: &PlatformArgs<'_>,
    allow_custom_host: bool,
    report: &mut Report,
) -> Result<(), CliError> {
    let project = project(ctx, allow_custom_host, report)?;
    let store = match project.store.clone() {
        Some(store) => store,
        None => {
            return Err(project.store_warning.unwrap_or_else(|| {
                CliError::new("credentials_unavailable", "No credentials store")
            }))
        }
    };
    let lock = store.lock(STORE_LOCK_TIMEOUT)?;
    let loaded = lock.load()?;
    let mut file = loaded.file;
    let removed = file.remove(project.host.origin(), &project.root);
    lock.save(&file)?;
    drop(lock);
    report.status = Status::LoggedOut;
    report.removed = Some(removed.is_some());
    report.credentials = Some(CredentialsReport {
        path: store.path().display().to_string(),
        saved: false,
        recovery: false,
        moved_corrupt_file: loaded.moved_corrupt_file.map(|p| p.display().to_string()),
    });
    let Some(removed) = removed else {
        report.tell_the_user = format!(
            "No Xybrid credentials were saved for this project on {}; nothing changed.",
            project.host.origin()
        );
        return Ok(());
    };
    let env_note = match &removed.env_file {
        Some(record) => {
            let path = if record.path.is_absolute() {
                record.path.clone()
            } else {
                project.root.join(&record.path)
            };
            let format = EnvFormat::for_path(&path).unwrap_or(EnvFormat::Dotenv);
            let still = still_configures(&path, format, &record.variable);
            report.env_file = Some(crate::platform::report::EnvFileReport {
                path: record.path.display().to_string(),
                variable: record.variable.clone(),
                format,
                written: false,
                already_configured: still,
                tracked: None,
                gitignored: false,
                gitignore_updated: false,
                suggestions: Vec::new(),
            });
            if still {
                format!(
                    " {} still sets {}, so the app keeps using the key until you delete that line.",
                    record.path.display(),
                    record.variable
                )
            } else {
                format!(
                    " {} no longer sets {}.",
                    record.path.display(),
                    record.variable
                )
            }
        }
        None => {
            " No project file was recorded for it; check your env files for XYBRID_API_KEY.".into()
        }
    };
    let env_key = match ctx.explicit_key.as_ref().map(|k| k.source) {
        Some(KeySource::Env) => " XYBRID_API_KEY is still set in this environment.",
        _ => "",
    };
    let claim = if removed.claimable(Utc::now()) {
        " Its claim link was saved only there, so it is gone from this machine too."
    } else {
        ""
    };
    report.tell_the_user = format!(
        "Removed the saved Xybrid credentials for this project from {}. This did not revoke the \
         key on the server.{env_note}{env_key}{claim}",
        store.path().display()
    );
    Ok(())
}

fn still_configures(path: &Path, format: EnvFormat, variable: &str) -> bool {
    matches!(env_file::read(path), Ok(Some(text)) if env_file::lookup(format, &text, variable) == Lookup::Present)
}
