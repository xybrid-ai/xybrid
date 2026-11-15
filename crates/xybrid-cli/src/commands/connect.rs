//! `xybrid connect`: connect this project to a provisional Xybrid workspace.
//!
//! Built for coding agents: no browser, no account. In order:
//!
//! 1. **Preflight**, before any identity or server mutation: CI markers
//!    (`--allow-ci` overrides only these), privacy opt-outs
//!    (`XYBRID_TELEMETRY_OPTOUT`, `DO_NOT_TRACK`, `OTEL_SDK_DISABLED=true`),
//!    the host policy, a tracked `--write-env` target, keys that already
//!    exist (flag, `XYBRID_API_KEY`, a nonempty value in the file), a
//!    writable private credentials store, and the `.gitignore` entry.
//! 2. **Reuse** this project's stored key; else **add a project** to another
//!    repository's unclaimed workspace through its claim token; else solve a
//!    challenge and **provision** a new workspace. Idempotency state is saved
//!    before each POST, so a crash or a lost response retries the same
//!    request instead of creating a second workspace.
//! 3. **Save** the credential privately, then write the key and a comment
//!    with the claim command and dates (never the token) to the project file,
//!    after checking again that Git neither tracks it nor stopped ignoring it.
//!
//! `connect` never starts an exporter and never prints the raw key. Its JSON
//! and exit codes are documented in [`crate::platform::report`].

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use clap::Args;
use uuid::Uuid;

use crate::platform::client::{
    safe_link, safe_name, ApiFailure, ClientInfo, ConnectBody, PlatformClient, Provision,
    WorkspaceStatus,
};
use crate::platform::credentials::{
    Credential, CredentialsFile, EnvFileRecord, NamedRef, PendingKind, PendingRequest, Secret,
    Store, StoreLock, CREDENTIALS_FILE_ENV,
};
use crate::platform::env_file::{self, EnvFormat, Lookup};
use crate::platform::git::{self, Repository};
use crate::platform::host::{self, PlatformHost, CONSOLE_URL_ENV};
use crate::platform::notices::human_date;
use crate::platform::report::{
    ensure_period, CredentialsReport, EnvFileReport, KeyReport, NextStep, PlatformReport, Report,
    Status, VerificationReport, WorkspaceReport,
};
use crate::platform::resolve::{KeySource, ResolvedKey};
use crate::platform::{ci_marker, install_id, key_preview, opt_out, pow, shell, CliError};

/// How long to wait for another `xybrid connect` holding the store lock.
const STORE_LOCK_TIMEOUT: Duration = Duration::from_secs(90);
/// The platform honors an `Idempotency-Key` retry for ten minutes after the
/// first attempt; stop a minute early to allow for clock skew.
const RETRY_WINDOW_SECS: i64 = 9 * 60;
/// Retries of `503 agent_signup_busy` (which sends `Retry-After: 2`).
const MAX_BUSY_RETRIES: u32 = 3;

/// Flags of `xybrid connect` (and the hidden `xybrid auth signup`).
#[derive(Args, Debug, Clone, Default)]
pub(crate) struct ConnectArgs {
    /// Self-reported agent name, e.g. claude-code (detected from CLAUDECODE=1 when possible)
    #[arg(long, value_name = "NAME")]
    pub agent: Option<String>,

    /// Write the key to this untracked project file (.env, .env.local,
    /// local.properties, Secrets.xcconfig, ...) and gitignore it
    #[arg(long, value_name = "PATH")]
    pub write_env: Option<PathBuf>,

    /// Variable to set in --write-env (default XYBRID_API_KEY, or xybrid.apiKey
    /// for .properties; e.g. EXPO_PUBLIC_XYBRID_API_KEY for Expo)
    #[arg(long, value_name = "VAR", requires = "write_env")]
    pub env_name: Option<String>,

    /// Name for a new workspace or project (default: the repository's directory name)
    #[arg(long, value_name = "NAME")]
    pub project_name: Option<String>,

    /// Create a new workspace even if this project has saved credentials
    /// (never replaces a key already in the project file)
    #[arg(long)]
    pub new: bool,

    /// Connect even when a CI environment is detected (privacy opt-outs still apply)
    #[arg(long)]
    pub allow_ci: bool,

    /// Allow a --platform-url other than the Xybrid production or staging API
    /// (HTTPS required unless localhost; XYBRID_CONSOLE_URL sets its console)
    #[arg(long)]
    pub allow_custom_host: bool,

    /// Print one JSON object (schema_version, status, tell_the_user, ...)
    #[arg(long)]
    pub json: bool,
}

/// What `connect` and `auth` take from the global flags.
pub(crate) struct PlatformArgs<'a> {
    pub platform_url: &'a str,
    /// `--api-key` or `XYBRID_API_KEY`.
    pub explicit_key: Option<ResolvedKey>,
    pub quiet: bool,
}

/// Run `xybrid connect` and return the exit code.
pub(crate) fn run(ctx: &PlatformArgs<'_>, args: &ConnectArgs) -> i32 {
    let mut report = Report::new("connect");
    if let Err(err) = connect(ctx, args, &mut report) {
        report.fail(&err);
    }
    report.emit(args.json, ctx.quiet)
}

/// Parse `--platform-url`, apply the custom-host policy and pick a custom
/// host's console from `XYBRID_CONSOLE_URL`.
pub(crate) fn platform_host(raw: &str, allow_custom_host: bool) -> Result<PlatformHost, CliError> {
    let host = PlatformHost::parse(raw)?;
    host.require_allowed(allow_custom_host)?;
    host.with_custom_console(std::env::var(CONSOLE_URL_ENV).ok().as_deref())
}

fn connect(
    ctx: &PlatformArgs<'_>,
    args: &ConnectArgs,
    report: &mut Report,
) -> Result<(), CliError> {
    if !args.allow_ci {
        if let Some(marker) = ci_marker() {
            report.status = Status::CiSkipped;
            report.tell_the_user = format!(
                "Skipped connecting to the Xybrid console because this looks like a CI run \
                 ({marker} is set). On-device inference works without a key. Run `xybrid \
                 connect` from your own terminal to get one."
            );
            return Ok(());
        }
    }
    if let Some(opt_out) = opt_out() {
        report.status = Status::OptedOut;
        report.tell_the_user = format!(
            "Skipped connecting to the Xybrid console because {opt_out} opts out of \
             telemetry. On-device inference works without a key."
        );
        return Ok(());
    }

    let host = platform_host(ctx.platform_url, args.allow_custom_host)?;
    report.platform = Some(PlatformReport::from_host(&host));
    let cwd = std::env::current_dir().map_err(|err| {
        CliError::new(
            "project_root_unavailable",
            format!("Could not read the current directory: {err}"),
        )
    })?;
    let root = git::project_root(&cwd)?;
    let repo = git::repository_root(&cwd).map(Repository::at);
    let target = args
        .write_env
        .as_deref()
        .map(|raw| EnvTarget::prepare(&cwd, repo.as_ref(), raw, args.env_name.as_deref()))
        .transpose()?;

    if let (Some(target), Some(repo)) = (&target, &repo) {
        let relative = target.relative.as_deref().unwrap_or_default();
        if repo.is_tracked(relative)? {
            let suggestions = target.suggestions(repo);
            report.status = Status::EnvFileTracked;
            report.env_file = Some(target.report(Some(true), false, false, suggestions.clone()));
            // `suggestions` are repository-relative, like `env_file.path`;
            // the command names the file the way `--write-env` did.
            let alternative = suggestions
                .first()
                .map(|s| {
                    let name = s.rsplit('/').next().unwrap_or(s);
                    let path = args
                        .write_env
                        .as_deref()
                        .and_then(Path::parent)
                        .map_or_else(|| PathBuf::from(name), |dir| dir.join(name));
                    match connect_command(&host, args, Some(&path), true) {
                        Ok(command) => format!(" such as {s} ({command})"),
                        Err(unquotable) => format!(
                            " such as {s} (pass it to --write-env; no command is shown because {})",
                            unquotable.reason()
                        ),
                    }
                })
                .unwrap_or_default();
            report.tell_the_user = format!(
                "Didn't connect: {} is tracked by Git, so a key written there would be \
                 committed. Use an untracked file{alternative}. On-device inference works \
                 without a key.",
                target.display
            );
            return Ok(());
        }
    }

    if !args.new {
        if let Some(key) = &ctx.explicit_key {
            report.status = Status::Existing;
            report.key = Some(KeyReport {
                source: KeyReport::source_name(key.source),
                preview: Some(key_preview(key.key.expose())),
                scope: None,
            });
            if let Some(target) = &target {
                let ignored = is_ignored(repo.as_ref(), target)?;
                report.env_file = Some(target.report(Some(false), ignored, false, Vec::new()));
            }
            report.next_steps = next_steps(&host, KeyHome::Explicit(key.source));
            report.tell_the_user = format!(
                "This environment already provides a Xybrid key ({}); `xybrid connect` changed \
                 nothing. Run `xybrid auth status` to check it.",
                key.source.describe()
            );
            return Ok(());
        }
    }

    // A key already in the project file is never replaced. If it is the key
    // saved for this project, carry on to check it with the platform.
    let file_holds_saved_key = !args.new
        && target
            .as_ref()
            .is_some_and(|t| t.lookup == Lookup::Present && holds_saved_key(&host, &root, t));
    if let Some(target) = target
        .as_ref()
        .filter(|t| t.lookup == Lookup::Present && !file_holds_saved_key)
    {
        if args.new {
            return Err(CliError::new(
                "env_file_has_key",
                format!(
                    "{} already sets {}; --new never replaces a project key. Remove that line or \
                     choose another --write-env file",
                    target.display, target.variable
                ),
            ));
        }
        let updated = match &repo {
            Some(repo) => repo.ensure_ignored(target.relative.as_deref().unwrap_or_default())?,
            None => false,
        };
        report.status = Status::Existing;
        report.key = Some(KeyReport {
            source: "env_file",
            preview: None,
            scope: None,
        });
        report.env_file = Some(target.report(Some(false), repo.is_some(), updated, Vec::new()));
        report.next_steps = next_steps(&host, KeyHome::UnsavedFile(target));
        report.tell_the_user = format!(
            "{} already sets {}; `xybrid connect` never replaces it and changed nothing{}.",
            target.display,
            target.variable,
            if updated {
                " except adding it to .gitignore"
            } else {
                ""
            }
        );
        return Ok(());
    }
    if let Some(target) = &target {
        if let Lookup::Ambiguous(why) = target.lookup {
            return Err(env_file::ambiguous(&target.display, &target.variable, why));
        }
    }

    // Recovery state must be writable before anything is created remotely.
    let store = Store::locate()?;
    guard_store_location(&store)?;
    let lock = store.lock(STORE_LOCK_TIMEOUT)?;
    let loaded = lock.load()?;
    report.credentials = Some(CredentialsReport {
        path: store.path().display().to_string(),
        saved: false,
        recovery: false,
        moved_corrupt_file: loaded
            .moved_corrupt_file
            .as_ref()
            .map(|p| p.display().to_string()),
    });
    let mut gitignore_updated = false;
    if let Some(target) = &target {
        env_file::check_writable(&target.path)?;
        if let Some(repo) = &repo {
            gitignore_updated =
                repo.ensure_ignored(target.relative.as_deref().unwrap_or_default())?;
        }
        // Reported even if provisioning fails below: .gitignore may have changed.
        report.env_file =
            Some(target.report(Some(false), repo.is_some(), gitignore_updated, Vec::new()));
    }

    let mut session = Session {
        args,
        report,
        client: PlatformClient::new(host.clone()),
        host,
        root,
        repo,
        target,
        gitignore_updated,
        store,
        lock,
        file: loaded.file,
    };
    if !args.new {
        if let Some(credential) = session
            .file
            .find(session.host.origin(), &session.root)
            .cloned()
        {
            return session.reuse(credential);
        }
        // A connect that may already have created this project's workspace
        // (crash, lost response) is retried before anything else.
        if session.fresh_pending(PendingKind::Connect).is_some() {
            return session.provision();
        }
        // Likewise a project request that may already have added this
        // project: it is retried in the workspace it was sent to, whatever
        // the store offers now (another repository may have connected a newer
        // workspace, recorded a claim or logged out meanwhile).
        let resumed = session
            .fresh_pending(PendingKind::Project)
            .and_then(|pending| pending.claim_token);
        if let Some(claim_token) = &resumed {
            let member = session
                .file
                .workspace_member(session.host.origin(), claim_token)
                .cloned();
            if session.add_project(claim_token.clone(), member.as_ref())? {
                return Ok(());
            }
        }
        let now = Utc::now();
        if let Some(workspace) = session
            .file
            .workspace_for_new_project(session.host.origin(), &session.root, now)
            .filter(|w| w.claim_token.as_ref() != resumed.as_ref())
            .cloned()
        {
            if let Some(claim_token) = workspace.claim_token.clone() {
                if session.add_project(claim_token, Some(&workspace))? {
                    return Ok(());
                }
            }
        }
    }
    session.provision()
}

/// Whether `target` assigns the key saved for this host and repository.
/// An unreadable store or file counts as no.
fn holds_saved_key(host: &PlatformHost, root: &Path, target: &EnvTarget) -> bool {
    let Some(credential) = Store::locate()
        .and_then(|store| store.read())
        .ok()
        .and_then(|file| file.find(host.origin(), root).cloned())
    else {
        return false;
    };
    matches!(
        env_file::read(&target.path),
        Ok(Some(text)) if env_file::assigns(
            target.format,
            &text,
            &target.variable,
            credential.api_key.expose(),
        )
    )
}

/// Keep a credentials store that sits in a Git working tree (any repository,
/// not only this project's) out of commits. Its directory also receives the
/// lock, the install ID, notice dates and any corrupt copy, so the whole
/// directory is ignored; a repository root, or a directory holding tracked
/// files, is refused instead.
fn guard_store_location(store: &Store) -> Result<(), CliError> {
    let Some(directory) = store.path().parent().and_then(canonical_lenient) else {
        return Ok(());
    };
    let Some(root) = directory
        .ancestors()
        .find(|dir| dir.exists())
        .and_then(git::repository_root)
    else {
        return Ok(());
    };
    let Some(relative) = git::relative_slash_path(&root, &directory) else {
        return Err(CliError::new(
            "credentials_in_repository",
            format!(
                "{} would keep the credentials file directly in the Git repository {}; point \
                 {CREDENTIALS_FILE_ENV} to a directory of its own (outside the repository, or \
                 e.g. .xybrid/credentials.json)",
                store.path().display(),
                root.display()
            ),
        ));
    };
    let repo = Repository::at(root);
    if repo.is_tracked(&relative)? {
        return Err(CliError::new(
            "credentials_tracked",
            format!(
                "The credentials directory {relative} holds files tracked by Git; point \
                 {CREDENTIALS_FILE_ENV} to a directory of its own"
            ),
        ));
    }
    repo.ensure_ignored(&relative)?;
    Ok(())
}

/// `path` with its longest existing prefix canonicalized and the rest kept,
/// so a directory `lock` has yet to create is placed correctly. `None` if a
/// missing component is not a plain name.
fn canonical_lenient(path: &Path) -> Option<PathBuf> {
    let mut missing = Vec::new();
    let mut existing = path;
    loop {
        if let Ok(canonical) = git::canonicalize(existing) {
            return Some(
                missing
                    .iter()
                    .rev()
                    .fold(canonical, |dir, part| dir.join(part)),
            );
        }
        missing.push(existing.file_name()?);
        existing = existing.parent()?;
    }
}

/// `path` is outside the repository `connect` runs in (`root`).
fn outside_repository(path: &Path, root: Option<&Path>) -> CliError {
    let root = root.map_or_else(String::new, |r| format!(" ({})", r.display()));
    CliError::new(
        "env_file_outside_repository",
        format!(
            "{} is outside this Git repository{root}, so it can't be gitignored here; choose a \
             file inside it",
            path.display()
        ),
    )
}

fn is_ignored(repo: Option<&Repository>, target: &EnvTarget) -> Result<bool, CliError> {
    match (repo, &target.relative) {
        (Some(repo), Some(relative)) => repo.is_ignored(relative),
        _ => Ok(false),
    }
}

/// A validated `--write-env` target.
struct EnvTarget {
    path: PathBuf,
    /// Repository-relative (`/`-separated) inside a repository, else the path.
    display: String,
    relative: Option<String>,
    format: EnvFormat,
    variable: String,
    lookup: Lookup,
}

impl EnvTarget {
    fn prepare(
        cwd: &Path,
        repo: Option<&Repository>,
        raw: &Path,
        env_name: Option<&str>,
    ) -> Result<Self, CliError> {
        let joined = cwd.join(raw);
        let name = joined.file_name().ok_or_else(|| {
            CliError::new(
                "invalid_env_file",
                format!("--write-env {} must name a file", raw.display()),
            )
        })?;
        let parent = joined.parent().unwrap_or(cwd);
        let parent = git::canonicalize(parent).map_err(|err| {
            CliError::new(
                "env_file_unwritable",
                format!("The directory {} is not usable: {err}", parent.display()),
            )
        })?;
        let path = parent.join(name);
        let format = EnvFormat::require_for_path(&path)?;
        let variable = env_name.unwrap_or(format.default_variable()).to_string();
        format.validate_variable(&variable)?;
        // The file's own repository decides whether it is tracked and where it
        // is ignored. Only the repository `connect` runs in is checked and
        // gitignored, so a file in any other one (a nested repository, or a
        // repository below a directory that is not one) is refused.
        let own_repository = git::repository_root(&parent);
        if own_repository.as_deref() != repo.map(Repository::root) {
            let Some(own) = own_repository else {
                return Err(outside_repository(&path, repo.map(Repository::root)));
            };
            let here = match repo {
                Some(repo) => format!(
                    "this command runs in the repository {}",
                    repo.root().display()
                ),
                None => format!("{} is not in a Git repository", cwd.display()),
            };
            return Err(CliError::new(
                "env_file_outside_repository",
                format!(
                    "{} belongs to the Git repository {}, but {here}, so it can't be checked or \
                     gitignored from here. Run `xybrid connect` from {}",
                    path.display(),
                    own.display(),
                    own.display()
                ),
            ));
        }
        let relative = match repo {
            Some(repo) => Some(
                git::relative_slash_path(repo.root(), &path)
                    .ok_or_else(|| outside_repository(&path, Some(repo.root())))?,
            ),
            None => None,
        };
        let lookup = env_file::read(&path)?.map_or(Lookup::Absent, |text| {
            env_file::lookup(format, &text, &variable)
        });
        Ok(Self {
            display: relative
                .clone()
                .unwrap_or_else(|| path.display().to_string()),
            path,
            relative,
            format,
            variable,
            lookup,
        })
    }

    /// Untracked files in the same directory that suit the format.
    fn suggestions(&self, repo: &Repository) -> Vec<String> {
        let name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let candidates: &[&str] = match self.format {
            EnvFormat::Dotenv => &[".env.local", ".env.development.local"],
            EnvFormat::Properties => &["local.properties", "xybrid.local.properties"],
            EnvFormat::Xcconfig => &["Secrets.xcconfig", "Secrets.local.xcconfig"],
        };
        let prefix = self
            .relative
            .as_deref()
            .and_then(|r| r.rsplit_once('/'))
            .map(|(dir, _)| format!("{dir}/"))
            .unwrap_or_default();
        candidates
            .iter()
            .filter(|c| **c != name)
            .map(|c| format!("{prefix}{c}"))
            .filter(|c| !repo.is_tracked(c).unwrap_or(true))
            .take(1)
            .collect()
    }

    fn report(
        &self,
        tracked: Option<bool>,
        gitignored: bool,
        gitignore_updated: bool,
        suggestions: Vec<String>,
    ) -> EnvFileReport {
        EnvFileReport {
            path: self.display.clone(),
            variable: self.variable.clone(),
            format: self.format,
            written: false,
            already_configured: self.lookup == Lookup::Present,
            tracked,
            gitignored,
            gitignore_updated,
            suggestions,
        }
    }
}

/// Why provisioning stopped.
enum Failed {
    /// Boxed: an [`ApiFailure`] carries the whole error body.
    Api(Box<ApiFailure>),
    /// A provisioning POST got no usable answer (transport failure, 5xx, a
    /// malformed 2xx): it may have been applied, and its retry state is kept
    /// so the next run repeats it instead of creating another.
    Unanswered(Box<ApiFailure>),
    Cli(CliError),
}

impl From<ApiFailure> for Failed {
    fn from(failure: ApiFailure) -> Self {
        Failed::Api(Box::new(failure))
    }
}

impl From<CliError> for Failed {
    fn from(err: CliError) -> Self {
        Failed::Cli(err)
    }
}

/// Everything `connect` holds once preflight passed, including the store
/// lock, so concurrent runs in one project serialize instead of each
/// creating a workspace.
struct Session<'a> {
    args: &'a ConnectArgs,
    report: &'a mut Report,
    client: PlatformClient,
    host: PlatformHost,
    root: PathBuf,
    /// The repository `connect` runs in (and `--write-env` belongs to).
    repo: Option<Repository>,
    target: Option<EnvTarget>,
    gitignore_updated: bool,
    store: Store,
    lock: StoreLock,
    file: CredentialsFile,
}

impl Session<'_> {
    fn save(&self) -> Result<(), CliError> {
        self.lock.save(&self.file)
    }

    /// Reuse this project's stored credential, after asking the server
    /// whether it still works. A transport failure is not evidence of
    /// revocation: the stored deadlines are used instead.
    fn reuse(mut self, credential: Credential) -> Result<(), CliError> {
        match self.client.workspace(&credential.api_key) {
            Ok(status) => {
                let credential = refresh_credential(credential, &status);
                self.file.put(credential.clone(), Utc::now());
                if let (true, Some(token)) = (status.claimed, &credential.claim_token) {
                    self.file.mark_claimed(token);
                }
                self.save()?;
                self.report.workspace = Some(workspace_from_server(&status));
                self.report.verification = Some(VerificationReport::from_server(
                    status.verification.as_ref(),
                ));
                if status.expired {
                    self.report_expired(&credential);
                    return Ok(());
                }
                self.finish_existing(credential)
            }
            Err(failure) if is_workspace_expired(&failure) => {
                self.report.workspace = Some(workspace_from_credential(&credential, "stored"));
                self.report_expired(&credential);
                Ok(())
            }
            // Only the platform's own answer proves the key dead: a 401 from
            // a proxy or load balancer says nothing about it.
            Err(failure) if is_invalid_api_key(&failure) => {
                self.file.retire(
                    self.host.origin(),
                    &self.root,
                    "rejected by the platform",
                    Utc::now(),
                );
                self.save()?;
                Err(CliError::new(
                    "stored_credential_rejected",
                    format!(
                        "The saved key for this project was rejected by {} ({}); it may have \
                         been revoked or its workspace deleted. It was set aside in {}. If the \
                         workspace can still be claimed, `xybrid auth claim` shows its saved \
                         link; otherwise run `xybrid connect --new` to create a new workspace",
                        self.host.origin(),
                        failure.describe(),
                        self.store.path().display()
                    ),
                ))
            }
            Err(failure @ ApiFailure::Redirect { .. }) => {
                Err(CliError::new("redirect_refused", failure.describe()))
            }
            Err(_) => {
                self.report.verification = Some(VerificationReport::state("unavailable"));
                self.report.workspace = Some(workspace_from_credential(&credential, "stored"));
                if credential.key_expired(Utc::now()) {
                    self.report_expired(&credential);
                    return Ok(());
                }
                self.finish_existing(credential)
            }
        }
    }

    fn finish_existing(mut self, mut credential: Credential) -> Result<(), CliError> {
        self.report.key = Some(KeyReport {
            source: "credentials",
            preview: Some(key_preview(credential.api_key.expose())),
            scope: credential.scope.clone(),
        });
        self.mark_saved(&credential);
        self.report.claim_url = claim_url_for(&self.host, &credential, Utc::now());
        if self.report.verification.is_none() {
            self.report.verification = Some(VerificationReport::not_checked());
        }
        let written = match self.write_project_file(&credential) {
            Ok(written) => written,
            Err(err) => {
                self.project_file_failed(&err, &credential, false);
                return Ok(());
            }
        };
        if written {
            credential.env_file = self.env_record();
            self.file.put(credential.clone(), Utc::now());
            self.save()?;
        }
        self.report.status = Status::Existing;
        self.report.next_steps = next_steps(&self.host, self.key_home());
        let name = org_name(&credential);
        let file_note = match (&self.target, written) {
            (Some(target), true) => format!(
                " Its key was written to {}{}.",
                target.display,
                self.ignore_note()
            ),
            (Some(target), false) => format!(" {} already had it.", target.display),
            (None, _) => String::new(),
        };
        let mut tell = if credential.is_provisional() {
            format!("This project is already connected to the provisional Xybrid workspace \u{201c}{name}\u{201d}.{file_note}")
        } else {
            format!("This project is already connected to the Xybrid workspace \u{201c}{name}\u{201d}.{file_note}")
        };
        tell.push_str(EXPORT_NOTE);
        tell.push_str(&self.claim_sentences(&credential));
        self.report.tell_the_user = tell;
        Ok(())
    }

    /// Add a project for this repository to another repository's unclaimed
    /// workspace, through its claim token. `member` is the saved credential
    /// of a repository in that workspace, when one is still saved. Returns
    /// `false` to fall through to a new workspace.
    fn add_project(
        &mut self,
        claim_token: Secret,
        member: Option<&Credential>,
    ) -> Result<bool, CliError> {
        match self.request_project(&claim_token) {
            Ok(provision) => {
                self.store_provision(provision, Some(claim_token), true)?;
                Ok(true)
            }
            Err(Failed::Cli(err)) => Err(err),
            Err(Failed::Unanswered(failure)) => {
                self.report_failure(&failure, Some(PendingKind::Project));
                Ok(true)
            }
            Err(Failed::Api(failure)) => match failure.code() {
                Some("workspace_expired") => match member {
                    Some(workspace) => {
                        self.report_other_workspace_expired(workspace);
                        Ok(true)
                    }
                    // No repository here uses it any more, so there is
                    // nothing to claim it from: start a new workspace.
                    None => Ok(false),
                },
                Some("workspace_claimed") => {
                    self.update_workspace_entries(&claim_token, |c| c.claimed = true)?;
                    Ok(false)
                }
                Some("claim_token_invalid") => {
                    self.update_workspace_entries(&claim_token, |c| c.claim_token = None)?;
                    Ok(false)
                }
                Some("project_limit_reached") => Ok(false),
                _ => {
                    self.report_failure(&failure, None);
                    Ok(true)
                }
            },
        }
    }

    fn provision(mut self) -> Result<(), CliError> {
        match self.request_connect() {
            Ok(provision) => {
                let claim_token = provision.claim_token.clone();
                self.store_provision(provision, claim_token, false)
            }
            Err(Failed::Cli(err)) => Err(err),
            Err(Failed::Api(failure)) => {
                self.report_failure(&failure, None);
                Ok(())
            }
            Err(Failed::Unanswered(failure)) => {
                self.report_failure(&failure, Some(PendingKind::Connect));
                Ok(())
            }
        }
    }

    fn update_workspace_entries(
        &mut self,
        claim_token: &Secret,
        update: impl Fn(&mut Credential),
    ) -> Result<(), CliError> {
        for credential in &mut self.file.credentials {
            if credential.claim_token.as_ref() == Some(claim_token) {
                update(credential);
            }
        }
        self.save()
    }

    /// Save new credentials (recovery first), then write the project file.
    fn store_provision(
        &mut self,
        provision: Provision,
        claim_token: Option<Secret>,
        new_project: bool,
    ) -> Result<(), CliError> {
        env_file::validate_key(provision.api_key.expose())?;
        let now = Utc::now();
        let kind = if new_project {
            PendingKind::Project
        } else {
            PendingKind::Connect
        };
        let mut credential = Credential {
            platform_origin: self.host.origin().to_string(),
            repository_root: self.root.clone(),
            api_key: provision.api_key.clone(),
            scope: provision.scope.clone(),
            organization: Some(clean_ref(&provision.organization)),
            project: Some(clean_ref(&provision.project)),
            key_expires_at: Some(provision.key_expires_at),
            claim_expires_at: Some(provision.claim_expires_at),
            claim_token: claim_token.filter(|t| host::is_claim_token(t.expose())),
            source: detect_source(self.args.agent.as_deref()),
            console_origin: self
                .host
                .is_custom()
                .then(|| self.host.console_origin().map(str::to_string))
                .flatten(),
            env_file: None,
            claimed: false,
            terms_url: safe_link(provision.terms_url.as_deref()),
            privacy_url: safe_link(provision.privacy_url.as_deref()),
            created_at: now,
        };
        self.file
            .clear_pending(self.host.origin(), &self.root, kind);
        self.file.put(credential.clone(), now);
        self.save()?;
        self.report.status = Status::Created;
        self.report.key = Some(KeyReport {
            source: "created",
            preview: Some(key_preview(credential.api_key.expose())),
            scope: credential.scope.clone(),
        });
        self.report.workspace = Some(WorkspaceReport {
            new_project_in_existing_workspace: new_project,
            replayed: provision.replayed,
            limits: provision.limits.clone(),
            ..workspace_from_credential(&credential, "provision")
        });
        self.mark_saved(&credential);
        self.report.claim_url = claim_url_for(&self.host, &credential, now);
        self.report.verification = Some(VerificationReport::not_checked());
        self.report.next_steps = next_steps(&self.host, self.key_home());

        let written = match self.write_project_file(&credential) {
            Ok(written) => written,
            Err(err) => {
                self.project_file_failed(&err, &credential, true);
                return Ok(());
            }
        };
        if written {
            credential.env_file = self.env_record();
            self.file.put(credential.clone(), now);
            self.save()?;
        }
        let name = org_name(&credential);
        let saved_to = match (&self.target, written) {
            (Some(target), true) => format!("{}{}", target.display, self.ignore_note()),
            (Some(target), false) => format!(
                "the private credentials store ({} already set {})",
                target.display, target.variable
            ),
            (None, _) => "the private credentials store (no project file was written)".into(),
        };
        let mut tell = if new_project {
            let project = credential
                .project
                .as_ref()
                .map_or("this project", |p| p.name.as_str());
            format!(
                "Added the project \u{201c}{project}\u{201d} to your provisional Xybrid workspace \
                 \u{201c}{name}\u{201d} and saved its restricted key to {saved_to}."
            )
        } else {
            format!(
                "Created a provisional Xybrid workspace \u{201c}{name}\u{201d} and saved its \
                 restricted key to {saved_to}."
            )
        };
        if self.host.is_custom() {
            tell.push_str(&format!(
                " It lives on the custom platform host {}, not the Xybrid production service.",
                self.host.origin()
            ));
        }
        tell.push_str(EXPORT_NOTE);
        tell.push_str(&self.claim_sentences(&credential));
        let links: Vec<String> = [
            credential
                .terms_url
                .as_deref()
                .map(|u| format!("Terms: {u}.")),
            credential
                .privacy_url
                .as_deref()
                .map(|u| format!("Privacy: {u}.")),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !links.is_empty() {
            tell.push(' ');
            tell.push_str(&links.join(" "));
        }
        self.report.tell_the_user = tell;
        Ok(())
    }

    /// The project file could not be written after the credential was saved.
    /// Fail without hiding what exists: the workspace, the privately saved
    /// key, its claim link, and how to finish.
    fn project_file_failed(&mut self, err: &CliError, credential: &Credential, created: bool) {
        let name = org_name(credential);
        let store = self.store.path().display().to_string();
        let done = if created {
            format!(
                "Created the provisional Xybrid workspace \u{201c}{name}\u{201d} and saved its \
                 restricted key privately in {store}"
            )
        } else {
            format!(
                "This project is connected to the Xybrid workspace \u{201c}{name}\u{201d} (its key \
                 is saved privately in {store})"
            )
        };
        let file = self
            .target
            .as_ref()
            .map_or_else(|| "the project file".to_string(), |t| t.display.clone());
        let retry = self.rerun(false);
        self.report.status = Status::Error;
        self.report.error = Some(crate::platform::report::ErrorReport {
            code: err.code.to_string(),
            message: err.message.clone(),
        });
        self.report.next_steps = Vec::new();
        self.report.tell_the_user = format!(
            "{done}, but couldn't write it to {file}: {} Fix that and run {retry}; it reuses \
             the saved key instead of creating another workspace.{}",
            ensure_period(&err.message),
            self.claim_sentences(credential)
        );
    }

    /// How to run this `connect` again from the same directory (see
    /// [`run_again`]). `repeat_new` keeps `--new`, for a rerun that must
    /// repeat a request whose answer was lost; a rerun that should reuse the
    /// credential this run saved drops it.
    fn rerun(&self, repeat_new: bool) -> String {
        let command = connect_command(
            &self.host,
            self.args,
            self.args.write_env.as_deref(),
            repeat_new,
        );
        run_again(command, self.args.new && !repeat_new)
    }

    /// Where the app finds the key once this run succeeds.
    fn key_home(&self) -> KeyHome<'_> {
        self.target.as_ref().map_or(KeyHome::Store, KeyHome::File)
    }

    fn mark_saved(&mut self, credential: &Credential) {
        if let Some(report) = &mut self.report.credentials {
            report.saved = true;
            report.recovery = credential.claim_token.is_some();
        }
    }

    fn ignore_note(&self) -> &'static str {
        match (self.repo.is_some(), self.target.is_some()) {
            (true, true) => " (gitignored)",
            (false, true) => " (not in a Git repository, so not gitignored)",
            _ => "",
        }
    }

    fn env_record(&self) -> Option<EnvFileRecord> {
        let target = self.target.as_ref()?;
        Some(EnvFileRecord {
            path: target
                .relative
                .as_ref()
                .map_or_else(|| target.path.clone(), PathBuf::from),
            variable: target.variable.clone(),
        })
    }

    /// Write the key into `--write-env` unless it gained a value meanwhile.
    ///
    /// Preflight checked that Git neither tracks nor fails to ignore the
    /// file, but platform requests and proof of work take a while: a checkout
    /// or a `.gitignore` edit may have changed that since. Both are checked
    /// again right before the key goes in, and the write is refused if either
    /// fails (the key is already saved privately).
    fn write_project_file(&mut self, credential: &Credential) -> Result<bool, CliError> {
        let Some(target) = &self.target else {
            return Ok(false);
        };
        let text = env_file::read(&target.path)?.unwrap_or_default();
        let lookup = env_file::lookup(target.format, &text, &target.variable);
        let mut report = target.report(
            Some(false),
            self.repo.is_some(),
            self.gitignore_updated,
            Vec::new(),
        );
        if lookup == Lookup::Present {
            report.already_configured = true;
            self.report.env_file = Some(report);
            return Ok(false);
        }
        if let (Some(repo), Some(relative)) = (&self.repo, target.relative.as_deref()) {
            let refusal = if repo.is_tracked(relative)? {
                report.tracked = Some(true);
                report.gitignored = false;
                Some(CliError::new(
                    "env_file_tracked",
                    format!(
                        "{} became tracked by Git while connecting, so a key written there \
                         would be committed; nothing was written to it. Untrack it or choose \
                         another --write-env file",
                        target.display
                    ),
                ))
            } else if !repo.is_ignored(relative)? {
                report.gitignored = false;
                Some(CliError::new(
                    "env_file_not_ignored",
                    format!(
                        "{} stopped being ignored by Git while connecting (its .gitignore entry \
                         changed); nothing was written to it",
                        target.display
                    ),
                ))
            } else {
                None
            };
            if let Some(err) = refusal {
                self.report.env_file = Some(report);
                return Err(err);
            }
        }
        let updated = env_file::upsert(
            target.format,
            &text,
            &target.variable,
            credential.api_key.expose(),
            &file_comment(credential),
        )?;
        env_file::write_atomic(&target.path, &updated)?;
        report.written = true;
        self.report.env_file = Some(report);
        Ok(true)
    }

    /// How to claim the workspace, and both of its deadlines: it can be claimed
    /// until the claim deadline (when it is deleted), while its key stops
    /// working at the earlier key deadline.
    fn claim_sentences(&self, credential: &Credential) -> String {
        let now = Utc::now();
        if !credential.is_provisional() {
            return String::new();
        }
        let (Some(key_deadline), Some(claim_deadline)) =
            (credential.key_expires_at, credential.claim_expires_at)
        else {
            return String::new();
        };
        let claim = match claim_url_for(&self.host, credential, now) {
            Some(url) => format!(
                " Claim it with GitHub by {}: {url} (this link is private: don't put it in \
                 commits, PRs or issues).",
                human_date(claim_deadline)
            ),
            None if credential.claim_token.is_some() => format!(
                " Claim it with GitHub by {} with `xybrid auth claim`.",
                human_date(claim_deadline)
            ),
            None => format!(
                " Its claim link was not saved on this machine, so it can't be claimed from \
                 here; the key works until {}.",
                human_date(key_deadline)
            ),
        };
        let recover = if credential.claim_token.is_some() {
            " Recover the link any time with `xybrid auth claim`."
        } else {
            ""
        };
        format!(
            "{claim} Until it is claimed, platform access stops on {} and the workspace is \
             scheduled for deletion on {}. On-device inference keeps working.{recover}",
            human_date(key_deadline),
            human_date(claim_deadline)
        )
    }

    fn report_expired(&mut self, credential: &Credential) {
        let now = Utc::now();
        self.report.key = Some(KeyReport {
            source: "credentials",
            preview: Some(key_preview(credential.api_key.expose())),
            scope: credential.scope.clone(),
        });
        self.mark_saved(credential);
        self.report.claim_url = claim_url_for(&self.host, credential, now);
        let lead = format!(
            "The provisional Xybrid workspace \u{201c}{}\u{201d} expired{}: its key no longer \
             reaches the platform.",
            org_name(credential),
            stopped_on(credential.key_expires_at)
        );
        let outcome = expired_outcome(
            &lead,
            credential.claim_expires_at,
            credential.claim_token.is_some(),
            self.report.claim_url.as_deref(),
            "`xybrid auth claim`",
            now,
        );
        outcome.apply(self.report);
    }

    /// Another repository's unclaimed workspace, which this project would
    /// have joined, expired. Nothing was saved for this project, and its key
    /// is not this project's.
    fn report_other_workspace_expired(&mut self, workspace: &Credential) {
        let now = Utc::now();
        let other = workspace.repository_root.display().to_string();
        self.report.workspace = Some(WorkspaceReport {
            project: None,
            ..workspace_from_credential(workspace, "stored")
        });
        self.report.key = None;
        self.report.claim_url = claim_url_for(&self.host, workspace, now);
        let lead = format!(
            "This project wasn't added to the provisional Xybrid workspace \u{201c}{}\u{201d} \
             that {other} uses, because the workspace expired{}: its key no longer reaches the \
             platform.",
            org_name(workspace),
            stopped_on(workspace.key_expires_at)
        );
        let outcome = expired_outcome(
            &lead,
            workspace.claim_expires_at,
            workspace.claim_token.is_some(),
            self.report.claim_url.as_deref(),
            &format!("`xybrid auth claim` in {other}"),
            now,
        );
        outcome.apply(self.report);
    }

    /// Report a platform failure. `unanswered` names the provisioning POST
    /// whose answer never arrived (see [`Failed::Unanswered`]): it may have
    /// been applied, so the report says so and how to recover it.
    fn report_failure(&mut self, failure: &ApiFailure, unanswered: Option<PendingKind>) {
        // No credential was saved, so a rerun with the same flags (`--new`
        // included) takes the same path: a kept request is repeated rather
        // than the old workspace's saved key reused.
        let rerun = self.rerun(true);
        let recover = format!(
            "Run {rerun} within the next few minutes: it repeats the same request, so it picks \
             up what was created instead of creating another."
        );
        match failure {
            ApiFailure::Offline(_) => {
                self.report.status = Status::Offline;
                self.report.tell_the_user = match unanswered {
                    Some(kind) => format!(
                        "Didn't get an answer from Xybrid, so {} {recover} On-device inference \
                         works without a key.",
                        match kind {
                            PendingKind::Connect => {
                                "a workspace may already have been created for this project."
                            }
                            PendingKind::Project => {
                                "this project may already have been added to the provisional \
                                 workspace."
                            }
                        }
                    ),
                    // Nothing that creates a workspace was sent.
                    None => format!(
                        "Couldn't reach Xybrid from this environment, so no workspace was \
                         created. On-device inference works without a key. Run {rerun} later \
                         to see traces in the console."
                    ),
                };
            }
            ApiFailure::Status { details, .. } if failure.code() == Some("agent_signup_closed") => {
                self.report.status = Status::SignupClosed;
                let upgrade = details
                    .get("min_cli_version")
                    .and_then(|v| v.as_str())
                    .filter(|min| version_less(env!("CARGO_PKG_VERSION"), min))
                    .map(|min| {
                        format!(
                            " This CLI is older than the required {min}; update it and try again."
                        )
                    })
                    .unwrap_or_default();
                let console = self
                    .host
                    .console_origin()
                    .map(|c| format!(" at {c}"))
                    .unwrap_or_default();
                self.report.tell_the_user = format!(
                    "Xybrid isn't creating workspaces for coding agents right now, so this \
                     project wasn't connected.{upgrade} On-device inference works without a \
                     key; to see traces, create a workspace and API key in the console{console}."
                );
            }
            _ => {
                let code = match failure {
                    ApiFailure::Redirect { .. } => "redirect_refused".to_string(),
                    ApiFailure::Invalid(_) => "invalid_platform_response".to_string(),
                    _ => failure
                        .code()
                        .map_or_else(|| "platform_error".to_string(), str::to_string),
                };
                self.report.status = Status::Error;
                self.report.error = Some(crate::platform::report::ErrorReport {
                    code,
                    message: failure.describe(),
                });
                self.report.tell_the_user = format!(
                    "Couldn't connect this project to the Xybrid console: {}{} On-device \
                     inference works without it.",
                    ensure_period(&failure.describe()),
                    if unanswered.is_some() {
                        format!(" The request may still have been applied. {recover}")
                    } else {
                        String::new()
                    }
                );
            }
        }
        if !matches!(self.report.status, Status::Error) {
            self.report.error = Some(crate::platform::report::ErrorReport {
                code: failure.code().unwrap_or("offline").to_string(),
                message: failure.describe(),
            });
        }
    }

    fn project_name(&self) -> Option<String> {
        let raw = self.args.project_name.clone().or_else(|| {
            self.root
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
        })?;
        let name = safe_name(&raw);
        (!name.is_empty()).then_some(name)
    }

    fn fresh_pending(&mut self, kind: PendingKind) -> Option<PendingRequest> {
        let pending = self
            .file
            .pending(self.host.origin(), &self.root, kind)
            .cloned()?;
        let age = Utc::now() - pending.started_at;
        if age.num_seconds() >= 0 && age.num_seconds() < RETRY_WINDOW_SECS {
            Some(pending)
        } else {
            self.file
                .clear_pending(self.host.origin(), &self.root, kind);
            None
        }
    }

    fn new_pending(&self, kind: PendingKind) -> PendingRequest {
        PendingRequest {
            platform_origin: self.host.origin().to_string(),
            repository_root: self.root.clone(),
            kind,
            idempotency_key: new_idempotency_key(),
            challenge: None,
            nonce: None,
            claim_token: None,
            project_name: self.project_name(),
            source: detect_source(self.args.agent.as_deref()),
            started_at: Utc::now(),
        }
    }

    /// Fetch and solve a challenge into `pending`, then persist it.
    fn solve_into(&mut self, pending: &mut PendingRequest) -> Result<(), Failed> {
        let challenge = self.client.challenge()?;
        if let Some(algorithm) = &challenge.algorithm {
            if algorithm != pow::ALGORITHM {
                return Err(CliError::new(
                    "unsupported_challenge",
                    format!("The platform asked for an unknown proof of work ({algorithm}); update the CLI"),
                )
                .into());
            }
        }
        if challenge.challenge.is_empty()
            || challenge.challenge.len() > 512
            || challenge.challenge.chars().any(char::is_control)
        {
            return Err(Failed::from(ApiFailure::Invalid(
                "malformed challenge".into(),
            )));
        }
        let nonce = pow::solve(&challenge.challenge, challenge.difficulty).ok_or_else(|| {
            CliError::new(
                "challenge_too_hard",
                format!(
                    "The platform asked for {} bits of proof of work; this CLI stops at {}",
                    challenge.difficulty,
                    pow::MAX_DIFFICULTY
                ),
            )
        })?;
        pending.challenge = Some(challenge.challenge);
        pending.nonce = Some(nonce.to_string());
        self.file.set_pending(pending.clone());
        self.save()?;
        Ok(())
    }

    /// `POST /v1/agent/connect`, retrying a saved request with its own
    /// `Idempotency-Key` when one is still inside the retry window.
    fn request_connect(&mut self) -> Result<Provision, Failed> {
        let mut pending = match self.fresh_pending(PendingKind::Connect) {
            Some(pending) => pending,
            None => {
                let mut pending = self.new_pending(PendingKind::Connect);
                self.solve_into(&mut pending)?;
                pending
            }
        };
        let install_id = install_id::get();
        let (mut refreshed, mut restarted, mut busy) = (false, false, 0);
        loop {
            let body = ConnectBody {
                source: pending.source.as_deref(),
                project_name: pending.project_name.as_deref(),
                install_id: Some(&install_id),
                client_info: ClientInfo::current(),
                challenge: pending.challenge.as_deref().unwrap_or_default(),
                nonce: pending.nonce.as_deref().unwrap_or_default(),
            };
            let failure = match self.client.connect(&pending.idempotency_key, &body) {
                Ok(provision) => return Ok(provision),
                Err(failure) => failure,
            };
            match failure.code() {
                Some("challenge_expired" | "challenge_invalid" | "proof_of_work_invalid")
                    if !refreshed =>
                {
                    refreshed = true;
                    self.solve_into(&mut pending)?;
                }
                Some(
                    "idempotency_key_expired"
                    | "idempotency_key_conflict"
                    | "workspace_claimed"
                    | "workspace_unavailable",
                ) if !restarted => {
                    restarted = true;
                    pending = self.new_pending(PendingKind::Connect);
                    self.solve_into(&mut pending)?;
                }
                Some("agent_signup_busy") if busy < MAX_BUSY_RETRIES => {
                    busy += 1;
                    std::thread::sleep(retry_delay(&failure));
                }
                _ => {
                    if failure.outcome_unknown() {
                        // Kept: the next run repeats this request.
                        return Err(Failed::Unanswered(Box::new(failure)));
                    }
                    self.file
                        .clear_pending(self.host.origin(), &self.root, PendingKind::Connect);
                    self.save()?;
                    return Err(failure.into());
                }
            }
        }
    }

    /// `POST /v1/agent/projects` with the workspace's claim token.
    fn request_project(&mut self, claim_token: &Secret) -> Result<Provision, Failed> {
        let mut pending = match self
            .fresh_pending(PendingKind::Project)
            .filter(|p| p.claim_token.as_ref() == Some(claim_token))
        {
            Some(pending) => pending,
            None => self.new_project_pending(claim_token)?,
        };
        let (mut restarted, mut busy) = (false, 0);
        loop {
            let failure = match self.client.create_project(
                &pending.idempotency_key,
                claim_token,
                pending.project_name.as_deref(),
            ) {
                Ok(provision) => return Ok(provision),
                Err(failure) => failure,
            };
            match failure.code() {
                Some("idempotency_key_expired" | "idempotency_key_conflict") if !restarted => {
                    restarted = true;
                    pending = self.new_project_pending(claim_token)?;
                }
                Some("agent_signup_busy") if busy < MAX_BUSY_RETRIES => {
                    busy += 1;
                    std::thread::sleep(retry_delay(&failure));
                }
                _ => {
                    if failure.outcome_unknown() {
                        // Kept: the next run repeats this request.
                        return Err(Failed::Unanswered(Box::new(failure)));
                    }
                    self.file
                        .clear_pending(self.host.origin(), &self.root, PendingKind::Project);
                    self.save()?;
                    return Err(failure.into());
                }
            }
        }
    }

    fn new_project_pending(&mut self, claim_token: &Secret) -> Result<PendingRequest, Failed> {
        let mut pending = self.new_pending(PendingKind::Project);
        pending.claim_token = Some(claim_token.clone());
        self.file.set_pending(pending.clone());
        self.save()?;
        Ok(pending)
    }
}

/// Appended to every success message: a key alone exports nothing.
const EXPORT_NOTE: &str = " Trace export is not configured by `xybrid connect`: the key alone \
    sends nothing until the app's observability setup (or `xybrid --observability`) enables it.";

fn retry_delay(failure: &ApiFailure) -> Duration {
    let secs = match failure {
        ApiFailure::Status { retry_after, .. } => retry_after.unwrap_or(2),
        _ => 2,
    };
    Duration::from_secs(secs.clamp(1, 5))
}

/// 256 bits of randomness as 64 URL-safe characters (the platform takes 32-128).
fn new_idempotency_key() -> Secret {
    Secret::new(format!(
        "{}{}",
        Uuid::new_v4().simple(),
        Uuid::new_v4().simple()
    ))
}

/// The self-reported agent: `--agent`, else a well-known agent environment.
pub(crate) fn detect_source(flag: Option<&str>) -> Option<String> {
    let raw = flag.map(str::to_string).or_else(|| {
        let set = |name: &str| std::env::var(name).is_ok_and(|v| !v.trim().is_empty());
        if std::env::var("CLAUDECODE").is_ok_and(|v| v.trim() == "1") {
            Some("claude-code".into())
        } else if set("CODEX_SANDBOX") {
            Some("codex".into())
        } else if std::env::var("GEMINI_CLI").is_ok_and(|v| v.trim() == "1") {
            Some("gemini-cli".into())
        } else {
            None
        }
    })?;
    let cleaned: String = raw
        .trim()
        .chars()
        .filter(|c| !c.is_control())
        .take(64)
        .collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// Whether dotted version `a` is lower than `b` (numeric components only).
fn version_less(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u64> {
        v.trim_start_matches('v')
            .split(['.', '-', '+'])
            .map_while(|part| part.parse().ok())
            .collect()
    };
    parse(a) < parse(b)
}

fn clean_ref(named: &NamedRef) -> NamedRef {
    NamedRef {
        slug: safe_name(&named.slug),
        name: safe_name(&named.name),
    }
}

fn org_name(credential: &Credential) -> String {
    credential
        .organization
        .as_ref()
        .map_or_else(|| "this project".to_string(), |o| o.name.clone())
}

/// The comment written above the key: claim command and dates, never the
/// token or link. The workspace can be claimed until the claim deadline; its
/// key stops at the earlier key deadline.
fn file_comment(credential: &Credential) -> Vec<String> {
    let name = org_name(credential);
    if !credential.is_provisional() {
        return vec![
            format!("Xybrid key for the workspace \"{name}\", written by `xybrid connect`."),
            "Never commit this file or share the key.".into(),
        ];
    }
    let date = |at: Option<DateTime<Utc>>| {
        at.map(|t| t.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "?".into())
    };
    vec![
        format!(
            "Xybrid provisional workspace \"{name}\": restricted client key written by `xybrid connect`."
        ),
        format!(
            "Claim it with GitHub by {} to keep it: run `xybrid auth claim`.",
            date(credential.claim_expires_at)
        ),
        format!(
            "Unclaimed, the key stops on {} (UTC) and the workspace is deleted on {}.",
            date(credential.key_expires_at),
            date(credential.claim_expires_at)
        ),
        "Never commit this file or share the key.".into(),
    ]
}

/// The claim link for a claimable credential, from the pinned console (or
/// the console chosen for a custom host when it was connected).
pub(crate) fn claim_url_for(
    host: &PlatformHost,
    credential: &Credential,
    now: DateTime<Utc>,
) -> Option<String> {
    if !credential.claimable(now) {
        return None;
    }
    let token = credential.claim_token.as_ref()?;
    if host.is_custom() {
        let console = credential
            .console_origin
            .as_deref()
            .or(host.console_origin())?;
        host::claim_url(console, token.expose())
    } else {
        host.claim_url(token.expose())
    }
}

/// Fold the server's lifecycle answer into a stored credential.
pub(crate) fn refresh_credential(
    mut credential: Credential,
    status: &WorkspaceStatus,
) -> Credential {
    credential.organization = Some(clean_ref(&status.organization));
    credential.project = Some(clean_ref(&status.project));
    if let Some(scope) = &status.scope {
        credential.scope = Some(scope.clone());
    }
    if status.claimed {
        credential.claimed = true;
    }
    if status.key_expires_at.is_some() {
        credential.key_expires_at = status.key_expires_at;
    }
    if status.claim_expires_at.is_some() {
        credential.claim_expires_at = status.claim_expires_at;
    }
    credential
}

pub(crate) fn workspace_from_server(status: &WorkspaceStatus) -> WorkspaceReport {
    WorkspaceReport {
        organization: Some(clean_ref(&status.organization)),
        project: Some(clean_ref(&status.project)),
        provisional: Some(status.provisional),
        claimed: Some(status.claimed),
        expired: Some(status.expired),
        key_expires_at: status.key_expires_at,
        claim_expires_at: status.claim_expires_at,
        lifecycle_source: Some("server"),
        ..WorkspaceReport::default()
    }
}

pub(crate) fn workspace_from_credential(
    credential: &Credential,
    source: &'static str,
) -> WorkspaceReport {
    WorkspaceReport {
        organization: credential.organization.clone(),
        project: credential.project.clone(),
        provisional: Some(credential.is_provisional()),
        claimed: Some(credential.claimed),
        expired: Some(credential.key_expired(Utc::now())),
        key_expires_at: credential.key_expires_at,
        claim_expires_at: credential.claim_expires_at,
        lifecycle_source: Some(source),
        terms_url: credential.terms_url.clone(),
        privacy_url: credential.privacy_url.clone(),
        ..WorkspaceReport::default()
    }
}

/// How to report an expired provisional workspace.
pub(crate) struct ExpiredOutcome {
    status: Status,
    error: Option<crate::platform::report::ErrorReport>,
    tell: String,
}

impl ExpiredOutcome {
    pub(crate) fn apply(self, report: &mut Report) {
        report.status = self.status;
        report.error = self.error;
        report.tell_the_user = self.tell;
    }

    #[cfg(test)]
    pub(crate) fn status_for_test(&self) -> Status {
        self.status
    }

    #[cfg(test)]
    pub(crate) fn tell_for_test(&self) -> &str {
        &self.tell
    }
}

/// The outcome for an expired provisional workspace, after `lead` (which
/// says what expired). While it can still be claimed: `workspace_expired`
/// (exit 7) with how to restore the key. Once the claim deadline passed,
/// nothing restores it: an error (`claim_expired`, exit 1).
pub(crate) fn expired_outcome(
    lead: &str,
    claim_deadline: Option<DateTime<Utc>>,
    has_token: bool,
    claim_url: Option<&str>,
    claim_command: &str,
    now: DateTime<Utc>,
) -> ExpiredOutcome {
    const LOCAL: &str = " On-device inference keeps working.";
    if let Some(deadline) = claim_deadline.filter(|at| now >= *at) {
        let ended = human_date(deadline);
        return ExpiredOutcome {
            status: Status::Error,
            error: Some(crate::platform::report::ErrorReport {
                code: "claim_expired".into(),
                message: format!(
                    "The workspace expired and its claim period ended on {ended}; it can no \
                     longer be claimed"
                ),
            }),
            tell: format!(
                "{lead} Its claim period ended on {ended}, so it can no longer be claimed and is \
                 scheduled for deletion. Run `xybrid connect --new` to create a new \
                 workspace.{LOCAL}"
            ),
        };
    }
    let by = claim_deadline
        .map(|at| format!(" by {}", human_date(at)))
        .unwrap_or_default();
    let restore = match (claim_url, has_token, claim_deadline) {
        (Some(url), _, _) => format!(
            " Claim it with GitHub{by} to restore the key: {url} (private link; also \
             {claim_command})."
        ),
        (None, true, _) => {
            format!(" Claim it with GitHub{by} to restore the key: run {claim_command}.")
        }
        (None, false, Some(at)) => format!(
            " It can be claimed until {} with its claim link, which is not saved on this machine.",
            human_date(at)
        ),
        (None, false, None) => " Its claim link is not saved on this machine.".into(),
    };
    ExpiredOutcome {
        status: Status::WorkspaceExpired,
        error: None,
        tell: format!(
            "{lead}{restore} To start over with a new workspace instead, run `xybrid connect \
             --new`.{LOCAL}"
        ),
    }
}

/// ` on <date>` for a known key deadline.
pub(crate) fn stopped_on(key_deadline: Option<DateTime<Utc>>) -> String {
    key_deadline
        .map(|at| format!(" on {}", human_date(at)))
        .unwrap_or_default()
}

/// The platform's own `401 invalid_api_key`: the key is unknown or revoked.
/// Any other 401 (a proxy, load balancer or auth layer in front of it) is not
/// evidence about the key.
pub(crate) fn is_invalid_api_key(failure: &ApiFailure) -> bool {
    matches!(failure, ApiFailure::Status { status: 401, .. })
        && failure.code() == Some("invalid_api_key")
}

/// `401 workspace_expired` (or the `X-Xybrid-Workspace: expired` marker).
pub(crate) fn is_workspace_expired(failure: &ApiFailure) -> bool {
    match failure {
        ApiFailure::Status {
            status,
            code,
            workspace,
            ..
        } => {
            matches!(*status, 401 | 403)
                && (code.as_deref() == Some("workspace_expired")
                    || workspace.as_deref().is_some_and(|w| w.trim() == "expired"))
        }
        _ => false,
    }
}

/// A flag whose value can't be written safely into a command for this
/// platform's shell (see [`shell`]), so no command is suggested.
#[derive(Debug, PartialEq, Eq)]
struct Unquotable(&'static str);

impl Unquotable {
    /// Why the report shows no command.
    fn reason(&self) -> String {
        format!("the {} value can't be quoted safely for this shell", self.0)
    }
}

/// ` flag value`, the value quoted as one word for this platform's shell.
fn push_flag(command: &mut String, flag: &'static str, value: &str) -> Result<(), Unquotable> {
    let words = shell::flag(flag, value).ok_or(Unquotable(flag))?;
    command.push(' ');
    command.push_str(&words);
    Ok(())
}

/// Global flags that select this host, for copy-pasteable commands.
fn host_flags(host: &PlatformHost) -> Result<String, Unquotable> {
    let mut flags = String::new();
    if host.environment() != crate::platform::host::Environment::Production {
        push_flag(&mut flags, "--platform-url", host.origin())?;
    }
    Ok(flags)
}

/// `xybrid connect` as a copy-pasteable command, in backticks, to run from
/// the directory this one ran in. It keeps the flags that decide what
/// `connect` does: the host, `--write-env` (the path as given, or
/// `write_env`), `--env-name`, `--allow-ci`, `--allow-custom-host`, and
/// `--new` when `repeat_new`. Each value is one word of this platform's
/// shell; a value that can't be written safely there (see [`shell`]) is an
/// error, so no command is shown rather than one that runs something else.
fn connect_command(
    host: &PlatformHost,
    args: &ConnectArgs,
    write_env: Option<&Path>,
    repeat_new: bool,
) -> Result<String, Unquotable> {
    let mut command = format!("`xybrid{} connect", host_flags(host)?);
    if repeat_new && args.new {
        command.push_str(" --new");
    }
    if let Some(path) = write_env {
        // A path that isn't Unicode can't be shown without changing it.
        let path = path.to_str().ok_or(Unquotable("--write-env"))?;
        push_flag(&mut command, "--write-env", path)?;
        if let Some(name) = &args.env_name {
            push_flag(&mut command, "--env-name", name)?;
        }
    }
    if args.allow_ci {
        command.push_str(" --allow-ci");
    }
    if host.is_custom() {
        command.push_str(" --allow-custom-host");
    }
    command.push('`');
    Ok(command)
}

/// `connect` to run again, for a sentence such as "Run {…} within the next
/// few minutes": `command` followed by "again", or, when a value can't be
/// written safely for this shell, the flags to repeat (all of them, except
/// `--new` when `drops_new`).
fn run_again(command: Result<String, Unquotable>, drops_new: bool) -> String {
    match command {
        Ok(command) => format!("{command} again"),
        Err(unquotable) => format!(
            "`xybrid connect` again with the same flags{} (no command is shown because {})",
            if drops_new { " except --new" } else { "" },
            unquotable.reason()
        ),
    }
}

/// Where the app finds the key, for the `wire_init` and `verify` steps.
enum KeyHome<'a> {
    /// The `--write-env` file holds the key `connect` saved.
    File(&'a EnvTarget),
    /// The file already held a key `connect` did not save.
    UnsavedFile(&'a EnvTarget),
    /// Only the private credentials store holds it.
    Store,
    /// `--api-key` or `XYBRID_API_KEY`.
    Explicit(KeySource),
}

fn next_steps(host: &PlatformHost, home: KeyHome<'_>) -> Vec<NextStep> {
    // A host that can't be written safely for this shell gets no commands.
    let flags = host_flags(host);
    let unshown = flags
        .as_ref()
        .err()
        .map(|unquotable| format!(" No command is shown because {}.", unquotable.reason()))
        .unwrap_or_default();
    let allow = if host.is_custom() {
        " --allow-custom-host"
    } else {
        ""
    };
    let traces = if host.is_custom() {
        " --traces-endpoint <your OTLP trace URL>"
    } else {
        ""
    };
    let source = match &home {
        KeyHome::File(target) | KeyHome::UnsavedFile(target) => {
            format!("read {} from {}", target.variable, target.display)
        }
        KeyHome::Store => "read the key from your app's own configuration (the CLI uses the saved \
                           credentials)"
            .to_string(),
        KeyHome::Explicit(KeySource::Env) => {
            "read XYBRID_API_KEY from the environment the app already uses".to_string()
        }
        KeyHome::Explicit(_) => "read the key from your app's own configuration".to_string(),
    };
    let verify_note = match &home {
        KeyHome::UnsavedFile(target) => format!(
            " `xybrid connect` did not save that key, so run the check with XYBRID_API_KEY set \
             to {} from {} (never print it).",
            target.variable, target.display
        ),
        KeyHome::Explicit(KeySource::Flag) => " Pass the same --api-key to the check.".to_string(),
        _ => String::new(),
    };
    vec![
        NextStep {
            kind: "wire_init",
            summary: format!(
                "Enable trace export explicitly with metadata-only content: {source} and pass \
                 it to the SDK's Xybrid observability preset (ObservabilityConfig::xybrid with a \
                 metadata-only content policy, or the binding's equivalent). The key alone \
                 exports nothing. Never hard-code or print it. Keep any observability the app \
                 already owns.{unshown}"
            ),
            command: flags.as_ref().ok().map(|flags| {
                format!(
                    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT=no_content xybrid{flags} \
                     --observability{traces} run --model <model-id> --input-text \"hello\""
                )
            }),
        },
        NextStep {
            kind: "verify",
            summary: format!(
                "After one real inference has flushed, check that a trace was stored for this \
                 project: verification.traces_received is true only then. none or unavailable \
                 is not success.{verify_note}{unshown}"
            ),
            command: flags
                .as_ref()
                .ok()
                .map(|flags| format!("xybrid{flags} auth status{allow} --json")),
        },
        NextStep {
            kind: "relay",
            summary: "Give the user tell_the_user in chat, including the claim link. Keep the \
                      link and the key out of commits, PRs, issues and logs."
                .into(),
            command: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idempotency_keys_fit_the_platform_contract() {
        let key = new_idempotency_key();
        let raw = key.expose();
        assert_eq!(raw.len(), 64);
        assert!(raw.bytes().all(|b| b.is_ascii_alphanumeric()));
        assert_ne!(new_idempotency_key(), key);
    }

    #[test]
    fn versions_compare_numerically() {
        assert!(version_less("0.9.0", "0.10.0"));
        assert!(version_less("0.18.0", "v0.18.1"));
        assert!(!version_less("0.18.0", "0.18.0"));
        assert!(!version_less("1.0.0", "0.99.0"));
        assert!(version_less("0.1.0-beta12", "0.2.0"));
    }

    /// The comment also dates each deadline by what it means: claim by the
    /// claim deadline, while the key stops at the earlier key deadline.
    #[test]
    fn file_comments_never_carry_the_claim_token() {
        let now = Utc::now();
        let token = "z".repeat(43);
        let deadline = |raw: &str| raw.parse::<DateTime<Utc>>().unwrap();
        let credential = Credential {
            platform_origin: host::PRODUCTION_API.into(),
            repository_root: "/repo".into(),
            api_key: Secret::new("sk_live_abcdefgh"),
            scope: Some("client".into()),
            organization: Some(NamedRef {
                slug: "my-app".into(),
                name: "my-app".into(),
            }),
            project: None,
            key_expires_at: Some(deadline("2026-11-21T12:00:00Z")),
            claim_expires_at: Some(deadline("2026-11-28T12:00:00Z")),
            claim_token: Some(Secret::new(token.clone())),
            source: None,
            console_origin: None,
            env_file: None,
            claimed: false,
            terms_url: None,
            privacy_url: None,
            created_at: now,
        };
        let lines = file_comment(&credential);
        let comment = lines.join("\n");
        assert!(!comment.contains(&token));
        assert!(!comment.contains("/claim/"));
        assert!(comment.contains("xybrid auth claim"));
        assert_eq!(
            lines[1],
            "Claim it with GitHub by 2026-11-28 to keep it: run `xybrid auth claim`."
        );
        assert_eq!(
            lines[2],
            "Unclaimed, the key stops on 2026-11-21 (UTC) and the workspace is deleted on \
             2026-11-28."
        );
        assert!(!comment.contains("by 2026-11-21"), "{comment}");
    }

    #[test]
    fn suggested_commands_keep_the_flags_that_decide_the_outcome() {
        let host = PlatformHost::parse("http://127.0.0.1:8080").unwrap();
        let args = ConnectArgs {
            agent: Some("claude-code".into()),
            write_env: Some(PathBuf::from("config/.env")),
            env_name: Some("EXPO_PUBLIC_XYBRID_API_KEY".into()),
            project_name: Some("demo".into()),
            new: true,
            allow_ci: true,
            allow_custom_host: true,
            json: true,
        };
        // Repeating a lost request keeps --new: without it the saved key of
        // the old workspace would be reused instead.
        assert_eq!(
            connect_command(&host, &args, args.write_env.as_deref(), true).unwrap(),
            "`xybrid --platform-url http://127.0.0.1:8080 connect --new --write-env config/.env \
             --env-name EXPO_PUBLIC_XYBRID_API_KEY --allow-ci --allow-custom-host`"
        );
        // Once a credential is saved, a rerun must reuse it, never replace it.
        assert_eq!(
            connect_command(&host, &args, args.write_env.as_deref(), false).unwrap(),
            "`xybrid --platform-url http://127.0.0.1:8080 connect --write-env config/.env \
             --env-name EXPO_PUBLIC_XYBRID_API_KEY --allow-ci --allow-custom-host`"
        );
        let production = PlatformHost::parse(host::PRODUCTION_API).unwrap();
        assert_eq!(
            connect_command(&production, &ConnectArgs::default(), None, true).unwrap(),
            "`xybrid connect`"
        );
    }

    /// `word` as this platform's shell quotes it (see `shell`).
    fn quoted(word: &str) -> String {
        if cfg!(windows) {
            format!("\"{word}\"")
        } else {
            format!("'{word}'")
        }
    }

    /// Every user-supplied value is one shell word, so a pasted command
    /// repeats the run instead of splitting a path or expanding `$HOME`.
    #[test]
    fn suggested_commands_quote_each_value_as_one_word() {
        let production = PlatformHost::parse(host::PRODUCTION_API).unwrap();
        let args = ConnectArgs::default();
        for path in ["ios/My App/Secrets.xcconfig", "My (App)/a&b.env"] {
            assert_eq!(
                connect_command(&production, &args, Some(Path::new(path)), true).unwrap(),
                format!("`xybrid connect --write-env {}`", quoted(path))
            );
        }
        #[cfg(unix)]
        assert_eq!(
            connect_command(&production, &args, Some(Path::new("it's $HOME/.env")), true).unwrap(),
            "`xybrid connect --write-env 'it'\\''s $HOME/.env'`"
        );
        // A host origin may hold shell syntax too: brackets of an IPv6
        // address, or `$(…)` in a hostname.
        for origin in ["http://[::1]:8080", "https://a$(id).example"] {
            let host = PlatformHost::parse(origin).unwrap();
            let command = connect_command(&host, &args, None, true);
            if cfg!(windows) && origin.contains('$') {
                assert_eq!(command, Err(Unquotable("--platform-url")));
            } else {
                assert_eq!(
                    command.unwrap(),
                    format!(
                        "`xybrid --platform-url {} connect --allow-custom-host`",
                        quoted(origin)
                    )
                );
            }
        }
        // A value starting with `-` stays attached to its flag.
        assert_eq!(
            connect_command(&production, &args, Some(Path::new("-x/.env")), true).unwrap(),
            format!("`xybrid connect {}`", quoted("--write-env=-x/.env"))
        );
    }

    /// A value that can't be written safely is not shown as a command: the
    /// copy names the flags to repeat instead.
    #[test]
    fn an_unquotable_value_suggests_no_command() {
        let production = PlatformHost::parse(host::PRODUCTION_API).unwrap();
        let args = ConnectArgs::default();
        for path in ["a\nb/.env", "a`b/.env", "a\u{202e}b/.env"] {
            assert_eq!(
                connect_command(&production, &args, Some(Path::new(path)), true),
                Err(Unquotable("--write-env")),
                "{path:?}"
            );
        }
        #[cfg(windows)]
        for path in ["%USERPROFILE%/.env", "$HOME/.env", "a\"b/.env"] {
            assert_eq!(
                connect_command(&production, &args, Some(Path::new(path)), true),
                Err(Unquotable("--write-env")),
                "{path:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let path = Path::new(std::ffi::OsStr::from_bytes(b"caf\xe9/.env"));
            assert_eq!(
                connect_command(&production, &args, Some(path), true),
                Err(Unquotable("--write-env"))
            );
        }
        // The copy says which flags to repeat, and that none is shown.
        assert_eq!(
            run_again(Err(Unquotable("--write-env")), false),
            "`xybrid connect` again with the same flags (no command is shown because the \
             --write-env value can't be quoted safely for this shell)"
        );
        assert!(run_again(Err(Unquotable("--write-env")), true)
            .starts_with("`xybrid connect` again with the same flags except --new ("));
        assert_eq!(
            run_again(Ok("`xybrid connect`".into()), true),
            "`xybrid connect` again"
        );
        // A host origin can hold a backtick: the next steps keep their
        // summaries, say why, and show no command.
        let host = PlatformHost::parse("https://a`b.example").unwrap();
        assert_eq!(
            connect_command(&host, &args, None, true),
            Err(Unquotable("--platform-url"))
        );
        for step in next_steps(&host, KeyHome::Store) {
            assert_eq!(step.command, None, "{}", step.kind);
            if step.kind != "relay" {
                assert!(
                    step.summary.ends_with(
                        " No command is shown because the --platform-url value can't be quoted \
                         safely for this shell."
                    ),
                    "{}",
                    step.summary
                );
            }
        }
    }

    #[test]
    fn explicit_agent_names_win_over_detection() {
        assert_eq!(detect_source(Some("cursor")).as_deref(), Some("cursor"));
        assert_eq!(detect_source(Some("  ")), None);
    }
}
