//! The private credentials store, `~/.xybrid/credentials.json` by default.
//!
//! Entries are keyed by canonical platform origin and repository root, so a
//! key never crosses hosts or projects. Each holds the restricted key, the
//! claim token (the recovery capability), deadlines and the self-reported
//! source. `XYBRID_CREDENTIALS_FILE` points the store elsewhere (tests,
//! sandboxes).
//!
//! On Unix the directory is created `0700` and the file is always replaced
//! by a fresh `0600` temporary file that was fsynced first, so readers see the
//! old or the new file and never a partial one. Writers serialize on an
//! advisory lock (`credentials.json.lock`). A corrupt file is moved aside,
//! never overwritten. On Windows the file lives in the user profile, which is
//! private to the user by default. Secrets are [`Secret`]s: their `Debug`
//! output is redacted and nothing here prints file contents.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::CliError;

/// Overrides the store location.
pub(crate) const CREDENTIALS_FILE_ENV: &str = "XYBRID_CREDENTIALS_FILE";
const FORMAT_VERSION: u32 = 1;
/// Retired (replaced, rejected) credentials kept for manual recovery.
const MAX_RETIRED: usize = 20;

/// A secret string: serialized as-is, redacted from `Debug`, never `Display`.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct Secret(String);

impl Secret {
    pub(crate) fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The raw value. Only for HTTP headers and the project file writer.
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

/// An organization or project as the platform names it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct NamedRef {
    pub slug: String,
    pub name: String,
}

/// The project file a credential was written to, for `auth logout`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct EnvFileRecord {
    pub path: PathBuf,
    pub variable: String,
}

/// One project's credentials on one platform host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Credential {
    pub platform_origin: String,
    pub repository_root: PathBuf,
    pub api_key: Secret,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub organization: Option<NamedRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<NamedRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_expires_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_expires_at: Option<DateTime<Utc>>,
    /// The claim capability for the whole workspace (shared by its projects).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_token: Option<Secret>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Console origin chosen for a custom host; pinned hosts recompute theirs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub console_origin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_file: Option<EnvFileRecord>,
    #[serde(default)]
    pub claimed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_url: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl Credential {
    pub(crate) fn matches(&self, origin: &str, root: &Path) -> bool {
        self.platform_origin == origin && self.repository_root == root
    }

    /// Unclaimed as far as this machine knows, with platform deadlines.
    pub(crate) fn is_provisional(&self) -> bool {
        !self.claimed && self.key_expires_at.is_some()
    }

    /// Provisional and past its key deadline.
    pub(crate) fn key_expired(&self, now: DateTime<Utc>) -> bool {
        self.is_provisional() && self.key_expires_at.is_some_and(|at| now >= at)
    }

    /// Provisional, before the claim deadline, with a stored claim token.
    pub(crate) fn claimable(&self, now: DateTime<Utc>) -> bool {
        !self.claimed
            && self.claim_token.is_some()
            && self.claim_expires_at.is_some_and(|at| now < at)
    }
}

/// Which request a [`PendingRequest`] is retrying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PendingKind {
    Connect,
    Project,
}

/// Idempotency state saved before a provisioning POST, so a crash or a lost
/// response retries with the same `Idempotency-Key` instead of creating a
/// second workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PendingRequest {
    pub platform_origin: String,
    pub repository_root: PathBuf,
    pub kind: PendingKind,
    pub idempotency_key: Secret,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    /// For [`PendingKind::Project`]: the workspace's claim token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_token: Option<Secret>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub started_at: DateTime<Utc>,
}

impl PendingRequest {
    fn matches(&self, origin: &str, root: &Path, kind: PendingKind) -> bool {
        self.platform_origin == origin && self.repository_root == root && self.kind == kind
    }
}

/// A credential moved out of use, kept for manual recovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RetiredCredential {
    pub reason: String,
    pub retired_at: DateTime<Utc>,
    pub credential: Credential,
}

/// The whole store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CredentialsFile {
    pub version: u32,
    #[serde(default)]
    pub credentials: Vec<Credential>,
    #[serde(default)]
    pub pending: Vec<PendingRequest>,
    #[serde(default)]
    pub retired: Vec<RetiredCredential>,
}

impl Default for CredentialsFile {
    fn default() -> Self {
        Self {
            version: FORMAT_VERSION,
            credentials: Vec::new(),
            pending: Vec::new(),
            retired: Vec::new(),
        }
    }
}

impl CredentialsFile {
    pub(crate) fn find(&self, origin: &str, root: &Path) -> Option<&Credential> {
        self.credentials.iter().find(|c| c.matches(origin, root))
    }

    /// Store `credential`, retiring a different key previously stored for the
    /// same host and repository.
    pub(crate) fn put(&mut self, credential: Credential, now: DateTime<Utc>) {
        if let Some(index) = self.index(&credential.platform_origin, &credential.repository_root) {
            let previous = self.credentials.remove(index);
            if previous.api_key != credential.api_key {
                self.push_retired(previous, "replaced", now);
            }
        }
        self.credentials.push(credential);
    }

    /// Move a credential out of use (for example, after the server rejected it).
    pub(crate) fn retire(
        &mut self,
        origin: &str,
        root: &Path,
        reason: &str,
        now: DateTime<Utc>,
    ) -> Option<Credential> {
        let index = self.index(origin, root)?;
        let credential = self.credentials.remove(index);
        self.push_retired(credential.clone(), reason, now);
        Some(credential)
    }

    /// Forget everything stored for this host and repository (`auth logout`).
    pub(crate) fn remove(&mut self, origin: &str, root: &Path) -> Option<Credential> {
        self.pending
            .retain(|p| !(p.platform_origin == origin && p.repository_root == root));
        self.retired.retain(|r| !r.credential.matches(origin, root));
        let index = self.index(origin, root)?;
        Some(self.credentials.remove(index))
    }

    pub(crate) fn pending(
        &self,
        origin: &str,
        root: &Path,
        kind: PendingKind,
    ) -> Option<&PendingRequest> {
        self.pending.iter().find(|p| p.matches(origin, root, kind))
    }

    pub(crate) fn set_pending(&mut self, request: PendingRequest) {
        self.clear_pending(
            &request.platform_origin,
            &request.repository_root,
            request.kind,
        );
        self.pending.push(request);
    }

    pub(crate) fn clear_pending(&mut self, origin: &str, root: &Path, kind: PendingKind) {
        self.pending.retain(|p| !p.matches(origin, root, kind));
    }

    /// The newest provisional workspace on this host, from another repository,
    /// whose claim token can add a project for `root`.
    pub(crate) fn workspace_for_new_project(
        &self,
        origin: &str,
        root: &Path,
        now: DateTime<Utc>,
    ) -> Option<&Credential> {
        self.credentials
            .iter()
            .filter(|c| c.platform_origin == origin && c.repository_root != root)
            .filter(|c| c.claimable(now))
            .max_by_key(|c| c.created_at)
    }

    fn index(&self, origin: &str, root: &Path) -> Option<usize> {
        self.credentials
            .iter()
            .position(|c| c.matches(origin, root))
    }

    fn push_retired(&mut self, credential: Credential, reason: &str, now: DateTime<Utc>) {
        self.retired.push(RetiredCredential {
            reason: reason.to_string(),
            retired_at: now,
            credential,
        });
        if self.retired.len() > MAX_RETIRED {
            let excess = self.retired.len() - MAX_RETIRED;
            self.retired.drain(..excess);
        }
    }
}

/// Where the store lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Store {
    path: PathBuf,
}

impl Store {
    /// `XYBRID_CREDENTIALS_FILE`, else `~/.xybrid/credentials.json`.
    pub(crate) fn locate() -> Result<Self, CliError> {
        if let Some(raw) = std::env::var_os(CREDENTIALS_FILE_ENV).filter(|v| !v.is_empty()) {
            let path = std::path::absolute(PathBuf::from(raw)).map_err(|err| {
                CliError::new(
                    "credentials_unavailable",
                    format!("{CREDENTIALS_FILE_ENV} is not a usable path: {err}"),
                )
            })?;
            return Ok(Self { path });
        }
        let home = dirs::home_dir().ok_or_else(|| {
            CliError::new(
                "credentials_unavailable",
                format!(
                    "No home directory for ~/.xybrid/credentials.json; set {CREDENTIALS_FILE_ENV}"
                ),
            )
        })?;
        Ok(Self::at(home.join(".xybrid").join("credentials.json")))
    }

    pub(crate) fn at(path: PathBuf) -> Self {
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Read without locking (writers replace the file atomically). A missing
    /// file is empty; a corrupt one is an error and is left untouched.
    pub(crate) fn read(&self) -> Result<CredentialsFile, CliError> {
        match std::fs::read(&self.path) {
            Ok(bytes) => parse(&bytes).map_err(|err| err.into_cli_error(&self.path)),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(CredentialsFile::default()),
            Err(err) => Err(io_error("read", &self.path, &err)),
        }
    }

    /// Take the writer lock, creating the private directory if needed. Waits
    /// up to `timeout` for another `xybrid` process to finish.
    pub(crate) fn lock(&self, timeout: Duration) -> Result<StoreLock, CliError> {
        let directory = self.directory();
        create_private_dir_all(directory).map_err(|err| io_error("create", directory, &err))?;
        ensure_private_directory(directory)?;
        let lock_path = lock_path(&self.path);
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options
            .open(&lock_path)
            .map_err(|err| io_error("open", &lock_path, &err))?;
        let deadline = Instant::now() + timeout;
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    return Err(CliError::new(
                        "credentials_busy",
                        format!(
                            "Another xybrid process is updating {}; try again",
                            self.path.display()
                        ),
                    ));
                }
                Err(std::fs::TryLockError::Error(err)) => {
                    return Err(io_error("lock", &lock_path, &err));
                }
            }
        }
        Ok(StoreLock {
            path: self.path.clone(),
            _lock: file,
        })
    }

    fn directory(&self) -> &Path {
        self.path.parent().unwrap_or_else(|| Path::new("."))
    }
}

/// Exclusive access to the store until dropped.
#[derive(Debug)]
pub(crate) struct StoreLock {
    path: PathBuf,
    _lock: File,
}

/// What [`StoreLock::load`] found.
#[derive(Debug)]
pub(crate) struct Loaded {
    pub file: CredentialsFile,
    /// Where an unreadable store was moved before starting a fresh one.
    pub moved_corrupt_file: Option<PathBuf>,
}

impl StoreLock {
    /// Load for an update. A corrupt file is renamed aside (keeping its
    /// private permissions) so its contents are never lost.
    pub(crate) fn load(&self) -> Result<Loaded, CliError> {
        let bytes = match std::fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                return Ok(Loaded {
                    file: CredentialsFile::default(),
                    moved_corrupt_file: None,
                })
            }
            Err(err) => return Err(io_error("read", &self.path, &err)),
        };
        match parse(&bytes) {
            Ok(file) => Ok(Loaded {
                file,
                moved_corrupt_file: None,
            }),
            Err(ParseError::Corrupt) => {
                let name = self
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "credentials.json".into());
                let aside = self.path.with_file_name(format!(
                    "{name}.corrupt-{}",
                    Utc::now().format("%Y%m%dT%H%M%S%.3fZ")
                ));
                std::fs::rename(&self.path, &aside)
                    .map_err(|err| io_error("move aside", &self.path, &err))?;
                Ok(Loaded {
                    file: CredentialsFile::default(),
                    moved_corrupt_file: Some(aside),
                })
            }
            Err(err) => Err(err.into_cli_error(&self.path)),
        }
    }

    /// Atomically replace the store with `file`.
    pub(crate) fn save(&self, file: &CredentialsFile) -> Result<(), CliError> {
        let directory = self.path.parent().unwrap_or_else(|| Path::new("."));
        let mut json = serde_json::to_vec_pretty(file)
            .map_err(|err| CliError::new("credentials_unwritable", err.to_string()))?;
        json.push(b'\n');
        let write = || -> io::Result<()> {
            // NamedTempFile is created 0600 (O_EXCL) on Unix.
            let mut temp = tempfile::Builder::new()
                .prefix(".credentials-")
                .suffix(".tmp")
                .tempfile_in(directory)?;
            temp.write_all(&json)?;
            temp.as_file().sync_all()?;
            temp.persist(&self.path).map_err(|err| err.error)?;
            sync_directory(directory);
            Ok(())
        };
        write().map_err(|err| io_error("write", &self.path, &err))
    }
}

/// Create `directory` (and missing parents) private to the user: `0700` on Unix.
pub(crate) fn create_private_dir_all(directory: &Path) -> io::Result<()> {
    if directory.as_os_str().is_empty() || directory.is_dir() {
        return Ok(());
    }
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(directory)
}

/// Refuse a directory other users could write to: they could swap the file.
fn ensure_private_directory(directory: &Path) -> Result<(), CliError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(directory)
            .map_err(|err| io_error("inspect", directory, &err))?
            .permissions()
            .mode();
        let shared_write = mode & 0o022 != 0;
        let sticky = mode & 0o1000 != 0;
        if shared_write && !sticky {
            return Err(CliError::new(
                "credentials_unsafe",
                format!(
                    "{} is writable by other users; keep credentials in a private directory \
                     (chmod go-w, or set {CREDENTIALS_FILE_ENV})",
                    directory.display()
                ),
            ));
        }
    }
    #[cfg(not(unix))]
    let _ = directory;
    Ok(())
}

fn lock_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_else(|| "credentials.json".into());
    name.push(".lock");
    path.with_file_name(name)
}

fn sync_directory(directory: &Path) {
    // Make the rename durable. Best effort: not every platform can open a
    // directory, and the data itself was already fsynced.
    #[cfg(unix)]
    if let Ok(dir) = File::open(directory) {
        let _ = dir.sync_all();
    }
    #[cfg(not(unix))]
    let _ = directory;
}

#[derive(Debug, PartialEq, Eq)]
enum ParseError {
    Corrupt,
    NewerVersion(u64),
}

impl ParseError {
    fn into_cli_error(self, path: &Path) -> CliError {
        match self {
            ParseError::Corrupt => CliError::new(
                "credentials_corrupt",
                format!("{} is not a valid xybrid credentials file", path.display()),
            ),
            ParseError::NewerVersion(version) => CliError::new(
                "credentials_unsupported",
                format!(
                    "{} was written by a newer xybrid (format {version}); upgrade the CLI",
                    path.display()
                ),
            ),
        }
    }
}

fn parse(bytes: &[u8]) -> Result<CredentialsFile, ParseError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| ParseError::Corrupt)?;
    match value.get("version").and_then(serde_json::Value::as_u64) {
        Some(version) if version > u64::from(FORMAT_VERSION) => {
            return Err(ParseError::NewerVersion(version))
        }
        Some(_) => {}
        None => return Err(ParseError::Corrupt),
    }
    serde_json::from_value(value).map_err(|_| ParseError::Corrupt)
}

fn io_error(action: &str, path: &Path, err: &io::Error) -> CliError {
    CliError::new(
        "credentials_unwritable",
        format!("Could not {action} {}: {err}", path.display()),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use chrono::TimeZone;

    const ORIGIN: &str = "https://api.xybrid.dev";

    /// A store path whose directory `lock` creates `0700`: a fresh temporary
    /// directory follows the umask, which may leave it group-writable.
    pub(crate) fn private_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("private").join("credentials.json")
    }

    fn at(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_800_000_000 + seconds, 0).unwrap()
    }

    fn credential(root: &str, key: &str) -> Credential {
        Credential {
            platform_origin: ORIGIN.into(),
            repository_root: PathBuf::from(root),
            api_key: Secret::new(key),
            scope: Some("client".into()),
            organization: Some(NamedRef {
                slug: "my-app".into(),
                name: "my-app".into(),
            }),
            project: Some(NamedRef {
                slug: "default".into(),
                name: "Default".into(),
            }),
            key_expires_at: Some(at(7 * 86_400)),
            claim_expires_at: Some(at(14 * 86_400)),
            claim_token: Some(Secret::new("t".repeat(43))),
            source: Some("claude-code".into()),
            console_origin: None,
            env_file: None,
            claimed: false,
            terms_url: None,
            privacy_url: None,
            created_at: at(0),
        }
    }

    #[test]
    fn debug_output_redacts_secrets() {
        let mut cred = credential("/repo", "sk_live_SECRETSECRETSECRET");
        cred.claim_token = Some(Secret::new("CLAIMTOKENCLAIMTOKENCLAIMTOKEN0123456789abc"));
        let mut file = CredentialsFile::default();
        file.put(cred, at(0));
        let debug = format!("{file:?}");
        assert!(!debug.contains("SECRETSECRET"), "{debug}");
        assert!(!debug.contains("CLAIMTOKEN"), "{debug}");
        assert!(debug.contains("Secret(<redacted>)"));
    }

    #[test]
    fn round_trips_with_private_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path().join("nested").join("credentials.json"));
        let lock = store.lock(Duration::from_secs(1)).unwrap();
        let mut file = lock.load().unwrap().file;
        file.put(credential("/repo", "sk_live_one"), at(0));
        lock.save(&file).unwrap();
        drop(lock);
        assert_eq!(store.read().unwrap(), file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(store.path()), 0o600);
            assert_eq!(mode(store.path().parent().unwrap()), 0o700);
            assert_eq!(mode(&lock_path(store.path())), 0o600);
        }
        // No stray temporary files are left next to it.
        let names: Vec<_> = std::fs::read_dir(store.path().parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().all(|n| !n.ends_with(".tmp")), "{names:?}");
    }

    #[test]
    fn entries_are_keyed_by_host_and_repository() {
        let mut file = CredentialsFile::default();
        file.put(credential("/a", "sk_live_a"), at(0));
        let mut staging = credential("/a", "sk_live_staging");
        staging.platform_origin = "https://staging.api.xybrid.dev".into();
        file.put(staging, at(0));
        file.put(credential("/b", "sk_live_b"), at(0));
        assert_eq!(file.credentials.len(), 3);
        assert_eq!(
            file.find(ORIGIN, Path::new("/a")).unwrap().api_key.expose(),
            "sk_live_a"
        );
        assert!(file.find("https://evil.example", Path::new("/a")).is_none());

        // Replacing a key retires the old one instead of losing its claim token.
        file.put(credential("/a", "sk_live_a2"), at(10));
        assert_eq!(file.credentials.len(), 3);
        assert_eq!(file.retired.len(), 1);
        assert_eq!(file.retired[0].reason, "replaced");
        assert_eq!(file.retired[0].credential.api_key.expose(), "sk_live_a");

        assert!(file.remove(ORIGIN, Path::new("/a")).is_some());
        assert!(file.retired.is_empty(), "logout forgets retired copies too");
        assert!(file.find(ORIGIN, Path::new("/a")).is_none());
    }

    #[test]
    fn another_repository_can_reuse_a_claimable_workspace() {
        let mut file = CredentialsFile::default();
        let mut claimed = credential("/claimed", "sk_live_c");
        claimed.claimed = true;
        file.put(claimed, at(0));
        assert!(file
            .workspace_for_new_project(ORIGIN, Path::new("/new"), at(1))
            .is_none());
        file.put(credential("/a", "sk_live_a"), at(0));
        let found = file
            .workspace_for_new_project(ORIGIN, Path::new("/new"), at(1))
            .unwrap();
        assert_eq!(found.api_key.expose(), "sk_live_a");
        assert!(file
            .workspace_for_new_project(ORIGIN, Path::new("/a"), at(1))
            .is_none());
        assert!(
            file.workspace_for_new_project(ORIGIN, Path::new("/new"), at(15 * 86_400))
                .is_none(),
            "past the claim deadline"
        );
    }

    #[test]
    fn concurrent_writers_never_lose_updates() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(private_path(&dir));
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for i in 0..8 {
                let (store, barrier) = (&store, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for round in 0..5 {
                        let lock = store.lock(Duration::from_secs(30)).unwrap();
                        let mut file = lock.load().unwrap().file;
                        assert!(lock.load().unwrap().moved_corrupt_file.is_none());
                        file.put(
                            credential(&format!("/repo-{i}"), &format!("sk_live_{i}_{round}")),
                            at(round),
                        );
                        lock.save(&file).unwrap();
                    }
                });
            }
        });
        let file = store.read().unwrap();
        assert_eq!(file.credentials.len(), 8);
        for i in 0..8 {
            let root = PathBuf::from(format!("/repo-{i}"));
            assert_eq!(
                file.find(ORIGIN, &root).unwrap().api_key.expose(),
                format!("sk_live_{i}_4")
            );
        }
    }

    #[test]
    fn a_corrupt_store_is_moved_aside_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(private_path(&dir));
        create_private_dir_all(store.path().parent().unwrap()).unwrap();
        std::fs::write(store.path(), b"{ not json").unwrap();
        assert_eq!(store.read().unwrap_err().code, "credentials_corrupt");

        let lock = store.lock(Duration::from_secs(1)).unwrap();
        let loaded = lock.load().unwrap();
        let aside = loaded.moved_corrupt_file.expect("moved aside");
        assert_eq!(std::fs::read(&aside).unwrap(), b"{ not json");
        assert!(loaded.file.credentials.is_empty());
        assert!(!store.path().exists());
    }

    #[test]
    fn a_newer_format_is_refused_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(private_path(&dir));
        create_private_dir_all(store.path().parent().unwrap()).unwrap();
        std::fs::write(store.path(), br#"{"version": 99, "credentials": "x"}"#).unwrap();
        assert_eq!(store.read().unwrap_err().code, "credentials_unsupported");
        let lock = store.lock(Duration::from_secs(1)).unwrap();
        assert_eq!(lock.load().unwrap_err().code, "credentials_unsupported");
        assert!(store.path().exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_shared_writable_directory_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        std::fs::create_dir(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();
        let store = Store::at(shared.join("credentials.json"));
        assert_eq!(
            store.lock(Duration::from_secs(1)).unwrap_err().code,
            "credentials_unsafe"
        );
    }

    #[test]
    fn a_held_lock_times_out_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(private_path(&dir));
        let _held = store.lock(Duration::from_secs(1)).unwrap();
        assert_eq!(
            store.lock(Duration::from_millis(100)).unwrap_err().code,
            "credentials_busy"
        );
    }

    #[test]
    fn pending_requests_are_per_host_repository_and_kind() {
        let mut file = CredentialsFile::default();
        let pending = PendingRequest {
            platform_origin: ORIGIN.into(),
            repository_root: PathBuf::from("/a"),
            kind: PendingKind::Connect,
            idempotency_key: Secret::new("k".repeat(64)),
            challenge: Some("c".into()),
            nonce: Some("1".into()),
            claim_token: None,
            project_name: None,
            source: None,
            started_at: at(0),
        };
        file.set_pending(pending.clone());
        file.set_pending(pending.clone());
        assert_eq!(file.pending.len(), 1);
        assert!(file
            .pending(ORIGIN, Path::new("/a"), PendingKind::Project)
            .is_none());
        assert_eq!(
            file.pending(ORIGIN, Path::new("/a"), PendingKind::Connect),
            Some(&pending)
        );
        file.clear_pending(ORIGIN, Path::new("/a"), PendingKind::Connect);
        assert!(file.pending.is_empty());
    }
}
