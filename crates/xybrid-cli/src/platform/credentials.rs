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
//!
//! Another account must not be able to plant or swap the saved key, so on
//! Unix the store is refused, with the `chmod` that fixes it, when:
//!
//! - its directory belongs to another account (other than root);
//! - its directory is writable by every account without the sticky bit;
//! - its directory is group-writable, unless the group is this account's own
//!   private group (directory group = primary group = uid, the user-private-
//!   group convention behind a `002` umask, so `~/.xybrid` keeps working on
//!   stock Linux accounts);
//! - the file belongs to another account, or its group or other accounts have
//!   any permission on it (a planted file in a sticky directory such as
//!   `/tmp` is caught here).

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

    /// Whether the entry for this host and repository still holds `key`.
    /// A read-check-write done without the lock (for example `auth status`
    /// asking the server about the key it read) only writes back when this
    /// holds, so it never retires or overwrites a key saved meanwhile.
    pub(crate) fn holds_key(&self, origin: &str, root: &Path, key: &Secret) -> bool {
        self.find(origin, root).is_some_and(|c| &c.api_key == key)
    }

    /// Record that the workspace behind `claim_token` was claimed, for every
    /// repository that joined it. Returns whether anything changed.
    pub(crate) fn mark_claimed(&mut self, claim_token: &Secret) -> bool {
        let mut changed = false;
        for credential in &mut self.credentials {
            if !credential.claimed && credential.claim_token.as_ref() == Some(claim_token) {
                credential.claimed = true;
                changed = true;
            }
        }
        changed
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

    /// The newest saved credential on this host whose workspace `claim_token`
    /// claims: a repository already in that workspace, whatever its state.
    pub(crate) fn workspace_member(
        &self,
        origin: &str,
        claim_token: &Secret,
    ) -> Option<&Credential> {
        self.credentials
            .iter()
            .filter(|c| c.platform_origin == origin && c.claim_token.as_ref() == Some(claim_token))
            .max_by_key(|c| c.created_at)
    }

    /// The newest retired credential for this host and repository that can
    /// still be claimed: the recovery path after the platform rejected a key.
    pub(crate) fn retired_claimable(
        &self,
        origin: &str,
        root: &Path,
        now: DateTime<Utc>,
    ) -> Option<&Credential> {
        self.retired
            .iter()
            .rev()
            .map(|r| &r.credential)
            .find(|c| c.matches(origin, root) && c.claimable(now))
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
    /// file is empty; a corrupt one is an error and is left untouched. An
    /// existing file is only trusted in a private directory, owned by this
    /// account and with private permissions.
    pub(crate) fn read(&self) -> Result<CredentialsFile, CliError> {
        match read_trusted(&self.path)? {
            Some(bytes) => {
                ensure_private_directory(self.directory())?;
                parse(&bytes).map_err(|err| err.into_cli_error(&self.path))
            }
            None => Ok(CredentialsFile::default()),
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
        let Some(bytes) = read_trusted(&self.path)? else {
            return Ok(Loaded {
                file: CredentialsFile::default(),
                moved_corrupt_file: None,
            });
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

/// The account this process runs as, for the ownership checks.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Account {
    pub uid: u32,
    /// The process's group: the login's primary group.
    pub gid: u32,
}

#[cfg(unix)]
impl Account {
    pub(crate) fn current() -> Self {
        // SAFETY: geteuid and getegid take no arguments, always succeed and
        // touch no memory; std offers no safe equivalent.
        let (uid, gid) = unsafe { (libc::geteuid(), libc::getegid()) };
        Self { uid, gid }
    }

    /// The user-private-group convention: a primary group of its own, named
    /// by the same number as the account.
    fn has_private_group(self) -> bool {
        self.gid == self.uid
    }
}

/// Who can add, remove or replace files in a directory.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryAccess {
    /// Only this account (or root).
    Private,
    /// Others can add files but the sticky bit stops them replacing this
    /// account's; a file they planted fails the file ownership check.
    Sticky,
    /// Group-writable, but the group is this account's own private group.
    UserPrivateGroup,
    /// Another account owns it and can replace anything in it.
    OtherOwner { uid: u32 },
    /// Every account can write it and there is no sticky bit.
    WorldWritable,
    /// A group other accounts may belong to can write it.
    SharedGroup { gid: u32 },
}

#[cfg(unix)]
impl DirectoryAccess {
    fn of(mode: u32, owner: u32, group: u32, account: Account) -> Self {
        if owner != account.uid && owner != 0 {
            return Self::OtherOwner { uid: owner };
        }
        if mode & 0o022 == 0 {
            return Self::Private;
        }
        if mode & 0o1000 != 0 {
            return Self::Sticky;
        }
        if mode & 0o002 != 0 {
            return Self::WorldWritable;
        }
        if owner == account.uid && group == account.gid && account.has_private_group() {
            return Self::UserPrivateGroup;
        }
        Self::SharedGroup { gid: group }
    }

    #[cfg(test)]
    fn is_safe(self) -> bool {
        matches!(self, Self::Private | Self::Sticky | Self::UserPrivateGroup)
    }
}

/// Refuse a directory in which another account could plant or swap the file.
#[cfg(unix)]
fn ensure_private_directory(directory: &Path) -> Result<(), CliError> {
    use std::os::unix::fs::MetadataExt;
    let metadata =
        std::fs::metadata(directory).map_err(|err| io_error("inspect", directory, &err))?;
    let mode = metadata.mode() & 0o7777;
    let access = DirectoryAccess::of(mode, metadata.uid(), metadata.gid(), Account::current());
    let dir = directory.display();
    let quoted = shell_quote(directory);
    let message = match access {
        DirectoryAccess::Private | DirectoryAccess::Sticky | DirectoryAccess::UserPrivateGroup => {
            return Ok(())
        }
        DirectoryAccess::OtherOwner { uid } => format!(
            "The credentials directory {dir} belongs to another account (uid {uid}), which could \
             replace the saved key. Set {CREDENTIALS_FILE_ENV} to a file in a directory you own"
        ),
        DirectoryAccess::WorldWritable => format!(
            "The credentials directory {dir} is writable by every account on this machine (mode \
             {mode:04o}), so another account could replace the saved key. Make it private with \
             `chmod go-w {quoted}` and run the command again, or set {CREDENTIALS_FILE_ENV} to a \
             file in a directory only you can write"
        ),
        DirectoryAccess::SharedGroup { gid } => format!(
            "The credentials directory {dir} is writable by its group (gid {gid}, mode {mode:04o}), \
             which is not your own private group, so another account in it could replace the \
             saved key. Make it private with `chmod go-w {quoted}` and run the command again, or \
             set {CREDENTIALS_FILE_ENV} to a file in a directory only you can write"
        ),
    };
    Err(CliError::new("credentials_unsafe", message))
}

#[cfg(not(unix))]
fn ensure_private_directory(_directory: &Path) -> Result<(), CliError> {
    Ok(())
}

/// Open and read the store file, trusting it only if this account owns it
/// and nobody else has any permission on it. `None` if it does not exist.
fn read_trusted(path: &Path) -> Result<Option<Vec<u8>>, CliError> {
    use std::io::Read;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        // A FIFO planted in a shared directory must not block the open.
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let mut file = match options.open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(io_error("read", path, &err)),
    };
    // Checked on the open handle, so the bytes read are the bytes checked.
    let metadata = file
        .metadata()
        .map_err(|err| io_error("inspect", path, &err))?;
    ensure_private_file(path, &metadata)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|err| io_error("read", path, &err))?;
    Ok(Some(bytes))
}

/// Why a credentials file is not trusted.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileProblem {
    /// Another account created it (planted, in a shared directory).
    OtherOwner { uid: u32 },
    /// Its group or other accounts have some permission on it.
    Shared,
}

#[cfg(unix)]
impl FileProblem {
    fn of(mode: u32, owner: u32, account: Account) -> Option<Self> {
        if owner != account.uid {
            Some(Self::OtherOwner { uid: owner })
        } else if mode & 0o077 != 0 {
            Some(Self::Shared)
        } else {
            None
        }
    }
}

/// Refuse a credentials file another account owns, or that other accounts
/// can access at all.
fn ensure_private_file(path: &Path, metadata: &std::fs::Metadata) -> Result<(), CliError> {
    if !metadata.is_file() {
        return Err(CliError::new(
            "credentials_unsafe",
            format!(
                "The credentials file {} is not a regular file",
                path.display()
            ),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let mode = metadata.mode() & 0o7777;
        let message = match FileProblem::of(mode, metadata.uid(), Account::current()) {
            None => None,
            Some(FileProblem::OtherOwner { uid }) => Some(format!(
                "The credentials file {} belongs to another account (uid {uid}), so its key is \
                 not trusted. Set {CREDENTIALS_FILE_ENV} to a file in a directory only you can \
                 write",
                path.display()
            )),
            Some(FileProblem::Shared) => Some(format!(
                "The credentials file {} can be accessed by other accounts (mode {mode:04o}). \
                 Make it private with `chmod 600 {}` and run the command again",
                path.display(),
                shell_quote(path)
            )),
        };
        if let Some(message) = message {
            return Err(CliError::new("credentials_unsafe", message));
        }
    }
    Ok(())
}

/// Whether this account owns the file behind `metadata`: state such as the
/// install ID is only trusted from a file this account wrote.
#[cfg(unix)]
pub(crate) fn owned_by_current_account(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.uid() == Account::current().uid
}

/// Off Unix the user profile is private to the user.
#[cfg(not(unix))]
pub(crate) fn owned_by_current_account(_metadata: &std::fs::Metadata) -> bool {
    true
}

/// `path` as one POSIX shell word, for copy-pasteable fixes.
#[cfg(unix)]
fn shell_quote(path: &Path) -> String {
    let raw = path.display().to_string();
    if !raw.is_empty()
        && raw
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"/._-+".contains(&b))
    {
        raw
    } else {
        format!("'{}'", raw.replace('\'', "'\\''"))
    }
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

    /// Write `bytes` as a store file would be: `0600` in a `0700` directory.
    fn write_private(store: &Store, bytes: &[u8]) {
        create_private_dir_all(store.path().parent().unwrap()).unwrap();
        std::fs::write(store.path(), bytes).unwrap();
        set_mode(store.path(), 0o600);
    }

    #[cfg(unix)]
    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    #[cfg(not(unix))]
    fn set_mode(_path: &Path, _mode: u32) {}

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

        assert_eq!(
            file.retired_claimable(ORIGIN, Path::new("/a"), at(10))
                .unwrap()
                .api_key
                .expose(),
            "sk_live_a"
        );
        assert!(file
            .retired_claimable(ORIGIN, Path::new("/a"), at(15 * 86_400))
            .is_none());
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
        write_private(&store, b"{ not json");
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
        write_private(&store, br#"{"version": 99, "credentials": "x"}"#);
        assert_eq!(store.read().unwrap_err().code, "credentials_unsupported");
        let lock = store.lock(Duration::from_secs(1)).unwrap();
        assert_eq!(lock.load().unwrap_err().code, "credentials_unsupported");
        assert!(store.path().exists());
    }

    #[cfg(unix)]
    #[test]
    fn directory_policy_follows_ownership_and_the_user_private_group() {
        use DirectoryAccess::*;
        // A user-private-group account (uid 1000, primary group 1000).
        let upg = Account {
            uid: 1000,
            gid: 1000,
        };
        // An account whose primary group is shared (e.g. `users`, `staff`).
        let shared = Account {
            uid: 1000,
            gid: 100,
        };
        for (mode, owner, group, account, expected) in [
            (0o700, 1000, 1000, upg, Private),
            (0o755, 1000, 100, shared, Private),
            (0o755, 0, 0, upg, Private),
            // umask 002 on a user-private-group system: accepted.
            (0o775, 1000, 1000, upg, UserPrivateGroup),
            (0o770, 1000, 1000, upg, UserPrivateGroup),
            // Group-writable by a group that is not the account's own.
            (0o775, 1000, 100, shared, SharedGroup { gid: 100 }),
            (0o775, 1000, 100, upg, SharedGroup { gid: 100 }),
            (0o775, 1000, 1000, shared, SharedGroup { gid: 1000 }),
            (0o775, 0, 1000, upg, SharedGroup { gid: 1000 }),
            // Writable by everyone: only with the sticky bit.
            (0o777, 1000, 1000, upg, WorldWritable),
            (0o1777, 0, 0, upg, Sticky),
            (0o1775, 1000, 100, shared, Sticky),
            // Another account's directory, whatever its mode.
            (0o700, 1001, 1001, upg, OtherOwner { uid: 1001 }),
            (0o1777, 1001, 1001, upg, OtherOwner { uid: 1001 }),
        ] {
            let access = DirectoryAccess::of(mode, owner, group, account);
            assert_eq!(access, expected, "{mode:o} {owner}:{group} {account:?}");
            assert_eq!(
                access.is_safe(),
                matches!(expected, Private | Sticky | UserPrivateGroup)
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn file_policy_requires_this_account_and_no_shared_bits() {
        let me = Account {
            uid: 1000,
            gid: 1000,
        };
        assert_eq!(FileProblem::of(0o600, 1000, me), None);
        assert_eq!(FileProblem::of(0o400, 1000, me), None);
        for mode in [0o640, 0o604, 0o660, 0o644, 0o610] {
            assert_eq!(FileProblem::of(mode, 1000, me), Some(FileProblem::Shared));
        }
        assert_eq!(
            FileProblem::of(0o600, 1001, me),
            Some(FileProblem::OtherOwner { uid: 1001 })
        );
        assert_eq!(
            FileProblem::of(0o600, 0, me),
            Some(FileProblem::OtherOwner { uid: 0 })
        );
    }

    #[cfg(unix)]
    #[test]
    fn world_writable_directories_are_refused_with_the_fix() {
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("world");
        std::fs::create_dir(&shared).unwrap();
        set_mode(&shared, 0o777);
        let store = Store::at(shared.join("credentials.json"));
        let err = store.lock(Duration::from_secs(1)).unwrap_err();
        assert_eq!(err.code, "credentials_unsafe");
        let fix = format!("chmod go-w {}", shell_quote(&shared));
        assert!(err.message.contains(&fix), "{}", err.message);
        assert!(
            err.message.contains(CREDENTIALS_FILE_ENV),
            "{}",
            err.message
        );

        // A file there is never read either, even a private one of ours.
        std::fs::write(store.path(), b"{\"version\": 1}").unwrap();
        set_mode(store.path(), 0o600);
        assert_eq!(store.read().unwrap_err().code, "credentials_unsafe");
    }

    /// The real directory created with a `002` umask: accepted exactly when
    /// its group is this account's private group, else refused with the fix.
    #[cfg(unix)]
    #[test]
    fn a_group_writable_directory_is_accepted_only_for_a_private_group() {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::tempdir().unwrap();
        let group = dir.path().join("group");
        std::fs::create_dir(&group).unwrap();
        set_mode(&group, 0o775);
        let metadata = std::fs::metadata(&group).unwrap();
        let account = Account::current();
        let private_group = metadata.gid() == account.gid && account.gid == account.uid;

        let store = Store::at(group.join("credentials.json"));
        match store.lock(Duration::from_secs(1)) {
            Ok(lock) => {
                assert!(private_group, "accepted a shared group: {account:?}");
                let mut file = lock.load().unwrap().file;
                file.put(credential("/repo", "sk_live_one"), at(0));
                lock.save(&file).unwrap();
                drop(lock);
                assert_eq!(store.read().unwrap(), file);
            }
            Err(err) => {
                assert!(!private_group, "refused a private group: {}", err.message);
                assert_eq!(err.code, "credentials_unsafe");
                let fix = format!("chmod go-w {}", shell_quote(&group));
                assert!(err.message.contains(&fix), "{}", err.message);
            }
        }
    }

    /// `/tmp`-like: anyone can add a file, nobody can replace ours. Our own
    /// private file is used; the ownership check covers planted ones.
    #[cfg(unix)]
    #[test]
    fn a_sticky_shared_directory_keeps_working_for_private_files() {
        let dir = tempfile::tempdir().unwrap();
        let sticky = dir.path().join("sticky");
        std::fs::create_dir(&sticky).unwrap();
        set_mode(&sticky, 0o1777);
        let store = Store::at(sticky.join("credentials.json"));
        let lock = store.lock(Duration::from_secs(1)).unwrap();
        let mut file = lock.load().unwrap().file;
        file.put(credential("/repo", "sk_live_one"), at(0));
        lock.save(&file).unwrap();
        drop(lock);
        assert_eq!(store.read().unwrap(), file);
    }

    #[cfg(unix)]
    #[test]
    fn a_file_other_accounts_can_access_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(private_path(&dir));
        let mut file = CredentialsFile::default();
        file.put(credential("/repo", "sk_live_one"), at(0));
        write_private(&store, &serde_json::to_vec(&file).unwrap());
        assert_eq!(store.read().unwrap(), file);

        for mode in [0o644, 0o640, 0o620] {
            set_mode(store.path(), mode);
            let err = store.read().unwrap_err();
            assert_eq!(err.code, "credentials_unsafe", "{mode:o}");
            assert!(
                err.message
                    .contains(&format!("chmod 600 {}", shell_quote(store.path()))),
                "{}",
                err.message
            );
            let lock = store.lock(Duration::from_secs(1)).unwrap();
            assert_eq!(lock.load().unwrap_err().code, "credentials_unsafe");
            assert!(store.path().exists(), "never moved aside or replaced");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_in_place_of_the_file_is_refused_without_blocking() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(private_path(&dir));
        create_private_dir_all(store.path().parent().unwrap()).unwrap();
        let made = std::process::Command::new("mkfifo")
            .arg(store.path())
            .status();
        if !made.is_ok_and(|s| s.success()) {
            eprintln!("skipping: mkfifo is not available");
            return;
        }
        assert_eq!(store.read().unwrap_err().code, "credentials_unsafe");
    }

    #[cfg(unix)]
    #[test]
    fn fixes_quote_paths_for_the_shell() {
        assert_eq!(
            shell_quote(Path::new("/home/me/.xybrid")),
            "/home/me/.xybrid"
        );
        assert_eq!(shell_quote(Path::new("/a b/c")), "'/a b/c'");
        assert_eq!(shell_quote(Path::new("/it's")), "'/it'\\''s'");
    }

    #[test]
    fn a_claim_marks_every_repository_of_the_workspace() {
        let mut file = CredentialsFile::default();
        file.put(credential("/a", "sk_live_a"), at(0));
        file.put(credential("/b", "sk_live_b"), at(0));
        let mut other = credential("/c", "sk_live_c");
        other.claim_token = Some(Secret::new("o".repeat(43)));
        file.put(other, at(0));
        assert!(file.mark_claimed(&Secret::new("t".repeat(43))));
        let claimed: Vec<bool> = file.credentials.iter().map(|c| c.claimed).collect();
        assert_eq!(claimed, [true, true, false]);
        assert!(
            !file.mark_claimed(&Secret::new("t".repeat(43))),
            "no change"
        );

        assert!(file.holds_key(ORIGIN, Path::new("/a"), &Secret::new("sk_live_a")));
        assert!(!file.holds_key(ORIGIN, Path::new("/a"), &Secret::new("sk_live_b")));
        assert!(!file.holds_key(ORIGIN, Path::new("/z"), &Secret::new("sk_live_a")));
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
