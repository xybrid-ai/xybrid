//! The CLI's one key resolver.
//!
//! `--api-key`, then `XYBRID_API_KEY`, then the stored credentials for the
//! current canonical platform origin and repository root. Arbitrary dotenv
//! files are never loaded. Finding a key configures cloud authentication
//! only: trace export still needs an explicit `--observability`.

use std::path::Path;

use serde::Serialize;

use super::credentials::{Credential, Secret, Store};
use super::host::PlatformHost;
use super::CliError;

/// Where the key came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum KeySource {
    Flag,
    Env,
    Credentials,
}

impl KeySource {
    pub(crate) fn describe(self) -> &'static str {
        match self {
            KeySource::Flag => "--api-key",
            KeySource::Env => "XYBRID_API_KEY",
            KeySource::Credentials => "saved credentials",
        }
    }
}

/// A resolved key and, for stored keys, the credential it came from.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedKey {
    pub key: Secret,
    pub source: KeySource,
    pub credential: Option<Credential>,
}

/// An explicit key from the flag or environment (Clap merges the two).
pub(crate) fn explicit(value: Option<&str>, from_flag: bool) -> Option<ResolvedKey> {
    let value = value.map(str::trim).filter(|v| !v.is_empty())?;
    Some(ResolvedKey {
        key: Secret::new(value),
        source: if from_flag {
            KeySource::Flag
        } else {
            KeySource::Env
        },
        credential: None,
    })
}

/// The stored credential for exactly this origin and repository root.
pub(crate) fn stored(
    store: &Store,
    host: &PlatformHost,
    root: &Path,
) -> Result<Option<Credential>, CliError> {
    Ok(store.read()?.find(host.origin(), root).cloned())
}

/// The saved credential behind `key`: its own for a stored key, else the one
/// saved for this origin and repository when it holds the same key (an
/// exported project `.env`, say), so the claim reminder still sees the saved
/// deadlines. Best effort: an unreadable store is no credential.
pub(crate) fn stored_matching(
    key: &ResolvedKey,
    host: Option<&PlatformHost>,
    root: Option<&Path>,
) -> Option<Credential> {
    if let Some(credential) = &key.credential {
        return Some(credential.clone());
    }
    let store = Store::locate().ok()?;
    stored(&store, host?, root?)
        .ok()
        .flatten()
        .filter(|credential| credential.api_key == key.key)
}

/// Flag, then environment, then stored credentials. A store that cannot be
/// read is reported as a warning, never as a missing key silently.
pub(crate) fn resolve(
    explicit_key: Option<ResolvedKey>,
    host: Option<&PlatformHost>,
    root: Option<&Path>,
) -> (Option<ResolvedKey>, Option<String>) {
    if explicit_key.is_some() {
        return (explicit_key, None);
    }
    let (Some(host), Some(root)) = (host, root) else {
        return (None, None);
    };
    let store = match Store::locate() {
        Ok(store) => store,
        Err(_) => return (None, None),
    };
    match stored(&store, host, root) {
        Ok(Some(credential)) => (
            Some(ResolvedKey {
                key: credential.api_key.clone(),
                source: KeySource::Credentials,
                credential: Some(credential),
            }),
            None,
        ),
        Ok(None) => (None, None),
        Err(err) => (
            None,
            Some(format!("Ignoring saved Xybrid credentials: {err}")),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_keys_win_and_record_their_source() {
        let flag = explicit(Some("sk_live_flag"), true).unwrap();
        assert_eq!(flag.source, KeySource::Flag);
        let env = explicit(Some(" sk_live_env "), false).unwrap();
        assert_eq!(env.source, KeySource::Env);
        assert_eq!(env.key.expose(), "sk_live_env");
        assert!(explicit(Some("   "), false).is_none());
        assert!(explicit(None, true).is_none());

        let host = PlatformHost::parse("https://api.xybrid.dev").unwrap();
        let (resolved, warning) = resolve(Some(flag), Some(&host), Some(Path::new("/repo")));
        assert_eq!(resolved.unwrap().source, KeySource::Flag);
        assert!(warning.is_none());
    }

    #[test]
    fn stored_keys_never_cross_hosts_or_repositories() {
        use super::super::credentials::tests::private_path;
        use super::super::credentials::{CredentialsFile, NamedRef};
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(private_path(&dir));
        let lock = store.lock(std::time::Duration::from_secs(1)).unwrap();
        let mut file = CredentialsFile::default();
        file.put(
            Credential {
                platform_origin: "https://api.xybrid.dev".into(),
                repository_root: "/repo".into(),
                api_key: Secret::new("sk_live_stored"),
                scope: None,
                organization: Some(NamedRef {
                    slug: "a".into(),
                    name: "a".into(),
                }),
                project: None,
                key_expires_at: None,
                claim_expires_at: None,
                claim_token: None,
                source: None,
                console_origin: None,
                env_file: None,
                claimed: false,
                terms_url: None,
                privacy_url: None,
                created_at: chrono::Utc::now(),
            },
            chrono::Utc::now(),
        );
        lock.save(&file).unwrap();
        drop(lock);

        let prod = PlatformHost::parse("https://api.xybrid.dev").unwrap();
        let staging = PlatformHost::parse("https://staging.api.xybrid.dev").unwrap();
        assert!(stored(&store, &prod, Path::new("/repo")).unwrap().is_some());
        assert!(stored(&store, &staging, Path::new("/repo"))
            .unwrap()
            .is_none());
        assert!(stored(&store, &prod, Path::new("/other"))
            .unwrap()
            .is_none());
    }
}
