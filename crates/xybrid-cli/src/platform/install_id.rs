//! Random CLI installation ID, sent only with `xybrid connect`.
//!
//! The platform stores an HMAC of it as one abuse-control input. It is a
//! random UUID kept in `~/.xybrid/install-id` (next to
//! `XYBRID_CREDENTIALS_FILE` when that is set), never a hostname, MAC address
//! or hardware fingerprint, and it is never exported as telemetry (no
//! `device.id`, no resource attribute). If the file cannot be written, or an
//! existing one is unreadable, an ephemeral random ID is used instead and the
//! file is left as it is.
//!
//! Recovered from the install-ID helper discarded in #737 (`abeec6e3`).

use std::io::{self, Write};
use std::path::Path;

use uuid::Uuid;

use super::credentials::create_private_dir_all;

/// Read or create the install ID, falling back to an ephemeral UUID.
pub(crate) fn get() -> String {
    super::state_dir()
        .and_then(|directory| load_or_create(&directory).ok())
        .unwrap_or_else(Uuid::new_v4)
        .to_string()
}

fn read_id(path: &Path) -> io::Result<Uuid> {
    let content = std::fs::read_to_string(path)?;
    let id = Uuid::parse_str(content.trim())
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    if id.get_version() != Some(uuid::Version::Random) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "install ID must be a random UUID",
        ));
    }
    Ok(id)
}

fn load_or_create(directory: &Path) -> io::Result<Uuid> {
    let path = directory.join("install-id");
    match read_id(&path) {
        Ok(id) => return Ok(id),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }

    create_private_dir_all(directory)?;
    let id = Uuid::new_v4();
    // NamedTempFile is created 0600 on Unix.
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    writeln!(file, "{id}")?;
    // Reach the disk before the rename. Otherwise a crash can publish an empty
    // file, which is rejected without being replaced and would never heal.
    file.as_file().sync_all()?;
    // Publish a complete file without overwriting another CLI process's ID.
    match file.persist_noclobber(&path) {
        Ok(_) => Ok(id),
        Err(err) if err.error.kind() == io::ErrorKind::AlreadyExists => read_id(&path),
        Err(err) => Err(err.error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persists_random_identity_and_reuses_it() {
        let home = tempfile::tempdir().unwrap();
        let directory = home.path().join(".xybrid");
        let id = load_or_create(&directory).unwrap();
        assert_eq!(id.get_version(), Some(uuid::Version::Random));
        assert_eq!(load_or_create(&directory).unwrap(), id);
        assert_eq!(read_id(&directory.join("install-id")).unwrap(), id);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&directory.join("install-id")), 0o600);
            assert_eq!(mode(&directory), 0o700);
        }
    }

    #[test]
    fn separate_installations_have_separate_identities() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        assert_ne!(
            load_or_create(a.path()).unwrap(),
            load_or_create(b.path()).unwrap()
        );
    }

    #[test]
    fn concurrent_first_runs_share_one_identity() {
        let home = tempfile::tempdir().unwrap();
        let barrier = std::sync::Barrier::new(8);
        let ids = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        load_or_create(&home.path().join(".xybrid")).unwrap()
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(ids.iter().all(|id| *id == ids[0]));
    }

    #[test]
    fn rejects_corrupt_or_nonrandom_ids_without_overwriting_them() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("install-id");
        for value in [
            "host.example.test",
            "",
            "00000000-0000-0000-0000-000000000000",
        ] {
            std::fs::write(&path, value).unwrap();
            assert_eq!(
                load_or_create(directory.path()).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
            assert_eq!(std::fs::read_to_string(&path).unwrap(), value);
        }
    }
}
