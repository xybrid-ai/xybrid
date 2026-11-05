//! Random CLI installation identity, persisted separately from model caches.
//!
//! Telemetry uses this instead of a hostname or hardware fingerprint. If the
//! home directory cannot be written, an ephemeral random ID keeps inference usable.

use std::io::{self, Write};
use std::path::Path;
use uuid::Uuid;

/// Read or create `~/.xybrid/install-id`, falling back to an ephemeral UUID.
pub(crate) fn get() -> String {
    dirs::home_dir()
        .and_then(|home| load_or_create(&home.join(".xybrid")).ok())
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

    std::fs::create_dir_all(directory)?;
    let id = Uuid::new_v4();
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    writeln!(file, "{id}")?;
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
