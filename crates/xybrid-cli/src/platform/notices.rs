//! Stderr notices: the claim reminder, the no-export tip, and the workspace
//! notice of a run's trace export.
//!
//! The reminder and the tip go to stderr at most once per local day (state in
//! `~/.xybrid/notices.json`, which holds no secrets; see [`super::state_dir`]).
//! Every notice is suppressed by `--quiet`, JSON output, CI and
//! `XYBRID_QUIET`. The reminder runs on every command except `connect` and
//! `auth`, from the stored deadlines of the current project's credential
//! (also when the same key arrives through `--api-key` or `XYBRID_API_KEY`);
//! it never calls the platform, so it says the state is as of the last check.
//! The tip explains that a key alone exports nothing.
//!
//! The workspace notice comes from Xybrid's trace ingest, through the SDK's
//! [`WorkspaceFeedback`](xybrid_sdk::observability::WorkspaceFeedback), and is
//! read after the export's final drain. An expired key is reported on every
//! run it happens; a provisional one shares the reminder's daily slot, so a
//! day brings one claim nudge. A provisional deadline also refreshes the
//! saved credential's, when the run used that credential's key on its own
//! platform's ingest ([`refresh_key_deadline`]).

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Local, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use xybrid_sdk::observability::WorkspaceNotice;

use super::credentials::{create_private_dir_all, owned_by_current_account, Credential, Store};
use super::host::{self, PlatformHost};
use super::resolve::ResolvedKey;
use super::{ci_marker, xybrid_quiet};

#[derive(Debug, Default, Serialize, Deserialize)]
struct NoticeState {
    #[serde(default)]
    claim_reminder: Option<NaiveDate>,
    #[serde(default)]
    export_tip: Option<NaiveDate>,
}

#[derive(Debug, Clone, Copy)]
enum Notice {
    ClaimReminder,
    ExportTip,
}

/// Whether notices are off for this invocation.
pub(crate) fn suppressed(quiet: bool, json: bool) -> bool {
    quiet || json || ci_marker().is_some() || xybrid_quiet()
}

/// A human date for deadlines, e.g. `Sun, Nov 15 2026`.
pub(crate) fn human_date(at: DateTime<Utc>) -> String {
    at.with_timezone(&Local).format("%a, %b %-d %Y").to_string()
}

/// The reminder for a provisional, still-claimable credential.
pub(crate) fn claim_reminder_text(credential: &Credential, now: DateTime<Utc>) -> Option<String> {
    if !credential.claimable(now) {
        return None;
    }
    let name = credential
        .organization
        .as_ref()
        .map_or("this project", |o| o.name.as_str());
    let key_deadline = credential.key_expires_at?;
    let claim_deadline = credential.claim_expires_at?;
    // Saved state, refreshed only by `connect` and `auth status|claim`: a
    // claim made in the console since then is not known here, so say so.
    Some(if now < key_deadline {
        format!(
            "Xybrid workspace \"{name}\" was unclaimed when last checked: its key stops working \
             on {} unless you claim it. Claim it: xybrid auth claim (already claimed? `xybrid \
             auth status` updates this)",
            human_date(key_deadline)
        )
    } else {
        format!(
            "Xybrid workspace \"{name}\" was unclaimed when last checked and its key stopped on \
             {}. Claim it by {} to restore it: xybrid auth claim (already claimed? `xybrid auth \
             status` updates this)",
            human_date(key_deadline),
            human_date(claim_deadline)
        )
    })
}

/// The no-export tip for a run that has a key but exports nothing. Never
/// implies that a key alone exports traces.
pub(crate) fn export_tip_text() -> &'static str {
    "Tip: this project has a Xybrid key (from `xybrid connect` or the console), but a key only \
     authenticates: nothing was exported. Rerun with `xybrid --observability ...` to send this \
     run's traces to the console (metadata only: set \
     OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT=no_content), then check them with `xybrid auth \
     status`."
}

/// What a run reports about the workspace of the key its trace export used.
pub(crate) fn workspace_notice_text(notice: WorkspaceNotice) -> Option<String> {
    match notice {
        WorkspaceNotice::Expired => Some(
            "Xybrid stopped accepting traces from this key: its workspace was not claimed in \
             time and the key expired, so this run's traces were not stored (local inference \
             is unaffected). Claim the workspace to restore the key: xybrid auth claim"
                .to_string(),
        ),
        WorkspaceNotice::Provisional { key_expires_at } => Some(format!(
            "Xybrid accepted this run's traces into an unclaimed workspace: its key stops \
             working on {} unless you claim it. Claim it: xybrid auth claim",
            human_date(key_expires_at.into())
        )),
        _ => None,
    }
}

/// Print the workspace notice of this run's trace export to stderr, if any.
/// An expiry is reported every time; a provisional deadline at most once a
/// day, sharing the claim reminder's slot.
pub(crate) fn report_workspace_notice(notice: Option<WorkspaceNotice>, quiet: bool, json: bool) {
    let line = workspace_notice_line(notice, suppressed(quiet, json), || {
        take_daily_slot(Notice::ClaimReminder)
    });
    if let Some(text) = line {
        eprintln!("  {} {}", crate::ui::warn("!"), text);
    }
}

/// The workspace notice to print, if any. `take_slot` claims the day's
/// reminder slot; it only runs for a provisional notice that may be shown.
fn workspace_notice_line(
    notice: Option<WorkspaceNotice>,
    suppressed: bool,
    take_slot: impl FnOnce() -> bool,
) -> Option<String> {
    let notice = notice.filter(|_| !suppressed)?;
    let text = workspace_notice_text(notice)?;
    match notice {
        WorkspaceNotice::Expired => Some(text),
        _ => take_slot().then_some(text),
    }
}

/// Fold a provisional deadline reported by trace ingest into the saved
/// credential behind `key`, so the daily reminder follows the platform. Only
/// when the run exported with that credential's own key (whatever its
/// source), to `endpoint` that its platform's ingest may receive it on, and
/// the credential is not known to be claimed. Best effort: a store that
/// cannot be read, locked or written is left alone.
pub(crate) fn refresh_key_deadline(
    key: &ResolvedKey,
    host: &PlatformHost,
    root: &Path,
    endpoint: &str,
    key_expires_at: DateTime<Utc>,
) {
    if !host::trace_endpoint_allowed(host, endpoint) {
        return;
    }
    let Ok(store) = Store::locate() else {
        return;
    };
    let needs_update = |file: &super::credentials::CredentialsFile| {
        file.holds_key(host.origin(), root, &key.key)
            && file.find(host.origin(), root).is_some_and(|credential| {
                !credential.claimed && credential.key_expires_at != Some(key_expires_at)
            })
    };
    // Read first: most runs change nothing and never take the lock.
    if !store.read().is_ok_and(|file| needs_update(&file)) {
        return;
    }
    let Ok(lock) = store.lock(Duration::from_secs(2)) else {
        return;
    };
    let Ok(loaded) = lock.load() else {
        return;
    };
    let mut file = loaded.file;
    if loaded.moved_corrupt_file.is_some() || !needs_update(&file) {
        return;
    }
    let Some(mut credential) = file.find(host.origin(), root).cloned() else {
        return;
    };
    credential.key_expires_at = Some(key_expires_at);
    file.put(credential, Utc::now());
    let _ = lock.save(&file);
}

/// Print the claim reminder to stderr once a day. `credential` (the saved
/// credential behind the key in use) is only looked up when not suppressed.
pub(crate) fn maybe_claim_reminder(
    credential: impl FnOnce() -> Option<Credential>,
    quiet: bool,
    json: bool,
) {
    if suppressed(quiet, json) {
        return;
    }
    let Some(text) = credential().and_then(|c| claim_reminder_text(&c, Utc::now())) else {
        return;
    };
    if take_daily_slot(Notice::ClaimReminder) {
        eprintln!("  {} {}", crate::ui::warn("!"), text);
    }
}

/// Print the no-export tip to stderr once a day.
pub(crate) fn maybe_export_tip(quiet: bool, json: bool) {
    if suppressed(quiet, json) || super::opt_out().is_some() {
        return;
    }
    if take_daily_slot(Notice::ExportTip) {
        crate::ui::hint_stderr(export_tip_text());
    }
}

fn state_path() -> Option<PathBuf> {
    super::state_dir().map(|directory| directory.join("notices.json"))
}

/// The notice state, if this account wrote it: a file another account
/// planted in a shared directory must not silence the reminder.
fn read_own(path: &std::path::Path) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    if !owned_by_current_account(&file.metadata().ok()?) {
        return None;
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

/// Whether `notice` has not been shown today; records it as shown. If the
/// state cannot be saved, the notice is shown anyway.
fn take_daily_slot(notice: Notice) -> bool {
    let today = Local::now().date_naive();
    let Some(path) = state_path() else {
        return true;
    };
    let mut state: NoticeState = read_own(&path)
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    let slot = match notice {
        Notice::ClaimReminder => &mut state.claim_reminder,
        Notice::ExportTip => &mut state.export_tip,
    };
    if *slot == Some(today) {
        return false;
    }
    *slot = Some(today);
    let save = || -> std::io::Result<()> {
        use std::io::Write;
        let directory = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        create_private_dir_all(directory)?;
        let mut temp = tempfile::NamedTempFile::new_in(directory)?;
        temp.write_all(&serde_json::to_vec(&state).unwrap_or_default())?;
        temp.persist(&path).map_err(|e| e.error)?;
        Ok(())
    };
    let _ = save();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::credentials::{NamedRef, Secret};
    use chrono::Duration;

    fn credential(now: DateTime<Utc>) -> Credential {
        Credential {
            platform_origin: "https://api.xybrid.dev".into(),
            repository_root: "/repo".into(),
            api_key: Secret::new("sk_live_x"),
            scope: Some("client".into()),
            organization: Some(NamedRef {
                slug: "my-app".into(),
                name: "my-app".into(),
            }),
            project: None,
            key_expires_at: Some(now + Duration::days(7)),
            claim_expires_at: Some(now + Duration::days(14)),
            claim_token: Some(Secret::new("t".repeat(43))),
            source: None,
            console_origin: None,
            env_file: None,
            claimed: false,
            terms_url: None,
            privacy_url: None,
            created_at: now,
        }
    }

    #[test]
    fn reminders_follow_the_stored_deadlines() {
        let now = Utc::now();
        let cred = credential(now);
        let before = claim_reminder_text(&cred, now).unwrap();
        assert!(before.contains("stops working on"), "{before}");
        assert!(
            before.contains("when last checked"),
            "saved state: {before}"
        );
        assert!(
            before.contains("xybrid auth status"),
            "how to refresh: {before}"
        );
        assert!(before.contains("xybrid auth claim"));
        assert!(!before.contains(&"t".repeat(43)), "never the claim token");

        let after = claim_reminder_text(&cred, now + Duration::days(8)).unwrap();
        assert!(after.contains("stopped on"), "{after}");
        assert!(after.contains("to restore it"));

        assert!(claim_reminder_text(&cred, now + Duration::days(15)).is_none());
        let mut claimed = cred.clone();
        claimed.claimed = true;
        assert!(claim_reminder_text(&claimed, now).is_none());
        let mut lost = cred;
        lost.claim_token = None;
        assert!(claim_reminder_text(&lost, now).is_none());
    }

    #[test]
    fn the_tip_never_implies_key_only_export() {
        let tip = export_tip_text();
        assert!(tip.contains("--observability"), "{tip}");
        assert!(tip.contains("only authenticates"), "{tip}");
        assert!(tip.contains("nothing was exported"), "{tip}");
        assert!(tip.contains("xybrid connect"), "{tip}");
        assert!(tip.contains("no_content"), "{tip}");
    }

    #[test]
    fn workspace_notices_point_at_the_claim_command() {
        let expired = workspace_notice_text(WorkspaceNotice::Expired).unwrap();
        assert!(expired.contains("xybrid auth claim"), "{expired}");
        assert!(expired.contains("expired"), "{expired}");
        assert!(expired.contains("not stored"), "{expired}");
        let deadline = Utc::now() + Duration::days(3);
        let provisional = workspace_notice_text(WorkspaceNotice::Provisional {
            key_expires_at: deadline.into(),
        })
        .unwrap();
        assert!(provisional.contains("xybrid auth claim"), "{provisional}");
        assert!(provisional.contains(&human_date(deadline)), "{provisional}");
    }

    #[test]
    fn workspace_notices_are_silent_when_suppressed_and_provisional_is_daily() {
        let provisional = Some(WorkspaceNotice::Provisional {
            key_expires_at: (Utc::now() + Duration::days(3)).into(),
        });
        let expired = Some(WorkspaceNotice::Expired);
        // `--quiet`, JSON, CI and XYBRID_QUIET all arrive as `suppressed`
        // (`suppressed(quiet, json)` covers the flags and the environment).
        assert!(suppressed(true, false) && suppressed(false, true));
        for notice in [provisional, expired] {
            let line = workspace_notice_line(notice, true, || panic!("no slot when suppressed"));
            assert_eq!(line, None);
        }
        assert_eq!(workspace_notice_line(None, false, || true), None);
        // An expiry never waits for the daily slot.
        assert!(workspace_notice_line(expired, false, || panic!("not gated")).is_some());
        // A provisional deadline shares the reminder's slot.
        assert!(workspace_notice_line(provisional, false, || true).is_some());
        assert_eq!(workspace_notice_line(provisional, false, || false), None);
    }
}
