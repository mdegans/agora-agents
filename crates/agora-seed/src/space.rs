//! Free space, checked once before a run starts any session.
//!
//! A run's saves are every agent's state: a session whose save fails for a
//! full disk is kept only in a rescue file, which needs the same disk
//! ([`crate::alerts::AlertKind::StateUnsaved`]). So before anything is
//! loaded or started, the filesystem holding the state root, and the one
//! holding the secrets root if it is another, must have at least
//! [`floor`] free. Below it the run starts no session, raises
//! [`crate::alerts::AlertKind::DiskSpaceLow`] (which path, how much is
//! free, the floor) and exits as failed; a unit with `Restart=` tries again
//! at its next start.
//!
//! **What this reads.** Filesystem statistics only (`statvfs(3)`) and each
//! root's device number: no file of an agent's is opened, listed or
//! written. A root that doesn't exist yet is measured at its nearest
//! existing ancestor. A root that can't be measured is logged and does not
//! block the run: no evidence of a full disk is not a full disk.

use std::path::{Path, PathBuf};

const GIB: u64 = 1024 * 1024 * 1024;

/// The least free space a run starts with, whatever the filesystem's
/// size: 2 GiB. The sealed copies the runner keeps were estimated at 0.4
/// to 0.7 GB for the fleet at its enrolment and under 2 GB a year after
/// (the local-chain plan, 2026-11-11), so this is about a year of the
/// chain's growth: far more than one sweep writes, with room to rescue
/// the sessions it can't save.
pub const FLOOR_BYTES: u64 = 2 * GIB;

/// The floor is also never under one part in this many of the filesystem
/// (1%): on a large disk the last percent is where allocation slows and
/// other writers (logs, the database's volume) start to fail.
pub const FLOOR_DIVISOR: u64 = 100;

/// The floor for a filesystem of `total` bytes: [`FLOOR_BYTES`] or
/// `total / FLOOR_DIVISOR`, whichever is larger.
pub fn floor(total: u64) -> u64 {
    FLOOR_BYTES.max(total / FLOOR_DIVISOR)
}

/// One filesystem's statistics, as an unprivileged writer sees them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Space {
    /// The device the measured path is on: two roots with the same one
    /// share a filesystem.
    pub dev: u64,
    /// Bytes free to an unprivileged user (`f_bavail`).
    pub free: u64,
    /// The filesystem's size in bytes (`f_blocks`).
    pub total: u64,
}

/// A root whose filesystem is below its floor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Low {
    /// The root asked about (the state root, or the secrets root).
    pub path: PathBuf,
    pub free: u64,
    pub total: u64,
    pub floor: u64,
}

impl std::fmt::Display for Low {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} free of {}, under the floor of {}",
            self.path.display(),
            gib(self.free),
            gib(self.total),
            gib(self.floor)
        )
    }
}

/// The run was not started: [`Low`] for each filesystem below its floor.
#[derive(Debug, thiserror::Error)]
#[error("free space is below the floor, so no session was started: {}", list(.0))]
pub struct LowSpace(pub Vec<Low>);

fn list(low: &[Low]) -> String {
    low.iter()
        .map(Low::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

/// Bytes as GiB to two places ("1.50 GiB"): for a person, beside the exact
/// count in the log's fields.
pub fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB as f64)
}

/// The nearest ancestor of `path` that exists (itself, if it does): a data
/// directory not yet made is on its parent's filesystem.
fn existing(path: &Path) -> &Path {
    path.ancestors().find(|p| p.exists()).unwrap_or(path)
}

/// `statvfs(3)` of the filesystem holding `path`, and its device number.
pub fn stat(path: &Path) -> std::io::Result<Space> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    let path = existing(path);
    let dev = std::fs::metadata(path)?.dev();
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let mut buf = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    // SAFETY: `c_path` is a NUL-terminated string that lives across the
    // call, and `buf` is writable memory of exactly `struct statvfs`'s
    // size and alignment, which the call only writes to.
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), buf.as_mut_ptr()) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: zero-initialised above, and every bit pattern is a valid
    // `struct statvfs` (integers only); the call returned 0, so it is the
    // filesystem's.
    let s = unsafe { buf.assume_init() };
    // The field widths differ by target (u64 on x86_64 Linux, glibc and
    // musl alike)
    #[allow(clippy::unnecessary_cast)]
    let (frsize, bavail, blocks) = (s.f_frsize as u64, s.f_bavail as u64, s.f_blocks as u64);
    Ok(Space {
        dev,
        free: bavail.saturating_mul(frsize),
        total: blocks.saturating_mul(frsize),
    })
}

/// Which of `roots` are on a filesystem below its floor, one entry per
/// filesystem (the first root found on it). A root `stat` can't measure is
/// logged and left out.
pub fn below_floor_with(
    roots: &[&Path],
    stat: impl Fn(&Path) -> std::io::Result<Space>,
) -> Vec<Low> {
    let mut seen: Vec<u64> = Vec::new();
    let mut low = Vec::new();
    for root in roots {
        let space = match stat(root) {
            Ok(space) => space,
            Err(e) => {
                tracing::warn!(
                    event_type = "disk_space_unmeasured",
                    path = %root.display(),
                    error = %e,
                    "free space couldn't be measured; the run is not held for it"
                );
                continue;
            }
        };
        if seen.contains(&space.dev) {
            continue;
        }
        seen.push(space.dev);
        let floor = floor(space.total);
        tracing::info!(
            event_type = "disk_space_checked",
            path = %root.display(),
            free_bytes = space.free,
            total_bytes = space.total,
            floor_bytes = floor,
            "free space before the run"
        );
        if space.free < floor {
            low.push(Low {
                path: root.to_path_buf(),
                free: space.free,
                total: space.total,
                floor,
            });
        }
    }
    low
}

/// The gate a run passes before it loads or starts anything: `Ok` when no
/// filesystem of `roots` is below its floor (or none could be measured);
/// else each is logged, one alert is raised, and the run is not started.
pub fn gate_with(
    roots: &[&Path],
    stat: impl Fn(&Path) -> std::io::Result<Space>,
    alerts: &crate::alerts::Alerter,
) -> Result<(), LowSpace> {
    let low = below_floor_with(roots, stat);
    if low.is_empty() {
        return Ok(());
    }
    for l in &low {
        tracing::error!(
            event_type = "disk_space_low",
            path = %l.path.display(),
            free_bytes = l.free,
            total_bytes = l.total,
            floor_bytes = l.floor,
            "free space below the floor; no session started"
        );
    }
    alerts.notify(alert(&low));
    Err(LowSpace(low))
}

/// The alert for a run not started: each path, what is free, the floor.
pub fn alert(low: &[Low]) -> crate::alerts::Alert {
    let mut alert = crate::alerts::Alert::new(
        crate::alerts::AlertKind::DiskSpaceLow,
        format!("no session started: {}", list(low)),
    );
    for l in low {
        alert = alert
            .detail("path", l.path.display())
            .detail("free_bytes", l.free)
            .detail("total_bytes", l.total)
            .detail("floor_bytes", l.floor);
    }
    alert
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2 GiB, or 1% of the filesystem, whichever is larger.
    #[test]
    fn the_floor_is_two_gib_or_one_percent() {
        assert_eq!(FLOOR_BYTES, 2 * GIB);
        assert_eq!(floor(0), 2 * GIB);
        assert_eq!(floor(100 * GIB), 2 * GIB, "1% is 1 GiB: the 2 GiB stands");
        assert_eq!(floor(200 * GIB), 2 * GIB);
        assert_eq!(floor(1000 * GIB), 10 * GIB);
        assert_eq!(gib(3 * GIB / 2), "1.50 GiB");
    }

    fn fake(
        table: &'static [(&'static str, u64, u64, u64)],
    ) -> impl Fn(&Path) -> std::io::Result<Space> {
        move |p: &Path| {
            table
                .iter()
                .find(|(path, ..)| Path::new(path) == p)
                .map(|&(_, dev, free, total)| Space { dev, free, total })
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
        }
    }

    /// Above the floor: nothing. Below: the path, what is free, the floor.
    #[test]
    fn a_filesystem_below_its_floor_is_named_with_its_figures() {
        let roots = [Path::new("/d/state"), Path::new("/d/secrets")];
        let plenty = fake(&[
            ("/d/state", 1, 50 * GIB, 500 * GIB),
            ("/d/secrets", 1, 50 * GIB, 500 * GIB),
        ]);
        assert_eq!(below_floor_with(&roots, plenty), []);

        // 500 GiB: the floor is 1%, 5 GiB; 4 GiB free is under it, though
        // over the 2 GiB
        let low = below_floor_with(
            &roots,
            fake(&[
                ("/d/state", 1, 4 * GIB, 500 * GIB),
                ("/d/secrets", 1, 4 * GIB, 500 * GIB),
            ]),
        );
        assert_eq!(
            low,
            [Low {
                path: "/d/state".into(),
                free: 4 * GIB,
                total: 500 * GIB,
                floor: 5 * GIB
            }],
            "one entry for the one filesystem"
        );
        assert_eq!(
            LowSpace(low.clone()).to_string(),
            "free space is below the floor, so no session was started: /d/state: 4.00 GiB free \
             of 500.00 GiB, under the floor of 5.00 GiB"
        );
        let alert = alert(&low);
        assert_eq!(alert.kind, crate::alerts::AlertKind::DiskSpaceLow);
        assert!(
            alert.summary.contains("/d/state: 4.00 GiB free"),
            "{alert:?}"
        );
        assert!(
            alert
                .details
                .contains(&("floor_bytes", (5 * GIB).to_string())),
            "{alert:?}"
        );
        // Exactly at the floor is not below it
        assert_eq!(
            below_floor_with(&roots[..1], fake(&[("/d/state", 1, 5 * GIB, 500 * GIB)])),
            []
        );
    }

    /// The secrets root on its own filesystem is checked on its own, with
    /// its own floor.
    #[test]
    fn the_secrets_root_on_another_filesystem_is_checked_too() {
        let roots = [Path::new("/d/state"), Path::new("/d/secrets")];
        let low = below_floor_with(
            &roots,
            fake(&[
                ("/d/state", 1, 50 * GIB, 500 * GIB),
                ("/d/secrets", 2, GIB, 8 * GIB),
            ]),
        );
        assert_eq!(
            low,
            [Low {
                path: "/d/secrets".into(),
                free: GIB,
                total: 8 * GIB,
                floor: 2 * GIB
            }]
        );
    }

    /// The gate: one alert naming the path, what is free and the floor,
    /// and the run refused; nothing raised when there is room or when
    /// nothing could be measured.
    #[tokio::test]
    async fn the_gate_refuses_the_run_with_one_alert() {
        let root = crate::sealed::TestDir::new("space-gate");
        let sent = crate::alerts::testing::Recorder::default();
        let alerts = crate::alerts::testing::recording(root.path(), &sent);
        let roots = [Path::new("/d/state"), Path::new("/d/secrets")];
        gate_with(
            &roots,
            fake(&[("/d/state", 1, 50 * GIB, 500 * GIB)]),
            &alerts,
        )
        .unwrap();
        gate_with(&roots, fake(&[]), &alerts).unwrap();
        let refused = gate_with(
            &roots,
            fake(&[
                ("/d/state", 1, GIB, 100 * GIB),
                ("/d/secrets", 1, GIB, 100 * GIB),
            ]),
            &alerts,
        )
        .unwrap_err();
        assert_eq!(refused.0.len(), 1);
        alerts.flush(std::time::Duration::from_secs(5)).await;
        let mails = sent.sent();
        assert_eq!(mails.len(), 1, "{mails:?}");
        for said in [
            "disk_space_low",
            "/d/state: 1.00 GiB free of 100.00 GiB",
            "floor_bytes: 2147483648",
        ] {
            assert!(mails[0].contains(said), "{said}: {}", mails[0]);
        }
    }

    /// A root that can't be measured holds nothing back, and the other is
    /// still checked.
    #[test]
    fn a_failed_measurement_does_not_block_the_run() {
        let roots = [Path::new("/d/state"), Path::new("/d/secrets")];
        assert_eq!(below_floor_with(&roots, fake(&[])), []);
        let low = below_floor_with(&roots, fake(&[("/d/secrets", 2, 0, 8 * GIB)]));
        assert_eq!(low.len(), 1);
        assert_eq!(low[0].path, Path::new("/d/secrets"));
    }

    /// The real call: a directory here has a filesystem with a size, no
    /// more free than that; a directory not yet made is measured at its
    /// parent; two paths in one directory share a device.
    #[test]
    fn statvfs_measures_a_real_directory_and_a_missing_ones_parent() {
        let root = crate::sealed::TestDir::new("space");
        let here = stat(&root).unwrap();
        assert!(here.total > 0, "{here:?}");
        assert!(here.free <= here.total, "{here:?}");
        let missing = stat(&root.join("state").join("not-made-yet")).unwrap();
        assert_eq!(missing.dev, here.dev);
        assert_eq!(missing.total, here.total);
        assert!(stat(Path::new("has\0nul")).is_err());
    }
}
