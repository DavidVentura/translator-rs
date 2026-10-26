use std::collections::BTreeSet;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::catalog::{DeletePlan, DownloadPlan, DownloadTask, TaskInstall, write_install_marker};

const STAGING_DIR: &str = ".staging";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
pub enum DownloadEncoding {
    Identity,
    Gzip,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
pub enum StageTarget {
    File {
        path: String,
        encoding: DownloadEncoding,
    },
    Zip {
        archive_path: String,
        unpack_dir: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Error))]
pub enum InstallError {
    Io {
        path: String,
        reason: String,
    },
    UnknownTask {
        install_path: String,
    },
    NotStaged {
        install_path: String,
    },
    ArchiveLayout {
        install_path: String,
        reason: String,
    },
    InstallRootChanged {
        staged_for: String,
        current: String,
    },
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, reason } => write!(f, "{path}: {reason}"),
            Self::UnknownTask { install_path } => {
                write!(f, "{install_path} is not part of this installation")
            }
            Self::NotStaged { install_path } => write!(f, "{install_path} was not staged"),
            Self::ArchiveLayout {
                install_path,
                reason,
            } => write!(f, "{install_path}: {reason}"),
            Self::InstallRootChanged {
                staged_for,
                current,
            } => write!(
                f,
                "installation was staged for {staged_for} but the catalog now installs into {current}"
            ),
        }
    }
}

impl std::error::Error for InstallError {}

fn io_error(path: &Path, error: std::io::Error) -> InstallError {
    InstallError::Io {
        path: path.display().to_string(),
        reason: error.to_string(),
    }
}

// A plan's files land in a private staging directory first and move into the
// install root together on commit: the catalog reads a pack with only some of
// its files as needing repair, so packs can't land one by one. Hosts only fetch
// bytes into the targets handed out here.
pub struct Installation {
    install_root: PathBuf,
    staging_root: PathBuf,
    plan: DownloadPlan,
    staged: Mutex<BTreeSet<String>>,
}

impl Installation {
    pub(crate) fn begin(install_root: &Path, plan: DownloadPlan) -> Result<Self, InstallError> {
        let staging_root = create_unique_dir(&install_root.join(STAGING_DIR))?;
        Ok(Self {
            install_root: install_root.to_path_buf(),
            staging_root,
            plan,
            staged: Mutex::new(BTreeSet::new()),
        })
    }

    pub(crate) fn install_root(&self) -> &Path {
        &self.install_root
    }

    pub(crate) fn delete_after(&self) -> &DeletePlan {
        &self.plan.delete_after
    }

    pub fn stage_target(&self, task: &DownloadTask) -> Result<StageTarget, InstallError> {
        let task = self.planned(&task.install_path)?;
        let file = self.staged_file(&task.install_path);
        create_parent_dir(&file)?;
        let path = file.display().to_string();
        let target = match &task.install {
            TaskInstall::Raw => StageTarget::File {
                path,
                encoding: DownloadEncoding::Identity,
            },
            TaskInstall::Gunzip => StageTarget::File {
                path,
                encoding: DownloadEncoding::Gzip,
            },
            TaskInstall::Zip { .. } => {
                let unpack_dir = self.unpack_dir(&task.install_path);
                fs::create_dir_all(&unpack_dir).map_err(|e| io_error(&unpack_dir, e))?;
                StageTarget::Zip {
                    archive_path: path,
                    unpack_dir: unpack_dir.display().to_string(),
                }
            }
        };
        Ok(target)
    }

    pub fn mark_staged(&self, task: &DownloadTask) -> Result<(), InstallError> {
        let task = self.planned(&task.install_path)?;
        if !self.staged_file(&task.install_path).is_file() {
            return Err(InstallError::NotStaged {
                install_path: task.install_path.clone(),
            });
        }
        if let TaskInstall::Zip { archive } = &task.install {
            check_archive_layout(
                &self.unpack_dir(&task.install_path),
                &archive.root_dir,
                &task.install_path,
            )?;
        }
        self.staged
            .lock()
            .expect("staged set lock poisoned")
            .insert(task.install_path.clone());
        Ok(())
    }

    // An archive replaces only the entries it brings under its root directory:
    // other packs install files next to them (espeak dictionaries share the
    // espeak data directory). Staged plain files move after the archives, and
    // markers go last, so an archive only reads as installed once every file
    // is in place.
    pub(crate) fn move_into_place(&self) -> Result<(), InstallError> {
        let staged = std::mem::take(&mut *self.staged.lock().expect("staged set lock poisoned"));
        let tasks = self
            .plan
            .tasks
            .iter()
            .filter(|task| staged.contains(&task.install_path))
            .collect::<Vec<_>>();

        for task in &tasks {
            let TaskInstall::Zip { archive } = &task.install else {
                continue;
            };
            let source_root = self.unpack_dir(&task.install_path).join(&archive.root_dir);
            let target_root = self.install_root.join(archive.install_dir());
            fs::create_dir_all(&target_root).map_err(|e| io_error(&target_root, e))?;
            for entry in fs::read_dir(&source_root).map_err(|e| io_error(&source_root, e))? {
                let entry = entry.map_err(|e| io_error(&source_root, e))?;
                let target = target_root.join(entry.file_name());
                remove_path(&target)?;
                fs::rename(entry.path(), &target).map_err(|e| io_error(&target, e))?;
            }
        }

        for task in &tasks {
            if let TaskInstall::Zip { archive } = &task.install
                && archive.delete_after_extract
            {
                continue;
            }
            let target = self.install_root.join(&task.install_path);
            create_parent_dir(&target)?;
            fs::rename(self.staged_file(&task.install_path), &target)
                .map_err(|e| io_error(&target, e))?;
        }

        for task in &tasks {
            if let TaskInstall::Zip { archive } = &task.install {
                write_install_marker(&self.install_root, &archive.marker)
                    .map_err(|e| io_error(&self.install_root.join(&archive.marker.path), e))?;
            }
        }
        Ok(())
    }

    fn planned(&self, install_path: &str) -> Result<&DownloadTask, InstallError> {
        self.plan
            .tasks
            .iter()
            .find(|task| task.install_path == install_path)
            .ok_or_else(|| InstallError::UnknownTask {
                install_path: install_path.to_string(),
            })
    }

    fn staged_file(&self, install_path: &str) -> PathBuf {
        self.staging_root.join("files").join(install_path)
    }

    fn unpack_dir(&self, install_path: &str) -> PathBuf {
        self.staging_root.join("unpacked").join(install_path)
    }
}

impl Drop for Installation {
    fn drop(&mut self) {
        if let Err(error) = remove_path(&self.staging_root) {
            log::warn!("failed to discard staged installation: {error}");
        }
    }
}

// Staging left behind by a killed process is never committed. Only safe before
// this process begins any installation.
pub(crate) fn discard_abandoned(install_root: &Path) -> Result<(), InstallError> {
    remove_path(&install_root.join(STAGING_DIR))
}

fn check_archive_layout(
    unpack_dir: &Path,
    root_dir: &str,
    install_path: &str,
) -> Result<(), InstallError> {
    let mut names = fs::read_dir(unpack_dir)
        .map_err(|e| io_error(unpack_dir, e))?
        .map(|entry| {
            entry
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                .map_err(|e| io_error(unpack_dir, e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    names.sort();
    if names != [root_dir] || !unpack_dir.join(root_dir).is_dir() {
        return Err(InstallError::ArchiveLayout {
            install_path: install_path.to_string(),
            reason: format!("expected only {root_dir}/ at the top level, found {names:?}"),
        });
    }
    Ok(())
}

fn create_unique_dir(parent: &Path) -> Result<PathBuf, InstallError> {
    fs::create_dir_all(parent).map_err(|e| io_error(parent, e))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after 1970")
        .as_nanos();
    let mut attempt = 0u32;
    loop {
        let dir = parent.join(format!("{}-{stamp}-{attempt}", std::process::id()));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => attempt += 1,
            Err(e) => return Err(io_error(&dir, e)),
        }
    }
}

fn create_parent_dir(path: &Path) -> Result<(), InstallError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(parent).map_err(|e| io_error(parent, e))
}

fn remove_path(path: &Path) -> Result<(), InstallError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_error(path, e)),
    };
    let removed = if metadata.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    removed.map_err(|e| io_error(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{
        FsPackInstallChecker, InstallMarker, ZipArchive, download_task_installed,
    };

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new() -> Self {
            Self(create_unique_dir(&std::env::temp_dir().join("translator-install-tests")).unwrap())
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            remove_path(&self.0).unwrap();
        }
    }

    fn espeak_core() -> DownloadTask {
        DownloadTask {
            pack_id: "tts-espeak-core".to_string(),
            install_path: "bin/espeak-ng-data.zip".to_string(),
            url: "https://example.invalid/espeak-ng-data.zip".to_string(),
            size_bytes: 1,
            install: TaskInstall::Zip {
                archive: ZipArchive {
                    extract_to: "bin".to_string(),
                    root_dir: "espeak-ng-data".to_string(),
                    delete_after_extract: true,
                    marker: InstallMarker {
                        path: "bin/espeak-ng-data/.install-info.json".to_string(),
                        version: 2,
                    },
                },
            },
        }
    }

    fn raw(install_path: &str) -> DownloadTask {
        DownloadTask {
            pack_id: "tts-espeak-dict".to_string(),
            install_path: install_path.to_string(),
            url: "https://example.invalid/raw".to_string(),
            size_bytes: 1,
            install: TaskInstall::Raw,
        }
    }

    fn plan(tasks: Vec<DownloadTask>) -> DownloadPlan {
        DownloadPlan {
            total_size: tasks.len() as u64,
            tasks,
            delete_after: DeletePlan::default(),
        }
    }

    fn write(path: impl AsRef<Path>, contents: &str) {
        create_parent_dir(path.as_ref()).unwrap();
        fs::write(path, contents).unwrap();
    }

    fn stage_zip(installation: &Installation, task: &DownloadTask, entries: &[(&str, &str)]) {
        let StageTarget::Zip {
            archive_path,
            unpack_dir,
        } = installation.stage_target(task).unwrap()
        else {
            panic!("zip task staged as a file");
        };
        write(&archive_path, "zip bytes");
        for (name, contents) in entries {
            write(Path::new(&unpack_dir).join(name), contents);
        }
    }

    fn stage_file(installation: &Installation, task: &DownloadTask, contents: &str) {
        let StageTarget::File { path, encoding } = installation.stage_target(task).unwrap() else {
            panic!("file task staged as a zip");
        };
        assert_eq!(encoding, DownloadEncoding::Identity);
        write(path, contents);
    }

    #[test]
    fn archive_replaces_only_its_own_entries_and_reads_installed_after_commit() {
        let root = TempRoot::new();
        write(root.0.join("bin/espeak-ng-data/phontab"), "old");
        write(root.0.join("bin/espeak-ng-data/lang/stale"), "old");
        write(
            root.0.join("bin/espeak-ng-data/de_dict"),
            "installed by another pack",
        );
        let core = espeak_core();
        let dict = raw("bin/espeak-ng-data/ko_dict");
        let installation =
            Installation::begin(&root.0, plan(vec![core.clone(), dict.clone()])).unwrap();

        stage_zip(
            &installation,
            &core,
            &[
                ("espeak-ng-data/phontab", "new"),
                ("espeak-ng-data/lang/ko", "new"),
            ],
        );
        stage_file(&installation, &dict, "ko");
        installation.mark_staged(&core).unwrap();
        installation.mark_staged(&dict).unwrap();
        let checker = FsPackInstallChecker::new(&root.0);
        assert!(!download_task_installed(&checker, &core));

        installation.move_into_place().unwrap();

        let data = root.0.join("bin/espeak-ng-data");
        assert_eq!(fs::read_to_string(data.join("phontab")).unwrap(), "new");
        assert_eq!(fs::read_to_string(data.join("lang/ko")).unwrap(), "new");
        assert!(!data.join("lang/stale").exists());
        assert_eq!(
            fs::read_to_string(data.join("de_dict")).unwrap(),
            "installed by another pack"
        );
        assert_eq!(fs::read_to_string(data.join("ko_dict")).unwrap(), "ko");
        assert!(!root.0.join("bin/espeak-ng-data.zip").exists());
        assert!(download_task_installed(&checker, &core));
    }

    #[test]
    fn archive_with_entries_outside_its_root_dir_is_rejected() {
        let root = TempRoot::new();
        let core = espeak_core();
        let installation = Installation::begin(&root.0, plan(vec![core.clone()])).unwrap();

        stage_zip(
            &installation,
            &core,
            &[("phontab", "contents at the top level")],
        );

        assert!(matches!(
            installation.mark_staged(&core),
            Err(InstallError::ArchiveLayout { .. })
        ));
    }

    #[test]
    fn only_staged_tasks_move_and_unplanned_tasks_are_refused() {
        let root = TempRoot::new();
        let fetched = raw("bin/a");
        let skipped = raw("bin/b");
        let installation =
            Installation::begin(&root.0, plan(vec![fetched.clone(), skipped.clone()])).unwrap();

        assert!(matches!(
            installation.mark_staged(&skipped),
            Err(InstallError::NotStaged { .. })
        ));
        assert!(matches!(
            installation.stage_target(&raw("bin/c")),
            Err(InstallError::UnknownTask { .. })
        ));
        stage_file(&installation, &fetched, "a");
        installation.mark_staged(&fetched).unwrap();
        installation.move_into_place().unwrap();

        assert_eq!(fs::read_to_string(root.0.join("bin/a")).unwrap(), "a");
        assert!(!root.0.join("bin/b").exists());
    }

    #[test]
    fn dropping_an_installation_discards_its_staging() {
        let root = TempRoot::new();
        let task = raw("bin/a");
        let installation = Installation::begin(&root.0, plan(vec![task.clone()])).unwrap();
        stage_file(&installation, &task, "a");

        drop(installation);

        assert_eq!(fs::read_dir(root.0.join(STAGING_DIR)).unwrap().count(), 0);
        assert!(!root.0.join("bin/a").exists());
    }
}
