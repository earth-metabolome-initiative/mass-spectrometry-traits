use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use mass_spectrometry::fuzzing::{run_linear_entropy_case, LinearEntropyHarnessOutcome};

const ARTIFACTS_DIR: &str = "fuzz/artifacts/linear_entropy";
const FIXTURE_CRASH_DIR: &str = "tests/fixtures/fuzz/linear_entropy/crashes";

#[test]
fn replay_all_linear_entropy_crashes() {
    let artifacts_dir = Path::new(ARTIFACTS_DIR);
    let fixture_dir = Path::new(FIXTURE_CRASH_DIR);
    fs::create_dir_all(fixture_dir).expect("failed to create fixture crash directory");

    if artifacts_dir.is_dir() {
        copy_crash_files(artifacts_dir, fixture_dir)
            .expect("failed to copy crashes from artifacts into fixture directory");
    }

    let mut crash_files =
        collect_files_recursive(fixture_dir).expect("failed to scan fixture crash directory");
    crash_files.sort();

    for crash_file in &crash_files {
        let bytes = fs::read(crash_file).unwrap_or_else(|error| {
            panic!(
                "failed to read crash file `{}`: {error}",
                crash_file.display()
            )
        });
        let outcome = run_linear_entropy_case(&bytes);
        assert_eq!(
            outcome,
            LinearEntropyHarnessOutcome::Checked,
            "unexpected replay outcome for `{}`",
            crash_file.display()
        );
    }

    eprintln!(
        "replayed {} crash case(s) from `{}`",
        crash_files.len(),
        fixture_dir.display()
    );
}

fn copy_crash_files(source_root: &Path, destination_root: &Path) -> io::Result<()> {
    for source_file in collect_files_recursive(source_root)? {
        let relative = source_file
            .strip_prefix(source_root)
            .expect("source file should be under source root");
        let destination = destination_root.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&source_file, &destination)?;
    }
    Ok(())
}

fn collect_files_recursive(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !root.exists() {
        return Ok(files);
    }
    collect_files_recursive_inner(root, &mut files)?;
    Ok(files)
}

fn collect_files_recursive_inner(root: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive_inner(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}
