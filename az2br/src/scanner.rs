// scanner.rs — walks a directory and runs patterns against each file

use crate::patterns::{compile_patterns, PatternMatch, PATTERNS};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use walkdir::WalkDir;

static SCANNABLE_EXTENSIONS: &[&str] = &["py", "js", "ts", "mjs", "cjs", "ipynb", "tf", "yaml", "yml", "json", "env"];

pub fn scan_path(root: &str) -> Result<Vec<PatternMatch>> {
    let compiled = compile_patterns();
    let mut findings: Vec<PatternMatch> = Vec::new();

    // Count files first for progress bar
    let file_count = WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && is_scannable(e.path()))
        .count();

    let pb = ProgressBar::new(file_count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && is_scannable(e.path()))
    {
        let path = entry.path();
        pb.set_message(path.file_name().unwrap_or_default().to_string_lossy().to_string());

        if let Ok(content) = std::fs::read_to_string(path) {
            for (line_num, line) in content.lines().enumerate() {
                for (pattern_idx, regex) in &compiled {
                    if regex.is_match(line) {
                        let (category, name, _, severity, migration_note) = &PATTERNS[*pattern_idx];
                        findings.push(PatternMatch {
                            category: category.to_string(),
                            name: name.to_string(),
                            severity: severity.clone(),
                            migration_note: migration_note.to_string(),
                            file: path.to_string_lossy().to_string(),
                            line: line_num + 1,
                            snippet: line.trim().to_string(),
                        });
                    }
                }
            }
        }
        pb.inc(1);
    }

    pb.finish_with_message("scan complete");
    Ok(findings)
}

fn is_scannable(path: &std::path::Path) -> bool {
    // Skip hidden dirs, node_modules, .venv, __pycache__, target
    let path_str = path.to_string_lossy();
    if path_str.contains("node_modules")
        || path_str.contains("/.venv/")
        || path_str.contains("/.git/")
        || path_str.contains("/__pycache__/")
        || path_str.contains("/target/")
        || path_str.contains("/.tox/")
    {
        return false;
    }

    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| SCANNABLE_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}
