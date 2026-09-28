// reporter.rs — formats scan findings as Markdown or JSON

use crate::patterns::{PatternMatch, Severity};
use colored::*;
use std::collections::HashMap;

pub fn print_summary(findings: &[PatternMatch], path: &str) {
    if findings.is_empty() {
        println!("{}", "✅ No Azure OpenAI patterns found.".green().bold());
        return;
    }

    let high = findings.iter().filter(|f| f.severity == Severity::High).count();
    let medium = findings.iter().filter(|f| f.severity == Severity::Medium).count();
    let low = findings.iter().filter(|f| f.severity == Severity::Low).count();

    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", "  az2br scan — Migration Readiness Report".bold());
    println!("{}\n", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    println!("📁 Scanned: {}", path.cyan());
    println!("🔍 Total findings: {}", findings.len().to_string().bold());
    println!(
        "   {} {} {}",
        format!("● {} HIGH", high).red().bold(),
        format!("● {} MEDIUM", medium).yellow(),
        format!("● {} LOW", low).blue()
    );
    println!();

    // Group by category
    let mut by_category: HashMap<&str, Vec<&PatternMatch>> = HashMap::new();
    for f in findings {
        by_category.entry(f.category.as_str()).or_default().push(f);
    }

    let order = ["auth", "completions", "embeddings", "tools", "agents", "vector_search", "config"];
    for cat in &order {
        if let Some(items) = by_category.get(cat) {
            println!("{}", format!("▸ {} ({})", cat.to_uppercase(), items.len()).bold());
            for item in items {
                let sev = match item.severity {
                    Severity::High => format!("[{}]", item.severity).red().bold(),
                    Severity::Medium => format!("[{}]", item.severity).yellow(),
                    Severity::Low => format!("[{}]", item.severity).blue(),
                };
                println!(
                    "  {} {}:{} — {}",
                    sev,
                    item.file.cyan(),
                    item.line.to_string().dimmed(),
                    item.name
                );
                println!("      {}", item.snippet.dimmed());
                println!("      💡 {}", item.migration_note.italic());
                println!();
            }
        }
    }
}

pub fn to_markdown(findings: &[PatternMatch], path: &str) -> String {
    let mut md = String::new();
    md.push_str("# az2br Migration Readiness Report\n\n");
    md.push_str(&format!("**Scanned:** `{}`\n\n", path));

    let high = findings.iter().filter(|f| f.severity == Severity::High).count();
    let medium = findings.iter().filter(|f| f.severity == Severity::Medium).count();
    let low = findings.iter().filter(|f| f.severity == Severity::Low).count();

    md.push_str(&format!(
        "**Total findings:** {} — 🔴 {} HIGH · 🟠 {} MEDIUM · 🔵 {} LOW\n\n",
        findings.len(), high, medium, low
    ));

    md.push_str("---\n\n");

    let mut by_category: HashMap<&str, Vec<&PatternMatch>> = HashMap::new();
    for f in findings {
        by_category.entry(f.category.as_str()).or_default().push(f);
    }

    let order = ["auth", "completions", "embeddings", "tools", "agents", "vector_search", "config"];
    for cat in &order {
        if let Some(items) = by_category.get(cat) {
            md.push_str(&format!("## {} ({})\n\n", cat.to_uppercase(), items.len()));
            md.push_str("| Severity | File | Line | Pattern | Migration Note |\n");
            md.push_str("|----------|------|------|---------|----------------|\n");
            for item in items {
                let sev_icon = match item.severity {
                    Severity::High => "🔴",
                    Severity::Medium => "🟠",
                    Severity::Low => "🔵",
                };
                md.push_str(&format!(
                    "| {} {} | `{}` | {} | {} | {} |\n",
                    sev_icon, item.severity, item.file, item.line, item.name, item.migration_note
                ));
            }
            md.push('\n');
        }
    }

    md
}

pub fn to_json(findings: &[PatternMatch]) -> String {
    use serde_json::{json, Value};
    let items: Vec<Value> = findings.iter().map(|f| json!({
        "category": f.category,
        "name": f.name,
        "severity": f.severity.to_string(),
        "file": f.file,
        "line": f.line,
        "snippet": f.snippet,
        "migration_note": f.migration_note
    })).collect();
    serde_json::to_string_pretty(&json!({ "findings": items, "total": findings.len() }))
        .unwrap_or_default()
}
