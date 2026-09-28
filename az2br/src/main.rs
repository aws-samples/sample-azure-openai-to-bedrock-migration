// main.rs — az2br CLI entry point

mod patterns;
mod reporter;
mod scanner;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::fs;

#[derive(Parser)]
#[command(
    name = "az2br",
    about = "Azure OpenAI → Amazon Bedrock migration toolkit",
    version = "0.1.0",
    long_about = "Scan your codebase for Azure OpenAI patterns and generate a prioritized migration plan.\n\nSee https://github.com/aws-samples/sample-azure-openai-to-bedrock-migration for full documentation."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan a directory for Azure OpenAI patterns
    Scan {
        /// Path to scan (default: current directory)
        #[arg(default_value = ".")]
        path: String,

        /// Output format: terminal (default), markdown, json
        #[arg(short, long, default_value = "terminal")]
        output: String,

        /// Save report to a file
        #[arg(short = 'f', long)]
        file: Option<String>,

        /// Only show HIGH severity findings
        #[arg(long)]
        high_only: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Scan { path, output, file, high_only } => {
            println!("🔍 Scanning {} ...\n", path);

            let mut findings = scanner::scan_path(&path)?;

            if high_only {
                findings.retain(|f| f.severity == patterns::Severity::High);
            }

            match output.as_str() {
                "terminal" | "term" => {
                    reporter::print_summary(&findings, &path);
                }
                "markdown" | "md" => {
                    let md = reporter::to_markdown(&findings, &path);
                    if let Some(ref out_file) = file {
                        fs::write(out_file, &md)?;
                        println!("✅ Report saved to {}", out_file);
                    } else {
                        println!("{}", md);
                    }
                }
                "json" => {
                    let json = reporter::to_json(&findings);
                    if let Some(ref out_file) = file {
                        fs::write(out_file, &json)?;
                        println!("✅ Report saved to {}", out_file);
                    } else {
                        println!("{}", json);
                    }
                }
                _ => {
                    eprintln!("Unknown output format '{}'. Use: terminal, markdown, json", output);
                    std::process::exit(1);
                }
            }

            // Exit with non-zero if HIGH findings exist (useful for CI)
            let high_count = findings.iter().filter(|f| f.severity == patterns::Severity::High).count();
            if high_count > 0 {
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
