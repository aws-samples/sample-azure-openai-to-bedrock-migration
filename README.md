# Azure OpenAI to Amazon Bedrock Migration Playbook

[![License: MIT-0](https://img.shields.io/badge/License-MIT--0-blue.svg)](LICENSE)
[![Contributing](https://img.shields.io/badge/contributing-guidelines-green)](CONTRIBUTING.md)

A field playbook and open-source toolkit for migrating Azure OpenAI workloads to Amazon Bedrock — **keeping your existing GPT models** while gaining AWS enterprise controls, platform portability, and the freedom to upgrade or swap models with a config change instead of a code rewrite.

> AWS hosts GPT models natively on Bedrock via the OpenAI partnership. You don't have to change your model — you change *where it runs*.

---

## Why This Playbook

Azure OpenAI applications are tightly coupled to Azure-managed endpoints, authentication models, and regional constraints. Every model upgrade, cost comparison, or provider swap is expensive engineering work.

**This playbook gives you:**

- `az2br` — a Rust CLI that scans your codebase for Azure OpenAI patterns and generates a prioritized migration report
- Step-by-step migration guides covering chat completions, embeddings, function calling, agents, and vector search
- A regression test harness to validate parity between Azure and Bedrock before cutover
- Real before/after code — not pseudocode

Once on Bedrock's Converse API or AgentCore, model upgrades and A/B tests become config changes, not code rewrites.

---

## Repository Layout

```
sample-azure-openai-to-bedrock-migration/
│
├── az2br/                          # Rust CLI scanner and migration toolkit
│   ├── src/
│   │   ├── main.rs                 # CLI entry point (scan command, output flags)
│   │   ├── patterns.rs             # 22 Azure OAI patterns across 7 categories
│   │   ├── scanner.rs              # File walker with progress bar
│   │   └── reporter.rs             # Terminal, Markdown, and JSON output
│   ├── Cargo.toml
│   └── README.md                   # az2br quick start and usage
│
├── migration/                      # Step-by-step migration guides (Python examples)
│   ├── 01_chat_completions/        # Azure completions → Bedrock OpenAI-compatible endpoint / Converse API
│   ├── 02_embeddings/              # Embedding migration + dimension mismatch handling
│   ├── 03_tool_use/                # Function calling + Azure server tools → Bedrock
│   ├── 04_agents/                  # Azure AI Agent Service → Bedrock AgentCore + Strands
│   │   ├── azure_ai_agent_service/
│   │   ├── langchain/
│   │   ├── autogen/
│   │   └── semantic_kernel/
│   └── 05_data_plane/              # Azure AI Search → OpenSearch / Bedrock Knowledge Bases
│
├── harness/                        # Regression test harness (record / replay / diff)
│
├── docs/
│   ├── model-mapping.md            # GPT → Bedrock model ID mapping table
│   ├── api-mapping.md              # Azure OAI API surface → Bedrock equivalents
│   └── tool-mapping.md             # Azure server tools → Bedrock equivalents
│
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── LICENSE
```

---

## Quick Start — `az2br scan`

```bash
# Build from source (requires Rust)
cd az2br
cargo build --release

# Scan your codebase
./target/release/az2br scan /path/to/your/app

# Save a Markdown migration report
./target/release/az2br scan ./src --output markdown --file migration-report.md

# CI mode — exits with code 1 if HIGH severity findings exist
./target/release/az2br scan . --high-only
```

**Example terminal output:**

```
🔍 Scanning ./src ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  az2br scan — Migration Readiness Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Scanned: ./src
🔍 Total findings: 8 — ● 5 HIGH  ● 2 MEDIUM  ● 1 LOW

▸ AUTH (3)
  [HIGH] app/client.py:12 — AzureOpenAI client init
      client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")...
      💡 Replace with OpenAI SDK pointed at Bedrock's OpenAI-compatible endpoint, or boto3 bedrock-runtime.

▸ TOOLS (2)
  [MEDIUM] app/agent.py:45 — web_search server tool
      {"type": "web_search"}
      💡 Bedrock has a built-in Web Search tool (OpenAI Responses API, GPT-5.x) — near drop-in. See migration/03_tool_use/server-tools.md.
  ...
```

> After building, run `source ~/.cargo/env` if `cargo` / `az2br` aren't on your
> `PATH` yet (fresh Rust install in the same shell).

---

## Prerequisites & Setup

**Rust (for `az2br`):** install via [rustup](https://rustup.rs/); then, in the
same shell, `source ~/.cargo/env` so `cargo` is on your `PATH`.

**Python (for the migration examples and the parity harness):** the Python
dependencies live in [`harness/requirements.txt`](harness/requirements.txt) —
there is **no** root `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r harness/requirements.txt   # openai, boto3, aws-bedrock-token-generator, pytest
```

**Run the offline harness tests:**

```bash
python -m pytest -q harness/
```

**Verify the `after/` (Bedrock) migration examples run live** against your
account with real inference calls:

```bash
python harness/smoke_test_after.py
```

See [`harness/README.md`](harness/README.md) for the full record / replay /
diff workflow. Note that `--cases` is a **top-level** flag on `harness.py` and
must appear **before** the subcommand, e.g.:

```bash
python harness/harness.py --cases my_cases.jsonl replay --provider bedrock-converse --label b
```

---

## What `az2br` Scans For

| Category | Patterns detected |
|----------|-------------------|
| **auth** | `AzureOpenAI` client init, `AZURE_OPENAI_API_KEY`, Azure AD/Entra credentials |
| **completions** | `chat.completions.create`, Azure deployment names, `engine=` parameter |
| **embeddings** | `embeddings.create`, `text-embedding-ada-002` model references |
| **tools** | Function calling, `tool_choice`, Azure server tools (`web_search`, `code_interpreter`, `file_search`) |
| **agents** | Azure AI Agent Service, `AzureChatOpenAI`, Semantic Kernel Azure connector |
| **vector_search** | `SearchClient`, `azure.search.documents` |
| **config** | `api_version=`, `azure_deployment=` parameters |

---

## Migration Guides

After scanning, follow the step-by-step guides in `migration/`:

| Guide | What it covers |
|-------|---------------|
| `01_chat_completions/` | Two paths: keep the OpenAI SDK pointed at Bedrock's OpenAI-compatible endpoint (smallest diff), or rewrite to the portable Converse API. Model mapping table included. |
| `02_embeddings/` | Move to Titan Text Embeddings v2 or keep OpenAI embeddings via Bedrock. Handles dimension mismatch (ada-002 1536d → Titan v2 1024d). |
| `03_tool_use/` | Convert Azure `tools` + `tool_choice` to Bedrock `toolConfig`. Maps Azure-only server tools to Bedrock equivalents. |
| `04_agents/` | Migrate Azure AI Agent Service, LangChain-Azure, AutoGen, and Semantic Kernel to Bedrock AgentCore + Strands Agents SDK. |
| `05_data_plane/` | Migrate Azure AI Search indexes to Amazon OpenSearch or Bedrock Knowledge Bases. |

---

## Contributing

We welcome contributions — new patterns, migration examples, and harness improvements. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Security

For responsible disclosure of security issues, see [SECURITY.md](SECURITY.md).

---

## License

This project is licensed under the [MIT-0 License](LICENSE) — you can use, copy, modify, and distribute this code without restriction, including in commercial products, without requiring attribution.
