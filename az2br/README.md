# az2br — Azure OpenAI to Amazon Bedrock CLI

Scan your codebase for Azure OpenAI patterns and generate a prioritized migration plan.

## Quick Start

```bash
# Install (once Cargo is available)
cargo install az2br

# Scan current directory
az2br scan .

# Scan a specific path, save Markdown report
az2br scan ./src --output markdown --file migration-report.md

# Scan and only show HIGH severity (CI-friendly, exits 1 if any HIGH found)
az2br scan . --high-only

# JSON output for programmatic processing
az2br scan . --output json
```

## What It Scans For

| Category | Patterns |
|----------|----------|
| **auth** | AzureOpenAI client init, AZURE_OPENAI_API_KEY, Azure AD/Entra auth |
| **completions** | chat.completions.create, Azure deployment names, engine parameter |
| **embeddings** | embeddings.create, text-embedding-ada-002 model references |
| **tools** | Function calling, tool_choice, Azure server-side tools (web_search, code_interpreter, file_search) |
| **agents** | Azure AI Agent Service, LangChain AzureChatOpenAI, Semantic Kernel Azure connector |
| **vector_search** | Azure AI Search (SearchClient) |
| **config** | api_version, azure_deployment parameters |

## Security Note

az2br output includes matched source lines verbatim. Do not pipe output to
untrusted log aggregators or shared systems without redacting code snippets.
A matched line may contain a hardcoded secret (for example an
`AZURE_OPENAI_API_KEY` assignment), and terminal, Markdown, and JSON reports all
reproduce that line as-is. Treat generated reports as potentially sensitive:
review before sharing, and keep them out of version control.

## Building from Source

```bash
cd az2br
cargo build --release
./target/release/az2br scan .
```

## Migration Guides

After scanning, see the `migration/` folder for step-by-step guides:

- `01_chat_completions/` — Azure completions → Bedrock OpenAI-compatible endpoint or Converse API
- `02_embeddings/` — Embedding migration and dimension mismatch handling
- `03_tool_use/` — Function calling and server-side tool migration
- `04_agents/` — Azure AI Agent Service → Bedrock AgentCore + Strands
- `05_data_plane/` — Azure AI Search → Amazon OpenSearch / Bedrock Knowledge Bases
