// patterns.rs — Azure OpenAI pattern definitions
// Each pattern: (category, description, regex, migration_note)

use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    High,    // Must fix — functional breakage if not migrated
    Medium,  // Should fix — behavioral difference possible
    Low,     // Nice to fix — cosmetic or optional improvement
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::High => write!(f, "HIGH"),
            Severity::Medium => write!(f, "MEDIUM"),
            Severity::Low => write!(f, "LOW"),
        }
    }
}

pub struct PatternMatch {
    pub category: String,
    pub name: String,
    pub severity: Severity,
    pub migration_note: String,
    pub file: String,
    pub line: usize,
    pub snippet: String,
}

/// All Azure OpenAI patterns to scan for
pub static PATTERNS: &[(&str, &str, &str, Severity, &str)] = &[
    // ── Auth / Client Initialization ──────────────────────────────────────
    (
        "auth",
        "AzureOpenAI client init",
        r"AzureOpenAI\s*\(",
        Severity::High,
        "Replace with OpenAI(base_url=.../openai/v1, api_key=...) pointed at Bedrock's OpenAI-compatible endpoint, or a boto3 bedrock-runtime client (Converse API). See migration/01_chat_completions/.",
    ),
    (
        "auth",
        "Azure API key env var",
        r"AZURE_OPENAI_API_KEY",
        Severity::High,
        "Replace with AWS credentials (IAM role or AWS_ACCESS_KEY_ID/SECRET). No API key needed on Bedrock.",
    ),
    (
        "auth",
        "Azure endpoint env var",
        r"AZURE_OPENAI_ENDPOINT|azure_endpoint\s*=",
        Severity::High,
        "Replace with Bedrock endpoint (region-based, no custom URL needed).",
    ),
    (
        "auth",
        "Azure OpenAI import",
        r"from openai import AzureOpenAI|from openai\.lib\.azure import",
        Severity::High,
        "Replace with: from openai import OpenAI (point base_url at Bedrock's OpenAI-compatible endpoint) or import boto3 (Converse API).",
    ),
    (
        "auth",
        "Azure AD / Entra token auth",
        r"AzureAD|azure\.identity|DefaultAzureCredential|ClientSecretCredential",
        Severity::High,
        "Replace Azure AD auth with IAM role or assume_role. See migration/01_chat_completions/.",
    ),

    // ── Chat Completions ──────────────────────────────────────────────────
    (
        "completions",
        "chat.completions.create",
        r"\.chat\.completions\.create\s*\(",
        Severity::Medium,
        "Call shape is unchanged on Bedrock's OpenAI-compatible endpoint (Path A, minimal change: swap client init + auth). For the portable Path B, map to bedrock_client.converse(). See migration/01_chat_completions/.",
    ),
    (
        "completions",
        "Azure deployment name (model ID)",
        r#"model\s*=\s*["'][a-z0-9-]*gpt[a-z0-9-]*["']"#,
        Severity::Medium,
        "Azure uses deployment names; Bedrock uses model IDs. Map to a Bedrock model ID (e.g. 'us.openai.gpt-5.6-terra', an inference-profile id) — see the OpenAI-models-on-Bedrock page and docs/model-mapping.md.",
    ),
    (
        "completions",
        "engine parameter (legacy)",
        r#"engine\s*=\s*["']"#,
        Severity::Medium,
        "Legacy Azure parameter. Replace with model= and use Bedrock model ID.",
    ),

    // ── Embeddings ────────────────────────────────────────────────────────
    (
        "embeddings",
        "embeddings.create",
        r"\.embeddings\.create\s*\(",
        Severity::High,
        "Migrate to Bedrock Titan Text Embeddings v2 (1024d), or keep OpenAI embeddings via Bedrock's OpenAI-compatible endpoint. Note: dimension mismatch if switching models — re-embed existing vector stores. See migration/02_embeddings/.",
    ),
    (
        "embeddings",
        "text-embedding-ada-002",
        r"text-embedding-ada-002|text-embedding-3",
        Severity::Medium,
        "ada-002 = 1536 dimensions. Migrate to Amazon Titan Text Embeddings V2 (amazon.titan-embed-text-v2:0, 1024d — supports 256/512/1024). Re-indexing is mandatory: Titan V2 does not offer a 1536-d setting, so the dimensions differ from ada-002 and existing vectors must be re-embedded. See migration/02_embeddings/dimension-mapping.md.",
    ),

    // ── Function Calling / Tool Use ───────────────────────────────────────
    (
        "tools",
        "function calling (tools param)",
        r#"["']type["']\s*:\s*["']function["']"#,
        Severity::High,
        "Convert Azure function calling to Bedrock toolConfig format. See migration/03_tool_use/.",
    ),
    (
        "tools",
        "tool_choice parameter",
        r"tool_choice\s*=",
        Severity::Medium,
        "Map tool_choice to Bedrock toolChoice field. 'auto' → {'auto': {}}, 'required' → {'any': {}}.",
    ),
    (
        "tools",
        "web_search server tool",
        r#"["']type["']\s*:\s*["']web_search["']"#,
        Severity::Medium,
        "Bedrock now has a built-in Web Search tool: add {\"type\":\"web_search\"} on the OpenAI Responses API (GPT-5.x, bedrock-mantle endpoint) — near drop-in. Server-side tools are not available on bedrock-runtime, Converse, or InvokeModel; else use a client-side search tool. See migration/03_tool_use/server-tools.md.",
    ),
    (
        "tools",
        "code_interpreter server tool (Azure-only)",
        r#"["']type["']\s*:\s*["']code_interpreter["']"#,
        Severity::High,
        "Azure server-side tool. Replace with AgentCore Runtime sandbox or AWS Lambda-backed tool. See migration/03_tool_use/server-tools.md.",
    ),
    (
        "tools",
        "file_search server tool (Azure-only)",
        r#"["']type["']\s*:\s*["']file_search["']"#,
        Severity::High,
        "Azure server-side tool. Replace with Bedrock Knowledge Bases or Amazon OpenSearch. See migration/03_tool_use/server-tools.md.",
    ),

    // ── Agents ────────────────────────────────────────────────────────────
    (
        "agents",
        "Azure AI Agent Service",
        r"AIProjectClient|azure\.ai\.projects|AgentClient|azure_agent",
        Severity::High,
        "Migrate to Amazon Bedrock AgentCore + Strands Agents SDK. See migration/04_agents/azure_ai_agent_service/README.md.",
    ),
    (
        "agents",
        "LangChain AzureChatOpenAI",
        r"AzureChatOpenAI|AzureOpenAIEmbeddings",
        Severity::High,
        "Replace with ChatBedrock (LangChain) or Strands Agents + AgentCore Runtime. See migration/04_agents/langchain/README.md.",
    ),
    (
        "agents",
        "Semantic Kernel Azure connector",
        r"AzureChatCompletion|semantic_kernel.*azure",
        Severity::High,
        "Replace with Bedrock connector for Semantic Kernel or migrate to Strands. See migration/04_agents/semantic_kernel/README.md.",
    ),

    // ── Vector Search / AI Search ─────────────────────────────────────────
    (
        "vector_search",
        "Azure AI Search",
        r"SearchClient|azure\.search\.documents|AzureKeyCredential.*search",
        Severity::High,
        "Migrate index to Amazon OpenSearch Serverless or Bedrock Knowledge Bases. See migration/05_data_plane/ai-search-to-opensearch.md.",
    ),

    // ── Config / Misc ─────────────────────────────────────────────────────
    (
        "config",
        "api_version parameter",
        r#"api_version\s*=\s*["']20\d\d"#,
        Severity::Low,
        "Azure-specific API versioning. Not needed on Bedrock — remove this parameter.",
    ),
    (
        "config",
        "azure_deployment parameter",
        r"azure_deployment\s*=",
        Severity::Medium,
        "Azure-specific. Replace with Bedrock model ID.",
    ),
];

/// Compile all patterns into regexes (called once at startup)
pub fn compile_patterns() -> Vec<(usize, Regex)> {
    PATTERNS
        .iter()
        .enumerate()
        .filter_map(|(i, (_, _, pattern, _, _))| {
            Regex::new(pattern).ok().map(|r| (i, r))
        })
        .collect()
}
