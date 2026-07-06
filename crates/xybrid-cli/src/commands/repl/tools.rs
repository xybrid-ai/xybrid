//! Built-in tools for the REPL's tool-calling loop.
//!
//! Three tools ship with the REPL: `web_search` (pluggable provider,
//! keyless Wikipedia by default), `fetch_url` (SSRF-gated page fetch), and
//! `current_time` (local clock, no network). Tool definitions ride
//! `GenerationConfig::with_tools`; execution happens here, in app code —
//! the model only ever sees the JSON results.

use std::io::Read;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::time::Duration;

use xybrid_core::gateway::{Tool, ToolCall};
use xybrid_core::ir::ToolCallResult;

/// Snippets to keep per search. More context helps synthesis but a small
/// model's context window fills fast.
pub(crate) const RESULTS_PER_SEARCH: usize = 4;
/// Cap on the lead-article summary a search returns.
const SUMMARY_CHAR_CAP: usize = 900;
/// Cap on the text handed back from `fetch_url` (characters, post-strip).
const FETCH_CHAR_CAP: usize = 4_000;
/// Cap on the raw bytes read from a fetched page. Generous because
/// script/style blocks and navigation chrome are stripped before the
/// character cap applies.
const FETCH_BYTE_CAP: u64 = 512 * 1024;
const SEARCH_TIMEOUT: Duration = Duration::from_secs(20);
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

const USER_AGENT: &str = concat!(
    "xybrid-cli/",
    env!("CARGO_PKG_VERSION"),
    " (+https://xybrid.dev)"
);

// =============================================================================
// Search providers
// =============================================================================

/// Where `web_search` sends its queries.
///
/// Wikipedia is the default because it needs no key and does not bot-block;
/// keyed providers give open-web coverage. All providers return the same
/// shape, so the loop and the model are provider-agnostic.
#[derive(Debug, Clone)]
pub(crate) enum SearchProvider {
    Wikipedia,
    Tavily { key: String },
    Brave { key: String },
}

/// One search hit, provider-normalized.
struct SearchHit {
    title: String,
    snippet: String,
    url: Option<String>,
}

struct SearchResults {
    /// Lead summary / direct answer, when the provider has one.
    summary: Option<String>,
    hits: Vec<SearchHit>,
}

impl SearchProvider {
    /// Resolve the provider from the environment.
    ///
    /// `XYBRID_SEARCH_PROVIDER` selects (`wikipedia` | `tavily` | `brave`);
    /// keyed providers read `TAVILY_API_KEY` / `BRAVE_API_KEY`. Anything
    /// missing or unknown falls back to Wikipedia, with a warning string the
    /// caller can surface.
    pub(crate) fn from_env() -> (Self, Option<String>) {
        Self::resolve(
            std::env::var("XYBRID_SEARCH_PROVIDER").ok().as_deref(),
            std::env::var("TAVILY_API_KEY").ok().as_deref(),
            std::env::var("BRAVE_API_KEY").ok().as_deref(),
        )
    }

    fn resolve(
        provider: Option<&str>,
        tavily_key: Option<&str>,
        brave_key: Option<&str>,
    ) -> (Self, Option<String>) {
        fn non_empty(v: Option<&str>) -> Option<&str> {
            v.map(str::trim).filter(|s| !s.is_empty())
        }
        match non_empty(provider).map(str::to_ascii_lowercase).as_deref() {
            None | Some("wikipedia") => (Self::Wikipedia, None),
            Some("tavily") => match non_empty(tavily_key) {
                Some(key) => (Self::Tavily { key: key.to_string() }, None),
                None => (
                    Self::Wikipedia,
                    Some("XYBRID_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is not set; using wikipedia".to_string()),
                ),
            },
            Some("brave") => match non_empty(brave_key) {
                Some(key) => (Self::Brave { key: key.to_string() }, None),
                None => (
                    Self::Wikipedia,
                    Some("XYBRID_SEARCH_PROVIDER=brave but BRAVE_API_KEY is not set; using wikipedia".to_string()),
                ),
            },
            Some(other) => (
                Self::Wikipedia,
                Some(format!(
                    "unknown XYBRID_SEARCH_PROVIDER '{other}' (expected wikipedia|tavily|brave); using wikipedia"
                )),
            ),
        }
    }

    /// Short label for banners and `/tools` output.
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Wikipedia => "wikipedia (keyless)",
            Self::Tavily { .. } => "tavily",
            Self::Brave { .. } => "brave",
        }
    }

    fn search(&self, query: &str, max: usize) -> Result<SearchResults, String> {
        match self {
            Self::Wikipedia => wikipedia_search(query, max),
            Self::Tavily { key } => tavily_search(key, query, max),
            Self::Brave { key } => brave_search(key, query, max),
        }
    }
}

/// MediaWiki full-text search plus the top article's REST summary extract
/// (richer than the search snippet, and usually where the answer sits).
fn wikipedia_search(query: &str, max: usize) -> Result<SearchResults, String> {
    let agent = ureq::builder().timeout(SEARCH_TIMEOUT).build();

    let search: serde_json::Value = agent
        .get("https://en.wikipedia.org/w/api.php")
        .query("action", "query")
        .query("list", "search")
        .query("srsearch", query)
        .query("srlimit", &max.to_string())
        .query("format", "json")
        .set("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| format!("search request failed: {e}"))?
        .into_json()
        .map_err(|e| format!("search parse failed: {e}"))?;

    let hits: Vec<SearchHit> = search["query"]["search"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|hit| {
            let title = hit["title"].as_str()?.to_string();
            let snippet = strip_html(hit["snippet"].as_str()?);
            let url = Some(format!(
                "https://en.wikipedia.org/wiki/{}",
                title.replace(' ', "_")
            ));
            Some(SearchHit {
                title,
                snippet,
                url,
            })
        })
        .collect();

    let summary = hits
        .first()
        .and_then(|hit| wikipedia_summary(&agent, &hit.title));

    Ok(SearchResults { summary, hits })
}

/// The Wikipedia REST summary extract for `title` (best-effort — a missing
/// page or transport error just yields no summary).
fn wikipedia_summary(agent: &ureq::Agent, title: &str) -> Option<String> {
    let url = format!(
        "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
        title.replace(' ', "_")
    );
    let value: serde_json::Value = agent
        .get(&url)
        .set("User-Agent", USER_AGENT)
        .call()
        .ok()?
        .into_json()
        .ok()?;
    value["extract"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(SUMMARY_CHAR_CAP).collect())
}

fn tavily_search(key: &str, query: &str, max: usize) -> Result<SearchResults, String> {
    let response: serde_json::Value = ureq::builder()
        .timeout(SEARCH_TIMEOUT)
        .build()
        .post("https://api.tavily.com/search")
        .set("User-Agent", USER_AGENT)
        .send_json(serde_json::json!({
            "api_key": key,
            "query": query,
            "max_results": max,
            "include_answer": true,
        }))
        .map_err(|e| format!("tavily request failed: {e}"))?
        .into_json()
        .map_err(|e| format!("tavily parse failed: {e}"))?;

    let hits = response["results"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|hit| {
            Some(SearchHit {
                title: hit["title"].as_str()?.to_string(),
                snippet: hit["content"].as_str().unwrap_or_default().to_string(),
                url: hit["url"].as_str().map(str::to_string),
            })
        })
        .collect();

    let summary = response["answer"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(SUMMARY_CHAR_CAP).collect());

    Ok(SearchResults { summary, hits })
}

fn brave_search(key: &str, query: &str, max: usize) -> Result<SearchResults, String> {
    let response: serde_json::Value = ureq::builder()
        .timeout(SEARCH_TIMEOUT)
        .build()
        .get("https://api.search.brave.com/res/v1/web/search")
        .query("q", query)
        .query("count", &max.to_string())
        .set("Accept", "application/json")
        .set("X-Subscription-Token", key)
        .set("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| format!("brave request failed: {e}"))?
        .into_json()
        .map_err(|e| format!("brave parse failed: {e}"))?;

    let hits = response["web"]["results"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|hit| {
            Some(SearchHit {
                title: hit["title"].as_str()?.to_string(),
                snippet: strip_html(hit["description"].as_str().unwrap_or_default()),
                url: hit["url"].as_str().map(str::to_string),
            })
        })
        .collect();

    Ok(SearchResults {
        summary: None,
        hits,
    })
}

// =============================================================================
// Tool definitions
// =============================================================================

/// The built-in tool declarations offered to the model.
pub(crate) fn builtin_tools() -> Vec<Tool> {
    vec![
        Tool::function(
            "web_search",
            "Search the web for current or factual information and return the \
             top result snippets.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query."
                    }
                },
                "required": ["query"]
            }),
        ),
        Tool::function(
            "fetch_url",
            "Fetch a specific public http(s) URL you already know and return its \
             readable text. For answering questions, prefer web_search.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full http:// or https:// URL to fetch."
                    }
                },
                "required": ["url"]
            }),
        ),
        Tool::function(
            "current_time",
            "Get the current local date and time. Takes no arguments.",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        ),
    ]
}

// =============================================================================
// fetch_url — SSRF-gated page fetch
// =============================================================================

/// Split `url` into (scheme, authority-and-rest); rejects non-http(s)
/// schemes and credential-bearing authorities.
fn host_and_port(url: &str) -> Result<(String, u16), String> {
    let lower = url.to_ascii_lowercase();
    let (default_port, rest) = if let Some(rest) = lower.strip_prefix("https://") {
        (443u16, rest)
    } else if let Some(rest) = lower.strip_prefix("http://") {
        (80u16, rest)
    } else {
        return Err("only http:// and https:// URLs are supported".to_string());
    };

    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .to_string();
    if authority.is_empty() {
        return Err("URL has no host".to_string());
    }
    if authority.contains('@') {
        return Err("URLs with embedded credentials are not supported".to_string());
    }

    // Bracketed IPv6 authority: [::1]:8080 or [::1]
    if let Some(rest) = authority.strip_prefix('[') {
        let Some((host, tail)) = rest.split_once(']') else {
            return Err("malformed IPv6 host".to_string());
        };
        let port = match tail.strip_prefix(':') {
            Some(p) => p.parse().map_err(|_| "invalid port".to_string())?,
            None => default_port,
        };
        return Ok((host.to_string(), port));
    }

    match authority.rsplit_once(':') {
        Some((host, port)) => Ok((
            host.to_string(),
            port.parse().map_err(|_| "invalid port".to_string())?,
        )),
        None => Ok((authority, default_port)),
    }
}

/// True for address ranges `fetch_url` must never touch: loopback, private,
/// link-local (cloud metadata endpoints live there), CGNAT, unspecified.
fn ip_is_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let octets = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                // CGNAT 100.64.0.0/10
                || (octets[0] == 100 && (octets[1] & 0xc0) == 64)
        }
        IpAddr::V6(v6) => {
            // IPv4-mapped addresses smuggle a v4 target through a v6 socket.
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return ip_is_blocked(IpAddr::V4(mapped));
            }
            let seg = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                // unique-local fc00::/7
                || (seg[0] & 0xfe00) == 0xfc00
                // link-local fe80::/10
                || (seg[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Resolve `netloc` and reject any answer in a blocked range. Installed as
/// the ureq agent's resolver so the check applies to the addresses actually
/// dialed — no gap between a pre-check and the request (DNS rebinding).
fn guarded_resolve(netloc: &str) -> std::io::Result<Vec<SocketAddr>> {
    let addrs: Vec<SocketAddr> = netloc.to_socket_addrs()?.collect();
    if addrs.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "host did not resolve",
        ));
    }
    if let Some(blocked) = addrs.iter().find(|a| ip_is_blocked(a.ip())) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "refusing to fetch private/internal address {}",
                blocked.ip()
            ),
        ));
    }
    Ok(addrs)
}

/// Fetch a public page and return its readable text (tags stripped,
/// whitespace collapsed, capped). Redirects are reported back to the model
/// rather than followed, so every hop goes through the resolver guard.
fn fetch_url_impl(url: &str) -> Result<serde_json::Value, String> {
    // Cheap static validation first (scheme, credentials, parseability)…
    host_and_port(url)?;

    // …then the real enforcement lives in the resolver, which sees the
    // addresses the request will actually dial.
    let agent = ureq::builder()
        .timeout(FETCH_TIMEOUT)
        .redirects(0)
        .resolver(guarded_resolve as fn(&str) -> std::io::Result<Vec<SocketAddr>>)
        .build();

    let response = match agent.get(url).set("User-Agent", USER_AGENT).call() {
        Ok(response) => response,
        Err(ureq::Error::Status(code, response)) => {
            return Err(format!(
                "request returned HTTP {code} {}",
                response.status_text()
            ));
        }
        Err(e) => return Err(format!("request failed: {e}")),
    };

    if (300..400).contains(&response.status()) {
        let location = response.header("Location").unwrap_or("(missing)");
        return Ok(serde_json::json!({
            "url": url,
            "note": format!(
                "the server redirected to {location}; call fetch_url with that URL if it looks right"
            ),
        }));
    }

    let mut raw = Vec::new();
    response
        .into_reader()
        .take(FETCH_BYTE_CAP)
        .read_to_end(&mut raw)
        .map_err(|e| format!("read failed: {e}"))?;
    let page = String::from_utf8_lossy(&raw);

    // Skip the document head — it is all scripts and metadata, and on
    // large pages it would otherwise eat the whole character budget.
    let body = match page.to_ascii_lowercase().find("<body") {
        Some(at) => &page[at..],
        None => &page[..],
    };
    let text: String = readable_text(body).chars().take(FETCH_CHAR_CAP).collect();

    Ok(serde_json::json!({ "url": url, "content": text }))
}

// =============================================================================
// Shared helpers
// =============================================================================

/// Split HTML into cleaned text runs (one per text node): tags dropped —
/// including the *contents* of `<script>`/`<style>` blocks — entities
/// decoded, whitespace collapsed, empty runs discarded.
fn text_runs(fragment: &str) -> Vec<String> {
    // ASCII-lowered copy for case-insensitive matching; byte offsets are
    // identical to the original, so `find` positions transfer safely.
    let lower = fragment.to_ascii_lowercase();
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < fragment.len() {
        let rest = &lower[i..];
        if rest.starts_with("<script") || rest.starts_with("<style") {
            // Skip the whole block, contents included.
            let close = if rest.starts_with("<script") {
                "</script"
            } else {
                "</style"
            };
            let Some(rel) = rest.find(close) else {
                break; // unclosed block — drop the remainder
            };
            match lower[i + rel..].find('>') {
                Some(gt) => i += rel + gt + 1,
                None => break,
            }
            continue;
        }
        if rest.starts_with('<') {
            match rest.find('>') {
                Some(gt) => i += gt + 1,
                None => break,
            }
            continue;
        }
        // Plain text run: up to the next tag.
        let end = rest.find('<').map(|at| i + at).unwrap_or(fragment.len());
        let run = fragment[i..end]
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#x27;", "'")
            .replace("&#39;", "'")
            .replace("&nbsp;", " ");
        let run = run.split_whitespace().collect::<Vec<_>>().join(" ");
        if !run.is_empty() {
            runs.push(run);
        }
        i = end;
    }
    runs
}

/// Flatten HTML to a single line of text. Good for short fragments (search
/// snippets); use [`readable_text`] for whole pages.
fn strip_html(fragment: &str) -> String {
    text_runs(fragment).join(" ")
}

/// Minimum run length that counts as prose, and the minimum total prose
/// before navigation-sized runs get dropped.
const PROSE_RUN_MIN_CHARS: usize = 80;
const PROSE_TOTAL_MIN_CHARS: usize = 400;

/// Extract the readable text of a whole page: when enough paragraph-shaped
/// runs exist, keep only those — navigation chrome, menus, and link lists
/// are short runs and drown out the content otherwise. Falls back to all
/// text when a page has no prose-shaped runs (then there is nothing better
/// to offer).
fn readable_text(html: &str) -> String {
    let runs = text_runs(html);
    let prose: Vec<&str> = runs
        .iter()
        .filter(|run| run.chars().count() >= PROSE_RUN_MIN_CHARS)
        .map(String::as_str)
        .collect();
    let prose_total: usize = prose.iter().map(|run| run.chars().count()).sum();
    if prose_total >= PROSE_TOTAL_MIN_CHARS {
        prose.join("\n")
    } else {
        runs.join(" ")
    }
}

/// Stable dedup key for a tool call: name plus canonicalized arguments
/// (serde_json maps sort keys, so semantically-equal calls collide).
pub(crate) fn dedup_key(call: &ToolCall) -> String {
    let canonical = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
        .map(|v| v.to_string())
        .unwrap_or_else(|_| call.function.arguments.clone());
    format!("{}({canonical})", call.function.name)
}

/// Compact one-line rendering of a call for the `⚙` display line, e.g.
/// `web_search("dune first published")`.
pub(crate) fn display_call(call: &ToolCall) -> String {
    let args: serde_json::Value =
        serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::Value::Null);
    if let Some(arg) = args
        .get("query")
        .or_else(|| args.get("url"))
        .and_then(|v| v.as_str())
    {
        return format!("{}({arg:?})", call.function.name);
    }
    // Generic fallback (user tools): show the compact arguments object when
    // it is short enough to read.
    match &args {
        serde_json::Value::Object(map) if !map.is_empty() => {
            let compact = args.to_string();
            if compact.chars().count() <= 48 {
                format!("{}({compact})", call.function.name)
            } else {
                format!("{}(…)", call.function.name)
            }
        }
        _ => format!("{}()", call.function.name),
    }
}

/// The set of tools live in a REPL session: the built-ins plus any
/// user-declared tools from `--tools-file`.
pub(crate) struct ToolBox {
    pub(crate) provider: SearchProvider,
    pub(crate) user_tools: Vec<UserTool>,
}

impl ToolBox {
    /// All tool declarations to offer the model.
    pub(crate) fn definitions(&self) -> Vec<Tool> {
        let mut tools = builtin_tools();
        tools.extend(self.user_tools.iter().map(|tool| {
            Tool::function(
                tool.name.clone(),
                tool.description.clone(),
                tool.parameters.clone(),
            )
        }));
        tools
    }

    /// Run one tool call and shape the JSON the model reads back. User
    /// tools cannot shadow built-ins — collisions are rejected at load.
    pub(crate) fn execute(&self, call: &ToolCall) -> ToolCallResult {
        if let Some(tool) = self
            .user_tools
            .iter()
            .find(|tool| tool.name == call.function.name)
        {
            let content = match run_user_command(tool, &call.function.arguments) {
                Ok(value) => value,
                Err(err) => serde_json::json!({ "error": err }),
            };
            return ToolCallResult {
                call_id: call.id.clone(),
                name: call.function.name.clone(),
                content,
            };
        }
        execute_builtin(call, &self.provider)
    }
}

// =============================================================================
// User tools (--tools-file)
// =============================================================================

const BUILTIN_TOOL_NAMES: [&str; 3] = ["web_search", "fetch_url", "current_time"];
/// Cap on the bytes read from a user tool's stdout/stderr.
const USER_TOOL_OUTPUT_CAP: u64 = 64 * 1024;

fn default_user_tool_parameters() -> serde_json::Value {
    serde_json::json!({ "type": "object", "properties": {} })
}

fn default_user_tool_timeout() -> u64 {
    30
}

/// A user-declared tool from `--tools-file`: a JSON-schema'd function the
/// model may call, backed by a command the user authored.
///
/// The executable and its fixed arguments come from the file only; the
/// model's arguments are delivered as a JSON object on stdin — never
/// interpolated into the command line, and never via a shell.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UserTool {
    pub(crate) name: String,
    pub(crate) description: String,
    /// JSON Schema for the arguments object.
    #[serde(default = "default_user_tool_parameters")]
    pub(crate) parameters: serde_json::Value,
    /// Command argv: `["./scripts/weather.sh"]` or `["python3", "tool.py"]`.
    pub(crate) command: Vec<String>,
    /// Seconds before the command is killed.
    #[serde(default = "default_user_tool_timeout")]
    pub(crate) timeout_secs: u64,
}

/// Load and validate a `--tools-file` (JSON by default, YAML for
/// `.yaml`/`.yml` extensions).
pub(crate) fn load_user_tools(path: &std::path::Path) -> anyhow::Result<Vec<UserTool>> {
    let content = std::fs::read_to_string(path)?;
    let is_yaml = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml"));
    let tools: Vec<UserTool> = if is_yaml {
        serde_yaml::from_str(&content)?
    } else {
        serde_json::from_str(&content)?
    };
    validate_user_tools(&tools)?;
    Ok(tools)
}

fn validate_user_tools(tools: &[UserTool]) -> anyhow::Result<()> {
    let mut seen = std::collections::HashSet::new();
    for tool in tools {
        let valid_name = !tool.name.is_empty()
            && tool
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !tool.name.starts_with(|c: char| c.is_ascii_digit());
        if !valid_name {
            anyhow::bail!(
                "invalid tool name {:?}: use letters, digits, and underscores, not starting with a digit",
                tool.name
            );
        }
        if BUILTIN_TOOL_NAMES.contains(&tool.name.as_str()) {
            anyhow::bail!("tool name {:?} collides with a built-in tool", tool.name);
        }
        if !seen.insert(tool.name.as_str()) {
            anyhow::bail!("duplicate tool name {:?}", tool.name);
        }
        if tool.description.trim().is_empty() {
            anyhow::bail!(
                "tool {:?} needs a description (the model relies on it)",
                tool.name
            );
        }
        if tool.command.is_empty() {
            anyhow::bail!("tool {:?} has an empty command", tool.name);
        }
    }
    Ok(())
}

/// Run a user tool's command: model arguments as JSON on stdin, stdout is
/// the result (parsed as JSON when possible, wrapped otherwise). Killed
/// after `timeout_secs`.
fn run_user_command(tool: &UserTool, arguments: &str) -> Result<serde_json::Value, String> {
    use std::io::Write as _;
    use std::process::Stdio;

    let mut child = std::process::Command::new(&tool.command[0])
        .args(&tool.command[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to start {:?}: {e}", tool.command[0]))?;

    // Deliver the arguments and close stdin so line-oriented scripts see EOF.
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(arguments.as_bytes());
    }

    // Read the pipes on their own threads — draining concurrently avoids a
    // deadlock when the child writes more than a pipe buffer.
    let stdout_reader = child.stdout.take().map(spawn_capped_reader);
    let stderr_reader = child.stderr.take().map(spawn_capped_reader);

    let deadline = std::time::Instant::now() + Duration::from_secs(tool.timeout_secs);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("timed out after {}s", tool.timeout_secs));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(format!("wait failed: {e}")),
        }
    };

    let stdout = stdout_reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();
    let stderr = stderr_reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();

    if !status.success() {
        let detail = stderr.trim();
        return Err(format!(
            "command exited with {status}{}{}",
            if detail.is_empty() { "" } else { ": " },
            detail.chars().take(400).collect::<String>()
        ));
    }

    let stdout = stdout.trim();
    if stdout.is_empty() {
        return Ok(serde_json::json!({ "output": "" }));
    }
    Ok(serde_json::from_str(stdout).unwrap_or_else(|_| serde_json::json!({ "output": stdout })))
}

fn spawn_capped_reader<R: std::io::Read + Send + 'static>(
    source: R,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = source.take(USER_TOOL_OUTPUT_CAP).read_to_end(&mut buffer);
        String::from_utf8_lossy(&buffer).into_owned()
    })
}

/// Run one built-in tool call and shape the JSON the model reads back.
fn execute_builtin(call: &ToolCall, provider: &SearchProvider) -> ToolCallResult {
    let args: serde_json::Value =
        serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::Value::Null);

    let content = match call.function.name.as_str() {
        "web_search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            if query.is_empty() {
                serde_json::json!({ "error": "web_search requires a non-empty query" })
            } else {
                match provider.search(query, RESULTS_PER_SEARCH) {
                    Ok(results) if !results.hits.is_empty() => serde_json::json!({
                        "query": query,
                        "summary": results.summary.unwrap_or_default(),
                        "results": results
                            .hits
                            .iter()
                            .map(|hit| serde_json::json!({
                                "title": hit.title,
                                "snippet": hit.snippet,
                                "url": hit.url,
                            }))
                            .collect::<Vec<_>>(),
                    }),
                    Ok(_) => {
                        serde_json::json!({ "query": query, "results": [], "note": "no results" })
                    }
                    Err(err) => serde_json::json!({ "query": query, "error": err }),
                }
            }
        }
        "fetch_url" => {
            let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
            if url.is_empty() {
                serde_json::json!({ "error": "fetch_url requires a url" })
            } else {
                match fetch_url_impl(url) {
                    Ok(content) => content,
                    Err(err) => serde_json::json!({ "url": url, "error": err }),
                }
            }
        }
        "current_time" => {
            let now = chrono::Local::now();
            serde_json::json!({
                "iso": now.to_rfc3339(),
                "human": now.format("%A, %B %-d, %Y at %H:%M %Z").to_string(),
            })
        }
        other => serde_json::json!({
            "error": format!("unsupported call: {other}")
        }),
    };

    ToolCallResult {
        call_id: call.id.clone(),
        name: call.function.name.clone(),
        content,
    }
}

/// Compact summary of a call+result, appended to the running findings log
/// that is folded back into the user turn each iteration. For searches it
/// leads with the article summary (where the answer usually is), then the
/// hit list; other tools fall back to their JSON content.
pub(crate) fn summarize(result: &ToolCallResult) -> String {
    match result.name.as_str() {
        "web_search" => {
            let query = result
                .content
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let summary = result
                .content
                .get("summary")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let hits = result
                .content
                .get("results")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|r| {
                            let title = r.get("title")?.as_str()?;
                            let snippet = r.get("snippet")?.as_str()?;
                            Some(format!("- {title}: {snippet}"))
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_else(|| result.content.to_string());
            match summary {
                Some(summary) => format!("Search {query:?}:\n{summary}\n{hits}"),
                None => format!("Search {query:?}:\n{hits}"),
            }
        }
        "fetch_url" => {
            let url = result
                .content
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let body = result
                .content
                .get("content")
                .or_else(|| result.content.get("note"))
                .or_else(|| result.content.get("error"))
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            format!("Fetched {url}:\n{body}")
        }
        _ => {
            // current_time and user tools: lead with a human-readable field
            // when present, else the (capped) JSON content.
            let body = result
                .content
                .get("human")
                .or_else(|| result.content.get("output"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| result.content.to_string());
            format!(
                "{}: {}",
                result.name,
                body.chars().take(1200).collect::<String>()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_core::gateway::FunctionCall;

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: "call_0".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn provider_resolution_defaults_and_fallbacks() {
        // Default and explicit wikipedia need no key.
        assert!(matches!(
            SearchProvider::resolve(None, None, None),
            (SearchProvider::Wikipedia, None)
        ));
        assert!(matches!(
            SearchProvider::resolve(Some("Wikipedia"), None, None),
            (SearchProvider::Wikipedia, None)
        ));

        // Keyed providers activate only when their key is present…
        assert!(matches!(
            SearchProvider::resolve(Some("tavily"), Some("tk"), None),
            (SearchProvider::Tavily { .. }, None)
        ));
        assert!(matches!(
            SearchProvider::resolve(Some("brave"), None, Some("bk")),
            (SearchProvider::Brave { .. }, None)
        ));

        // …and fall back to wikipedia (with a warning) when it is missing.
        let (provider, warning) = SearchProvider::resolve(Some("tavily"), None, None);
        assert!(matches!(provider, SearchProvider::Wikipedia));
        assert!(warning.unwrap().contains("TAVILY_API_KEY"));

        let (provider, warning) = SearchProvider::resolve(Some("altavista"), None, None);
        assert!(matches!(provider, SearchProvider::Wikipedia));
        assert!(warning.unwrap().contains("altavista"));
    }

    #[test]
    fn ssrf_guard_blocks_internal_ranges() {
        let blocked = [
            "127.0.0.1",
            "10.0.0.8",
            "172.16.3.4",
            "192.168.1.1",
            "169.254.169.254", // cloud metadata
            "100.64.0.1",      // CGNAT
            "0.0.0.0",
            "::1",
            "::",
            "fe80::1",
            "fc00::1",
            "::ffff:127.0.0.1", // IPv4-mapped loopback
            "::ffff:10.0.0.1",
        ];
        for addr in blocked {
            assert!(
                ip_is_blocked(addr.parse().unwrap()),
                "{addr} must be blocked"
            );
        }

        let allowed = ["93.184.216.34", "1.1.1.1", "2606:4700:4700::1111"];
        for addr in allowed {
            assert!(
                !ip_is_blocked(addr.parse().unwrap()),
                "{addr} must be allowed"
            );
        }
    }

    #[test]
    fn host_parsing_rejects_bad_urls() {
        assert_eq!(
            host_and_port("https://example.com/a?b#c").unwrap(),
            ("example.com".to_string(), 443)
        );
        assert_eq!(
            host_and_port("http://example.com:8080/x").unwrap(),
            ("example.com".to_string(), 8080)
        );
        assert_eq!(
            host_and_port("http://[::2]:9000/x").unwrap(),
            ("::2".to_string(), 9000)
        );

        assert!(host_and_port("ftp://example.com").is_err());
        assert!(host_and_port("file:///etc/passwd").is_err());
        assert!(host_and_port("https://user:pw@example.com").is_err());
        assert!(host_and_port("https://").is_err());
    }

    #[test]
    fn strip_html_drops_tags_and_decodes_entities() {
        assert_eq!(
            strip_html("<span class=\"x\">Rust &amp; Cargo</span>\n  rocks"),
            "Rust & Cargo rocks"
        );
    }

    #[test]
    fn strip_html_drops_script_and_style_contents() {
        let html = "<p>before</p>\n\
                    <SCRIPT type=\"text/javascript\">var x = \"<junk>\";</SCRIPT>\n\
                    <style>.a { color: red; }</style>\n\
                    <p>after</p>";
        assert_eq!(strip_html(html), "before after");

        // An unclosed script block drops the remainder instead of leaking it.
        assert_eq!(strip_html("ok<script>var leak = 1;"), "ok");
    }

    #[test]
    fn readable_text_prefers_prose_over_navigation_chrome() {
        let paragraph = "The 2022 FIFA World Cup final was an association football match \
                         that determined the winner of the tournament, played between \
                         Argentina and France at Lusail Stadium.";
        let html = format!(
            "<nav><a>Jump to content</a><a>Main menu</a><a>move to sidebar</a>\
             <a>Navigation</a><a>Main page</a><a>Contents</a></nav>\
             <p>{paragraph}</p><p>{paragraph}</p><p>{paragraph}</p>"
        );
        let text = readable_text(&html);
        assert!(text.contains("Lusail Stadium"));
        assert!(
            !text.contains("Jump to content"),
            "nav chrome must be filtered when prose exists: {text:?}"
        );

        // A page with no prose-shaped runs falls back to everything.
        let sparse = "<a>one</a><a>two</a>";
        assert_eq!(readable_text(sparse), "one two");
    }

    #[test]
    fn builtin_tool_schemas_are_well_formed() {
        let tools = builtin_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(names, ["web_search", "fetch_url", "current_time"]);
        for tool in &tools {
            assert_eq!(tool.tool_type, "function");
            assert!(tool.function.description.is_some());
            let params = tool.function.parameters.as_ref().unwrap();
            assert_eq!(params["type"], "object");
        }
    }

    #[test]
    fn dedup_key_is_stable_across_argument_formatting() {
        let a = call("web_search", r#"{"query": "dune author"}"#);
        let b = call("web_search", r#"{ "query":"dune author" }"#);
        assert_eq!(dedup_key(&a), dedup_key(&b));

        let c = call("web_search", r#"{"query": "dune sequel"}"#);
        assert_ne!(dedup_key(&a), dedup_key(&c));
    }

    #[test]
    fn execute_builtin_handles_unknown_and_local_tools() {
        let provider = SearchProvider::Wikipedia;

        let unknown = execute_builtin(&call("launch_missiles", "{}"), &provider);
        assert!(unknown.content["error"]
            .as_str()
            .unwrap()
            .contains("unsupported"));

        let time = execute_builtin(&call("current_time", "{}"), &provider);
        assert!(time.content["iso"].as_str().is_some());
        assert!(time.content["human"].as_str().is_some());

        // Missing arguments error without touching the network.
        let empty_search = execute_builtin(&call("web_search", "{}"), &provider);
        assert!(empty_search.content["error"].as_str().is_some());
        let empty_fetch = execute_builtin(&call("fetch_url", "{}"), &provider);
        assert!(empty_fetch.content["error"].as_str().is_some());
    }

    fn user_tool(name: &str, command: &[&str]) -> UserTool {
        UserTool {
            name: name.to_string(),
            description: "test tool".to_string(),
            parameters: default_user_tool_parameters(),
            command: command.iter().map(|s| s.to_string()).collect(),
            timeout_secs: 5,
        }
    }

    #[test]
    fn user_tools_files_parse_json_and_yaml_with_defaults() {
        let json = r#"[{
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
            "command": ["./weather.sh", "--fast"]
        }]"#;
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("tools.json");
        std::fs::write(&json_path, json).unwrap();
        let tools = load_user_tools(&json_path).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].command, ["./weather.sh", "--fast"]);
        assert_eq!(tools[0].timeout_secs, 30); // default

        let yaml = "- name: lookup\n  description: Lookup something.\n  command: [\"./lookup\"]\n";
        let yaml_path = dir.path().join("tools.yaml");
        std::fs::write(&yaml_path, yaml).unwrap();
        let tools = load_user_tools(&yaml_path).unwrap();
        assert_eq!(tools[0].name, "lookup");
        assert_eq!(tools[0].parameters["type"], "object"); // default schema
    }

    #[test]
    fn user_tool_validation_rejects_bad_specs() {
        // Builtin collision.
        let err = validate_user_tools(&[user_tool("web_search", &["./x"])]).unwrap_err();
        assert!(err.to_string().contains("built-in"));

        // Duplicate names.
        let err =
            validate_user_tools(&[user_tool("a", &["./x"]), user_tool("a", &["./y"])]).unwrap_err();
        assert!(err.to_string().contains("duplicate"));

        // Invalid identifier.
        assert!(validate_user_tools(&[user_tool("bad name!", &["./x"])]).is_err());
        assert!(validate_user_tools(&[user_tool("9lives", &["./x"])]).is_err());

        // Empty command.
        assert!(validate_user_tools(&[user_tool("ok", &[])]).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn user_tool_command_receives_args_on_stdin_and_returns_stdout() {
        // Echoes stdin back — JSON in, JSON out.
        let tool = user_tool("echo_args", &["/bin/cat"]);
        let value = run_user_command(&tool, r#"{"city": "Paris"}"#).unwrap();
        assert_eq!(value["city"], "Paris");

        // Non-JSON stdout is wrapped.
        let tool = user_tool("greet", &["/bin/sh", "-c", "echo hello"]);
        let value = run_user_command(&tool, "{}").unwrap();
        assert_eq!(value["output"], "hello");

        // Non-zero exit surfaces stderr.
        let tool = user_tool("fail", &["/bin/sh", "-c", "echo boom >&2; exit 3"]);
        let err = run_user_command(&tool, "{}").unwrap_err();
        assert!(err.contains("boom"), "stderr missing from: {err}");
    }

    #[cfg(unix)]
    #[test]
    fn user_tool_command_is_killed_on_timeout() {
        let mut tool = user_tool("sleepy", &["/bin/sh", "-c", "sleep 30"]);
        tool.timeout_secs = 1;
        let err = run_user_command(&tool, "{}").unwrap_err();
        assert!(err.contains("timed out"), "got: {err}");
    }

    #[test]
    fn toolbox_dispatches_user_tools_and_builtins() {
        let toolbox = ToolBox {
            provider: SearchProvider::Wikipedia,
            user_tools: vec![user_tool("my_tool", &["/nonexistent-command-xyz"])],
        };

        // Definitions merge builtins + user tools.
        let names: Vec<String> = toolbox
            .definitions()
            .into_iter()
            .map(|t| t.function.name)
            .collect();
        assert_eq!(
            names,
            ["web_search", "fetch_url", "current_time", "my_tool"]
        );

        // User dispatch reaches the command runner (spawn fails → error
        // content, not a panic and not the builtin "unsupported" arm).
        let result = toolbox.execute(&call("my_tool", "{}"));
        assert!(result.content["error"]
            .as_str()
            .unwrap()
            .contains("failed to start"));

        // Unknown names still fall through to the builtin error arm.
        let result = toolbox.execute(&call("nope", "{}"));
        assert!(result.content["error"]
            .as_str()
            .unwrap()
            .contains("unsupported"));
    }
}
