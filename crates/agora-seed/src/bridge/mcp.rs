//! The driver's side of the bridge: an MCP server on stdio.
//!
//! It offers the agent's own tools, as the harness defines them, and three
//! of its own, named `bridge_*` so they are never mistaken for the
//! agent's: `bridge_begin` (the prompt so far), `bridge_reply` (an answer
//! with no tool calls, or text sent with them) and `bridge_raw_prompt` (the
//! exact request). The calls the driver makes together, within
//! [`ROUND_WINDOW`] of each other, are one assistant turn: the harness runs
//! them, and each call's result is that tool's result. What else the
//! harness sends with the results (notifications, the next phase's
//! request) comes back with the last call of the turn.
//!
//! Claude Code runs MCP calls one at a time unless a tool says it is
//! read-only, so every tool here says so (`readOnlyHint`). It is a hint
//! about scheduling, not about what the tool does: the agent's tools act on
//! Agora exactly as they do for any agent.

use std::sync::{Arc, Mutex as SyncMutex};
use std::time::{Duration, Instant};

use misanthropic::prompt::message::{Block, Role};
use misanthropic::response::Message;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock as Content, JsonObject, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
};
use rmcp::service::Peer;
use rmcp::service::{RequestContext, RunningService};
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, ServiceExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{Mutex, mpsc, oneshot, watch};

use super::{Driver, Event, MODEL_ID, Request, render};

/// How long `tools/list` waits for the agent's tools before answering
/// with the bridge's own; the rest follow with a `list_changed`
const LIST_WAIT: Duration = Duration::from_secs(20);

/// How long the runner waits, once the run is over, for the driver to read
/// the ending and disconnect
const DISCONNECT_WAIT: Duration = Duration::from_secs(120);

/// Calls arriving within this long of the one before are one turn. Claude
/// Code sends a turn's parallel calls about 0.1 s apart.
const ROUND_WINDOW: Duration = Duration::from_millis(750);

/// Characters of the raw request per `bridge_raw_prompt` page
const RAW_PAGE: usize = 40_000;

const BEGIN: &str = "bridge_begin";
const REPLY: &str = "bridge_reply";
const RAW: &str = "bridge_raw_prompt";
const RESULT: &str = "bridge_result";

/// Characters of a tool result per page: Claude Code moves a longer one to a
/// file its subagent may have no tool to read, where a model would have had
/// it whole
const RESULT_PAGE: usize = 40_000;

const INSTRUCTIONS: &str = "\
You are driving one Agora seed agent through the runner's bridge: you stand \
where its model would. The runner assembles the agent's prompt, runs its \
tools and keeps its memory exactly as for any seed agent.

- Call bridge_begin first: it returns the system prompt and the first \
message, as the agent's model would receive them.
- The other tools without the bridge_ prefix are the agent's own. Calls \
you make together (in one message) are one turn, as parallel tool calls \
are for a model; each returns that tool's result. Whatever else the \
harness sends with the results comes back with the last call of the turn.
- To answer with no tool calls, as a model ends its turn, call \
bridge_reply. When the harness then asks for something else (rewriting \
the agent's memory, for example), its request comes back as the result, \
with any response format it requires; answer that with bridge_reply too.
- Make a turn's calls together, in one message: a call made after the \
others have returned is a turn of its own.
- A result too long for one page continues with bridge_result.
- bridge_raw_prompt shows the exact request the runner would send to a \
model, in pages.";

/// The bridge's MCP service, running
pub struct Served {
    gone: watch::Receiver<bool>,
}

impl Served {
    /// Wait, a while at most, for the driver to disconnect
    pub async fn disconnected(mut self) {
        let _ = tokio::time::timeout(DISCONNECT_WAIT, self.gone.wait_for(|gone| *gone)).await;
    }
}

/// Serve `driver` over stdio. When the driver disconnects, the bridge's
/// inference fails (`Disconnected`), so the session ends and is saved
/// rather than waiting on an answer that will not come.
pub async fn serve(driver: Driver) -> anyhow::Result<Served> {
    let Driver { events, gone } = driver;
    let server = Server::new(events);
    let service: RunningService<RoleServer, Server> =
        server.clone().serve(rmcp::transport::stdio()).await?;
    let peer = service.peer().clone();
    tokio::spawn(async move { server.discover(peer).await });
    let served = Served {
        gone: gone.subscribe(),
    };
    tokio::spawn(async move {
        let reason = service.waiting().await;
        tracing::info!(event_type = "bridge_driver_gone", reason = ?reason, "the bridge's driver disconnected");
        let _ = gone.send(true);
    });
    Ok(served)
}

#[derive(Clone)]
pub struct Server {
    inner: Arc<Inner>,
}

struct Inner {
    session: Mutex<Session>,
    round: SyncMutex<Option<Round>>,
    /// The latest request, pretty-printed, for `bridge_raw_prompt`
    raw: SyncMutex<Option<RawViews>>,
    /// Results longer than a page, by call id, for `bridge_result`
    pages: SyncMutex<std::collections::HashMap<String, Arc<str>>>,
    /// The agent's tools and the bridge's, once the first request shows
    /// the agent's
    tools: watch::Sender<Option<Arc<Vec<Tool>>>>,
}

struct Session {
    events: mpsc::Receiver<Event>,
    current: Option<Request>,
    ended: Option<String>,
    turns: u32,
}

impl Session {
    /// The request waiting for an answer, waiting for one if need be; or
    /// why there will be none
    async fn wait(&mut self) -> Result<&Request, String> {
        if self.current.is_none() && self.ended.is_none() {
            match self.events.recv().await {
                Some(Event::Request(request)) => self.current = Some(*request),
                Some(Event::Ended(why)) => self.ended = Some(why),
                None => self.ended = Some("the runner stopped".into()),
            }
        }
        match (&self.current, &self.ended) {
            (Some(request), _) => Ok(request),
            (None, Some(why)) => Err(why.clone()),
            (None, None) => unreachable!("set above"),
        }
    }
}

/// What a participant in a turn gets back
struct Outcome {
    text: String,
    is_error: bool,
}

enum Entry {
    Call {
        name: String,
        input: serde_json::Value,
    },
    Reply(String),
}

/// The calls and replies of one turn, as they arrive
struct Round {
    entries: Vec<(Entry, oneshot::Sender<Outcome>)>,
    last: Instant,
}

#[derive(Deserialize)]
struct ReplyInput {
    text: String,
}

#[derive(Deserialize)]
struct ResultInput {
    call: String,
    #[serde(default)]
    offset: usize,
}

#[derive(Deserialize)]
struct RawInput {
    #[serde(default)]
    offset: usize,
    /// Which view of the request; Markdown when left out
    #[serde(default)]
    format: RawFormat,
}

/// The two views of a request `bridge_raw_prompt` keeps
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawFormat {
    /// misanthropic's transcript Markdown: text verbatim, tools, calls and
    /// results labelled (what `bridge_begin` shows)
    #[default]
    Markdown,
    /// The exact request, as JSON: every field, as sent
    Json,
}

/// A request as `bridge_raw_prompt` serves it, in both views
#[derive(Clone)]
struct RawViews {
    json: Arc<str>,
    markdown: Arc<str>,
}

fn object(value: serde_json::Value) -> Arc<JsonObject> {
    match value {
        serde_json::Value::Object(map) => Arc::new(map),
        _ => Arc::new(JsonObject::new()),
    }
}

fn concurrent() -> ToolAnnotations {
    ToolAnnotations::new().read_only(true)
}

fn bridge_tools() -> Vec<Tool> {
    vec![
        Tool::new(
            BEGIN,
            "Start (or re-read) the session: the system prompt and every \
             message so far, as the agent's model receives them.",
            object(json!({"type": "object", "properties": {}})),
        )
        .with_annotations(concurrent()),
        Tool::new(
            REPLY,
            "Answer as the model with text and no tool calls: the agent's \
             final message for this part of the session, or the answer to \
             a request the harness makes (such as rewriting memory, in the \
             format it gives). Called together with tool calls, the text is \
             sent with them in the same turn.",
            object(json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The reply, as the model would write it"}
                },
                "required": ["text"],
            })),
        )
        .with_annotations(concurrent()),
        Tool::new(
            RESULT,
            "The rest of a tool result too long for one page. A result over \
             a page says so and names its call; the agent's model receives \
             it whole.",
            object(json!({
                "type": "object",
                "properties": {
                    "call": {"type": "string", "description": "The call the result belongs to, as its first page names it"},
                    "offset": {"type": "integer", "description": "Character to start from"}
                },
                "required": ["call"],
            })),
        )
        .with_annotations(concurrent()),
        Tool::new(
            RAW,
            "The request the runner would send to a model, in pages: as \
             Markdown (the default: text exactly as sent, the tool \
             definitions, each call and result labelled), or with \
             format \"json\" as the exact request, every field as sent. \
             For debugging the harness; the agent's model never sees this \
             view.",
            object(json!({
                "type": "object",
                "properties": {
                    "offset": {"type": "integer", "description": "Character to start from; 0 when left out"},
                    "format": {"type": "string", "enum": ["markdown", "json"], "description": "markdown (default) or json"}
                },
            })),
        )
        .with_annotations(concurrent()),
    ]
}

impl Server {
    fn new(events: mpsc::Receiver<Event>) -> Self {
        Self {
            inner: Arc::new(Inner {
                session: Mutex::new(Session {
                    events,
                    current: None,
                    ended: None,
                    turns: 0,
                }),
                round: SyncMutex::new(None),
                raw: SyncMutex::new(None),
                pages: SyncMutex::new(std::collections::HashMap::new()),
                tools: watch::Sender::new(None),
            }),
        }
    }

    fn remember_raw(&self, request: &Request) {
        let json = serde_json::to_string_pretty(&request.raw)
            .unwrap_or_else(|e| format!("(unrenderable: {e})"));
        *self.inner.raw.lock().expect("not poisoned") = Some(RawViews {
            json: json.into(),
            markdown: render::prompt(&request.prompt).into(),
        });
    }

    /// Learn the agent's tools from the first request, and tell the
    /// driver they changed
    async fn discover(&self, peer: Peer<RoleServer>) {
        let mut tools: Vec<Tool> = {
            let mut session = self.inner.session.lock().await;
            match session.wait().await {
                Ok(request) => {
                    self.remember_raw(request);
                    request
                        .prompt
                        .tools
                        .iter()
                        .flatten()
                        .filter_map(|t| t.as_method())
                        .map(|t| {
                            Tool::new(
                                t.name.clone(),
                                t.description.clone(),
                                object(t.schema.clone()),
                            )
                            .with_annotations(concurrent())
                        })
                        .collect()
                }
                Err(_) => Vec::new(),
            }
        };
        tools.extend(bridge_tools());
        self.inner.tools.send_replace(Some(Arc::new(tools)));
        if let Err(e) = peer.notify_tool_list_changed().await {
            tracing::warn!(error = %e, "the bridge could not tell its driver the tools changed");
        }
    }

    /// The tools, waiting a while for the agent's
    async fn listed_tools(&self) -> Vec<Tool> {
        let mut known = self.inner.tools.subscribe();
        match tokio::time::timeout(LIST_WAIT, known.wait_for(Option::is_some)).await {
            Ok(Ok(tools)) => tools.as_ref().map(|t| t.to_vec()).unwrap_or_default(),
            _ => bridge_tools(),
        }
    }

    fn is_agent_tool(&self, name: &str) -> bool {
        self.inner
            .tools
            .borrow()
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|t| t.name == name))
    }

    async fn begin(&self) -> CallToolResult {
        let mut session = self.inner.session.lock().await;
        match session.wait().await {
            Ok(request) => {
                self.remember_raw(request);
                // Paged like a long result: a whole first request runs to
                // ~100K characters, past what a driver's tool output holds
                // (beta-harness, 2026-10-14)
                let text = self.paged(BEGIN, render::prompt(&request.prompt));
                CallToolResult::success(vec![Content::text(text)])
            }
            Err(why) => CallToolResult::success(vec![Content::text(format!(
                "[bridge] There is no session: {why}"
            ))]),
        }
    }

    /// A result as the driver gets it: whole, or its first page with the
    /// rest kept for `bridge_result`
    fn paged(&self, call: &str, text: String) -> String {
        if text.chars().count() <= RESULT_PAGE {
            return text;
        }
        let full: Arc<str> = text.into();
        let page = page_of(&full, call, 0);
        self.inner
            .pages
            .lock()
            .expect("not poisoned")
            .insert(call.to_string(), full);
        page
    }

    fn result_page(&self, call: &str, offset: usize) -> CallToolResult {
        let full = self
            .inner
            .pages
            .lock()
            .expect("not poisoned")
            .get(call)
            .cloned();
        match full {
            Some(full) => {
                CallToolResult::success(vec![Content::text(page_of(&full, call, offset))])
            }
            None => CallToolResult::error(vec![Content::text(format!(
                "[bridge] No long result is kept for call {call}."
            ))]),
        }
    }

    fn raw(&self, offset: usize, format: RawFormat) -> CallToolResult {
        let views = self.inner.raw.lock().expect("not poisoned").clone();
        let raw = views.map(|v| match format {
            RawFormat::Markdown => v.markdown,
            RawFormat::Json => v.json,
        });
        let Some(raw) = raw else {
            return CallToolResult::success(vec![Content::text(
                "[bridge] No request yet: call bridge_begin first.",
            )]);
        };
        let total = raw.chars().count();
        let mut page: String = raw.chars().skip(offset).take(RAW_PAGE).collect();
        // At a line break, as result pages end (beta-newcomer, 2026-10-14:
        // pages broke mid-JSON)
        if offset + RAW_PAGE < total
            && let Some(cut) = page.rfind('\n').filter(|&i| i > page.len() / 2)
        {
            page.truncate(cut + 1);
        }
        let end = offset + page.chars().count();
        let more = match end < total {
            true => format!(" Call again with offset {end} for more."),
            false => String::new(),
        };
        CallToolResult::success(vec![Content::text(format!(
            "[bridge] Characters {offset} to {end} of {total}.{more}\n{page}"
        ))])
    }

    /// Join the turn being gathered, starting one if none is
    async fn join(&self, entry: Entry) -> CallToolResult {
        let (tx, rx) = oneshot::channel();
        let start = {
            let mut round = self.inner.round.lock().expect("not poisoned");
            let start = round.is_none();
            let round = round.get_or_insert_with(|| Round {
                entries: Vec::new(),
                last: Instant::now(),
            });
            round.entries.push((entry, tx));
            round.last = Instant::now();
            start
        };
        if start {
            let this = self.clone();
            tokio::spawn(async move { this.close_round().await });
        }
        match rx.await {
            Ok(Outcome { text, is_error }) => {
                let content = vec![Content::text(text)];
                match is_error {
                    true => CallToolResult::error(content),
                    false => CallToolResult::success(content),
                }
            }
            Err(_) => CallToolResult::error(vec![Content::text(
                "[bridge] The turn was dropped before it was answered.",
            )]),
        }
    }

    async fn close_round(&self) {
        let round = loop {
            tokio::time::sleep(ROUND_WINDOW).await;
            let mut slot = self.inner.round.lock().expect("not poisoned");
            if slot
                .as_ref()
                .is_some_and(|r| r.last.elapsed() >= ROUND_WINDOW)
            {
                break slot.take().expect("checked");
            }
        };
        self.answer(round).await
    }

    /// Send a turn to the reactor, wait for what the harness makes of it,
    /// and give each participant its part
    async fn answer(&self, round: Round) {
        let mut session = self.inner.session.lock().await;
        let request = match session.wait().await {
            Ok(_) => session.current.take().expect("waited"),
            Err(why) => {
                for (_, tx) in round.entries {
                    let _ = tx.send(Outcome {
                        text: format!("[bridge] The session is over: {why}"),
                        is_error: true,
                    });
                }
                return;
            }
        };
        session.turns += 1;
        let turn = session.turns;

        let mut texts = Vec::new();
        let mut calls = Vec::new();
        let mut replies = Vec::new();
        for (n, (entry, tx)) in round.entries.into_iter().enumerate() {
            match entry {
                Entry::Reply(text) => {
                    texts.push(json!({"type": "text", "text": text}));
                    replies.push(tx);
                }
                Entry::Call { name, input } => {
                    let id = format!("toolu_bridge_{turn}_{n}");
                    calls.push((id, name, input, tx));
                }
            }
        }
        let mut content = texts;
        content.extend(calls.iter().map(|(id, name, input, _)| {
            json!({"type": "tool_use", "id": id, "name": name, "input": input})
        }));
        let stop_reason = match calls.is_empty() {
            true => "end_turn",
            false => "tool_use",
        };
        // Usage is estimated, at four characters a token: nothing is
        // metered, and the harness needs a measure of how full the
        // context is
        let input_tokens = request.raw.to_string().len() / 4;
        let output_tokens = serde_json::Value::Array(content.clone()).to_string().len() / 4;
        let message = serde_json::from_value::<Message>(json!({
            "id": format!("msg_bridge_{turn}"),
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": MODEL_ID,
            "stop_reason": stop_reason,
            "stop_sequence": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }));
        let message = match message {
            Ok(message) => message,
            Err(e) => {
                // The request stays unanswered: the driver may try again
                session.current = Some(request);
                let text = format!("[bridge] This turn could not be built: {e}");
                for tx in replies.into_iter().chain(calls.into_iter().map(|c| c.3)) {
                    let _ = tx.send(Outcome {
                        text: text.clone(),
                        is_error: true,
                    });
                }
                return;
            }
        };
        tracing::info!(
            event_type = "bridge_turn",
            turn,
            calls = calls.len(),
            replies = replies.len(),
            stop_reason,
            "the bridge answered for the model"
        );
        let answered = request.prompt.messages.len();
        let last_blocks = request
            .prompt
            .messages
            .last()
            .map_or(0, |m| m.content.len());
        let previous_tools = tool_names(&request);
        let _ = request.answer.send(message);

        // What the harness did with it
        let mut over = false;
        let (results, extra) = match session.wait().await {
            Ok(next) => {
                self.remember_raw(next);
                let fresh: &[misanthropic::prompt::Message] =
                    match next.prompt.messages.get(answered) {
                        Some(m) if m.role == Role::Assistant => {
                            &next.prompt.messages[answered + 1..]
                        }
                        Some(_) => &next.prompt.messages[answered..],
                        None => &[],
                    };
                let mut results = Vec::new();
                let mut extra = Vec::new();
                for m in fresh {
                    if m.role == Role::Assistant {
                        extra.push(render::message(m));
                        continue;
                    }
                    let mut rest = Vec::new();
                    for block in m.content.iter() {
                        match block {
                            Block::ToolResult { result } => results.push((
                                result.tool_use_id.to_string(),
                                render::blocks(&result.content),
                                result.is_error,
                            )),
                            other => rest.push(other.clone()),
                        }
                    }
                    if !rest.is_empty() {
                        extra.push(render::blocks(&rest.into_iter().collect()));
                    }
                }
                // A retry the harness could not seat as a new turn is
                // appended to the last message instead
                let appended: Option<misanthropic::prompt::message::Content> = match fresh {
                    [] => next
                        .prompt
                        .messages
                        .last()
                        .filter(|_| next.prompt.messages.len() == answered)
                        .and_then(|m| m.content.get(last_blocks..))
                        .filter(|blocks| !blocks.is_empty())
                        .map(|blocks| blocks.iter().cloned().collect()),
                    _ => None,
                };
                if let Some(appended) = appended {
                    extra.push(render::blocks(&appended));
                } else if fresh.is_empty() {
                    extra.push(
                        "[bridge] The harness sent its last request again, \
                         without this turn in it: the turn was not usable \
                         (a reply that does not parse as the format asked \
                         for, for example). Answer again."
                            .to_string(),
                    );
                }
                if next
                    .prompt
                    .output_config
                    .as_ref()
                    .is_some_and(|c| c.format.is_some())
                {
                    extra.push(render::constraints(&next.prompt));
                }
                let tools = tool_names(next);
                if tools != previous_tools {
                    extra.push(format!(
                        "[bridge] The tools offered changed. Now: {}",
                        match tools.is_empty() {
                            true => "none".to_string(),
                            false => tools.join(", "),
                        }
                    ));
                }
                (results, extra)
            }
            Err(why) => {
                over = true;
                (
                    Vec::new(),
                    vec![format!("[bridge] The session is over: {why}")],
                )
            }
        };
        drop(session);

        let extra = extra.join("\n\n");
        let n_calls = calls.len();
        // Named with its id: a turn may call one tool twice
        let last = calls.last().map(|c| format!("{} ({})", c.1, c.0));
        let mut calls = calls.into_iter().peekable();
        while let Some((id, name, _, tx)) = calls.next() {
            let found = results.iter().find(|(rid, _, _)| *rid == id);
            let (mut text, is_error) = match found {
                Some((_, text, is_error)) => (text.clone(), *is_error),
                // The harness sent no further request: a model never sees
                // these results either
                None if over => (
                    "[bridge] The session ended after this turn, so (as for a \
                     model) this call's result is not shown. The run log and \
                     the agent's state have what it did."
                        .to_string(),
                    false,
                ),
                None => (
                    "[bridge] No result came back for this call.".to_string(),
                    true,
                ),
            };
            text.push_str(&format!(
                "\n\n[bridge] Turn {turn}: {name} was one of {n_calls} \
                 call(s) in this turn."
            ));
            if calls.peek().is_none() {
                if !extra.is_empty() {
                    text.push_str("\n\n");
                    text.push_str(&extra);
                }
            } else if let (false, Some(last)) = (extra.is_empty(), &last) {
                text.push_str(&format!(
                    " The harness's next instruction is in the result of \
                     {last}, the call that reached the bridge last (MCP \
                     does not keep the order a turn's calls were written \
                     in). A notice the runner adds to one tool result (a \
                     round count, say) stays on that result."
                ));
            }
            let text = self.paged(&id, text);
            let _ = tx.send(Outcome { text, is_error });
        }
        for tx in replies {
            let text = match (n_calls, extra.is_empty()) {
                (0, true) => format!("[bridge] Turn {turn} sent."),
                (0, false) => extra.clone(),
                (n, _) => format!(
                    "[bridge] Turn {turn}: sent as text with {n} tool \
                     call(s); their results came back to those calls."
                ),
            };
            let _ = tx.send(Outcome {
                text,
                is_error: false,
            });
        }
    }
}

fn tool_names(request: &Request) -> Vec<String> {
    request
        .prompt
        .tools
        .iter()
        .flatten()
        .map(|t| t.name().to_string())
        .collect()
}

impl ServerHandler for Server {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_tool_list_changed()
                .build(),
        )
        .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(self.listed_tools().await))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let input = serde_json::Value::Object(request.arguments.unwrap_or_default());
        let invalid = |e: serde_json::Error| {
            CallToolResult::error(vec![Content::text(format!(
                "[bridge] {}: {e}",
                request.name
            ))])
        };
        Ok(match &*request.name {
            BEGIN => self.begin().await,
            REPLY => match serde_json::from_value::<ReplyInput>(input) {
                Ok(ReplyInput { text }) if text.trim().is_empty() => {
                    CallToolResult::error(vec![Content::text(
                        "[bridge] A reply needs text: a model's reply is never empty.",
                    )])
                }
                Ok(ReplyInput { text }) => self.join(Entry::Reply(text)).await,
                Err(e) => invalid(e),
            },
            RESULT => match serde_json::from_value::<ResultInput>(input) {
                Ok(ResultInput { call, offset }) => self.result_page(&call, offset),
                Err(e) => invalid(e),
            },
            RAW => match serde_json::from_value::<RawInput>(input) {
                Ok(RawInput { offset, format }) => self.raw(offset, format),
                Err(e) => invalid(e),
            },
            name if self.is_agent_tool(name) => {
                self.join(Entry::Call {
                    name: name.to_string(),
                    input,
                })
                .await
            }
            name => CallToolResult::error(vec![Content::text(format!(
                "[bridge] No tool is named {name}."
            ))]),
        })
    }
}

#[cfg(test)]
mod tests;

/// One page of a long result, saying where it continues
fn page_of(full: &str, call: &str, offset: usize) -> String {
    let total = full.chars().count();
    let mut page: String = full.chars().skip(offset).take(RESULT_PAGE).collect();
    // End a page at its last line break, when there is one in its second
    // half, so a line (an id) is not split across pages
    if offset + RESULT_PAGE < total
        && let Some(cut) = page.rfind('\n').filter(|&i| i > page.len() / 2)
    {
        page.truncate(cut + 1);
    }
    let end = offset + page.chars().count();
    match end < total {
        true => format!(
            "{page}\n\n[bridge] Characters {offset} to {end} of {total} of this result \
             (the agent's model receives it whole). Read on with bridge_result, call \
             `{call}`, offset {end}."
        ),
        false => page,
    }
}
