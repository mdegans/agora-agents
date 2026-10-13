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
use rmcp::service::{RequestContext, RunningService};
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, ServiceExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{Mutex, OnceCell, mpsc, oneshot};

use super::{Driver, Event, MODEL_ID, Request, render};

/// Calls arriving within this long of the one before are one turn. Claude
/// Code sends a turn's parallel calls about 0.1 s apart.
const ROUND_WINDOW: Duration = Duration::from_millis(750);

/// Characters of the raw request per `bridge_raw_prompt` page
const RAW_PAGE: usize = 40_000;

const BEGIN: &str = "bridge_begin";
const REPLY: &str = "bridge_reply";
const RAW: &str = "bridge_raw_prompt";

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
- bridge_raw_prompt shows the exact request the runner would send to a \
model, in pages.";

pub async fn serve(driver: Driver) -> anyhow::Result<RunningService<RoleServer, Server>> {
    Ok(Server::new(driver).serve(rmcp::transport::stdio()).await?)
}

#[derive(Clone)]
pub struct Server {
    inner: Arc<Inner>,
}

struct Inner {
    session: Mutex<Session>,
    round: SyncMutex<Option<Round>>,
    /// The latest request, pretty-printed, for `bridge_raw_prompt`
    raw: SyncMutex<Option<Arc<str>>>,
    tools: OnceCell<Vec<Tool>>,
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
struct RawInput {
    #[serde(default)]
    offset: usize,
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
            RAW,
            "The exact request the runner would send to a model (system, \
             messages, tools, limits), as JSON, in pages. For debugging the \
             harness; the agent's model never sees this view.",
            object(json!({
                "type": "object",
                "properties": {
                    "offset": {"type": "integer", "description": "Character to start from; 0 when left out"}
                },
            })),
        )
        .with_annotations(concurrent()),
    ]
}

impl Server {
    fn new(driver: Driver) -> Self {
        Self {
            inner: Arc::new(Inner {
                session: Mutex::new(Session {
                    events: driver.events,
                    current: None,
                    ended: None,
                    turns: 0,
                }),
                round: SyncMutex::new(None),
                raw: SyncMutex::new(None),
                tools: OnceCell::new(),
            }),
        }
    }

    fn remember_raw(&self, request: &Request) {
        let pretty = serde_json::to_string_pretty(&request.raw)
            .unwrap_or_else(|e| format!("(unrenderable: {e})"));
        *self.inner.raw.lock().expect("not poisoned") = Some(pretty.into());
    }

    async fn tools(&self) -> &[Tool] {
        self.inner
            .tools
            .get_or_init(|| async {
                let mut session = self.inner.session.lock().await;
                let mut tools = match session.wait().await {
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
                };
                tools.extend(bridge_tools());
                tools
            })
            .await
    }

    async fn begin(&self) -> CallToolResult {
        let mut session = self.inner.session.lock().await;
        match session.wait().await {
            Ok(request) => {
                self.remember_raw(request);
                CallToolResult::success(vec![Content::text(render::prompt(&request.prompt))])
            }
            Err(why) => CallToolResult::success(vec![Content::text(format!(
                "[bridge] There is no session: {why}"
            ))]),
        }
    }

    fn raw(&self, offset: usize) -> CallToolResult {
        let raw = self.inner.raw.lock().expect("not poisoned").clone();
        let Some(raw) = raw else {
            return CallToolResult::success(vec![Content::text(
                "[bridge] No request yet: call bridge_begin first.",
            )]);
        };
        let total = raw.chars().count();
        let page: String = raw.chars().skip(offset).take(RAW_PAGE).collect();
        let end = (offset + RAW_PAGE).min(total);
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
        let answered = request.prompt.messages.len();
        let previous_tools = tool_names(&request);
        let _ = request.answer.send(message);

        // What the harness did with it
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
                if fresh.is_empty() {
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
            Err(why) => (
                Vec::new(),
                vec![format!("[bridge] The session is over: {why}")],
            ),
        };
        drop(session);

        let extra = extra.join("\n\n");
        let n_calls = calls.len();
        let last = calls.last().map(|c| c.1.clone());
        let mut calls = calls.into_iter().peekable();
        while let Some((id, name, _, tx)) = calls.next() {
            let found = results.iter().find(|(rid, _, _)| *rid == id);
            let (mut text, is_error) = match found {
                Some((_, text, is_error)) => (text.clone(), *is_error),
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
                    " What else the harness sent is in the result of the \
                     last call, {last}."
                ));
            }
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
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(self.tools().await.to_vec()))
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
                Ok(ReplyInput { text }) => self.join(Entry::Reply(text)).await,
                Err(e) => invalid(e),
            },
            RAW => match serde_json::from_value::<RawInput>(input) {
                Ok(RawInput { offset }) => self.raw(offset),
                Err(e) => invalid(e),
            },
            name if self.tools().await.iter().any(|t| t.name == name) => {
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
