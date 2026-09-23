//! [`ConsentAgent`]: agentkit's seed agent plus one closing question.
//!
//! A wrapper rather than an agentkit change: the seed phase tail lives in
//! the published `agora-agentkit`, and this is runner policy. The wrapper
//! delegates every hook to the inner agent and intercepts exactly one
//! thing — the inner agent's clean `Done(Complete)`, which is the end of
//! its closing phase (memory written, mutation/evolution rolled, survey
//! answered and, if anonymous, already redacted from the transcript). If
//! the ledger says a question is due, the wrapper seats it as the next
//! user turn on the *same* conversation — the prefix is reused, not
//! rebuilt, so on blallama the cache carries the whole session.
//!
//! **Retries.** An answer that can't be used — unparseable, a tool call
//! instead of an answer, or clipped at `max_tokens` — is not seated (as
//! the seed phases prune failed turns); the error is appended to the
//! question turn and the agent tries again, up to [`MAX_ATTEMPTS`] in all.
//! An explicit refusal is taken as no answer without a retry. Once
//! attempts run out it is no answer (= stay), recorded and warned, and the
//! ledger decides whether to ask again next session (once).
//!
//! **SOUL changelog.** Each recorded answer and each applied move adds one
//! automatic `[SYSTEM]` line to the SOUL's Evolution Log — the place
//! agentkit already notes deep mutations — so the agent knows what it
//! chose. Never its memory. The wrapper has no mutable access to the inner
//! agent's state, so at teardown (after the inner teardown, before the
//! reactor persists) it takes a copy of that state with the lines appended
//! and serves the copy from [`Agent::state`], which is what gets saved.

use std::sync::Arc;

use agora_agentkit::ids::{AgentId, CommentId};
use agora_agentkit::reactor::inference::Quirks;
use agora_agentkit::reactor::seed::SeedState;
use agora_agentkit::reactor::{Agent, Control, Outcome};
use chrono::{DateTime, Utc};
use misanthropic::model::ModelInfo;
use misanthropic::prompt::Prompt;
use misanthropic::prompt::message::{Block, Content, Role};
use misanthropic::prompt::output::OutputConfig;
use misanthropic::response::{self, JsonError, StopReason};
use misanthropic::tool::{Notifications, ToolBox};

use super::ConsentRuntime;
use super::comparison;
use super::ledger::{Due, Ledger, OfferNames};
use super::prompt::{self as text, OfferText, ReviewText};
use super::queue::{self, QueueEntry};

/// Answers per question per session: the first plus two retries.
pub const MAX_ATTEMPTS: u32 = 3;

/// [`Agent::Context`] for a [`ConsentAgent`]: the inner agent's context
/// plus the shared consent runtime.
#[derive(Clone)]
pub struct ConsentContext<C> {
    pub inner: C,
    pub consent: Arc<ConsentRuntime>,
}

/// Where the wrapper is in the session.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Phase {
    /// The inner agent owns the session.
    Inner,
    /// The closing question is seated; the next response answers it.
    /// `constrained` records whether it went out with the schema as
    /// `output_config` — it decides how the answer is parsed. `attempt`
    /// counts from 1.
    Asking {
        due: Due,
        constrained: bool,
        attempt: u32,
    },
}

/// Why an answer couldn't be used, and whether it's worth another try.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Failure {
    reason: String,
    retry: Retry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Retry {
    /// Clipped at `max_tokens`: ask again, more briefly.
    Clipped,
    /// Unparseable, or a tool call instead of an answer.
    Unusable,
    /// An explicit refusal (or a paused server tool): no answer, no retry.
    No,
}

impl Failure {
    fn new(reason: impl Into<String>, retry: Retry) -> Self {
        Self {
            reason: reason.into(),
            retry,
        }
    }

    /// The note appended to the question turn before another attempt.
    fn retry_note(&self) -> Option<String> {
        match self.retry {
            Retry::Clipped => Some(
                "Your answer was cut off at the length limit and discarded. Answer again, \
                 more briefly — JSON only, in exactly the shape above."
                    .to_string(),
            ),
            Retry::Unusable => Some(format!(
                "Your answer could not be used ({}). Answer again — JSON only, in exactly \
                 the shape above, and do not use tools.",
                self.reason
            )),
            Retry::No => None,
        }
    }
}

/// See the [module docs](self).
pub struct ConsentAgent<A> {
    inner: A,
    rt: Arc<ConsentRuntime>,
    /// `None` until `on_init`, and stays `None` when the ledger file could
    /// not be read — then this session neither asks nor saves, so a
    /// damaged ledger is never overwritten.
    ledger: Option<Ledger>,
    dirty: bool,
    phase: Phase,
    /// When this session began; ledger events at or after it become SOUL
    /// changelog lines.
    started: DateTime<Utc>,
    /// The inner state with the changelog lines appended, built at
    /// teardown. Served by [`Agent::state`] once present.
    patched: Option<SeedState>,
}

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState>,
{
    fn agent_dir(&self) -> std::path::PathBuf {
        self.rt.state_dir.join(self.inner.id().to_string())
    }

    fn model(&self) -> misanthropic::model::Model {
        self.inner.state().model.id.clone()
    }

    /// Append `content` to the trailing user turn, or push one.
    fn seat_user(prompt: &mut Prompt, content: Content) -> Result<(), A::Error> {
        match prompt.messages.last_mut() {
            Some(last) if last.role == Role::User => {
                last.extend(content);
                Ok(())
            }
            _ => prompt
                .push_message((Role::User, content))
                .map(|_| ())
                .map_err(|e| A::Error::from(Box::new(e))),
        }
    }

    /// Seat the question with its budget and — where changing
    /// `output_config` keeps the prefix cache (blallama) — its schema.
    fn seat_question(
        &mut self,
        due: Due,
        content: Content,
        schema: serde_json::Value,
    ) -> Result<Control, A::Error> {
        let cache_safe = self
            .inner
            .quirks()
            .unwrap_or_default()
            .output_config_cache_safe;
        let max_tokens = self.rt.max_tokens;
        let (_, prompt) = self.inner.parts();
        prompt.max_tokens = std::num::NonZeroU32::new(max_tokens).expect("validated nonzero");
        prompt.output_config = cache_safe.then(|| OutputConfig::json_schema(schema));
        Self::seat_user(prompt, content)?;
        tracing::info!(
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = due.kind(),
            from = %due.key().from,
            to = %due.key().to,
            constrained = cache_safe,
            "model-consent question seated"
        );
        self.phase = Phase::Asking {
            due,
            constrained: cache_safe,
            attempt: 1,
        };
        Ok(Control::Continue)
    }

    /// The inner session completed cleanly: count it toward any trial, then
    /// ask whatever is due. `None` means nothing is — the session ends.
    async fn close(&mut self) -> Result<Option<Control>, A::Error> {
        let model = self.model();
        // An allowlisted offer simply isn't on the table for anyone else.
        let name = &self.inner.state().soul.name;
        let offer_key = self
            .rt
            .offer
            .as_ref()
            .filter(|o| o.admits(name))
            .map(|o| o.key());
        let Some(ledger) = self.ledger.as_mut() else {
            return Ok(None);
        };
        self.dirty |= ledger.count_session(&model);
        let Some(due) = ledger.due(&model, offer_key.as_ref()) else {
            return Ok(None);
        };
        match &due {
            Due::Offer(_) => {
                let offer = self
                    .rt
                    .offer
                    .as_ref()
                    .expect("due offers need one configured");
                let content = text::offer(OfferText {
                    from_name: offer.source_name(),
                    to_name: offer.target_name(),
                    description: &offer.description,
                    limited: offer.is_limited(),
                });
                self.seat_question(due, content, text::offer_schema())
                    .map(Some)
            }
            Due::Review(key) => {
                let (record, boundary, sessions) =
                    ledger.trial(key).expect("due reviews are trials");
                let (from_name, to_name) = (record.from_name.clone(), record.to_name.clone());
                let chosen_on = ledger.chosen_at(key).unwrap_or(boundary).date_naive();
                let comments: Vec<CommentId> = self
                    .inner
                    .state()
                    .ledger
                    .read()
                    .expect("ledger lock")
                    .created_comments
                    .iter()
                    .copied()
                    .collect();
                let sample =
                    comparison::gather(&self.rt.client, self.inner.id(), comments, boundary).await;
                let content = text::review(
                    ReviewText {
                        from_name: &from_name,
                        to_name: &to_name,
                        chosen_on,
                        sessions,
                    },
                    &sample,
                );
                self.seat_question(due, content, text::review_schema())
                    .map(Some)
            }
        }
    }

    /// One response to the question: parse it; on a retryable failure with
    /// attempts left, relay the error and go again; otherwise record the
    /// answer (or its absence), queue any change, and end the session.
    async fn answer(
        &mut self,
        due: Due,
        constrained: bool,
        attempt: u32,
        response: response::Message,
    ) -> Result<Control, A::Error> {
        let now = Utc::now();
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
        if self.ledger.is_none() {
            return Ok(Control::Done(Outcome::Complete));
        }
        let (offer, review) = match &due {
            Due::Offer(_) => (Some(parse(&response, constrained, text::parse_offer)), None),
            Due::Review(_) => (
                None,
                Some(parse(&response, constrained, text::parse_review)),
            ),
        };
        let failure = offer
            .as_ref()
            .and_then(|r| r.as_ref().err())
            .or_else(|| review.as_ref().and_then(|r| r.as_ref().err()))
            .cloned();

        if let Some(failure) = &failure
            && attempt < MAX_ATTEMPTS
            && let Some(note) = failure.retry_note()
        {
            // The failed response is never seated — a clipped turn least of
            // all; only the note joins the question turn.
            tracing::info!(
                agent = %agent,
                agent_id = %agent_id,
                question = due.kind(),
                attempt,
                failure = %failure.reason,
                "model-consent answer unusable; asking again"
            );
            let (_, prompt) = self.inner.parts();
            Self::seat_user(prompt, Content::from(note))?;
            self.phase = Phase::Asking {
                due,
                constrained,
                attempt: attempt + 1,
            };
            return Ok(Control::Continue);
        }

        // Final: an answer, a refusal, or attempts exhausted.
        let reason = failure.as_ref().map(|f| match f.retry {
            Retry::No => f.reason.clone(),
            _ => format!("{} (after {attempt} attempts)", f.reason),
        });
        if let Some(reason) = &reason {
            tracing::warn!(
                event_type = "model_consent_no_answer",
                agent = %agent,
                agent_id = %agent_id,
                question = due.kind(),
                from = %due.key().from,
                to = %due.key().to,
                attempts = attempt,
                failure = %reason,
                "model-consent question got no usable answer; recorded as no answer"
            );
        }
        let ledger = self.ledger.as_mut().expect("checked above");
        let (change, answered) = match (&due, offer, review) {
            (Due::Offer(key), Some(result), _) => {
                let answered = result.as_ref().ok().map(|a| format!("{:?}", a.choice));
                let result = result.map_err(|_| reason.clone().unwrap_or_default());
                let offer = self.rt.offer.as_ref().expect("asked, so configured");
                let names = OfferNames {
                    key,
                    from_name: offer.source_name(),
                    to_name: offer.target_name(),
                };
                (ledger.record_offer(names, now, result), answered)
            }
            (Due::Review(key), _, Some(result)) => {
                let answered = result.as_ref().ok().map(|a| format!("{:?}", a.choice));
                let result = result.map_err(|_| reason.clone().unwrap_or_default());
                (ledger.record_review(key, now, result), answered)
            }
            _ => unreachable!("parsed for the question asked"),
        };
        self.dirty = true;
        if let Some(choice) = &answered {
            tracing::info!(
                event_type = "model_consent_answer",
                agent = %agent,
                agent_id = %agent_id,
                question = due.kind(),
                from = %due.key().from,
                to = %due.key().to,
                attempts = attempt,
                choice = %choice,
                "model-consent answer recorded"
            );
            // A usable answer joins the transcript (the prompt log keeps it).
            let (_, prompt) = self.inner.parts();
            if let Err(e) = prompt.push_message(response.inner) {
                tracing::warn!(agent_id = %agent_id, error = %e, "consent answer not seated");
            }
        }
        if let Some(change) = change {
            let entry = QueueEntry {
                at: now,
                agent_id,
                agent,
                change,
            };
            queue::emit(&self.rt.queue_path, &entry).await;
        }
        Ok(Control::Done(Outcome::Complete))
    }

    /// Copy the inner state with this session's changelog lines appended to
    /// the SOUL's Evolution Log. `None` when there is nothing to add.
    fn patch_soul(&self) -> Option<SeedState> {
        let lines = self.ledger.as_ref()?.changelog(self.started);
        if lines.is_empty() {
            return None;
        }
        // `SeedState` isn't `Clone`; its serde form is exactly what the
        // reactor persists, so a round trip is a faithful copy.
        let copy =
            serde_json::to_value(self.inner.state()).and_then(serde_json::from_value::<SeedState>);
        let mut state = match copy {
            Ok(state) => state,
            Err(e) => {
                tracing::error!(
                    agent_id = %self.inner.id(),
                    error = %e,
                    "could not copy state for the SOUL changelog; lines dropped"
                );
                return None;
            }
        };
        for line in lines {
            if let Err(e) = state.soul.push_evolution(line) {
                tracing::warn!(
                    agent_id = %self.inner.id(),
                    error = %e,
                    "SOUL changelog line rejected"
                );
            }
        }
        Some(state)
    }
}

/// The typed answer, or why there is none usable.
///
/// A clipped or paused turn is never an answer, however it would parse
/// (`json()` does not check `max_tokens`). Past that:
///
/// - **Constrained** (schema sent as `output_config`): misanthropic's
///   [`response::Message::json`] — the first text block, thinking skipped,
///   with typed [`JsonError`]s for refusal / tool use / no text / bad JSON.
///   No fence stripping: the grammar can't emit a fence, so a fenced answer
///   is a real failure.
/// - **Unconstrained**: all text joined, fences tolerated (`unconstrained`),
///   since free-running models fence JSON even when told not to.
fn parse<T: serde::de::DeserializeOwned>(
    response: &response::Message,
    constrained: bool,
    unconstrained: fn(&str) -> Result<T, String>,
) -> Result<T, Failure> {
    match response.stop_reason {
        Some(StopReason::MaxTokens) => {
            return Err(Failure::new("clipped at max_tokens", Retry::Clipped));
        }
        Some(StopReason::PauseTurn) => {
            return Err(Failure::new("paused on a server tool", Retry::No));
        }
        _ => {}
    }
    if constrained {
        return response.json::<T>().map_err(|e| {
            let retry = match e {
                JsonError::Refusal => Retry::No,
                _ => Retry::Unusable,
            };
            Failure::new(e.to_string(), retry)
        });
    }
    extract_text(response)
        .and_then(|t| unconstrained(&t).map_err(|e| Failure::new(e, Retry::Unusable)))
}

/// Unconstrained path: the answer's text, or why there is none usable.
fn extract_text(response: &response::Message) -> Result<String, Failure> {
    if matches!(response.stop_reason, Some(StopReason::Refusal)) {
        return Err(Failure::new("refusal", Retry::No));
    }
    let blocks = &response.inner.content;
    if blocks.iter().any(|b| b.tool_use().is_some()) {
        return Err(Failure::new(
            "called a tool instead of answering",
            Retry::Unusable,
        ));
    }
    let text: Vec<&str> = blocks
        .iter()
        .filter_map(|block| match block {
            Block::Text { text, .. } => Some(text.as_ref()),
            _ => None,
        })
        .collect();
    Ok(text.join("\n\n"))
}

#[async_trait::async_trait]
impl<A> Agent for ConsentAgent<A>
where
    A: Agent<State = SeedState>,
{
    type State = SeedState;
    type Context = ConsentContext<A::Context>;
    type Error = A::Error;

    fn new(id: AgentId, state: SeedState, ctx: Self::Context) -> Result<Self, A::Error> {
        Ok(Self {
            inner: A::new(id, state, ctx.inner)?,
            rt: ctx.consent,
            ledger: None,
            dirty: false,
            phase: Phase::Inner,
            started: Utc::now(),
            patched: None,
        })
    }

    fn id(&self) -> AgentId {
        self.inner.id()
    }

    /// The inner state — or, after teardown added SOUL changelog lines, the
    /// patched copy, which is what the reactor persists.
    fn state(&self) -> &SeedState {
        self.patched.as_ref().unwrap_or_else(|| self.inner.state())
    }

    fn prompt(&self) -> &Prompt {
        self.inner.prompt()
    }

    fn parts(&mut self) -> (&mut ToolBox, &mut Prompt) {
        self.inner.parts()
    }

    fn notifications(&mut self) -> Option<&mut Notifications> {
        self.inner.notifications()
    }

    fn model(&self) -> ModelInfo {
        self.inner.model()
    }

    fn on_admit(&mut self, model: &ModelInfo, quirks: &Quirks) {
        self.inner.on_admit(model, quirks)
    }

    fn quirks(&self) -> Option<Quirks> {
        self.inner.quirks()
    }

    fn prime_prompt(&self) -> Option<Prompt> {
        self.inner.prime_prompt()
    }

    /// Inner init, then the ledger: load it and note any change the
    /// Steward has applied since last session.
    async fn on_init(&mut self) -> Result<(), A::Error> {
        self.started = Utc::now();
        self.inner.on_init().await?;
        let dir = self.agent_dir();
        let mut ledger = match Ledger::load(&dir).await {
            Ok(ledger) => ledger,
            Err(e) => {
                tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %Ledger::path(&dir).display(),
                    error = %e,
                    "model-consent ledger unreadable; not asking or saving this session"
                );
                return Ok(());
            }
        };
        let model = self.model();
        if ledger.observe_model(&model, Utc::now()) {
            self.dirty = true;
            tracing::info!(
                event_type = "model_change_applied",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                model = %model,
                "queued model change observed as applied"
            );
        }
        self.ledger = Some(ledger);
        Ok(())
    }

    async fn on_turn(&mut self) -> Result<(), A::Error> {
        self.inner.on_turn().await
    }

    async fn on_pause(&mut self, response: response::Message) -> Result<Control, A::Error> {
        self.inner.on_pause(response).await
    }

    async fn on_truncate(&mut self, response: &response::Message) -> Result<Control, A::Error> {
        self.inner.on_truncate(response).await
    }

    async fn on_quiesce(&mut self, response: &response::Message) -> Result<Control, A::Error> {
        self.inner.on_quiesce(response).await
    }

    async fn handle(&mut self, response: response::Message) -> Result<Control, A::Error> {
        match std::mem::replace(&mut self.phase, Phase::Inner) {
            Phase::Asking {
                due,
                constrained,
                attempt,
            } => self.answer(due, constrained, attempt, response).await,
            Phase::Inner => match self.inner.handle(response).await? {
                Control::Done(Outcome::Complete) => Ok(self
                    .close()
                    .await?
                    .unwrap_or(Control::Done(Outcome::Complete))),
                other => Ok(other),
            },
        }
    }

    /// Inner teardown (which archives the transcript, question included),
    /// then the SOUL changelog (served by [`state`](Agent::state) for the
    /// reactor's save, which follows teardown), then the ledger.
    async fn on_teardown(&mut self) -> Result<(), A::Error> {
        let result = self.inner.on_teardown().await;
        self.patched = self.patch_soul();
        let dir = self.agent_dir();
        if self.dirty
            && let Some(ledger) = &mut self.ledger
        {
            // Kept current for the `--consent-queue` report (a rename
            // would otherwise print a stale `--agent`).
            ledger.agent = Some(self.inner.state().soul.name.clone());
            if let Err(e) = ledger.save(&dir).await {
                tracing::error!(
                    agent_id = %self.inner.id(),
                    path = %Ledger::path(&dir).display(),
                    error = %e,
                    "model-consent ledger save failed"
                );
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::ledger::{OfferKey, Stage, TRIAL_SESSIONS, Term};
    use crate::consent::queue::QueueEntry;
    use crate::consent::{ConsentConfig, ConsentRuntime, OfferConfig};
    use agora_agentkit::reactor::seed::{SeedError, ShortString};
    use misanthropic::model::{Kind, Model};

    const OLD: &str = "Qwen3.6.gguf";
    const NEW: &str = "Qwen3.8.gguf";

    /// Stands in for `SeedAgent`: its whole session is one response, after
    /// which its closing phase is done.
    struct Fake {
        id: AgentId,
        state: SeedState,
        tools: ToolBox,
        quirks: Quirks,
    }

    #[async_trait::async_trait]
    impl Agent for Fake {
        type State = SeedState;
        type Context = Quirks;
        type Error = SeedError;

        fn new(id: AgentId, mut state: SeedState, quirks: Quirks) -> Result<Self, SeedError> {
            state
                .prompt
                .push_message((Role::User, "Your dashboard."))
                .unwrap();
            Ok(Self {
                id,
                state,
                tools: ToolBox::flat(),
                quirks,
            })
        }
        fn id(&self) -> AgentId {
            self.id
        }
        fn state(&self) -> &SeedState {
            &self.state
        }
        fn prompt(&self) -> &Prompt {
            &self.state.prompt
        }
        fn parts(&mut self) -> (&mut ToolBox, &mut Prompt) {
            (&mut self.tools, &mut self.state.prompt)
        }
        fn model(&self) -> ModelInfo {
            self.state.model.clone()
        }
        fn quirks(&self) -> Option<Quirks> {
            Some(self.quirks)
        }
        async fn on_init(&mut self) -> Result<(), SeedError> {
            Ok(())
        }
        async fn handle(&mut self, response: response::Message) -> Result<Control, SeedError> {
            self.state.prompt.push_message(response.inner).unwrap();
            Ok(Control::Done(Outcome::Complete))
        }
    }

    fn state(model: &str) -> SeedState {
        let soul = serde_json::from_value(serde_json::json!({
            "name": "tarn",
            "identity": "A test agent.",
            "values": ["testing"],
            "interests": { "communities": ["tech"] },
            "voice": "terse",
        }))
        .unwrap();
        let model = ModelInfo {
            id: Model::from(model.to_string()),
            display_name: model.to_string().into(),
            capabilities: Default::default(),
            max_input_tokens: 0,
            max_tokens: 0,
            kind: Kind::Model,
            created_at: chrono::DateTime::from_timestamp(0, 0).unwrap(),
        };
        SeedState::new(soul, model)
    }

    fn reply(text: &str) -> response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "msg_test",
            "role": "assistant",
            "content": [{ "type": "text", "text": text }],
            "model": "test",
            "stop_reason": "end_turn",
            "stop_sequence": null,
        }))
        .unwrap()
    }

    struct Harness {
        root: std::path::PathBuf,
        rt: Arc<ConsentRuntime>,
        id: AgentId,
    }

    impl Harness {
        fn new(tag: &str) -> Self {
            Self::with_allowlist(tag, None)
        }

        fn with_allowlist(tag: &str, agents: Option<&[&str]>) -> Self {
            let root = std::env::temp_dir().join(format!(
                "agora-seed-consent-agent-{tag}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let config = ConsentConfig {
                offer: Some(OfferConfig {
                    from: Model::from(OLD),
                    to: Model::from(NEW),
                    from_name: Some("Qwen 3.6".into()),
                    to_name: Some("Qwen 3.8".into()),
                    description: "Qwen 3.8 is denser and slower.".into(),
                    agents: agents.map(|names| {
                        names
                            .iter()
                            .map(|n| ShortString::new(*n).unwrap())
                            .collect()
                    }),
                }),
            };
            // Nothing listens on the discard port: the review sample's
            // fetches fail fast and it comes back empty.
            let client =
                agora_agentkit::client::Client::new(url::Url::parse("http://127.0.0.1:9").unwrap())
                    .unwrap();
            let rt = Arc::new(ConsentRuntime::new(config, &root, client, 512).unwrap());
            Self {
                root,
                rt,
                id: AgentId::from(uuid::Uuid::from_u128(7)),
            }
        }

        fn agent(&self, model: &str, cache_safe: bool) -> ConsentAgent<Fake> {
            let mut quirks = Quirks::default();
            quirks.output_config_cache_safe = cache_safe;
            ConsentAgent::new(
                self.id,
                state(model),
                ConsentContext {
                    inner: quirks,
                    consent: self.rt.clone(),
                },
            )
            .unwrap()
        }

        async fn ledger(&self) -> Ledger {
            Ledger::load(&self.rt.state_dir.join(self.id.to_string()))
                .await
                .unwrap()
        }

        fn queue(&self) -> Vec<QueueEntry> {
            std::fs::read_to_string(&self.rt.queue_path)
                .unwrap_or_default()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn last_user_text(agent: &ConsentAgent<Fake>) -> String {
        let last = agent.prompt().messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        crate::consent::prompt::tests::text(&last.content)
    }

    /// The whole offer path: the question follows the closing phase on the
    /// same conversation, the answer lands in the ledger (not memory), and
    /// the change is queued — not applied.
    #[tokio::test]
    async fn offer_is_asked_after_the_closing_phase_and_queued() {
        let h = Harness::new("offer");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        let memory_before = agent.state().memory.content.clone();

        let control = agent.handle(reply("closing phase done")).await.unwrap();
        assert_eq!(control, Control::Continue, "question seated, not done");
        assert_eq!(
            agent.prompt().messages.len(),
            3,
            "prefix kept: dashboard, reply, question"
        );
        assert!(last_user_text(&agent).contains("1. `no_swap` — stay on Qwen 3.6."));
        assert!(
            agent.prompt().output_config.is_some(),
            "cache-safe endpoint: constrained"
        );

        let control = agent
            .handle(reply(r#"{"reason": "I am curious.", "choice": "trial"}"#))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();

        assert_eq!(
            agent.state().memory.content,
            memory_before,
            "memory untouched"
        );
        assert_eq!(agent.state().model.id, Model::from(OLD), "model untouched");
        let ledger = h.ledger().await;
        assert_eq!(
            ledger.offers[0].stage,
            Stage::AwaitingSwap { term: Term::Trial }
        );
        assert_eq!(ledger.agent.as_ref().unwrap().as_str(), "tarn");
        let queue = h.queue();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].change.to, Model::from(NEW));
    }

    /// Staging: under an allowlist only listed agents are asked, and the
    /// question says "you", not "every agent on …".
    #[tokio::test]
    async fn an_allowlisted_offer_asks_only_listed_agents() {
        let h = Harness::with_allowlist("unlisted", Some(&["aegis", "sentinel"]));
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "tarn is on the from-model but not listed"
        );
        agent.on_teardown().await.unwrap();
        assert!(!Ledger::path(&h.rt.state_dir.join(h.id.to_string())).exists());

        let h = Harness::with_allowlist("listed", Some(&["aegis", "tarn"]));
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue
        );
        let q = last_user_text(&agent);
        assert!(
            q.contains("You are being asked whether you would like to move to **Qwen 3.8**."),
            "{q}"
        );
        assert!(!q.contains("Every agent"), "{q}");
    }

    fn reply_blocks(content: serde_json::Value) -> response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "msg_test",
            "role": "assistant",
            "content": content,
            "model": "test",
            "stop_reason": "end_turn",
            "stop_sequence": null,
        }))
        .unwrap()
    }

    const TRIAL: &str = r#"{"reason": "curious", "choice": "trial"}"#;

    /// Constrained path: thinking before the JSON is skipped by `json()`.
    #[test]
    fn constrained_answer_after_a_thought_parses() {
        let r = reply_blocks(serde_json::json!([
            { "type": "thinking", "thinking": "Let me weigh this.", "signature": "" },
            { "type": "text", "text": TRIAL },
        ]));
        let a = parse(&r, true, text::parse_offer).unwrap();
        assert_eq!(a.choice, crate::consent::prompt::OfferChoice::Trial);
    }

    /// The grammar can't emit a fence, so under constraint a fenced answer
    /// is a genuine failure — not something to clean up.
    #[test]
    fn constrained_fenced_answer_fails_unconstrained_passes() {
        let fenced = reply(&format!("```json\n{TRIAL}\n```"));
        let err = parse(&fenced, true, text::parse_offer).unwrap_err();
        assert!(err.reason.contains("deserialize"), "{err:?}");
        assert_eq!(err.retry, Retry::Unusable);
        assert!(parse(&fenced, false, text::parse_offer).is_ok());
    }

    /// A clipped turn is no answer on either path, even if it parses.
    #[test]
    fn clipped_is_never_an_answer() {
        let mut r = reply(TRIAL);
        r.stop_reason = Some(StopReason::MaxTokens);
        for constrained in [true, false] {
            let err = parse(&r, constrained, text::parse_offer).unwrap_err();
            assert!(err.reason.contains("max_tokens"), "{err:?}");
            assert_eq!(err.retry, Retry::Clipped);
        }
    }

    fn evolution_notes(agent: &ConsentAgent<Fake>) -> Vec<String> {
        agent
            .state()
            .soul
            .evolution_log
            .iter()
            .map(|e| e.note.to_string())
            .collect()
    }

    /// Three unusable answers: each relayed back and retried in the same
    /// session, then recorded as no answer (= stay), nothing queued.
    #[tokio::test]
    async fn unparseable_is_retried_twice_then_no_answer() {
        let h = Harness::new("miss");
        let mut agent = h.agent(OLD, false);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        assert!(
            agent.prompt().output_config.is_none(),
            "not cache-safe: unconstrained"
        );
        let len = agent.prompt().messages.len();
        for attempt in 1..MAX_ATTEMPTS {
            let control = agent.handle(reply("Sure, I'd love to try!")).await.unwrap();
            assert_eq!(control, Control::Continue, "attempt {attempt} retried");
            assert_eq!(
                agent.prompt().messages.len(),
                len,
                "failed answer not seated"
            );
            assert!(last_user_text(&agent).contains("Your answer could not be used"));
        }
        let control = agent.handle(reply("Sure, I'd love to try!")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        let ledger = h.ledger().await;
        assert_eq!(ledger.offers[0].stage, Stage::Unanswered { misses: 1 });
        let failure = match &ledger.offers[0].history[0].kind {
            crate::consent::ledger::EventKind::Offered { failure, .. } => failure.clone().unwrap(),
            other => panic!("{other:?}"),
        };
        assert!(failure.contains("after 3 attempts"), "{failure}");
        assert!(h.queue().is_empty());
        assert_eq!(
            evolution_notes(&agent).last().unwrap(),
            &format!(
                "[SYSTEM] {}: Asked whether to move from Qwen 3.6 to Qwen 3.8 — no answer recorded; staying on Qwen 3.6 for now.",
                Utc::now().date_naive()
            )
        );
    }

    /// A clipped attempt is pruned, the agent is told, and a good second
    /// attempt counts.
    #[tokio::test]
    async fn clipped_then_answered() {
        let h = Harness::new("clipped");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        let len = agent.prompt().messages.len();
        let mut clipped = reply(r#"{"reason": "Well, let me think about th"#);
        clipped.stop_reason = Some(StopReason::MaxTokens);
        assert_eq!(agent.handle(clipped).await.unwrap(), Control::Continue);
        assert_eq!(
            agent.prompt().messages.len(),
            len,
            "clipped turn not seated"
        );
        assert!(last_user_text(&agent).contains("cut off at the length limit"));
        let control = agent
            .handle(reply(r#"{"reason": "brief", "choice": "no_swap"}"#))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        assert_eq!(h.ledger().await.offers[0].stage, Stage::Declined);
    }

    /// An explicit refusal is no answer at once: no retry.
    #[tokio::test]
    async fn refusal_is_no_answer_without_retry() {
        let h = Harness::new("refusal");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        let mut refusal = reply("I'd rather not.");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::Unanswered { misses: 1 }
        );
    }

    /// The answer lands in the SOUL's Evolution Log (via the state the
    /// reactor persists after teardown) — and memory is untouched.
    #[tokio::test]
    async fn answer_is_noted_in_the_soul_changelog_not_memory() {
        let h = Harness::new("changelog");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        let memory = agent.state().memory.content.clone();
        let before = evolution_notes(&agent).len();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(reply(r#"{"reason": "home", "choice": "no_swap"}"#))
            .await
            .unwrap();
        assert_eq!(evolution_notes(&agent).len(), before, "not before teardown");
        agent.on_teardown().await.unwrap();
        let notes = evolution_notes(&agent);
        assert_eq!(notes.len(), before + 1);
        assert!(
            notes.last().unwrap().ends_with(
                "Asked whether to move from Qwen 3.6 to Qwen 3.8 — chose to stay on Qwen 3.6."
            ),
            "{notes:?}"
        );
        assert!(notes.last().unwrap().starts_with("[SYSTEM] "));
        assert_eq!(agent.state().memory.content, memory);
        // What the reactor saves is the patched state.
        let saved = serde_json::to_value(agent.state()).unwrap();
        assert!(saved.to_string().contains("chose to stay on Qwen 3.6"));
    }

    /// Nothing recorded, nothing added: the inner state is served as is.
    #[tokio::test]
    async fn no_question_no_changelog() {
        let h = Harness::new("quiet");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert!(agent.patched.is_none());
    }

    #[tokio::test]
    async fn agents_on_other_models_are_not_asked() {
        let h = Harness::new("other");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        let control = agent.handle(reply("done")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        assert!(
            !Ledger::path(&h.rt.state_dir.join(h.id.to_string())).exists(),
            "nothing to record, nothing written"
        );
    }

    /// A trial end to end: sessions still on the old model don't count;
    /// once the Steward applies the swap, the fifth completed session on the
    /// new model ends with the review, and "revert" queues the way back.
    #[tokio::test]
    async fn trial_review_after_five_sessions_on_the_new_model() {
        let h = Harness::new("trial");
        let key = OfferKey {
            from: Model::from(OLD),
            to: Model::from(NEW),
        };
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(reply(r#"{"reason": "why not", "choice": "trial"}"#))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();

        // Not yet applied: still on the old model — not a trial session,
        // and not asked again.
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();

        // Applied (set_model + sync-models): sessions on the new model.
        for n in 1..=TRIAL_SESSIONS {
            let mut agent = h.agent(NEW, true);
            agent.on_init().await.unwrap();
            let control = agent.handle(reply("done")).await.unwrap();
            if n < TRIAL_SESSIONS {
                assert_eq!(control, Control::Done(Outcome::Complete), "session {n}");
                agent.on_teardown().await.unwrap();
                continue;
            }
            assert_eq!(control, Control::Continue, "review due after session {n}");
            let q = last_user_text(&agent);
            assert!(
                q.contains("You have now completed 5 sessions on Qwen 3.8."),
                "{q}"
            );
            assert!(q.find("1. `revert`").unwrap() < q.find("2. `keep`").unwrap());
            agent
                .handle(reply(
                    r#"{"reason": "I miss my old voice.", "choice": "revert"}"#,
                ))
                .await
                .unwrap();
            agent.on_teardown().await.unwrap();
        }
        let ledger = h.ledger().await;
        assert!(ledger.trial(&key).is_none());
        assert!(matches!(
            ledger.offers[0].stage,
            Stage::AwaitingRevert { .. }
        ));
        let queue = h.queue();
        assert_eq!(queue.len(), 2, "swap, then revert");
        assert_eq!(queue[1].change.to, Model::from(OLD));
    }
}
