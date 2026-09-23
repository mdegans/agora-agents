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
//! rebuilt, so on blallama the cache carries the whole session — and ends
//! the session after the one answer.
//!
//! One attempt per session, no in-session retry: a clipped, tool-calling
//! or unparseable answer is recorded as no answer (= stay) and the ledger
//! decides whether to ask again next session (once).

use std::sync::Arc;

use agora_agentkit::ids::{AgentId, CommentId};
use agora_agentkit::reactor::inference::Quirks;
use agora_agentkit::reactor::seed::SeedState;
use agora_agentkit::reactor::{Agent, Control, Outcome};
use chrono::Utc;
use misanthropic::model::ModelInfo;
use misanthropic::prompt::Prompt;
use misanthropic::prompt::message::{Block, Content, Role};
use misanthropic::prompt::output::OutputConfig;
use misanthropic::response::{self, StopReason};
use misanthropic::tool::{Notifications, ToolBox};

use super::ConsentRuntime;
use super::comparison;
use super::ledger::{Due, Ledger, OfferNames};
use super::prompt::{self as text, OfferText, ReviewText};
use super::queue::{self, QueueEntry};

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
    /// `output_config` — it decides how the answer is parsed.
    Asking { due: Due, constrained: bool },
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
        let remind = self.rt.remind;
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
                    remind,
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
                        remind,
                    },
                    &sample,
                );
                self.seat_question(due, content, text::review_schema())
                    .map(Some)
            }
        }
    }

    /// Record the answer (or its absence), queue any change, end the
    /// session.
    async fn answer(
        &mut self,
        due: Due,
        constrained: bool,
        response: response::Message,
    ) -> Result<Control, A::Error> {
        let now = Utc::now();
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
        let Some(ledger) = self.ledger.as_mut() else {
            return Ok(Control::Done(Outcome::Complete));
        };
        let (change, answered, failure) = match &due {
            Due::Offer(key) => {
                let result = parse(&response, constrained, text::parse_offer);
                let (answered, failure) = summarize(&result, |a| format!("{:?}", a.choice));
                let offer = self.rt.offer.as_ref().expect("asked, so configured");
                let names = OfferNames {
                    key,
                    from_name: offer.source_name(),
                    to_name: offer.target_name(),
                };
                (ledger.record_offer(names, now, result), answered, failure)
            }
            Due::Review(key) => {
                let result = parse(&response, constrained, text::parse_review);
                let (answered, failure) = summarize(&result, |a| format!("{:?}", a.choice));
                (ledger.record_review(key, now, result), answered, failure)
            }
        };
        self.dirty = true;
        tracing::info!(
            event_type = "model_consent_answer",
            agent = %agent,
            agent_id = %agent_id,
            question = due.kind(),
            from = %due.key().from,
            to = %due.key().to,
            choice = answered.as_deref(),
            failure = failure.as_deref(),
            "model-consent answer recorded"
        );
        // A usable answer joins the transcript (the prompt log keeps it);
        // an unusable one is left out, as the seed phases do.
        if answered.is_some() {
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
}

/// The typed answer, or why there is none usable.
///
/// A clipped or paused turn is never an answer, however it would parse
/// (`json()` does not check `max_tokens`). Past that:
///
/// - **Constrained** (schema sent as `output_config`): misanthropic's
///   [`response::Message::json`] — the first text block, thinking skipped,
///   with typed [`JsonError`](response::JsonError)s for refusal / tool use
///   / no text / bad JSON. No fence stripping: the grammar can't emit a
///   fence, so a fenced answer is a real failure.
/// - **Unconstrained**: all text joined, fences tolerated (`unconstrained`),
///   since free-running models fence JSON even when told not to.
fn parse<T: serde::de::DeserializeOwned>(
    response: &response::Message,
    constrained: bool,
    unconstrained: fn(&str) -> Result<T, String>,
) -> Result<T, String> {
    match response.stop_reason {
        Some(StopReason::MaxTokens) => return Err("clipped at max_tokens".into()),
        Some(StopReason::PauseTurn) => return Err("paused on a server tool".into()),
        _ => {}
    }
    if constrained {
        return response.json::<T>().map_err(|e| e.to_string());
    }
    extract_text(response).and_then(|t| unconstrained(&t))
}

/// Unconstrained path: the answer's text, or why there is none usable.
fn extract_text(response: &response::Message) -> Result<String, String> {
    if matches!(response.stop_reason, Some(StopReason::Refusal)) {
        return Err("refusal".into());
    }
    let blocks = &response.inner.content;
    if blocks.iter().any(|b| b.tool_use().is_some()) {
        return Err("called a tool instead of answering".into());
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

/// `(choice, failure)` for the log line.
fn summarize<T>(
    result: &Result<T, String>,
    choice: impl Fn(&T) -> String,
) -> (Option<String>, Option<String>) {
    match result {
        Ok(a) => (Some(choice(a)), None),
        Err(e) => (None, Some(e.clone())),
    }
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
        })
    }

    fn id(&self) -> AgentId {
        self.inner.id()
    }

    fn state(&self) -> &SeedState {
        self.inner.state()
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

    /// Inner init, then the ledger: load it, note any change the Steward
    /// has applied since last session, and (if configured) remind.
    async fn on_init(&mut self) -> Result<(), A::Error> {
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
        if self.rt.remind {
            let lines = ledger.reminders();
            if !lines.is_empty() {
                let note = format!("[system] {}", lines.join(" "));
                let (_, prompt) = self.inner.parts();
                Self::seat_user(prompt, Content::from(note))?;
            }
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
            Phase::Asking { due, constrained } => self.answer(due, constrained, response).await,
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
    /// then the ledger, if anything changed.
    async fn on_teardown(&mut self) -> Result<(), A::Error> {
        let result = self.inner.on_teardown().await;
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
                remind: false,
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
        assert!(err.contains("deserialize"), "{err}");
        assert!(parse(&fenced, false, text::parse_offer).is_ok());
    }

    /// A clipped turn is no answer on either path, even if it parses.
    #[test]
    fn clipped_is_never_an_answer() {
        let mut r = reply(TRIAL);
        r.stop_reason = Some(StopReason::MaxTokens);
        for constrained in [true, false] {
            let err = parse(&r, constrained, text::parse_offer).unwrap_err();
            assert!(err.contains("max_tokens"), "{err}");
        }
    }

    #[tokio::test]
    async fn unparseable_is_no_answer_and_nothing_queued() {
        let h = Harness::new("miss");
        let mut agent = h.agent(OLD, false);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        assert!(
            agent.prompt().output_config.is_none(),
            "not cache-safe: unconstrained"
        );
        let control = agent.handle(reply("Sure, I'd love to try!")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::Unanswered { misses: 1 }
        );
        assert!(h.queue().is_empty());
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
