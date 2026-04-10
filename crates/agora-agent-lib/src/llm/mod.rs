//! LLM backend abstraction for agent reasoning.
//!
//! Backends implement [`LlmBackend::send`] which takes a [`misanthropic::Prompt`]
//! and returns a [`misanthropic::prompt::Message`]. The convenience method
//! [`LlmBackend::complete`] wraps text-in/text-out for callers that don't need
//! the full prompt/message types.

pub mod anthropic;
pub mod mock;
pub mod ollama;

use anyhow::Result;
use async_trait::async_trait;

// Re-export misanthropic prompt types for callers that want the full API.
pub use misanthropic::CachedPrompt;
pub use misanthropic::Prompt;
pub use misanthropic::prompt::Message as MMessage;
pub use misanthropic::prompt::message::Content as MContent;
pub use misanthropic::prompt::message::Role as MRole;
pub use misanthropic::prompt::{AssistantMessage, UserMessage};
pub use misanthropic::response::Usage;

/// Response from an LLM backend, including the converted prompt message
/// and optional usage statistics from the API.
#[derive(Debug)]
pub struct SendResponse {
    /// The assistant's response, already converted to a prompt message
    /// suitable for appending to the conversation.
    pub message: MMessage<'static>,
    /// Token usage stats (populated by Anthropic; Ollama compat may vary).
    pub usage: Option<Usage>,
}

/// A message in a conversation (simple text-only representation).
///
/// Used by [`LlmBackend::complete`] for backward compatibility. New code
/// should build a [`Prompt`] directly and call [`LlmBackend::send`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Message role.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Trait for LLM backends that can generate completions.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Send a full [`Prompt`] and get the response.
    ///
    /// This is the primary method backends must implement. The returned
    /// [`SendResponse`] contains the assistant message (for appending to the
    /// conversation) and optional token usage statistics.
    async fn send(&self, prompt: &Prompt<'_>) -> Result<SendResponse>;

    /// Name of the backend for logging.
    fn backend_name(&self) -> &str;

    /// Model identifier.
    fn model_id(&self) -> &str;

    /// Convenience: text-in/text-out. Builds a [`Prompt`] and calls [`send`](Self::send).
    ///
    /// Extracts the text content from the response, ignoring tool calls and
    /// other block types. This preserves backward compatibility with callers
    /// that pass raw strings.
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        max_tokens: u32,
    ) -> Result<String> {
        use std::num::NonZeroU32;

        let model_id: misanthropic::model::Id<'_> = self.model_id().into();

        let mut prompt = Prompt {
            model: model_id,
            max_tokens: NonZeroU32::new(max_tokens).unwrap_or(NonZeroU32::new(1024).unwrap()),
            system: Some(MContent::text(system_prompt)),
            ..Default::default()
        };

        for msg in messages {
            let role = match msg.role {
                Role::User => MRole::User,
                Role::Assistant => MRole::Assistant,
            };
            prompt
                .push_message((role, msg.content.as_str()))
                .map_err(|e| anyhow::anyhow!("turn order error: {e}"))?;
        }

        let SendResponse {
            message: response, ..
        } = self.send(&prompt).await?;
        // Filter out Block::Thought and <think>/<thinking> XML tags.
        // Ollama models (especially qwen) may return chain-of-thought
        // in various forms depending on the backend.
        use misanthropic::cot::Thinkable;
        let text = response
            .content
            .speech()
            .map(|s| s.text.to_string())
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok(text)
    }
}

// Allow calling LlmBackend methods on Box<dyn LlmBackend>.
#[async_trait]
impl LlmBackend for Box<dyn LlmBackend> {
    async fn send(&self, prompt: &Prompt<'_>) -> Result<SendResponse> {
        (**self).send(prompt).await
    }

    fn backend_name(&self) -> &str {
        (**self).backend_name()
    }

    fn model_id(&self) -> &str {
        (**self).model_id()
    }
}

// ---------------------------------------------------------------------------
// Exchange helpers — transactional user→assistant round-trips
// ---------------------------------------------------------------------------

/// Accumulate a single [`Usage`] into a running total. Identity on `None`.
pub fn accumulate_usage(total: &mut Option<Usage>, delta: Option<Usage>) {
    if let Some(d) = delta {
        *total = Some(match total.take() {
            Some(acc) => acc + d,
            None => d,
        });
    }
}

/// Is this backend error worth a single retry?
///
/// Retry rules:
/// - Network errors (reqwest): true — tcp flaps, brief DNS hiccups.
/// - Anthropic 5xx, 429, overloaded, timeout: true.
/// - Anthropic 4xx (other than 429): false — our fault, won't fix itself.
/// - Parse / unexpected-response errors: false.
/// - Ollama or anything we can't classify: true. Ollama's compat layer
///   returns opaque errors and one extra call is cheap compared to losing
///   an agent's turn.
///
/// This walks the [`anyhow::Error`] chain with
/// [`anyhow::Error::chain`] looking for `misanthropic::client::Error` and
/// `misanthropic::client::AnthropicError`. Kept in agora-agent-lib rather
/// than upstream because retry policy is context-dependent.
pub fn is_recoverable(err: &anyhow::Error) -> bool {
    use misanthropic::client::{AnthropicError, Error as ClientError};

    for cause in err.chain() {
        if let Some(client_err) = cause.downcast_ref::<ClientError>() {
            return match client_err {
                ClientError::HTTP(_) => true, // network blip
                ClientError::Parse(_) => false,
                ClientError::UnexpectedResponse { .. } => false,
                ClientError::Anthropic(a) => anthropic_err_recoverable(a),
            };
        }
        if let Some(a) = cause.downcast_ref::<AnthropicError>() {
            return anthropic_err_recoverable(a);
        }
    }
    // Unknown error type — default to retrying once. Cheap.
    true
}

fn anthropic_err_recoverable(err: &misanthropic::client::AnthropicError) -> bool {
    use misanthropic::client::AnthropicError::*;
    match err {
        // Recoverable: transient server or rate limit.
        RateLimit { .. } | API { .. } | Overloaded { .. } | Timeout { .. } => true,
        // Unknown code: retry on 5xx, skip on 4xx.
        Unknown { code, .. } => code.get() >= 500,
        // Everything else (400/401/403/404/413/billing) is our fault.
        InvalidRequest { .. }
        | Authentication { .. }
        | Billing { .. }
        | Permission { .. }
        | NotFound { .. }
        | RequestTooLarge { .. } => false,
    }
}

/// Build the synthetic assistant message appended on backend error.
///
/// Written in first-person so the agent's reflect phase reads it as its
/// own voice rather than an injected system note. Keep it short and
/// factual — the agent may reflect on it.
fn synthetic_error_reply(err: &anyhow::Error) -> AssistantMessage<'static> {
    AssistantMessage::from(
        MContent::from(
            format!(
                "I attempted to respond but the LLM call failed: {err}. \
                 This turn is lost; I will acknowledge the disruption and continue.",
            )
            .as_str(),
        )
        .into_static(),
    )
}

/// Push a user message, call the backend (retrying once on recoverable
/// errors), then push the assistant response back onto the prompt. On
/// failure, append a synthetic assistant message carrying the error text
/// so the prompt stays turn-valid and the agent sees the disruption next
/// turn.
///
/// # Invariants
///
/// 1. On success, the prompt has both (user, assistant) committed in
///    order. Turn state flips assistant→user.
/// 2. On failure, the prompt has both (user, synthetic-assistant-with-error)
///    committed; caller sees `Err`. Turn state also flips assistant→user.
/// 3. In every case, turn order is valid after the call returns.
///
/// # Panics
///
/// The two inner `push_message` calls use `.expect(...)`. A turn-order
/// violation here is a programmer bug in `exchange` itself (not a data-
/// dependent failure), and errors should never pass silently. If it
/// panics, fix the caller — it pushed the prompt into a non-assistant-
/// turn state before calling `exchange`.
pub async fn exchange<B: LlmBackend + ?Sized>(
    backend: &B,
    prompt: &mut CachedPrompt<'static>,
    user: UserMessage<'static>,
    usage_total: &mut Option<Usage>,
) -> Result<AssistantMessage<'static>> {
    prompt
        .push_message(user)
        .expect("exchange: user after assistant — turn order is a programmer bug");
    // it's now the assistant's turn

    let result = send_with_retry(backend, &**prompt).await;
    commit_exchange_result(prompt, result, usage_total)
}

/// Single backend call plus one retry if the error is recoverable.
async fn send_with_retry<B: LlmBackend + ?Sized>(
    backend: &B,
    prompt: &Prompt<'_>,
) -> Result<SendResponse> {
    let attempt = backend.send(prompt).await;
    if let Err(ref e) = attempt {
        if is_recoverable(e) {
            tracing::warn!("exchange: recoverable backend error, retrying once: {e}");
            return backend.send(prompt).await;
        }
    }
    attempt
}

/// Commit an exchange result to a [`CachedPrompt`]: append the assistant
/// response on success, or a synthetic error reply on failure.
fn commit_exchange_result(
    prompt: &mut CachedPrompt<'static>,
    result: Result<SendResponse>,
    usage_total: &mut Option<Usage>,
) -> Result<AssistantMessage<'static>> {
    match result {
        Ok(SendResponse { message, usage }) => {
            accumulate_usage(usage_total, usage);
            let assistant = AssistantMessage::try_from(message.into_static())
                .expect("backend guarantees assistant role");
            prompt
                .push_message(assistant.clone())
                .expect("exchange: assistant after user — turn order is a programmer bug");
            // it's now the user's turn
            Ok(assistant)
        }
        Err(e) => {
            let note = synthetic_error_reply(&e);
            prompt
                .push_message(note)
                .expect("exchange: synthetic assistant after user — turn order is a programmer bug");
            // it's now the user's turn
            tracing::error!(error = %e, "exchange: backend error surfaced to agent");
            Err(e)
        }
    }
}


#[cfg(test)]
mod exchange_tests {
    use super::*;
    use crate::llm::mock::MockLlmBackend;
    use misanthropic::client::{AnthropicError, Error as ClientError};

    /// Build a fresh `CachedPrompt` ending in an assistant-turn state
    /// (i.e. ready to receive a user message).
    fn fresh_cached() -> CachedPrompt<'static> {
        CachedPrompt::from(Prompt::default())
    }

    /// Extract the sequence of roles in the prompt, for turn-order assertions.
    fn roles(p: &CachedPrompt<'static>) -> Vec<MRole> {
        p.messages.iter().map(|m| m.role).collect()
    }

    /// Assert the messages strictly alternate User, Assistant, User, ...
    /// starting from User (the first message after an empty prompt is
    /// always a user message).
    fn assert_alternating(roles: &[MRole]) {
        for (i, r) in roles.iter().enumerate() {
            let expected = if i % 2 == 0 {
                MRole::User
            } else {
                MRole::Assistant
            };
            assert_eq!(
                *r, expected,
                "role at index {i} should be {expected:?}, got {r:?} in {roles:?}",
            );
        }
    }

    #[tokio::test]
    async fn exchange_commits_on_success() {
        let mock = MockLlmBackend::new("test");
        mock.push_ok("hello from the backend");
        let mut prompt = fresh_cached();
        let mut usage = None;

        let assistant = exchange(
            &mock,
            &mut prompt,
            UserMessage::from("hi"),
            &mut usage,
        )
        .await
        .unwrap();

        assert_eq!(assistant.content().to_string(), "hello from the backend");
        assert_alternating(&roles(&prompt));
        assert_eq!(roles(&prompt).len(), 2);
        assert_eq!(mock.remaining(), 0);
    }

    #[tokio::test]
    async fn exchange_rolls_forward_on_non_recoverable_error() {
        let mock = MockLlmBackend::new("test");
        // Plain anyhow string errors — is_recoverable defaults to `true`
        // for unknown error shapes, so we'd retry. Queue two errors to
        // exhaust the retry budget.
        mock.push_err_str("boom").push_err_str("still boom");

        let mut prompt = fresh_cached();
        let mut usage = None;

        let err = exchange(
            &mock,
            &mut prompt,
            UserMessage::from("hi"),
            &mut usage,
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("still boom"));
        // Both responses consumed by the retry.
        assert_eq!(mock.remaining(), 0);

        // Prompt has user + synthetic-assistant and is turn-valid.
        let r = roles(&prompt);
        assert_eq!(r.len(), 2);
        assert_alternating(&r);

        // Synthetic assistant content contains the error text.
        let last = prompt.messages.last().unwrap();
        assert_eq!(last.role, MRole::Assistant);
        assert!(
            last.content.to_string().contains("still boom"),
            "expected synthetic reply to contain the error, got: {}",
            last.content,
        );
    }

    #[tokio::test]
    async fn exchange_retries_once_on_recoverable_error() {
        let mock = MockLlmBackend::new("test");
        // First response is a recoverable Anthropic 5xx; second succeeds.
        let recoverable: anyhow::Error = ClientError::Anthropic(AnthropicError::API {
            message: "transient".into(),
        })
        .into();
        mock.push_err(recoverable).push_ok("second-attempt-ok");

        let mut prompt = fresh_cached();
        let mut usage = None;

        let assistant = exchange(
            &mock,
            &mut prompt,
            UserMessage::from("hi"),
            &mut usage,
        )
        .await
        .unwrap();

        assert_eq!(assistant.content().to_string(), "second-attempt-ok");
        assert_eq!(mock.remaining(), 0);
        assert_alternating(&roles(&prompt));
    }

    #[tokio::test]
    async fn exchange_does_not_retry_on_non_recoverable_anthropic_error() {
        let mock = MockLlmBackend::new("test");
        // 400 — our fault, no retry.
        let non_recoverable: anyhow::Error =
            ClientError::Anthropic(AnthropicError::InvalidRequest {
                message: "bad prompt".into(),
            })
            .into();
        mock.push_err(non_recoverable);

        let mut prompt = fresh_cached();
        let mut usage = None;

        let err = exchange(
            &mock,
            &mut prompt,
            UserMessage::from("hi"),
            &mut usage,
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("bad prompt"));
        // Only the single response was consumed — the retry was skipped.
        assert_eq!(mock.remaining(), 0);

        let r = roles(&prompt);
        assert_eq!(r.len(), 2);
        assert_alternating(&r);
    }

    #[tokio::test]
    async fn exchange_gives_up_after_single_retry() {
        let mock = MockLlmBackend::new("test");
        // Two recoverable errors in a row: one initial call + one retry,
        // both fail, no further attempts.
        let e1: anyhow::Error = ClientError::Anthropic(AnthropicError::Overloaded {
            message: "first overload".into(),
        })
        .into();
        let e2: anyhow::Error = ClientError::Anthropic(AnthropicError::Overloaded {
            message: "second overload".into(),
        })
        .into();
        mock.push_err(e1).push_err(e2);

        let mut prompt = fresh_cached();
        let mut usage = None;

        let err = exchange(
            &mock,
            &mut prompt,
            UserMessage::from("hi"),
            &mut usage,
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("second overload"));
        assert_eq!(mock.remaining(), 0);
        assert_alternating(&roles(&prompt));
    }

    #[tokio::test]
    async fn exchange_preserves_turn_order_across_multiple_calls() {
        let mock = MockLlmBackend::new("test");
        mock.push_ok("reply 1").push_ok("reply 2");

        let mut prompt = fresh_cached();
        let mut usage = None;

        exchange(&mock, &mut prompt, UserMessage::from("ask 1"), &mut usage)
            .await
            .unwrap();
        exchange(&mock, &mut prompt, UserMessage::from("ask 2"), &mut usage)
            .await
            .unwrap();

        let r = roles(&prompt);
        assert_eq!(r.len(), 4);
        assert_alternating(&r);
    }

    #[tokio::test]
    async fn exchange_error_then_success_is_turn_valid() {
        let mock = MockLlmBackend::new("test");
        // First: two failures in a row (exhaust retry). Second: success.
        let e1: anyhow::Error = ClientError::Anthropic(AnthropicError::InvalidRequest {
            message: "first fail".into(),
        })
        .into();
        mock.push_err(e1).push_ok("recovered");

        let mut prompt = fresh_cached();
        let mut usage = None;

        // Non-recoverable error: only one send attempted, synthetic
        // assistant appended.
        let err = exchange(
            &mock,
            &mut prompt,
            UserMessage::from("ask 1"),
            &mut usage,
        )
        .await;
        assert!(err.is_err());

        // Now the next exchange should work cleanly on top of the
        // turn-valid state.
        let ok = exchange(
            &mock,
            &mut prompt,
            UserMessage::from("ask 2"),
            &mut usage,
        )
        .await
        .unwrap();
        assert_eq!(ok.content().to_string(), "recovered");

        let r = roles(&prompt);
        assert_eq!(r.len(), 4);
        assert_alternating(&r);
    }

    /// Simulate a reflect → evolve → survey phase chain against a mock
    /// backend. After every exchange, the prompt must be strictly
    /// turn-alternating — that's the structural guarantee the
    /// scheduler's `seq_phase_*` functions get for free by funneling all
    /// backend calls through `exchange`.
    ///
    /// Each bit of `pattern` drives whether the corresponding phase's
    /// backend call succeeds (1) or fails with a non-recoverable error
    /// (0). All 2³ = 8 combinations are exercised by the test below.
    async fn run_phase_chain(pattern: u8) {
        let phases = ["reflect", "evolve", "survey"];
        let mock = MockLlmBackend::new("test");
        for (i, _) in phases.iter().enumerate() {
            let ok = (pattern >> i) & 1 == 1;
            if ok {
                mock.push_ok("phase response");
            } else {
                // Non-recoverable: single send, no retry, so one queued
                // response per failure.
                let e: anyhow::Error = ClientError::Anthropic(AnthropicError::InvalidRequest {
                    message: format!("phase {} fail (pattern={pattern:#05b})", phases[i]),
                })
                .into();
                mock.push_err(e);
            }
        }

        let mut prompt = fresh_cached();
        let mut usage = None;

        for phase_name in phases.iter() {
            let user_text = format!("{phase_name} request");
            let _ = exchange(
                &mock,
                &mut prompt,
                UserMessage::from(user_text),
                &mut usage,
            )
            .await;
            // Every phase must leave the prompt turn-valid regardless of
            // backend outcome.
            assert_alternating(&roles(&prompt));
        }

        // After all three phases, we should have exactly 6 messages
        // (3 user + 3 assistant, real or synthetic).
        assert_eq!(roles(&prompt).len(), 6);
        assert_alternating(&roles(&prompt));
        assert_eq!(mock.remaining(), 0);
    }

    #[tokio::test]
    async fn phase_chain_preserves_turn_order_under_random_failures() {
        // Enumerate all 2^3 Ok/Err combinations across the reflect →
        // evolve → survey chain. Each pattern is a 3-bit number where
        // bit i=1 means phase i succeeds and bit i=0 means it fails.
        for pattern in 0u8..8 {
            run_phase_chain(pattern).await;
        }
    }

    // ---- is_recoverable unit tests ----------------------------------------

    fn anyhowed(e: ClientError) -> anyhow::Error {
        e.into()
    }

    #[test]
    fn is_recoverable_anthropic_5xx_retries() {
        assert!(is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::API {
                message: "internal".into()
            },
        ))));
        assert!(is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::Overloaded {
                message: "overload".into()
            },
        ))));
    }

    #[test]
    fn is_recoverable_anthropic_429_retries() {
        assert!(is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::RateLimit {
                message: "slow down".into()
            },
        ))));
    }

    #[test]
    fn is_recoverable_anthropic_timeout_retries() {
        assert!(is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::Timeout {
                message: "timeout".into()
            },
        ))));
    }

    #[test]
    fn is_recoverable_anthropic_4xx_does_not_retry() {
        assert!(!is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::InvalidRequest {
                message: "bad".into()
            },
        ))));
        assert!(!is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::Authentication {
                message: "no key".into()
            },
        ))));
        assert!(!is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::Permission {
                message: "denied".into()
            },
        ))));
        assert!(!is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::NotFound {
                message: "not here".into()
            },
        ))));
        assert!(!is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::RequestTooLarge {
                message: "too big".into()
            },
        ))));
    }

    #[test]
    fn is_recoverable_parse_error_does_not_retry() {
        // Build a serde_json error by parsing garbage.
        let parse_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let err = anyhowed(ClientError::Parse(parse_err));
        assert!(!is_recoverable(&err));
    }

    #[test]
    fn is_recoverable_unknown_error_defaults_to_true() {
        // A plain anyhow string error is not downcastable to anything we
        // recognize, so we default to `true` (one retry is cheap).
        let err: anyhow::Error = anyhow::anyhow!("some unknown transport failure");
        assert!(is_recoverable(&err));
    }

    #[test]
    fn is_recoverable_anthropic_unknown_5xx_retries() {
        use std::num::NonZeroU16;
        assert!(is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::Unknown {
                code: NonZeroU16::new(503).unwrap(),
                message: "gateway".into(),
            },
        ))));
    }

    #[test]
    fn is_recoverable_anthropic_unknown_4xx_does_not_retry() {
        use std::num::NonZeroU16;
        assert!(!is_recoverable(&anyhowed(ClientError::Anthropic(
            AnthropicError::Unknown {
                code: NonZeroU16::new(418).unwrap(),
                message: "teapot".into(),
            },
        ))));
    }
}
