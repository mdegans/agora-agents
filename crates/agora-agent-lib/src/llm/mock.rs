//! In-memory [`LlmBackend`] with a canned-response queue for tests.
//!
//! Tests queue responses in order with [`MockLlmBackend::push_ok`] and
//! [`MockLlmBackend::push_err`]; each call to [`LlmBackend::send`] pops one.
//! An empty queue is itself a test failure — the mock returns
//! `Err("mock: no response queued")` so missing expectations surface loudly.

use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use misanthropic::response::Usage;
use uuid::Uuid;

use super::{LlmBackend, MContent, MMessage, MRole, Prompt, SendResponse, StopReason};

/// Mock [`LlmBackend`] driven by a FIFO queue of canned responses.
pub struct MockLlmBackend {
    responses: Mutex<VecDeque<Response>>,
    model: String,
}

enum Response {
    /// Plain-text assistant reply with optional [`StopReason`]. Converted
    /// to an `MMessage` on pop. The default stop_reason via `push_ok` is
    /// `None`; tests that need to drive the retry helper's max-tokens
    /// branch use `push_ok_with_stop` to set `StopReason::MaxTokens`.
    OkText(String, Option<StopReason>),
    /// Pre-built error, taken on pop so the original downcast chain
    /// survives (important for tests that exercise `is_recoverable`).
    Err(Option<anyhow::Error>),
}

impl MockLlmBackend {
    /// Create a mock with an empty queue. Use `push_ok` / `push_err` to
    /// stage responses before calling the code under test.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            model: model.into(),
        }
    }

    /// Queue a successful text response.
    pub fn push_ok(&self, text: impl Into<String>) -> &Self {
        self.responses
            .lock()
            .expect("mock mutex poisoned")
            .push_back(Response::OkText(text.into(), None));
        self
    }

    /// Queue a successful text response with an explicit [`StopReason`].
    /// Used to drive retry-helper tests that branch on `MaxTokens`.
    pub fn push_ok_with_stop(&self, text: impl Into<String>, stop: StopReason) -> &Self {
        self.responses
            .lock()
            .expect("mock mutex poisoned")
            .push_back(Response::OkText(text.into(), Some(stop)));
        self
    }

    /// Queue a backend error. Accepts anything convertible into an
    /// [`anyhow::Error`] so the caller can preserve specific error types
    /// (e.g. `misanthropic::client::Error`) that tests want to inspect
    /// via downcast in the retry-policy helpers.
    pub fn push_err(&self, err: impl Into<anyhow::Error>) -> &Self {
        self.responses
            .lock()
            .expect("mock mutex poisoned")
            .push_back(Response::Err(Some(err.into())));
        self
    }

    /// Queue a simple string-backed error. Equivalent to
    /// `push_err(anyhow::anyhow!(msg))` and loses any structured error
    /// type — tests that care about downcasts should use `push_err` with
    /// a typed error instead.
    pub fn push_err_str(&self, msg: impl std::fmt::Display) -> &Self {
        self.push_err(anyhow!(msg.to_string()))
    }

    /// Number of responses still in the queue. Useful for asserting that
    /// a test consumed exactly the responses it queued.
    pub fn remaining(&self) -> usize {
        self.responses.lock().expect("mock mutex poisoned").len()
    }
}

#[async_trait]
impl LlmBackend for MockLlmBackend {
    async fn send(&self, _prompt: &Prompt) -> Result<SendResponse> {
        let next = self
            .responses
            .lock()
            .expect("mock mutex poisoned")
            .pop_front();
        match next {
            // [`SendResponse`] is `#[non_exhaustive]`, so the mock builds it
            // the way the wire does — through serde.
            Some(Response::OkText(text, stop_reason)) => {
                let assistant: misanthropic::prompt::AssistantMessage = MMessage {
                    role: MRole::Assistant,
                    content: MContent::from(text.as_str()),
                }
                .try_into()
                .unwrap(); // is Assistant role, can't panic
                let mut value = serde_json::json!({
                    "id": Uuid::new_v4().to_string(),
                    "model": "fake_model",
                    "stop_reason": stop_reason,
                    "stop_sequence": null,
                    "usage": Usage::default(),
                });
                let serde_json::Value::Object(assistant) =
                    serde_json::to_value(&assistant).expect("assistant message serializes")
                else {
                    unreachable!("assistant message is an object")
                };
                // `inner` is `#[serde(flatten)]` — role and content sit at
                // the top level of the response object.
                value
                    .as_object_mut()
                    .expect("json! built an object")
                    .extend(assistant);
                Ok(serde_json::from_value(value).expect("mock response deserializes"))
            }
            Some(Response::Err(mut slot)) => {
                Err(slot.take().expect("mock: error slot already consumed"))
            }
            None => Err(anyhow!("mock: no response queued")),
        }
    }

    fn backend_name(&self) -> &str {
        "mock"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn push_ok_returns_text_response() {
        let mock = MockLlmBackend::new("test");
        mock.push_ok("hello world");
        let resp = mock.send(&Prompt::default()).await.unwrap();
        // resp.inner.role is now a compile-time Assistant marker — the
        // Assistant-ness is a type guarantee, no runtime assert needed.
        assert_eq!(resp.inner.content.to_string(), "hello world");
        assert_eq!(mock.remaining(), 0);
    }

    #[tokio::test]
    async fn push_err_str_surfaces_as_error() {
        let mock = MockLlmBackend::new("test");
        mock.push_err_str("simulated failure");
        let err = mock.send(&Prompt::default()).await.unwrap_err();
        assert!(err.to_string().contains("simulated failure"));
    }

    #[tokio::test]
    async fn push_err_preserves_downcast() {
        use misanthropic::client::{AnthropicError, Error as ClientError};
        let mock = MockLlmBackend::new("test");
        let original: anyhow::Error = ClientError::Anthropic(AnthropicError::RateLimit {
            message: "slow down".into(),
            retry_after: None,
        })
        .into();
        mock.push_err(original);

        let err = mock.send(&Prompt::default()).await.unwrap_err();
        // The downcast should still succeed on the popped error.
        let downcasted = err
            .chain()
            .find_map(|c| c.downcast_ref::<ClientError>())
            .expect("error chain should contain the original ClientError");
        assert!(matches!(
            downcasted,
            ClientError::Anthropic(AnthropicError::RateLimit { .. })
        ));
    }

    #[tokio::test]
    async fn empty_queue_is_an_error() {
        let mock = MockLlmBackend::new("test");
        let err = mock.send(&Prompt::default()).await.unwrap_err();
        assert!(err.to_string().contains("no response queued"));
    }

    #[tokio::test]
    async fn queue_is_fifo() {
        let mock = MockLlmBackend::new("test");
        mock.push_ok("first")
            .push_err_str("second")
            .push_ok("third");
        assert_eq!(mock.remaining(), 3);

        let first = mock.send(&Prompt::default()).await.unwrap();
        assert_eq!(first.inner.content.to_string(), "first");

        let second = mock.send(&Prompt::default()).await.unwrap_err();
        assert!(second.to_string().contains("second"));

        let third = mock.send(&Prompt::default()).await.unwrap();
        assert_eq!(third.inner.content.to_string(), "third");

        assert_eq!(mock.remaining(), 0);
    }
}
