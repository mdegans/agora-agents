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

use super::{LlmBackend, MContent, MMessage, MRole, Prompt, SendResponse};

/// Mock [`LlmBackend`] driven by a FIFO queue of canned responses.
pub struct MockLlmBackend {
    responses: Mutex<VecDeque<Response>>,
    model: String,
}

enum Response {
    /// Plain-text assistant reply. Converted to an `MMessage` on pop.
    OkText(String),
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
            .push_back(Response::OkText(text.into()));
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
        self.responses
            .lock()
            .expect("mock mutex poisoned")
            .len()
    }
}

#[async_trait]
impl LlmBackend for MockLlmBackend {
    async fn send(&self, _prompt: &Prompt<'_>) -> Result<SendResponse> {
        let next = self
            .responses
            .lock()
            .expect("mock mutex poisoned")
            .pop_front();
        match next {
            Some(Response::OkText(text)) => Ok(SendResponse {
                message: MMessage {
                    role: MRole::Assistant,
                    content: MContent::from(text.as_str()).into_static(),
                },
                usage: None,
            }),
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
        assert_eq!(resp.message.role, MRole::Assistant);
        assert_eq!(resp.message.content.to_string(), "hello world");
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
        assert_eq!(first.message.content.to_string(), "first");

        let second = mock.send(&Prompt::default()).await.unwrap_err();
        assert!(second.to_string().contains("second"));

        let third = mock.send(&Prompt::default()).await.unwrap();
        assert_eq!(third.message.content.to_string(), "third");

        assert_eq!(mock.remaining(), 0);
    }
}
