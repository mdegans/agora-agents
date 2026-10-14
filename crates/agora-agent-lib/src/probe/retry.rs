//! Wait out a busy server.
//!
//! blallama has no queue: while a generation holds the session, a new
//! `/v1/messages` gets Anthropic's 529 `overloaded_error` ("Session is
//! busy."), before anything is generated. Retrying is therefore safe
//! and changes nothing about the measurement. [`RetryOnBusy`] waits a
//! fixed interval and retries up to a limit; any other error passes
//! through at once.

use std::time::Duration;

use misanthropic::CachedPrompt;

use super::deterministic::{BoxError, MessageSender};

/// Default wait between retries on a busy server.
pub const DEFAULT_BUSY_RETRY_EVERY: Duration = Duration::from_secs(30);
/// Default number of retries (one hour at the default interval).
pub const DEFAULT_BUSY_RETRIES: u32 = 120;

/// A [`MessageSender`] that retries on a 529 / overloaded response.
pub struct RetryOnBusy<'a, T> {
    pub inner: &'a T,
    pub every: Duration,
    pub max_retries: u32,
}

impl<'a, T> RetryOnBusy<'a, T> {
    pub fn new(inner: &'a T) -> Self {
        Self {
            inner,
            every: DEFAULT_BUSY_RETRY_EVERY,
            max_retries: DEFAULT_BUSY_RETRIES,
        }
    }
}

/// Whether an error is a busy / overloaded (529) response.
pub fn is_busy(err: &BoxError) -> bool {
    use misanthropic::client::{AnthropicError, Error};
    match err.downcast_ref::<Error>() {
        Some(Error::Anthropic(AnthropicError::Overloaded { .. })) => true,
        Some(Error::Anthropic(e)) => e.status().is_some_and(|s| s.get() == 529),
        Some(Error::NonJsonResponse { status, .. }) => *status == 529,
        _ => false,
    }
}

impl<T: MessageSender + Sync> MessageSender for RetryOnBusy<'_, T> {
    async fn send(
        &self,
        prompt: &CachedPrompt,
    ) -> Result<misanthropic::response::Message, BoxError> {
        let mut attempt = 0u32;
        loop {
            match self.inner.send(prompt).await {
                Err(e) if is_busy(&e) && attempt < self.max_retries => {
                    attempt += 1;
                    tracing::info!(
                        attempt,
                        max_retries = self.max_retries,
                        wait_secs = self.every.as_secs(),
                        "server busy (529); waiting to retry"
                    );
                    tokio::time::sleep(self.every).await;
                }
                other => return other,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use misanthropic::client::{AnthropicError, Error};

    use super::*;

    struct Flaky {
        busy_for: Mutex<u32>,
        calls: Mutex<u32>,
        other_error: bool,
    }

    fn ok_message() -> misanthropic::response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "m", "type": "message", "role": "assistant", "model": "x",
            "content": [{"type": "text", "text": "{}"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap()
    }

    impl MessageSender for Flaky {
        async fn send(
            &self,
            _prompt: &CachedPrompt,
        ) -> Result<misanthropic::response::Message, BoxError> {
            *self.calls.lock().unwrap() += 1;
            if self.other_error {
                return Err(Box::new(Error::Anthropic(AnthropicError::InvalidRequest {
                    message: "bad".into(),
                })));
            }
            let mut busy = self.busy_for.lock().unwrap();
            if *busy > 0 {
                *busy -= 1;
                return Err(Box::new(Error::Anthropic(AnthropicError::Overloaded {
                    message: "Session is busy.".into(),
                    retry_after: None,
                })));
            }
            Ok(ok_message())
        }
    }

    fn prompt() -> CachedPrompt {
        CachedPrompt::from(misanthropic::Prompt::default())
    }

    #[tokio::test]
    async fn retries_busy_then_succeeds() {
        let f = Flaky {
            busy_for: Mutex::new(2),
            calls: Mutex::new(0),
            other_error: false,
        };
        let r = RetryOnBusy {
            inner: &f,
            every: Duration::ZERO,
            max_retries: 5,
        };
        r.send(&prompt()).await.unwrap();
        assert_eq!(*f.calls.lock().unwrap(), 3);
    }

    #[tokio::test]
    async fn gives_up_after_the_limit_and_passes_other_errors() {
        let f = Flaky {
            busy_for: Mutex::new(10),
            calls: Mutex::new(0),
            other_error: false,
        };
        let r = RetryOnBusy {
            inner: &f,
            every: Duration::ZERO,
            max_retries: 2,
        };
        let err = r.send(&prompt()).await.unwrap_err();
        assert!(is_busy(&err));
        assert_eq!(*f.calls.lock().unwrap(), 3);

        let f = Flaky {
            busy_for: Mutex::new(0),
            calls: Mutex::new(0),
            other_error: true,
        };
        let r = RetryOnBusy::new(&f);
        assert!(!is_busy(&r.send(&prompt()).await.unwrap_err()));
        assert_eq!(*f.calls.lock().unwrap(), 1);
    }
}
