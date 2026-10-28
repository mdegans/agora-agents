//! An inference endpoint the runner can hold: a Messages API endpoint
//! (Anthropic, blallama or ollama), or the [`bridge`](crate::bridge).

use std::num::NonZeroUsize;

use agora_agentkit::reactor::inference::Quirks;
use agora_agentkit::reactor::{Inference, RetryAfter, anthropic};
use misanthropic::model::Models;
use misanthropic::response::Message;
use serde::Serialize;

use crate::bridge::{BridgeError, BridgeInference};

pub enum Endpoint {
    Messages(anthropic::Client),
    Bridge(BridgeInference),
}

#[derive(Debug, thiserror::Error)]
pub enum EndpointError {
    #[error(transparent)]
    Messages(#[from] misanthropic::client::Error),
    #[error(transparent)]
    Bridge(#[from] BridgeError),
}

impl RetryAfter for EndpointError {
    fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            Self::Messages(e) => e.retry_after(),
            Self::Bridge(e) => e.retry_after(),
        }
    }

    fn unreachable(&self) -> bool {
        match self {
            Self::Messages(e) => e.unreachable(),
            Self::Bridge(e) => e.unreachable(),
        }
    }
}

fn wrap<E: Into<EndpointError>>(
    results: Vec<Result<Message, E>>,
) -> Vec<Result<Message, EndpointError>> {
    results.into_iter().map(|r| r.map_err(Into::into)).collect()
}

#[async_trait::async_trait]
impl Inference for Endpoint {
    type Error = EndpointError;

    async fn infer<P>(&self, prompt: P) -> Result<Message, EndpointError>
    where
        P: Serialize + Send,
    {
        Ok(match self {
            Self::Messages(c) => c.infer(prompt).await?,
            Self::Bridge(b) => b.infer(prompt).await?,
        })
    }

    async fn infer_batch<P>(
        &self,
        prompts: &[&P],
    ) -> Result<Vec<Result<Message, EndpointError>>, EndpointError>
    where
        P: Serialize + Send + Sync,
    {
        Ok(match self {
            Self::Messages(c) => wrap(c.infer_batch(prompts).await?),
            Self::Bridge(b) => wrap(b.infer_batch(prompts).await?),
        })
    }

    async fn models(&self) -> Result<Models, EndpointError> {
        Ok(match self {
            Self::Messages(c) => c.models().await?,
            Self::Bridge(b) => b.models().await?,
        })
    }

    fn quirks(&self) -> Quirks {
        match self {
            Self::Messages(c) => c.quirks(),
            Self::Bridge(b) => b.quirks(),
        }
    }

    fn max_concurrency(&self) -> NonZeroUsize {
        match self {
            Self::Messages(c) => c.max_concurrency(),
            Self::Bridge(b) => b.max_concurrency(),
        }
    }
}

/// Run `attempt` until it succeeds, waiting `hold` between tries for as long
/// as the endpoint is [unreachable](RetryAfter::unreachable); any other error
/// returns at once. `None` tries once.
///
/// Startup's model discovery goes through this so a runner started while
/// blallama is down (the Mac rebooting, its GPU lent to Blender) waits for
/// it rather than exiting 1: under `Restart=always` that looped the unit and
/// mailed "unit failed" each hour after the 2026-10-28 reboot. The Steward:
/// keep retrying, with no timeout. Logged once when the wait starts and once
/// when it ends, so a day-long wait is two lines, not a thousand.
pub(crate) async fn hold_unreachable<T, E, F, Fut>(
    endpoint: &str,
    hold: Option<std::time::Duration>,
    mut attempt: F,
) -> Result<T, E>
where
    E: RetryAfter + std::fmt::Display,
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    let mut held: u64 = 0;
    loop {
        match attempt().await {
            Ok(t) => {
                if held > 0 {
                    tracing::info!(
                        event_type = "endpoint_reachable",
                        endpoint,
                        held,
                        "endpoint back; the run starts"
                    );
                }
                return Ok(t);
            }
            Err(e) if e.unreachable() && hold.is_some() => {
                let every = hold.expect("checked");
                if held == 0 {
                    tracing::warn!(
                        event_type = "endpoint_unreachable",
                        endpoint,
                        wait_secs = every.as_secs(),
                        error = %e,
                        "endpoint unreachable at startup; waiting for it, without limit"
                    );
                }
                held += 1;
                tokio::time::sleep(every).await;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::time::Duration;

    use super::*;

    #[derive(Debug, thiserror::Error)]
    enum Fake {
        #[error("refused")]
        Down,
        #[error("answered 401")]
        Answered,
    }

    impl RetryAfter for Fake {
        fn retry_after(&self) -> Option<Duration> {
            None
        }
        fn unreachable(&self) -> bool {
            matches!(self, Self::Down)
        }
    }

    /// Down for `n` tries, then `then` (`None`: up).
    async fn run(n: u32, hold: Option<Duration>, then: Option<Fake>) -> (Result<(), Fake>, u32) {
        let tries = Cell::new(0);
        let mut then = then;
        let r = hold_unreachable("test://", hold, || {
            tries.set(tries.get() + 1);
            let r = if tries.get() <= n {
                Err(Fake::Down)
            } else {
                then.take().map_or(Ok(()), Err)
            };
            async move { r }
        })
        .await;
        (r, tries.get())
    }

    const HOLD: Option<Duration> = Some(Duration::from_secs(150));

    #[tokio::test(start_paused = true)]
    async fn startup_waits_out_an_unreachable_endpoint() {
        // A thousand tries at 150 s is about 42 hours: there is no limit.
        let (r, tries) = run(1000, HOLD, None).await;
        assert!(r.is_ok());
        assert_eq!(tries, 1001);
    }

    #[tokio::test(start_paused = true)]
    async fn an_answered_error_is_not_held() {
        let (r, tries) = run(0, HOLD, Some(Fake::Answered)).await;
        assert!(matches!(r, Err(Fake::Answered)));
        assert_eq!(tries, 1);
        // Nor after a wait: the endpoint came back and refused us.
        let (r, tries) = run(2, HOLD, Some(Fake::Answered)).await;
        assert!(matches!(r, Err(Fake::Answered)));
        assert_eq!(tries, 3);
    }

    #[tokio::test(start_paused = true)]
    async fn no_hold_tries_once() {
        let (r, tries) = run(1, None, None).await;
        assert!(matches!(r, Err(Fake::Down)));
        assert_eq!(tries, 1);
    }
}
