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
