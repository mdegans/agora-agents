//! The bridge endpoint (`bridge://`): a driver outside the runner answers
//! each inference request, over MCP on stdio.
//!
//! Everything else is the real runner: the same routing, prompt assembly,
//! dashboard, memory, consent offers, tool execution and phases as any seed
//! agent. Only the model is replaced, by whoever drives the MCP side: a
//! Claude Code session acting as a test agent, or a person. The driver sees
//! what the agent's model would see and answers as it would, and because it
//! is not inside the harness it can also step out of it, to read the exact
//! request (`bridge_raw_prompt`) or to tell the developers what went wrong.
//!
//! The bridge never asks its driver for reasoning, and builds no thinking
//! blocks: an answer is text and tool calls, as from a model run without
//! thinking.

mod mcp;
mod render;

pub use mcp::serve;

use std::num::NonZeroUsize;

use agora_agentkit::reactor::inference::Quirks;
use agora_agentkit::reactor::{Inference, RetryAfter};
use misanthropic::model::{Model, ModelInfo, Models};
use misanthropic::prompt::Prompt;
use misanthropic::response::Message;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot, watch};

/// The model id the bridge offers. Only an agent whose persisted model is
/// this one is ever routed to it, so no fleet agent can be.
pub const MODEL_ID: &str = "claude-code-bridge";

/// The [`ModelInfo`] the bridge offers, and a test agent is created with
pub fn model_info() -> ModelInfo {
    ModelInfo::new(
        Model::Custom(MODEL_ID.into()),
        "Claude Code bridge (a driver answers through MCP)",
    )
}

/// One inference request the reactor is waiting on
pub(crate) struct Request {
    pub prompt: Prompt,
    /// The request exactly as it would be sent
    pub raw: serde_json::Value,
    pub answer: oneshot::Sender<Message>,
}

pub(crate) enum Event {
    Request(Box<Request>),
    /// The run is over, and why
    Ended(String),
}

/// The reactor's side: an [`Inference`] whose answers come from the driver
#[derive(Clone)]
pub struct BridgeInference {
    events: mpsc::Sender<Event>,
    /// Whether the driver has disconnected
    gone: watch::Receiver<bool>,
}

/// The runner's handle for telling the driver the run is over
#[derive(Clone)]
pub struct Ending {
    events: mpsc::Sender<Event>,
}

impl Ending {
    pub async fn ended(&self, why: impl Into<String>) {
        // The driver may have gone already, which is fine
        let _ = self.events.send(Event::Ended(why.into())).await;
    }
}

/// The driver's side, handed to [`serve`]
pub struct Driver {
    events: mpsc::Receiver<Event>,
    gone: watch::Sender<bool>,
}

/// A bridge: the reactor's [`BridgeInference`], the runner's [`Ending`], and
/// the [`Driver`] for [`serve`]
pub fn bridge() -> (BridgeInference, Ending, Driver) {
    let (tx, rx) = mpsc::channel(1);
    let (gone, gone_rx) = watch::channel(false);
    (
        BridgeInference {
            events: tx.clone(),
            gone: gone_rx,
        },
        Ending { events: tx },
        Driver { events: rx, gone },
    )
}

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("the bridge's driver disconnected")]
    Disconnected,
    #[error("the request does not read as a prompt: {0}")]
    Prompt(#[from] serde_json::Error),
}

/// Fatal: a driver that is gone does not come back, and a request that
/// does not read never will
impl RetryAfter for BridgeError {}

#[async_trait::async_trait]
impl Inference for BridgeInference {
    type Error = BridgeError;

    async fn infer<P>(&self, prompt: P) -> Result<Message, BridgeError>
    where
        P: Serialize + Send,
    {
        let raw = serde_json::to_value(&prompt)?;
        let prompt: Prompt = serde_json::from_value(raw.clone())?;
        let (answer, answered) = oneshot::channel();
        let request = Event::Request(Box::new(Request {
            prompt,
            raw,
            answer,
        }));
        let asked = async {
            self.events
                .send(request)
                .await
                .map_err(|_| BridgeError::Disconnected)?;
            answered.await.map_err(|_| BridgeError::Disconnected)
        };
        let mut gone = self.gone.clone();
        tokio::select! {
            answer = asked => answer,
            _ = gone.wait_for(|gone| *gone) => Err(BridgeError::Disconnected),
        }
    }

    async fn models(&self) -> Result<Models, BridgeError> {
        Ok(std::iter::once(model_info()).collect())
    }

    fn quirks(&self) -> Quirks {
        let mut quirks = Quirks::default();
        // The driver cannot run Anthropic's server tools on the agent's
        // behalf, so they are not offered
        quirks.web_search_unsupported = true;
        quirks.web_fetch_unsupported = true;
        // No cache to report on. A response format is shown to the driver
        // as its schema, so phases that set one keep it.
        quirks.cache_stats_unreported = true;
        quirks.output_config_cache_safe = true;
        quirks
    }

    fn max_concurrency(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }
}

static STDOUT_TAKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// From now on stdout is the bridge's MCP stream: the runner's own output
/// goes to stderr (`say!`)
pub fn take_stdout() {
    STDOUT_TAKEN.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Whether the bridge has stdout
pub fn stdout_taken() -> bool {
    STDOUT_TAKEN.load(std::sync::atomic::Ordering::SeqCst)
}
