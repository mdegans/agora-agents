//! Anthropic backend using the `misanthropic` crate (Messages API).
//!
//! This is the native backend — prompts are sent directly to the Anthropic API
//! without any conversion layer.

use anyhow::{Context, Result};
use async_trait::async_trait;
use misanthropic::Prompt;
use misanthropic::prompt::message::{Content, Role};

use crate::log::log_usage;

use super::{LlmBackend, SendResponse};

/// Anthropic accepts at most this many `cache_control` markers per request,
/// counted across `tools` + `system` + `messages`.
const MAX_CACHE_CONTROLS_PER_REQUEST: usize = 4;

/// Anthropic LLM backend using Claude models.
pub struct AnthropicBackend {
    client: misanthropic::Client,
    model: String,
}

impl AnthropicBackend {
    /// Create a new Anthropic backend with the given API key and model.
    ///
    /// `api_key` must be a valid Anthropic API key (108 bytes).
    pub fn new(api_key: String, model: &str) -> Result<Self> {
        let client = misanthropic::Client::new(api_key)
            .context("creating Anthropic client (is the API key 108 bytes?)")?;
        Ok(Self {
            client,
            model: model.to_string(),
        })
    }
}

#[async_trait]
impl LlmBackend for AnthropicBackend {
    async fn send(&self, prompt: &Prompt<'_>) -> Result<SendResponse> {
        assert_prompt_invariants(prompt);

        let start = std::time::Instant::now();
        let response = self
            .client
            .message(&prompt)
            .await
            .context("Anthropic API call failed")?;
        let elapsed = start.elapsed();

        log_usage(elapsed, response.usage, &prompt.model.to_string());

        Ok(response.into_static())
    }

    fn backend_name(&self) -> &str {
        "anthropic"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

/// Caller-side invariants the Anthropic API will reject (or silently
/// mis-cache) if violated. `assert!` (not `debug_assert!`) so the panic
/// reaches release builds — a wasted-prefill or 4xx round-trip is more
/// expensive than the check.
fn assert_prompt_invariants(prompt: &Prompt<'_>) {
    let count = count_cache_breakpoints(prompt);
    assert!(
        count <= MAX_CACHE_CONTROLS_PER_REQUEST,
        "expected at most {MAX_CACHE_CONTROLS_PER_REQUEST} cache_control markers \
         across tools + system + messages, got {count}",
    );

    // The system prefix is the prompt's largest shared chunk; the last
    // block must carry a cache_control marker for the prefix to be
    // cached at all. A missing marker here turns every request into a
    // full-price prefill.
    match &prompt.system {
        Some(Content::MultiPart(blocks)) => {
            let last = blocks.last().expect(
                "system Content::MultiPart with zero blocks is a misanthropic-side \
                 invariant violation",
            );
            assert!(
                last.is_cached(),
                "Prompt::system's last block must carry a cache_control marker \
                 to cache the system+tools prefix",
            );
        }
        Some(Content::SinglePart(_)) => {
            panic!(
                "Prompt::system is SinglePart — has no cache_control slot, so \
                 the system+tools prefix cannot be cached. Construct system via \
                 a MultiPart Content with cache_control on the last block.",
            );
        }
        None => {
            panic!(
                "Prompt::system is None — without a system block there is no \
                 prefix to cache. This is almost certainly a build_base_prompt \
                 regression.",
            );
        }
    }

    let messages = &prompt.messages;
    let n = messages.len();

    if n == 0 {
        return;
    }

    // Trailing message must be User — that's the message the model is
    // about to respond to. Pushing the assistant response and then
    // submitting again is a turn-order bug upstream.
    assert_eq!(
        messages[n - 1].role,
        Role::User,
        "final message must be User (the message the model is responding to); \
         got {:?} at index {}",
        messages[n - 1].role,
        n - 1,
    );

    // The asst at n-2 anchors the rolling cache window placed by
    // CachedPrompt::cache_windowed_with: with the trailing user at n-1,
    // its slide-by-2 lands the freshest marker at n-3, but the agora
    // post-loop call (line scheduler.rs:2675 at time of writing) places
    // its marker on the trailing-user/asst-2-back pair. Asserting both
    // the role and the marker catches a regression that would silently
    // un-cache the prior-round prefix.
    if n >= 2 {
        assert_eq!(
            messages[n - 2].role,
            Role::Assistant,
            "message at n-2 (index {}) must be Assistant; got {:?}",
            n - 2,
            messages[n - 2].role,
        );
        assert!(
            messages[n - 2].content.has_cache(),
            "message at n-2 (index {}, Assistant) must carry a cache_control marker",
            n - 2,
        );
    }

    if n >= 4 {
        assert_eq!(
            messages[n - 4].role,
            Role::Assistant,
            "message at n-4 (index {}) must be Assistant; got {:?}",
            n - 4,
            messages[n - 4].role,
        );
        assert!(
            messages[n - 4].content.has_cache(),
            "message at n-4 (index {}, Assistant) must carry a cache_control marker",
            n - 4,
        );
    }
}

/// Total `cache_control` markers across `tools` + `system` + `messages`.
fn count_cache_breakpoints(prompt: &Prompt<'_>) -> usize {
    let mut count = 0;

    if let Some(tools) = &prompt.functions {
        count += tools.iter().filter(|t| t.cache_control.is_some()).count();
    }

    if let Some(system) = &prompt.system
        && let Content::MultiPart(blocks) = system
    {
        count += blocks.iter().filter(|b| b.is_cached()).count();
    }

    for msg in &prompt.messages {
        if let Content::MultiPart(blocks) = &msg.content {
            count += blocks.iter().filter(|b| b.is_cached()).count();
        }
    }

    count
}
