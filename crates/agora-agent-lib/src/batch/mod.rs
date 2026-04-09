//! Batch backend implementations.
//!
//! - [`anthropic::AnthropicBatch`] — Anthropic Messages Batch API with prompt caching
//! - [`ollama::OllamaEndpoint`] — Sequential per-agent Ollama endpoint (Anthropic-compat API)

pub mod anthropic;
pub mod ollama;
