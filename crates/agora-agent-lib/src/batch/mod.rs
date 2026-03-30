//! Batch backend implementations.
//!
//! Provides [`BatchBackend`] implementations for different LLM providers:
//! - [`anthropic::AnthropicBatch`] — Anthropic Messages Batch API with prompt caching
//! - [`ollama::OllamaBatch`] — Sequential per model, parallel across models

pub mod anthropic;
pub mod ollama;
