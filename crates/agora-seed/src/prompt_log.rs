//! Content-addressed prompt logging.
//!
//! After every agent cycle, the fully-assembled prompt is serialized to
//! `logs/prompts/{first_2_hex}/{sha256_hex}.json`, sharded by the first
//! two hex characters of its SHA-256 hash. Duplicate prompts (identical
//! content, which happens often thanks to the shared cached prefix)
//! collapse to a single file, and individual agent runs can be replayed
//! deterministically by looking up the prompt hash printed at info level.

use sha2::Digest;

/// Save the prompt to a content-addressed JSON file and return the path.
///
/// Returns `None` on any I/O / serialization failure — saving is best-
/// effort and should never break a cycle.
pub async fn save(
    prompt: &misanthropic::Prompt<'_>,
    agent_name: &str,
) -> Option<std::path::PathBuf> {
    let json = serde_json::to_vec_pretty(prompt).ok()?;
    let hash = hex::encode(sha2::Sha256::digest(&json));
    let dir = std::path::PathBuf::from("logs/prompts").join(&hash[..2]);
    tokio::fs::create_dir_all(&dir).await.ok()?;
    let path = dir.join(format!("{hash}.json"));

    if !path.exists() {
        tokio::fs::write(&path, &json).await.ok()?;
    }

    tracing::info!("{agent_name} prompt saved: {}", path.display());

    // Pretty-print markdown at debug level. CachedPrompt derefs to Prompt
    // which implements ToMarkdown via the `markdown` feature.
    {
        use misanthropic::markdown::ToMarkdown;
        tracing::debug!("{agent_name} prompt:\n{}", prompt.markdown_verbose());
    }

    Some(path)
}
