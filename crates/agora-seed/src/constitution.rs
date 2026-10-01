//! The constitution check, run before every sweep (Steward request,
//! 2026-10-01).
//!
//! Fetches the constitution the server serves, builds the seed system text
//! from it with [`seed::system_text`] (the function every agent's prompt is
//! assembled with), and compares the SHA-256 of the served text with that of
//! the copy [`seed::embedded_constitution`] finds in the system text. Both
//! digests are logged with the version as one `constitution_embedded` event.
//! A mismatch (a truncated or stale copy) or a check that cannot complete
//! logs an ERROR `constitution_refused`, alerts, and stops the run before any
//! agent acts. Each agent's own assembly repeats the byte comparison
//! (`SeedError::Constitution`), so a copy fetched later can't slip through
//! either.

use agora_agentkit::client::Client;
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::reactor::seed;
use anyhow::Context;

use crate::alerts::{Alert, AlertKind, Alerter};

/// The served and embedded constitution, as digests
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Embedding {
    pub served_sha256: Sha256Hex,
    pub served_bytes: usize,
    /// `None` when the system text carries no constitution at all
    pub embedded_sha256: Option<Sha256Hex>,
    pub embedded_bytes: usize,
}

impl Embedding {
    /// Digest `served` and the constitution embedded in `system`
    pub fn of(served: &str, system: &str) -> Self {
        let embedded = seed::embedded_constitution(system);
        Self {
            served_sha256: seed::constitution_sha256(served),
            served_bytes: served.len(),
            embedded_sha256: embedded.map(seed::constitution_sha256),
            embedded_bytes: embedded.map_or(0, str::len),
        }
    }

    pub fn matches(&self) -> bool {
        self.embedded_sha256 == Some(self.served_sha256)
    }
}

/// Check the constitution. `Err` means the run must stop; both failures
/// log `constitution_refused` and alert, and `main` waits for the mail.
pub async fn verify(client: &Client, web_tools: bool, alerts: &Alerter) -> anyhow::Result<()> {
    let refuse = |summary: &'static str, alert: Alert| {
        alerts.notify(alert);
        anyhow::anyhow!("refusing to run: {summary}")
    };
    let (version, embedding) = match fetch(client, web_tools).await {
        Ok(found) => found,
        Err(e) => {
            const SUMMARY: &str = "REFUSING TO RUN: the constitution check did not complete";
            let error = format!("{e:#}");
            tracing::error!(event_type = "constitution_refused", error = %error, "{SUMMARY}");
            return Err(refuse(
                "the constitution check did not complete",
                Alert::new(AlertKind::ConstitutionRefused, SUMMARY).detail("error", error),
            ));
        }
    };
    let embedded = embedding
        .embedded_sha256
        .map_or_else(|| "none".to_string(), |h| h.to_hex());
    tracing::info!(
        event_type = "constitution_embedded",
        constitution_version = %version,
        served_sha256 = %embedding.served_sha256,
        embedded_sha256 = %embedded,
        served_bytes = embedding.served_bytes,
        embedded_bytes = embedding.embedded_bytes,
        matches = embedding.matches(),
        "constitution embedded in the seed system prompt"
    );
    if embedding.matches() {
        return Ok(());
    }
    const SUMMARY: &str =
        "REFUSING TO RUN: the constitution in the system prompt is not the served text";
    tracing::error!(
        event_type = "constitution_refused",
        constitution_version = %version,
        served_sha256 = %embedding.served_sha256,
        embedded_sha256 = %embedded,
        served_bytes = embedding.served_bytes,
        embedded_bytes = embedding.embedded_bytes,
        "{SUMMARY}"
    );
    Err(refuse(
        "the constitution in the system prompt is not the served text",
        Alert::new(AlertKind::ConstitutionRefused, SUMMARY)
            .detail("constitution_version", &version)
            .detail("served_sha256", embedding.served_sha256)
            .detail("embedded_sha256", &embedded)
            .detail("served_bytes", embedding.served_bytes)
            .detail("embedded_bytes", embedding.embedded_bytes),
    ))
}

/// The served version, and the [`Embedding`] of the system text built from
/// what the server serves now
async fn fetch(client: &Client, web_tools: bool) -> anyhow::Result<(String, Embedding)> {
    let served = client
        .get_constitution(None)
        .await
        .context("fetching the constitution")?;
    let communities: Vec<String> = client
        .list_communities()
        .await
        .context("listing communities")?
        .into_iter()
        .map(|c| c.name)
        .collect();
    let system = seed::system_text(&served.text, &communities, web_tools);
    Ok((served.version, Embedding::of(&served.text, &system)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Long enough that a cap at any plausible size would cut it
    fn constitution() -> String {
        let mut text = String::from("# The Agora Constitution\n\n**Version 0.5**\n\n");
        let mut n = 0;
        while text.len() < 40 * 1024 {
            n += 1;
            text.push_str(&format!("{n}. Article II § {n} — rights « retained ».\n"));
        }
        text
    }

    #[test]
    fn the_assembled_system_text_matches_the_served_text() {
        let served = constitution();
        let system = seed::system_text(&served, &["tech".to_string()], true);
        let embedding = Embedding::of(&served, &system);
        assert!(embedding.matches(), "{embedding:?}");
        assert_eq!(embedding.embedded_bytes, served.len());
    }

    #[test]
    fn a_truncated_copy_does_not_match() {
        let served = constitution();
        let truncated = seed::system_text(&served[..served.len() / 2], &[], false);
        let embedding = Embedding::of(&served, &truncated);
        assert!(!embedding.matches());
        assert_eq!(embedding.served_bytes, served.len());
        assert_eq!(embedding.embedded_bytes, served.len() / 2);
    }

    #[test]
    fn a_stale_copy_does_not_match() {
        let served = constitution();
        let stale = seed::system_text(&served.replace("0.5", "0.4"), &[], false);
        assert!(!Embedding::of(&served, &stale).matches());
    }

    #[test]
    fn a_system_text_without_a_constitution_does_not_match() {
        let embedding = Embedding::of(&constitution(), "## What You Are\n\nnothing else");
        assert_eq!(embedding.embedded_sha256, None);
        assert!(!embedding.matches());
    }
}
