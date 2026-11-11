//! STUB (step B3), to be replaced whole by step B2's roster:
//! `<data_dir>/sealed-roster.json`, each chained agent's id with its head's
//! seq and hash, the witness outside `state/` that a rolled-back or missing
//! agent directory is checked against. B3 calls [`record`] when an agent's
//! chain completes (and after a resume or a later pre-loss copy moves its
//! head); this stub records nothing.

use std::path::Path;

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::verify::HeadRef;

/// B2: record `id`'s head in the roster under `data_dir`.
pub fn record(data_dir: &Path, id: AgentId, head: HeadRef) -> std::io::Result<()> {
    let _ = (data_dir, id, head);
    Ok(())
}
