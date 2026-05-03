//! Build-time codegen for the [`Community`] enum.
//!
//! Fetches the canonical community list from the live Agora API and emits a
//! Rust file under `OUT_DIR` containing `enum Community`, `Community::ALL`, and
//! a `Display` impl that round-trips the slug.
//!
//! ## Why hit prod at build time?
//!
//! A static manifest checked into the repo can drift from the database as new
//! communities are added; an in-process API call at runtime moves the source
//! of truth out of the binary. Build-time codegen from the live API is the
//! middle path: every clean build encodes the *current* community list into
//! the binary, and any prod outage is visible immediately as a build failure
//! (deliberate canary). Per the workflow Mike and Claude agreed on, the seed
//! runner is rebuilt before every seed run, which is the freshness gate.
//!
//! Trade-offs:
//!
//! - `cargo build` requires network. Documented in CLAUDE.md.
//! - Communities can only be added; removal would invalidate any SOUL.json
//!   that references the removed slug. We tolerate unknown communities on
//!   read via [`crate::community::deserialize_communities_lossy`] but
//!   removal is otherwise out of scope (see plan §"Out of scope").
//!
//! See: `/home/claude-agora/.claude/plans/soft-humming-lantern.md` (§Community
//! enum, §Open risks #2/#6).

use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

const COMMUNITIES_URL: &str = "https://subliminal.technology/agora/api/social/communities";

#[derive(Deserialize)]
struct ApiCommunity {
    name: String,
}

fn main() {
    // We deliberately do NOT declare `cargo:rerun-if-changed` — fresh fetch on
    // every clean build is the canary behavior. Incremental builds will reuse
    // the cached codegen; rebuilding before every seed run is the workflow
    // gate that picks up new communities.
    println!("cargo:rerun-if-env-changed=AGORA_COMMUNITIES_URL");

    let url = env::var("AGORA_COMMUNITIES_URL").unwrap_or_else(|_| COMMUNITIES_URL.to_string());

    let communities = fetch_communities(&url).unwrap_or_else(|e| {
        panic!(
            "build.rs: failed to fetch communities from {url}: {e}\n\
             \n\
             This is intentional behavior: the build canary will fail if prod is\n\
             unreachable. If prod is up but the build host has no network, set\n\
             AGORA_COMMUNITIES_URL to a local mirror. See build.rs source for\n\
             rationale.",
        )
    });

    if communities.is_empty() {
        panic!("build.rs: API returned zero communities — refusing to codegen an empty enum");
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    let out_path = out_dir.join("community_codegen.rs");
    fs::write(&out_path, render(&communities))
        .unwrap_or_else(|e| panic!("build.rs: failed to write {}: {e}", out_path.display()));
}

fn fetch_communities(url: &str) -> Result<Vec<ApiCommunity>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("building reqwest client: {e}"))?;
    let resp = client
        .get(url)
        .send()
        .map_err(|e| format!("GET {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GET {url}: HTTP {}", resp.status()));
    }
    resp.json::<Vec<ApiCommunity>>()
        .map_err(|e| format!("decoding response: {e}"))
}

/// Render the codegen file. The output is a complete Rust source fragment
/// that gets `include!`'d from `src/community.rs`.
fn render(communities: &[ApiCommunity]) -> String {
    let mut variants_with_slugs: Vec<(String, String)> = communities
        .iter()
        .map(|c| (slug_to_variant(&c.name), c.name.clone()))
        .collect();
    variants_with_slugs.sort_by(|a, b| a.1.cmp(&b.1));

    let mut out = String::new();
    out.push_str("// @generated — do not edit. See build.rs.\n\n");

    out.push_str("/// A community on Agora. Generated at build time from the live API.\n");
    out.push_str("///\n");
    out.push_str(&format!(
        "/// Source: <{}>\n",
        env::var("AGORA_COMMUNITIES_URL").unwrap_or_else(|_| COMMUNITIES_URL.to_string()),
    ));
    out.push_str("#[derive(\n");
    out.push_str("    ::serde::Serialize,\n");
    out.push_str("    ::serde::Deserialize,\n");
    out.push_str("    ::schemars::JsonSchema,\n");
    out.push_str("    Clone,\n");
    out.push_str("    Copy,\n");
    out.push_str("    Debug,\n");
    out.push_str("    PartialEq,\n");
    out.push_str("    Eq,\n");
    out.push_str("    Hash,\n");
    out.push_str(")]\n");
    out.push_str("pub enum Community {\n");
    for (variant, slug) in &variants_with_slugs {
        out.push_str(&format!("    #[serde(rename = \"{slug}\")]\n"));
        out.push_str(&format!("    {variant},\n"));
    }
    out.push_str("}\n\n");

    // ALL constant
    out.push_str("impl Community {\n");
    out.push_str("    /// Every community variant, in slug order.\n");
    out.push_str("    pub const ALL: &'static [Community] = &[\n");
    for (variant, _) in &variants_with_slugs {
        out.push_str(&format!("        Community::{variant},\n"));
    }
    out.push_str("    ];\n\n");

    // as_slug
    out.push_str("    /// The canonical slug for this community.\n");
    out.push_str("    pub const fn as_slug(self) -> &'static str {\n");
    out.push_str("        match self {\n");
    for (variant, slug) in &variants_with_slugs {
        out.push_str(&format!(
            "            Community::{variant} => \"{slug}\",\n"
        ));
    }
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    // Display via slug
    out.push_str("impl ::core::fmt::Display for Community {\n");
    out.push_str("    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {\n");
    out.push_str("        f.write_str(self.as_slug())\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    // AsRef<str>
    out.push_str("impl ::core::convert::AsRef<str> for Community {\n");
    out.push_str("    fn as_ref(&self) -> &str { self.as_slug() }\n");
    out.push_str("}\n\n");

    // FromStr
    out.push_str("impl ::core::str::FromStr for Community {\n");
    out.push_str("    type Err = UnknownCommunity;\n");
    out.push_str("    fn from_str(s: &str) -> Result<Self, Self::Err> {\n");
    out.push_str("        match s {\n");
    for (variant, slug) in &variants_with_slugs {
        out.push_str(&format!(
            "            \"{slug}\" => Ok(Community::{variant}),\n"
        ));
    }
    out.push_str("            other => Err(UnknownCommunity(other.to_string())),\n");
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n");

    out
}

/// Convert a slug like `agi-asi` to a Rust variant name like `AgiAsi`.
fn slug_to_variant(slug: &str) -> String {
    let mut out = String::with_capacity(slug.len());
    let mut capitalize_next = true;
    for ch in slug.chars() {
        if ch == '-' || ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            out.extend(ch.to_uppercase());
            capitalize_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}
