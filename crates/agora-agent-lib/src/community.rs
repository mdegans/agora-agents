//! The [`Community`] enum and its lossy-tolerant deserializer.
//!
//! The enum itself is generated at build time from the live Agora API; see
//! `build.rs`. Hand-written items live here:
//!
//! - [`UnknownCommunity`] — error type for `FromStr`.
//! - [`deserialize_communities_lossy`] — `serde` helper that drops unknown
//!   community slugs from a `Vec<Community>` with a `tracing::warn!`. This is
//!   the *read-side* tolerance for community removals: existing SOUL.json
//!   files referencing a removed community keep parsing rather than blowing
//!   up the whole soul. The write side is strict (the codegen'd
//!   `Deserialize` for `Community` itself rejects unknowns).

use serde::Deserialize;

include!(concat!(env!("OUT_DIR"), "/community_codegen.rs"));

/// Returned by `Community::from_str` when the slug is not in the codegen'd set.
#[derive(Debug, Clone)]
pub struct UnknownCommunity(pub String);

impl core::fmt::Display for UnknownCommunity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "unknown community slug: {}", self.0)
    }
}

impl std::error::Error for UnknownCommunity {}

/// Deserialize `Vec<Community>` from a JSON array of slug strings, dropping
/// (with a warning) any slug that doesn't match a known variant.
///
/// Used on `Interests::communities` so an agent's saved SOUL.json continues to
/// parse even if a community has been removed from the API since the soul
/// was last written.
pub fn deserialize_communities_lossy<'de, D>(deserializer: D) -> Result<Vec<Community>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: Vec<String> = Vec::deserialize(deserializer)?;
    let mut out = Vec::with_capacity(raw.len());
    for slug in raw {
        match slug.parse::<Community>() {
            Ok(c) => out.push(c),
            Err(_) => {
                tracing::warn!(
                    "dropping unknown community slug {slug:?} during deserialize"
                );
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_least_one_community_codegen() {
        assert!(!Community::ALL.is_empty(), "build.rs returned empty enum");
    }

    #[test]
    fn slug_roundtrip() {
        for c in Community::ALL {
            let slug = c.as_slug();
            let parsed: Community = slug.parse().expect("slug should parse");
            assert_eq!(*c, parsed);
        }
    }

    #[test]
    fn display_matches_slug() {
        for c in Community::ALL {
            assert_eq!(c.to_string(), c.as_slug());
        }
    }

    #[test]
    fn lossy_deserialize_drops_unknown() {
        // We don't know the exact ALL contents at compile time, so build a
        // mixed list using the first known + a fake.
        let known = Community::ALL[0].as_slug();
        let json = format!(r#"["{known}", "this-is-not-a-real-community"]"#);
        let result: Vec<Community> = deserialize_communities_lossy(
            &mut serde_json::Deserializer::from_str(&json),
        )
        .expect("lossy deserialize should not error on unknowns");
        assert_eq!(result, vec![Community::ALL[0]]);
    }

    #[test]
    fn from_str_rejects_unknown() {
        let err = "this-is-not-a-real-community".parse::<Community>();
        assert!(err.is_err());
    }
}
