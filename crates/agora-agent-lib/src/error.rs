//! Agent-friendly translation of `serde_path_to_error::Error`.
//!
//! When an agent's JSON output fails to deserialize, we feed back a short
//! plain-language error so it can fix the next attempt. Agents can't count
//! columns ("strawberry problem"), so we strip line/column noise and lean
//! on the structured field path that `serde_path_to_error` extracts.
//!
//! For unknown-enum-variant errors (e.g., `interests.communities[2]: unknown
//! variant 'technology'`), we run Levenshtein against the variants serde
//! itself listed in the error message and surface a "did you mean" suggestion.
//! This works for any enum, not just `Community`.
//!
//! Note: serde error messages aren't structured; we pattern-match on the
//! standard `serde_json` shapes ("missing field `…`", "unknown field `…`",
//! "unknown variant `…`, expected one of `…`, `…`"). If those shapes change
//! upstream, the translations degrade gracefully back to passthrough.

use std::fmt;

use serde_path_to_error::Error as PathError;

/// Render a `serde_path_to_error::Error` into a short message intended for
/// inclusion in a retry-feedback prompt.
///
/// Output shape: `"field `path.to.field`: human-readable hint"` (or just the
/// hint, if the path is empty).
pub fn format_for_agent<E: fmt::Display>(err: &PathError<E>) -> String {
    let path = err.path().to_string();
    let inner = err.inner().to_string();
    let translated = translate(&inner);
    if path.is_empty() || path == "." {
        translated
    } else {
        format!("field `{path}`: {translated}")
    }
}

fn translate(msg: &str) -> String {
    // "missing field `voice`" → friendlier
    if let Some(rest) = msg.strip_prefix("missing field `")
        && let Some(name) = rest.split('`').next()
    {
        return format!("field `{name}` is missing — add it");
    }
    // "unknown field `foo`, expected one of `…`, `…`"
    if let Some(rest) = msg.strip_prefix("unknown field `")
        && let Some(name) = rest.split('`').next()
    {
        return format!("field `{name}` is not in the schema — remove it");
    }
    // "unknown variant `technology`, expected one of `agi-asi`, `tech`, …"
    if let Some(rest) = msg.strip_prefix("unknown variant `")
        && let Some((variant, after)) = split_first_backtick(rest)
    {
        let candidates = extract_expected_list(after);
        if !candidates.is_empty() {
            if let Some(closest) = closest_match(variant, &candidates) {
                return format!("`{variant}` is not a valid value. Did you mean `{closest}`?");
            }
            let abbrev = abbreviate(&candidates);
            return format!("`{variant}` is not a valid value. Valid options: {abbrev}");
        }
        return format!("`{variant}` is not a valid value");
    }
    // "invalid type: …, expected …" — already pretty descriptive
    if msg.starts_with("invalid type:") {
        return msg.to_string();
    }
    // ShortString error format: "exceeds N chars (got M) — shorten and try again"
    if msg.contains("exceeds") && msg.contains("chars (got") {
        return msg.to_string();
    }
    // Fallback: pass through as-is
    msg.to_string()
}

/// Given a string starting after the first backtick was already stripped,
/// return `(content_until_next_backtick, rest_after_that_backtick)`.
fn split_first_backtick(s: &str) -> Option<(&str, &str)> {
    let end = s.find('`')?;
    Some((&s[..end], &s[end + 1..]))
}

/// Extract the expected-variants list from a serde error tail like
/// `, expected one of `a`, `b`, `c``.
fn extract_expected_list(tail: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = tail;
    while let Some(open) = rest.find('`') {
        rest = &rest[open + 1..];
        if let Some(close) = rest.find('`') {
            out.push(rest[..close].to_string());
            rest = &rest[close + 1..];
        } else {
            break;
        }
    }
    out
}

/// Find the closest candidate by Levenshtein distance (case-insensitive).
/// Only returns a suggestion when distance ≤ 3, to avoid noisy guesses.
fn closest_match(needle: &str, haystack: &[String]) -> Option<String> {
    let needle_lower = needle.to_lowercase();
    let mut best: Option<(usize, &str)> = None;
    for cand in haystack {
        let dist = strsim::levenshtein(&needle_lower, &cand.to_lowercase());
        if dist > 3 {
            continue;
        }
        match best {
            Some((d, _)) if d <= dist => {}
            _ => best = Some((dist, cand.as_str())),
        }
    }
    best.map(|(_, s)| s.to_string())
}

/// Abbreviate a long candidate list as `a, b, c, …, z` (first 3 + last).
fn abbreviate(items: &[String]) -> String {
    if items.len() <= 6 {
        return items
            .iter()
            .map(|s| format!("`{s}`"))
            .collect::<Vec<_>>()
            .join(", ");
    }
    let mut parts: Vec<String> = items.iter().take(3).map(|s| format!("`{s}`")).collect();
    parts.push("…".to_string());
    parts.push(format!("`{}`", items.last().unwrap()));
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn deserialize_err<T: serde::de::DeserializeOwned>(json: &str) -> String {
        let de = &mut serde_json::Deserializer::from_str(json);
        let err: PathError<serde_json::Error> = serde_path_to_error::deserialize::<_, T>(de)
            .err()
            .expect("should fail");
        format_for_agent(&err)
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct MissingFieldFixture {
        a: String,
        b: String,
    }

    #[test]
    fn missing_field_message() {
        let msg = deserialize_err::<MissingFieldFixture>(r#"{"a": "x"}"#);
        assert!(msg.contains("`b`"), "got: {msg}");
        assert!(msg.contains("missing"), "got: {msg}");
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    #[serde(deny_unknown_fields)]
    struct UnknownFieldFixture {
        a: String,
    }

    #[test]
    fn unknown_field_message() {
        let msg = deserialize_err::<UnknownFieldFixture>(r#"{"a": "x", "z": 1}"#);
        assert!(msg.contains("`z`"), "got: {msg}");
        assert!(msg.contains("not in the schema"), "got: {msg}");
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    enum Color {
        Red,
        Green,
        Blue,
    }

    #[test]
    fn unknown_variant_with_did_you_mean() {
        let msg = deserialize_err::<Color>(r#""Reed""#);
        assert!(msg.contains("`Reed`"), "got: {msg}");
        assert!(msg.contains("Did you mean"), "got: {msg}");
        assert!(msg.contains("`Red`"), "got: {msg}");
    }

    #[test]
    fn unknown_variant_without_close_match_lists_options() {
        let msg = deserialize_err::<Color>(r#""xyzqq""#);
        assert!(msg.contains("`xyzqq`"), "got: {msg}");
        assert!(msg.contains("Valid options"), "got: {msg}");
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct ShortStringFixture {
        s: crate::ShortString<3>,
    }

    #[test]
    fn shortstring_overflow_passes_through_with_path() {
        let msg = deserialize_err::<ShortStringFixture>(r#"{"s": "hello"}"#);
        assert!(msg.contains("`s`"), "got: {msg}");
        assert!(msg.contains("exceeds 3 chars"), "got: {msg}");
        assert!(msg.contains("got 5"), "got: {msg}");
    }

    #[test]
    fn invalid_type_passes_through() {
        let msg = deserialize_err::<MissingFieldFixture>(r#"{"a": 5, "b": "x"}"#);
        assert!(msg.contains("`a`"), "got: {msg}");
        assert!(
            msg.contains("invalid type") || msg.contains("expected"),
            "got: {msg}"
        );
    }

    #[test]
    fn levenshtein_threshold_ignores_far_matches() {
        // 4-char distance shouldn't trigger "did you mean"
        let candidates = vec!["red".to_string(), "blue".to_string()];
        let result = closest_match("totallyunrelated", &candidates);
        assert_eq!(result, None);
    }

    #[test]
    fn abbreviate_short_list_inline() {
        let items = vec!["a".to_string(), "b".to_string()];
        let result = abbreviate(&items);
        assert_eq!(result, "`a`, `b`");
    }

    #[test]
    fn abbreviate_long_list_truncates() {
        let items: Vec<String> = (b'a'..=b'z').map(|c| (c as char).to_string()).collect();
        let result = abbreviate(&items);
        assert!(result.starts_with("`a`, `b`, `c`, …"));
        assert!(result.ends_with("`z`"));
    }
}
