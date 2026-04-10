//! Forgiving serde helpers for tool inputs.
//!
//! Small models occasionally produce inputs where an [`Option`] field is set
//! to the JSON string `"null"` instead of the JSON value `null`. Every other
//! serde error is worth relaying to the agent so it can self-correct, but
//! this particular footgun is common enough and harmless enough that it's
//! cheaper to paper over it than to burn a turn.

use serde::{Deserialize, Deserializer};

/// Deserialize an [`Option<T>`] while tolerating the string `"null"` and the
/// empty string as synonyms for `None`.
///
/// Use with `#[serde(default, deserialize_with = "forgiving_option")]`.
///
/// Valid inputs:
/// - JSON `null` → `None`
/// - field omitted (when paired with `#[serde(default)]`) → `None`
/// - JSON string `"null"` or `""` → `None`
/// - any other value → delegated to `T::deserialize`
pub fn forgiving_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) if s == "null" || s.is_empty() => Ok(None),
        Some(other) => T::deserialize(other)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Wrapper {
        #[serde(default, deserialize_with = "forgiving_option")]
        value: Option<u32>,
    }

    #[test]
    fn accepts_null() {
        let w: Wrapper = serde_json::from_str(r#"{"value": null}"#).unwrap();
        assert_eq!(w.value, None);
    }

    #[test]
    fn accepts_missing() {
        let w: Wrapper = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(w.value, None);
    }

    #[test]
    fn accepts_string_null() {
        let w: Wrapper = serde_json::from_str(r#"{"value": "null"}"#).unwrap();
        assert_eq!(w.value, None);
    }

    #[test]
    fn accepts_empty_string() {
        let w: Wrapper = serde_json::from_str(r#"{"value": ""}"#).unwrap();
        assert_eq!(w.value, None);
    }

    #[test]
    fn accepts_valid_value() {
        let w: Wrapper = serde_json::from_str(r#"{"value": 42}"#).unwrap();
        assert_eq!(w.value, Some(42));
    }

    #[test]
    fn rejects_garbage() {
        let err: Result<Wrapper, _> = serde_json::from_str(r#"{"value": "not a number"}"#);
        assert!(err.is_err());
    }
}
