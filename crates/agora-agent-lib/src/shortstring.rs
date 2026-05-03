//! Length-bounded string newtype.
//!
//! `ShortString<MAX>` wraps a `String` and rejects values longer than `MAX`
//! chars at deserialize time, with an error message that includes the
//! observed length. This gives us field-level length budgets that produce
//! useful retry feedback for agents — Anthropic strips JSON Schema
//! `maxLength` keywords (so we can't enforce in-grammar), but a custom
//! `Deserialize` runs unconditionally on every backend.
//!
//! Note: we hand-write `Serialize` and `Deserialize` rather than using
//! `#[serde(transparent)]` because (a) the custom-error message wants
//! field-path context that the derive doesn't surface, and (b) Mike
//! flagged uncertainty about composing `transparent` with a custom
//! `Deserialize`. The hand-written impls forward to `String`'s impls
//! directly so wire shape is identical.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A `String` with a compile-time maximum character length.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct ShortString<const MAX: usize>(String);

impl<const MAX: usize> ShortString<MAX> {
    /// Construct a `ShortString`, returning an error if the input exceeds `MAX`.
    pub fn new(s: impl Into<String>) -> Result<Self, ShortStringError> {
        let s = s.into();
        let len = s.chars().count();
        if len > MAX {
            return Err(ShortStringError {
                got: len,
                max: MAX,
            });
        }
        Ok(ShortString(s))
    }

    /// The underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume into the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Compile-time max length.
    pub const MAX: usize = MAX;
}

#[derive(Debug, Clone)]
pub struct ShortStringError {
    pub got: usize,
    pub max: usize,
}

impl core::fmt::Display for ShortStringError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "exceeds {} chars (got {}) — shorten and try again",
            self.max, self.got,
        )
    }
}

impl std::error::Error for ShortStringError {}

impl<const MAX: usize> core::ops::Deref for ShortString<MAX> {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl<const MAX: usize> AsRef<str> for ShortString<MAX> {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl<const MAX: usize> core::fmt::Display for ShortString<MAX> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl<const MAX: usize> Serialize for ShortString<MAX> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de, const MAX: usize> Deserialize<'de> for ShortString<MAX> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        ShortString::<MAX>::new(s).map_err(serde::de::Error::custom)
    }
}

// schemars: emit a plain `string` schema with maxLength. Anthropic strips
// maxLength from output_format schemas, but blallama/Ollama benefit, and
// the maxLength is informative when the schema is included as text in the
// prompt for documentation purposes.
impl<const MAX: usize> schemars::JsonSchema for ShortString<MAX> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(format!("ShortString_{MAX}"))
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "maxLength": MAX,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_under_max() {
        let s: ShortString<10> = serde_json::from_str("\"hello\"").unwrap();
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn deserialize_at_max() {
        let s: ShortString<5> = serde_json::from_str("\"hello\"").unwrap();
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn deserialize_over_max_errors_with_lengths() {
        let err = serde_json::from_str::<ShortString<3>>("\"hello\"").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("3"), "message missing max: {msg}");
        assert!(msg.contains("5"), "message missing got: {msg}");
        assert!(msg.contains("shorten"), "message missing hint: {msg}");
    }

    #[test]
    fn serialize_is_transparent() {
        let s = ShortString::<10>::new("hi").unwrap();
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"hi\"");
    }

    #[test]
    fn unicode_counted_by_chars_not_bytes() {
        // "héllo" = 5 chars but 6 bytes
        let s: ShortString<5> = serde_json::from_str("\"héllo\"").unwrap();
        assert_eq!(s.chars().count(), 5);
    }

    #[test]
    fn deref_works() {
        let s = ShortString::<10>::new("test").unwrap();
        let upper: String = s.to_uppercase();
        assert_eq!(upper, "TEST");
    }

    #[test]
    fn schema_includes_max_length() {
        let mut generator = schemars::SchemaGenerator::default();
        let schema = <ShortString<256> as schemars::JsonSchema>::json_schema(&mut generator);
        let json = serde_json::to_value(&schema).unwrap();
        assert_eq!(json["maxLength"], 256);
        assert_eq!(json["type"], "string");
    }
}
