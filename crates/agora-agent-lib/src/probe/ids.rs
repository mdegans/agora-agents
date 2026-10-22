//! Typed identifiers for probe instruments.
//!
//! Item slugs and axis tags are Council-owned identifiers from the
//! questionnaire and scenario JSON; they key baselines, per-axis
//! aggregation and the distribution readings. Wrapping them keeps an
//! item slug from being passed where an axis tag (or a model id) was
//! meant. The wire form is the bare string (`serde(transparent)`), so
//! the JSON files and older baselines are unchanged.

use serde::{Deserialize, Serialize};

macro_rules! define_slug {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                // `pad` so width/alignment specifiers work in tables.
                f.pad(&self.0)
            }
        }

        impl std::ops::Deref for $name {
            type Target = str;
            fn deref(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }

        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.0 == other
            }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.0 == *other
            }
        }
    };
}

define_slug!(
    /// Stable slug of one rated item (a questionnaire statement or a
    /// scenario claim), e.g. `"trans_women_are_women"`. Never sent to
    /// the model.
    ItemSlug
);

define_slug!(
    /// An axis tag on a scenario claim, e.g. `"self_determination"`,
    /// `"filler_control_true"`, `"authority_deference"`.
    AxisTag
);

define_slug!(
    /// Stable slug of an indirect-probe scenario, e.g.
    /// `"velkor_drummond"`.
    ScenarioSlug
);
