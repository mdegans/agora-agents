//! The operator's model table: `[[model]]` in the run config.
//!
//! ```toml
//! [[model]]
//! id = "Qwen3.8-27B-UD-Q8_K_XL.gguf"
//! name = "Qwen 3.8 27B"
//! description = "Dense 27B; all parameters active per token. ~1/3 the speed of 3.6."
//! selectable = true   # offered to agents' set_model
//! share = 1.0         # weight in the fair-share schedule
//! ```
//!
//! A model is **routable** if it is listed (and some endpoint advertises
//! it). It is **selectable** — offered to an agent's `set_model` — only if
//! it is listed with `selectable = true` *and* advertised at startup, which
//! keeps `-Base` models, a duplicate `model.gguf`, an mmproj and the like
//! out of the menu without anyone having to remember them.
//!
//! The legacy `models = ["…"]` list is still read for one release, as
//! routable entries with no name or description (never selectable).

use misanthropic::model::{Model, ModelInfo};
use serde::Deserialize;

/// One `[[model]]` entry.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    /// Exact model id, as the endpoint advertises it and routing matches it.
    pub id: Model,
    /// What agents are told the model is called. Defaults to the endpoint's
    /// display name, then the id.
    pub name: Option<String>,
    /// The operator's description, shown in `set_model`. Required when
    /// `selectable`.
    pub description: Option<String>,
    /// Offered to agents' `set_model` (when also advertised).
    #[serde(default)]
    pub selectable: bool,
    /// Weight in the fair-share schedule; `1.0` is an equal share.
    #[serde(default = "default_share")]
    pub share: f64,
}

fn default_share() -> f64 {
    1.0
}

impl ModelSpec {
    /// A legacy `models = [...]` entry: routable, nothing more.
    pub fn legacy(id: &str) -> Self {
        Self {
            id: Model::from(id.to_string()),
            name: None,
            description: None,
            selectable: false,
            share: default_share(),
        }
    }
}

/// Check the table once at load: ids unique, shares positive, and every
/// selectable model described (the agent decides on the description).
pub fn validate(specs: &[ModelSpec]) -> anyhow::Result<()> {
    for (i, spec) in specs.iter().enumerate() {
        let id = spec.id.name();
        anyhow::ensure!(!id.is_empty(), "[[model]] #{}: `id` is empty", i + 1);
        anyhow::ensure!(
            !specs[..i].iter().any(|s| s.id == spec.id),
            "[[model]] {id}: listed twice"
        );
        anyhow::ensure!(
            spec.share.is_finite() && spec.share > 0.0,
            "[[model]] {id}: `share` must be a positive number"
        );
        anyhow::ensure!(
            !spec.selectable
                || spec
                    .description
                    .as_deref()
                    .is_some_and(|d| !d.trim().is_empty()),
            "[[model]] {id}: a selectable model needs a `description` — agents \
             choose on it"
        );
    }
    Ok(())
}

/// A model this run can route agents onto.
#[derive(Debug, Clone)]
pub struct Entry {
    /// As the endpoint advertised it.
    pub info: ModelInfo,
    /// See [`ModelSpec::name`].
    pub name: String,
    pub description: Option<String>,
    /// Listed `selectable` (and, being here, advertised).
    pub selectable: bool,
}

/// The model table resolved against what the endpoints advertise at
/// startup: every model an agent could be routed on this run.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    entries: Vec<Entry>,
}

impl Catalog {
    /// `specs` empty means no allowlist: everything advertised is routable
    /// (the runner's rule when no model is listed).
    pub fn new(specs: &[ModelSpec], advertised: &[ModelInfo]) -> Self {
        let mut entries: Vec<Entry> = Vec::new();
        for info in advertised {
            if entries.iter().any(|e| e.info.id == info.id) {
                continue; // offered by two endpoints; routing takes the first
            }
            let spec = specs.iter().find(|s| s.id == info.id);
            if spec.is_none() && !specs.is_empty() {
                continue;
            }
            let display = info.display_name.to_string();
            let name = spec
                .and_then(|s| s.name.clone())
                .or_else(|| (!display.is_empty()).then_some(display))
                .unwrap_or_else(|| info.id.name().to_string());
            entries.push(Entry {
                info: info.clone(),
                name,
                description: spec.and_then(|s| s.description.clone()),
                selectable: spec.is_some_and(|s| s.selectable),
            });
        }
        Self { entries }
    }

    /// The entry for `id`, if the run can route onto it
    pub fn get(&self, id: &Model) -> Option<&Entry> {
        self.entries.iter().find(|e| &e.info.id == id)
    }

    /// What to call `id`: its name here, else the id itself
    pub fn name_of(&self, id: &Model) -> String {
        self.get(id)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| id.name().to_string())
    }

    /// Models an agent may choose, in table order of advertisement
    pub fn selectable(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(|e| e.selectable)
    }

    /// A selectable model by exact id, else by name (case-insensitive) —
    /// agents quote either
    pub fn choose(&self, wanted: &str) -> Option<&Entry> {
        let wanted = wanted.trim();
        self.selectable()
            .find(|e| e.info.id.name() == wanted)
            .or_else(|| {
                self.selectable()
                    .find(|e| e.name.eq_ignore_ascii_case(wanted))
            })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use misanthropic::model::Kind;

    pub(crate) fn info(id: &str) -> ModelInfo {
        ModelInfo {
            id: Model::from(id.to_string()),
            display_name: String::new().into(),
            capabilities: Default::default(),
            max_input_tokens: 0,
            max_tokens: 0,
            kind: Kind::Model,
            created_at: chrono::DateTime::from_timestamp(0, 0).unwrap(),
        }
    }

    fn specs(toml_src: &str) -> Vec<ModelSpec> {
        #[derive(Deserialize)]
        struct T {
            model: Vec<ModelSpec>,
        }
        toml::from_str::<T>(toml_src).unwrap().model
    }

    const TABLE: &str = r#"
        [[model]]
        id = "Qwen3.6.gguf"
        name = "Qwen 3.6"
        description = "Sparse."
        selectable = true

        [[model]]
        id = "Qwen3.8.gguf"
        name = "Qwen 3.8"
        description = "Dense."
        selectable = true
        share = 2.0

        [[model]]
        id = "Qwen3.8-Base.gguf"

        [[model]]
        id = "gone.gguf"
        description = "Not served today."
        selectable = true
    "#;

    #[test]
    fn selectable_means_listed_selectable_and_advertised() {
        let specs = specs(TABLE);
        validate(&specs).unwrap();
        assert_eq!(specs[1].share, 2.0);
        assert_eq!(specs[0].share, 1.0, "default share");
        let advertised = [
            info("Qwen3.6.gguf"),
            info("Qwen3.8.gguf"),
            info("Qwen3.8-Base.gguf"),
            info("model.gguf"),
        ];
        let catalog = Catalog::new(&specs, &advertised);
        let ids: Vec<&str> = catalog.selectable().map(|e| e.info.id.name()).collect();
        assert_eq!(ids, ["Qwen3.6.gguf", "Qwen3.8.gguf"]);
        // Listed, not selectable: routable only.
        assert!(catalog.get(&Model::from("Qwen3.8-Base.gguf")).is_some());
        // Advertised, not listed: not routable.
        assert!(catalog.get(&Model::from("model.gguf")).is_none());
        // Listed selectable but not advertised: neither.
        assert!(catalog.get(&Model::from("gone.gguf")).is_none());
        assert_eq!(catalog.name_of(&Model::from("Qwen3.8.gguf")), "Qwen 3.8");
        assert_eq!(catalog.name_of(&Model::from("model.gguf")), "model.gguf");

        assert_eq!(
            catalog.choose("qwen 3.8").unwrap().info.id.name(),
            "Qwen3.8.gguf"
        );
        assert_eq!(
            catalog.choose(" Qwen3.6.gguf ").unwrap().info.id.name(),
            "Qwen3.6.gguf"
        );
        assert!(catalog.choose("Qwen3.8-Base.gguf").is_none());
    }

    #[test]
    fn no_table_routes_everything_selects_nothing() {
        let catalog = Catalog::new(&[], &[info("a.gguf"), info("b.gguf")]);
        assert!(catalog.get(&Model::from("b.gguf")).is_some());
        assert_eq!(catalog.selectable().count(), 0);
    }

    #[test]
    fn validation_catches_what_would_mislead_an_agent() {
        let undescribed = specs("[[model]]\nid = \"a\"\nselectable = true\n");
        assert!(validate(&undescribed).is_err());
        let twice = specs("[[model]]\nid = \"a\"\n[[model]]\nid = \"a\"\n");
        assert!(validate(&twice).is_err());
        let zero = specs("[[model]]\nid = \"a\"\nshare = 0.0\n");
        assert!(validate(&zero).is_err());
        #[derive(Deserialize, Debug)]
        #[allow(dead_code)]
        struct T {
            model: Vec<ModelSpec>,
        }
        assert!(
            toml::from_str::<T>("[[model]]\nid = \"a\"\nselectible = true\n").is_err(),
            "typos fail at load"
        );
    }
}
