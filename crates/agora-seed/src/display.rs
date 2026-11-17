//! What each session's agent is shown others' text with, and the web tools
//! that follow from it.
//!
//! ```toml
//! [display]                      # the default for every agent
//! links = { mode = "defanged", allow = ["https://*.wikipedia.org"] }
//!
//! [display.profiles.open]        # a named profile: replaces the default whole
//! links = { mode = "raw" }
//!
//! [[model]]
//! id = "claude-haiku-5-5"
//! display = "open"               # a profile, "default", or a table in place
//!
//! [agent_display]                # one agent's choice, over its model's
//! "9726e530-7df8-4f5a-8705-1164fd092fdf" = "default"
//! ```
//!
//! `[display]` is agentkit's [`AgentDisplayProfiles`], read as it is; absent,
//! agentkit's default (every off-site link defanged). Of an agent's choice,
//! its model's and the default, the first there is applies, whole: nothing
//! is merged ([`AgentDisplayProfiles::resolve`]). Every choice is checked
//! when the config is loaded ([`DisplayPlan::new`]), so a misspelt profile
//! stops the run before any session starts.
//!
//! **The web tools.** Anthropic runs `web_fetch` and `web_search` itself, and
//! a link in a fetched page is fetchable on the next hop, so the tools' own
//! domain lists are the only gate there. A session's lists are its display
//! policy's ([`Links::fetch_domains`], [`Links::search_domains`]) unless the
//! operator wrote either list on that tool's `[seed.*]` table: then those are
//! used as written, and the policy's are not ([`WebSource::Operator`]). A
//! policy that cannot be sent as a list withholds the tool from the session
//! ([`ToolOffer::Withhold`]). A tool the operator did not enable stays off.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

use agora_agentkit::display::{
    AgentDisplayProfiles, DisplaySelection, DomainFilter, Links, ResolvedDisplay, ServerTool,
    ToolDomains, ToolOffer, UnknownProfile,
};
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::SeedConfig;
use misanthropic::model::Model;

use crate::models::ModelSpec;

/// Where a web tool's domain lists come from, for every session
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSource {
    /// The tool's `[seed.*]` table is absent: no session has it
    Off,
    /// Enabled, with no list written: each session's display policy sets
    /// the lists, or withholds the tool
    Policy,
    /// Enabled with `allowed_domains` or `blocked_domains` written: used as
    /// written, the display policy's list is not applied
    Operator,
}

/// [`WebSource`] for both tools
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebSources {
    pub fetch: WebSource,
    pub search: WebSource,
}

impl WebSources {
    fn of(&self, tool: ServerTool) -> WebSource {
        match tool {
            ServerTool::WebFetch => self.fetch,
            ServerTool::WebSearch => self.search,
        }
    }
}

/// The config table a tool is enabled by
fn table(tool: ServerTool) -> &'static str {
    match tool {
        ServerTool::WebFetch => "[seed.web_fetch]",
        ServerTool::WebSearch => "[seed.web_search]",
    }
}

/// A tool's name as the logs carry it
fn tool_field(tool: ServerTool) -> &'static str {
    match tool {
        ServerTool::WebFetch => "web_fetch",
        ServerTool::WebSearch => "web_search",
    }
}

/// The run's display config, checked: the profiles, each model's choice
/// and each agent's
#[derive(Debug, Clone, Default)]
pub struct DisplayPlan {
    profiles: AgentDisplayProfiles,
    /// `[[model]] display`, in table order
    models: Vec<(Model, DisplaySelection)>,
    /// `[agent_display]`
    agents: BTreeMap<AgentId, DisplaySelection>,
}

/// What became of one web tool in one session
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOutcome {
    /// Not enabled for the run
    Off,
    /// The operator's lists, as written
    Operator,
    /// The policy's lists (both `None`: unfiltered, as configured)
    Offered,
    /// The policy cannot be sent as a list: no such tool this session
    Withheld,
}

/// What [`DisplayPlan::apply`] did to one session's config
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionDisplay {
    pub fetch: ToolOutcome,
    pub search: ToolOutcome,
}

impl DisplayPlan {
    /// The plan, with every model's and agent's choice checked against the
    /// profiles there are. An error names the model or agent and the
    /// profile, and lists the profiles defined.
    pub fn new(
        profiles: AgentDisplayProfiles,
        specs: &[ModelSpec],
        agents: BTreeMap<AgentId, DisplaySelection>,
    ) -> anyhow::Result<Self> {
        let models: Vec<(Model, DisplaySelection)> = specs
            .iter()
            .filter_map(|s| Some((s.id.clone(), s.display.clone()?)))
            .collect();
        for (id, selection) in &models {
            profiles
                .check(selection)
                .map_err(|e| anyhow::anyhow!("[[model]] {}: `display`: {e}", id.name()))?;
        }
        for (id, selection) in &agents {
            profiles
                .check(selection)
                .map_err(|e| anyhow::anyhow!("[agent_display] {id}: {e}"))?;
        }
        Ok(Self {
            profiles,
            models,
            agents,
        })
    }

    fn model_selection(&self, model: &Model) -> Option<&DisplaySelection> {
        self.models
            .iter()
            .find(|(id, _)| id == model)
            .map(|(_, selection)| selection)
    }

    /// The display of `agent`'s session on `model`
    pub fn resolve(
        &self,
        model: &Model,
        agent: AgentId,
    ) -> Result<ResolvedDisplay<'_>, UnknownProfile> {
        self.profiles
            .resolve(self.model_selection(model), self.agents.get(&agent))
    }

    /// Set one session's display and web tools in `config`, which holds
    /// the run's (`[seed]`) web tools. `Err` only for a profile
    /// [`new`](Self::new) would have refused.
    pub fn apply(
        &self,
        config: &mut SeedConfig,
        web: WebSources,
        model: &Model,
        agent: AgentId,
        agent_name: &str,
    ) -> Result<SessionDisplay, UnknownProfile> {
        let resolved = self.resolve(model, agent)?;
        let links = resolved.config.links();
        config.display = resolved.config.clone();
        let withheld = |tool: ServerTool| {
            tracing::info!(
                event_type = "web_tool_withheld",
                agent = %agent_name,
                agent_id = %agent,
                model = %model.name(),
                tool = tool_field(tool),
                "the session's display policy has an allow list none of whose rules the tool \
                 takes, so the tool is withheld rather than offered unfiltered"
            );
        };
        let fetch = filter(
            &mut config.web_fetch,
            web.fetch,
            || links.fetch_domains(),
            |t| (&mut t.allowed_domains, &mut t.blocked_domains),
        );
        if fetch == ToolOutcome::Withheld {
            withheld(ServerTool::WebFetch);
        }
        let search = filter(
            &mut config.web_search,
            web.search,
            || links.search_domains(),
            |t| (&mut t.allowed_domains, &mut t.blocked_domains),
        );
        if search == ToolOutcome::Withheld {
            withheld(ServerTool::WebSearch);
        }
        Ok(SessionDisplay { fetch, search })
    }

    /// What every session is shown, for the operator at startup: the
    /// default, then each model's and each agent's choice, each with the
    /// two tools' domain filters, and which tools' lists are the
    /// operator's instead.
    ///
    /// [`Detail::Summary`] leaves out only the lists themselves (each
    /// rule, each domain sent): every line saying a rule was widened or
    /// not sent, a tool withheld or a list overridden is kept, and so is
    /// what every entry also reaches. [`Detail::Full`] (`--dry-run`) is
    /// agentkit's reports whole.
    pub fn report(&self, web: WebSources, detail: Detail) -> Report<'_> {
        Report {
            plan: self,
            web,
            detail,
        }
    }

    /// Each scope there is a choice for, and its resolution: the default,
    /// each model's, each agent's
    fn scopes(
        &self,
    ) -> impl Iterator<Item = (Scope<'_>, Result<ResolvedDisplay<'_>, UnknownProfile>)> {
        let default = std::iter::once((Scope::Default, self.profiles.resolve(None, None)));
        let models = self
            .models
            .iter()
            .map(|(id, s)| (Scope::Model(id), self.profiles.resolve(Some(s), None)));
        let agents = self
            .agents
            .iter()
            .map(|(id, s)| (Scope::Agent(*id), self.profiles.resolve(None, Some(s))));
        default.chain(models).chain(agents)
    }

    /// The same as structured events, one per scope, for a log reader. A
    /// scope whose sessions lose a web tool is a warning
    pub fn log(&self, web: WebSources) {
        for (scope, resolved) in self.scopes() {
            let resolved = match resolved {
                Ok(resolved) => resolved,
                // `new` checked every selection
                Err(e) => {
                    tracing::error!(
                        event_type = "display_config",
                        scope = %scope,
                        error = %e,
                        "display config does not resolve"
                    );
                    continue;
                }
            };
            let links = resolved.config.links();
            let (mode, rules) = match links {
                Links::Defanged { allow } => ("defanged", allow.len()),
                Links::Raw { deny } => ("raw", deny.len()),
            };
            let fetch = links.fetch_domains();
            let search = links.search_domains();
            let web_fetch = describe(&fetch, web.fetch);
            let web_search = describe(&search, web.search);
            let notable = |d: &str| d == "withheld" || d == "operator_lists_over_withheld";
            if notable(web_fetch) || notable(web_search) {
                tracing::warn!(
                    event_type = "display_config",
                    scope = %scope,
                    links_mode = mode,
                    link_rules = rules,
                    web_fetch,
                    web_fetch_notes = fetch.notes().len(),
                    web_search,
                    web_search_notes = search.notes().len(),
                    "display config: a web tool is withheld from these sessions, or offered with the operator's lists where the policy would withhold it"
                );
            } else {
                tracing::info!(
                    event_type = "display_config",
                    scope = %scope,
                    links_mode = mode,
                    link_rules = rules,
                    web_fetch,
                    web_fetch_notes = fetch.notes().len(),
                    web_search,
                    web_search_notes = search.notes().len(),
                    "display config"
                );
            }
        }
    }
}

/// Whose choice a resolution is
#[derive(Debug, Clone, Copy)]
enum Scope<'a> {
    Default,
    Model(&'a Model),
    Agent(AgentId),
}

impl fmt::Display for Scope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("default"),
            Self::Model(id) => write!(f, "[[model]] {}", id.name()),
            Self::Agent(id) => write!(f, "[agent_display] {id}"),
        }
    }
}

/// How much of the lists [`DisplayPlan::report`] prints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detail {
    /// Counts and every change, not the rules and domains themselves:
    /// what a run prints at every start
    Summary,
    /// Every rule and every domain sent: `--dry-run`
    Full,
}

/// [`DisplayPlan::report`]
pub struct Report<'a> {
    plan: &'a DisplayPlan,
    web: WebSources,
    detail: Detail,
}

impl fmt::Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let only_default = self.plan.models.is_empty() && self.plan.agents.is_empty();
        for (scope, resolved) in self.plan.scopes() {
            match scope {
                Scope::Default if only_default => writeln!(f, "== display: every agent")?,
                Scope::Default => writeln!(
                    f,
                    "== display: the default (every agent and model not listed below)"
                )?,
                Scope::Model(_) => writeln!(f, "== display: {scope}")?,
                Scope::Agent(_) => {
                    writeln!(f, "== display: {scope} (over its model's choice, if any)")?
                }
            }
            match resolved {
                Ok(resolved) => {
                    write!(f, "{}", Indented(&resolved, self.detail))?;
                    let links = resolved.config.links();
                    for domains in [links.fetch_domains(), links.search_domains()] {
                        write!(f, "{}", ToolReport(&domains, self.web, self.detail))?;
                    }
                }
                // `new` checked every selection
                Err(e) => writeln!(f, "   {e}")?,
            }
        }
        Ok(())
    }
}

/// One word for what a tool's lists are, for a log field
fn describe(domains: &ToolDomains, source: WebSource) -> &'static str {
    match source {
        WebSource::Off => "off",
        // Said apart, so a log reader sees what the operator's list overrode
        WebSource::Operator if domains.offer() == ToolOffer::Withhold => {
            "operator_lists_over_withheld"
        }
        WebSource::Operator => "operator_lists",
        WebSource::Policy => match domains.offer() {
            ToolOffer::Withhold => "withheld",
            ToolOffer::Offer {
                allowed_domains: Some(_),
                ..
            } => "allowed_domains",
            ToolOffer::Offer {
                blocked_domains: Some(_),
                ..
            } => "blocked_domains",
            ToolOffer::Offer { .. } => "unfiltered",
        },
    }
}

type DomainList = Option<Vec<Cow<'static, str>>>;

/// One tool of one session under its policy
fn filter<T>(
    tool: &mut Option<T>,
    source: WebSource,
    domains: impl FnOnce() -> ToolDomains,
    lists: impl FnOnce(&mut T) -> (&mut DomainList, &mut DomainList),
) -> ToolOutcome {
    let Some(on) = tool.as_mut() else {
        return ToolOutcome::Off;
    };
    match source {
        // The run's config has the tool, so its table is there: `Off` here
        // is a caller's mistake, and the tool is left as configured
        WebSource::Off | WebSource::Operator => ToolOutcome::Operator,
        WebSource::Policy => match domains().offer() {
            ToolOffer::Withhold => {
                *tool = None;
                ToolOutcome::Withheld
            }
            ToolOffer::Offer {
                allowed_domains,
                blocked_domains,
            } => {
                let (allowed, blocked) = lists(on);
                *allowed = allowed_domains;
                *blocked = blocked_domains;
                ToolOutcome::Offered
            }
        },
    }
}

/// A [`ResolvedDisplay`], indented under its heading. In a summary, its
/// lines listing each rule (`  allow …`, `  deny …`) are left out: the
/// count before them stays
struct Indented<'a>(&'a ResolvedDisplay<'a>, Detail);

impl fmt::Display for Indented<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(resolved, detail) = self;
        let rules: Vec<String> = match resolved.config.links() {
            Links::Defanged { allow } => allow.iter().map(|r| format!("  allow {r}")).collect(),
            Links::Raw { deny } => deny.iter().map(|r| format!("  deny {r}")).collect(),
        };
        resolved
            .to_string()
            .lines()
            .filter(|line| *detail == Detail::Full || !rules.iter().any(|r| r == line))
            .try_for_each(|line| writeln!(f, "   {line}"))
    }
}

/// A tool's [`ToolDomains`] report, or why it does not apply. In a
/// summary, the lines that are an entry sent (`  wikipedia.org`) are left
/// out; the count, what every entry also reaches, and each rule widened or
/// not sent stay
struct ToolReport<'a>(&'a ToolDomains, WebSources, Detail);

impl ToolReport<'_> {
    fn lines(&self) -> Vec<String> {
        let Self(domains, _, detail) = self;
        let entries: Vec<String> = match domains.filter() {
            DomainFilter::Allowed(list) | DomainFilter::Blocked(list) => {
                list.iter().map(|e| format!("  {e}")).collect()
            }
            DomainFilter::Unfiltered | DomainFilter::Withhold => Vec::new(),
        };
        let full = *detail == Detail::Full;
        domains
            .to_string()
            .lines()
            .filter(|line| full || !entries.iter().any(|e| e == line))
            .map(str::to_owned)
            .collect()
    }
}

impl fmt::Display for ToolReport<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(domains, web, _) = self;
        let tool = domains.tool();
        match web.of(tool) {
            WebSource::Off => {
                writeln!(f, "   {tool}: off ({} is absent)", table(tool))
            }
            WebSource::Operator => {
                writeln!(
                    f,
                    "   {tool}: the operator's lists in {} are used as written; the display \
                     policy's below is NOT applied",
                    table(tool)
                )?;
                self.lines()
                    .iter()
                    .try_for_each(|line| writeln!(f, "     (not applied) {line}"))
            }
            WebSource::Policy => self
                .lines()
                .iter()
                .try_for_each(|line| writeln!(f, "   {line}")),
        }
    }
}

#[cfg(test)]
mod tests;
