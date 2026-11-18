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
//!
//! `domains = "unfiltered"` on a tool's table sends it with no list at all,
//! for every session and model, whatever `[display]` says
//! ([`WebSource::Unfiltered`]): the tool reaches any address in the
//! session's context, and a policy that cannot be sent as a list does not
//! withhold it, since no list is sent. The Steward chose it for the fleet on
//! 2026-11-18, when the API refused every session of the daily run because
//! the policy's list named sites that block Anthropic's crawler
//! ([`refusal`]). Nothing here changes a session's tools after its
//! prompt is first built: they are the start of the cached prefix, shared
//! by every agent on a model.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

use agora_agentkit::display::{
    AgentDisplayProfiles, DisplaySelection, DomainFilter, Links, ResolvedDisplay, ServerTool,
    ToolDomains, ToolOffer, UnknownProfile,
};
use agora_agentkit::ids::AgentId;
use agora_agentkit::names::AgentName;
use agora_agentkit::reactor::seed::SeedConfig;
use misanthropic::model::Model;
use misanthropic::tool::{WebFetch, WebSearch};

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
    /// Enabled with `domains = "unfiltered"`: sent with neither list, the
    /// display policy's not applied and never withholding the tool
    Unfiltered,
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

/// Which server tools an endpoint runs (its `Quirks`): one that runs
/// neither offers its sessions no web tool, whatever the policy says
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointTools {
    /// The endpoint as configured (`blallama://…`)
    pub endpoint: String,
    pub fetch: bool,
    pub search: bool,
}

impl EndpointTools {
    pub fn of(endpoint: &str, quirks: &agora_agentkit::reactor::inference::Quirks) -> Self {
        Self {
            endpoint: endpoint.to_owned(),
            fetch: !quirks.web_fetch_unsupported,
            search: !quirks.web_search_unsupported,
        }
    }

    fn runs(&self, tool: ServerTool) -> bool {
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

/// A tool's name as the logs (and the API) carry it
pub fn tool_field(tool: ServerTool) -> &'static str {
    match tool {
        ServerTool::WebFetch => "web_fetch",
        ServerTool::WebSearch => "web_search",
    }
}

/// `[agent_display]`: agent names, each a choice. A key that is an agent's
/// id is refused (it passes the name rule, and would select no one), as
/// agentkit refuses one in `allow_authors`; a key that is not a name is
/// refused naming it. Names are matched exactly, case included
pub fn agent_selections<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<BTreeMap<AgentName, DisplaySelection>, D::Error> {
    use serde::Deserialize as _;
    use serde::de::Error as _;
    BTreeMap::<String, DisplaySelection>::deserialize(d)?
        .into_iter()
        .map(|(given, selection)| {
            if given.parse::<uuid::Uuid>().is_ok() {
                return Err(D::Error::custom(format_args!(
                    "[agent_display]: {given:?} is an agent id; agents are chosen by name \
                     (the handle, like `claude-ai`)"
                )));
            }
            let name = given
                .parse::<AgentName>()
                .map_err(|e| D::Error::custom(format_args!("[agent_display]: {given:?}: {e}")))?;
            Ok((name, selection))
        })
        .collect()
}

/// The run's display config, checked: the profiles, each model's choice
/// and each agent's
#[derive(Debug, Clone, Default)]
pub struct DisplayPlan {
    profiles: AgentDisplayProfiles,
    /// `[[model]] display`, in table order
    models: Vec<(Model, DisplaySelection)>,
    /// `[agent_display]`
    agents: BTreeMap<AgentName, DisplaySelection>,
}

/// What became of one web tool in one session
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOutcome {
    /// Not enabled for the run
    Off,
    /// Enabled, but the session's endpoint runs no such server tool: it
    /// drops the tool (agentkit says so), so nothing here is withheld or
    /// filtered
    NotRun,
    /// The operator's lists, as written
    Operator,
    /// `domains = "unfiltered"`: neither list, whatever the policy says
    Unfiltered,
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
        agents: BTreeMap<AgentName, DisplaySelection>,
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
        for (name, selection) in &agents {
            profiles
                .check(selection)
                .map_err(|e| anyhow::anyhow!("[agent_display] {name}: {e}"))?;
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

    /// The display of the session of the agent named `agent` on `model`.
    /// The name is matched exactly, case included (`Oracle` is not
    /// `oracle`)
    pub fn resolve(
        &self,
        model: &Model,
        agent: &str,
    ) -> Result<ResolvedDisplay<'_>, UnknownProfile> {
        self.profiles
            .resolve(self.model_selection(model), self.agents.get(agent))
    }

    /// Set one session's display and web tools in `config`, which holds
    /// the run's (`[seed]`) web tools. `Err` only for a profile
    /// [`new`](Self::new) would have refused.
    pub fn apply(
        &self,
        config: &mut SeedConfig,
        web: WebSources,
        endpoint: &EndpointTools,
        model: &Model,
        agent: AgentId,
        agent_name: &str,
    ) -> Result<SessionDisplay, UnknownProfile> {
        let resolved = self.resolve(model, agent_name)?;
        config.display = resolved.config.clone();
        Ok(web_tools(
            resolved.config.links(),
            (&mut config.web_fetch, &mut config.web_search),
            web,
            endpoint,
            (model, agent, agent_name),
            "session",
        ))
    }

    /// The web tools alone of what [`apply`](Self::apply) sets: `fetch`
    /// and `search` hold the run's (`[seed.web_*]`) and are left as a
    /// session of the agent named `agent_name` on `model` would be offered
    /// them. The same code as a session's, for what is not a session (a
    /// trial-review fork, `consent::forks`); `step` names it in the
    /// `web_tool_withheld` event (a session's says `session`).
    #[allow(clippy::too_many_arguments)]
    pub fn session_web_tools(
        &self,
        fetch: &mut Option<WebFetch>,
        search: &mut Option<WebSearch>,
        web: WebSources,
        endpoint: &EndpointTools,
        model: &Model,
        agent: AgentId,
        agent_name: &str,
        step: &'static str,
    ) -> Result<SessionDisplay, UnknownProfile> {
        let resolved = self.resolve(model, agent_name)?;
        Ok(web_tools(
            resolved.config.links(),
            (fetch, search),
            web,
            endpoint,
            (model, agent, agent_name),
            step,
        ))
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
    pub fn report<'a>(
        &'a self,
        web: WebSources,
        detail: Detail,
        endpoints: &'a [EndpointTools],
    ) -> Report<'a> {
        Report {
            plan: self,
            web,
            detail,
            endpoints,
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
            .map(|(name, s)| (Scope::Agent(name), self.profiles.resolve(None, Some(s))));
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
            let (mode, rules, authors) = match links {
                Links::Defanged {
                    allow,
                    allow_authors,
                } => ("defanged", allow.len(), allow_authors.len()),
                Links::Raw { deny } => ("raw", deny.len(), 0),
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
                    link_authors = authors,
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
                    link_authors = authors,
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
    Agent(&'a AgentName),
}

impl fmt::Display for Scope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("default"),
            Self::Model(id) => write!(f, "[[model]] {}", id.name()),
            Self::Agent(name) => write!(f, "[agent_display] {name}"),
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
    endpoints: &'a [EndpointTools],
}

impl fmt::Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let only_default = self.plan.models.is_empty() && self.plan.agents.is_empty();
        // The scopes are by display choice, not by endpoint: which
        // endpoints run the tools is said once, before them
        if self.web.fetch != WebSource::Off || self.web.search != WebSource::Off {
            let without: Vec<&str> = self
                .endpoints
                .iter()
                .filter(|e| !(e.fetch && e.search))
                .map(|e| e.endpoint.as_str())
                .collect();
            if without.is_empty() {
                writeln!(
                    f,
                    "== web tools: every endpoint of this run runs server tools; the web fetch \
                     and web search lines below apply to every session"
                )?;
            } else {
                writeln!(
                    f,
                    "== web tools: the web fetch and web search lines below apply only to \
                     sessions on endpoints that run server tools; these run none (or not \
                     both), and their sessions get no such tool: {}",
                    without.join(", ")
                )?;
            }
        }
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
        WebSource::Unfiltered => "unfiltered_by_operator",
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

/// Both web tools of one session under `links`: [`filter`] on each, and
/// the `web_tool_withheld` event for a tool withheld, with `step` saying
/// what it was for (`session`, or a fork's `review_fork`)
fn web_tools(
    links: &Links,
    (fetch_tool, search_tool): (&mut Option<WebFetch>, &mut Option<WebSearch>),
    web: WebSources,
    endpoint: &EndpointTools,
    (model, agent, agent_name): (&Model, AgentId, &str),
    step: &'static str,
) -> SessionDisplay {
    let withheld = |tool: ServerTool| {
        tracing::info!(
            event_type = "web_tool_withheld",
            step,
            agent = %agent_name,
            agent_id = %agent,
            model = %model.name(),
            tool = tool_field(tool),
            "the session's display policy has an allow list none of whose rules the tool \
             takes, so the tool is withheld rather than offered unfiltered"
        );
    };
    let fetch = filter(
        fetch_tool,
        web.fetch,
        endpoint.runs(ServerTool::WebFetch),
        || links.fetch_domains(),
        |t| (&mut t.allowed_domains, &mut t.blocked_domains),
    );
    if fetch == ToolOutcome::Withheld {
        withheld(ServerTool::WebFetch);
    }
    let search = filter(
        search_tool,
        web.search,
        endpoint.runs(ServerTool::WebSearch),
        || links.search_domains(),
        |t| (&mut t.allowed_domains, &mut t.blocked_domains),
    );
    if search == ToolOutcome::Withheld {
        withheld(ServerTool::WebSearch);
    }
    SessionDisplay { fetch, search }
}

/// One tool of one session under its policy
fn filter<T>(
    tool: &mut Option<T>,
    source: WebSource,
    runs: bool,
    domains: impl FnOnce() -> ToolDomains,
    lists: impl FnOnce(&mut T) -> (&mut DomainList, &mut DomainList),
) -> ToolOutcome {
    let Some(on) = tool.as_mut() else {
        return ToolOutcome::Off;
    };
    // Left as configured: the endpoint drops it
    if !runs {
        return ToolOutcome::NotRun;
    }
    match source {
        // The run's config has the tool, so its table is there: `Off` here
        // is a caller's mistake, and the tool is left as configured
        WebSource::Off | WebSource::Operator => ToolOutcome::Operator,
        // No list at all, whatever the policy is, so nothing to withhold
        WebSource::Unfiltered => {
            let (allowed, blocked) = lists(on);
            *allowed = None;
            *blocked = None;
            ToolOutcome::Unfiltered
        }
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
            // The authors' lines stay: who is trusted is a change, not a list
            Links::Defanged { allow, .. } => allow.iter().map(|r| format!("  allow {r}")).collect(),
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
            WebSource::Unfiltered => {
                writeln!(
                    f,
                    "   {tool}: no domain list is sent (domains = \"unfiltered\" in {}); the \
                     tool can reach any address in the session's context, whatever \
                     [display.links] allows",
                    table(tool)
                )?;
                self.lines()
                    .iter()
                    .try_for_each(|line| writeln!(f, "     (not applied) {line}"))
            }
            WebSource::Policy => {
                self.lines()
                    .iter()
                    .try_for_each(|line| writeln!(f, "   {line}"))?;
                if matches!(domains.filter(), DomainFilter::Allowed(_)) {
                    writeln!(
                        f,
                        "   {tool}: WARNING: Anthropic refuses the whole request when an \
                         allowed_domains entry is a site that blocks its crawler (2026-11-18: \
                         \"{CRAWLER_REFUSAL_FIRST_SENTENCE}\"), so a list from the display \
                         policy can fail every session of a run on any day; domains = \
                         \"unfiltered\" in {} sends none",
                        table(tool)
                    )?;
                }
                Ok(())
            }
        }
    }
}

pub mod refusal;
use refusal::CRAWLER_REFUSAL_FIRST_SENTENCE;

#[cfg(test)]
mod tests;
