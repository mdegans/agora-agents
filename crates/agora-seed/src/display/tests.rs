//! The run config's display tables, each session's resolution and web
//! tools, and the startup report, read from real TOML through the run
//! config itself ([`crate::RunConfig`], [`crate::display_plan`])

use super::*;

use agora_agentkit::display::{AgentDisplayConfig, LinkRule};

const BRIDGE_REACTOR: &str = "[[reactor]]\nendpoint = \"bridge://claude-code\"\n";
const REACTOR: &str = "[[reactor]]\nendpoint = \"anthropic://api.anthropic.com\"\n";

const HAIKU: &str = "claude-haiku-5-5";
const OLD_HAIKU: &str = "claude-haiku-4-5-20251001";

/// An endpoint that runs both server tools (Anthropic's)
fn anthropic() -> EndpointTools {
    EndpointTools {
        endpoint: "anthropic://api.anthropic.com".into(),
        fetch: true,
        search: true,
    }
}

/// An endpoint that runs neither (blallama, Ollama, the bridge)
fn local() -> EndpointTools {
    EndpointTools {
        endpoint: "blallama://h:1".into(),
        fetch: false,
        search: false,
    }
}

const ENDPOINTS: &[EndpointTools] = &[];

/// The agents' names: `[agent_display]` is keyed by name
const AGENT_NAME: &str = "beta-critic";
const OTHER_NAME: &str = "beta-social";

fn agent() -> AgentId {
    AgentId::from(uuid::Uuid::from_u128(
        0x9726e530_7df8_4f5a_8705_1164fd092fdf,
    ))
}

fn other_agent() -> AgentId {
    AgentId::from(uuid::Uuid::from_u128(
        0x4a67208e_0c37_4c1c_8072_8a71a630ab31,
    ))
}

fn model(id: &str) -> Model {
    Model::from(id.to_string())
}

fn run_config(src: &str) -> crate::RunConfig {
    toml::from_str(&format!("{src}\n{REACTOR}")).expect("config parses")
}

/// The plan, the run's `[seed]` config and its web sources, as `run`
/// builds them
fn load(src: &str) -> anyhow::Result<(DisplayPlan, SeedConfig, WebSources)> {
    let config: crate::RunConfig = toml::from_str(&format!("{src}\n{REACTOR}"))?;
    let specs = crate::model_specs(&config)?;
    let plan = crate::display_plan(&config, &specs)?;
    let seed = config
        .seed
        .to_config(std::path::Path::new("/nonexistent"), false)?;
    Ok((plan, seed, config.seed.web_sources()?))
}

fn allow(rules: &[&str]) -> AgentDisplayConfig {
    toml::from_str(&format!(
        "links = {{ mode = \"defanged\", allow = [{}] }}",
        rules
            .iter()
            .map(|r| format!("\"{r}\""))
            .collect::<Vec<_>>()
            .join(", ")
    ))
    .unwrap()
}

fn strings<'a>(list: &'a Option<Vec<Cow<'static, str>>>) -> Option<Vec<&'a str>> {
    list.as_ref()
        .map(|l| l.iter().map(|d| d.as_ref()).collect())
}

const MIXED: &str = r#"
[display]
links = { mode = "defanged", allow = ["https://*.wikipedia.org", "https://www.bbc.com/news"] }

[display.profiles.open]
links = { mode = "raw", deny = ["https://evil.example"] }

[display.profiles.local]
links = { mode = "defanged", allow = ["http://localhost"] }

[[model]]
id = "claude-haiku-5-5"
display = "open"

[[model]]
id = "claude-haiku-4-5-20251001"

[agent_display]
beta-critic = "default"
"#;

#[test]
fn no_display_table_is_agentkits_default_for_every_session() {
    let (plan, _, _) = load("").unwrap();
    let resolved = plan.resolve(&model(HAIKU), AGENT_NAME).unwrap();
    assert_eq!(resolved.config, &AgentDisplayConfig::default());
    assert!(resolved.chosen.is_none());
    assert_eq!(resolved.config.links(), &Links::default());
}

#[test]
fn a_model_names_a_profile_and_an_agent_takes_its_own_over_it() {
    let (plan, _, _) = load(MIXED).unwrap();
    // On the profile its model chose
    let on_profile = plan.resolve(&model(HAIKU), OTHER_NAME).unwrap();
    assert!(matches!(on_profile.config.links(), Links::Raw { deny } if deny.len() == 1));
    // A model with no choice is on the default
    let default = plan.resolve(&model(OLD_HAIKU), OTHER_NAME).unwrap();
    assert!(matches!(default.config.links(), Links::Defanged { allow, .. } if allow.len() == 2));
    // So is a model not in the table
    let unlisted = plan.resolve(&model("unlisted"), OTHER_NAME).unwrap();
    assert_eq!(unlisted.config, default.config);
    // The agent's own choice ("default") is taken over its model's
    let agent_choice = plan.resolve(&model(HAIKU), AGENT_NAME).unwrap();
    assert_eq!(agent_choice.config, default.config);
    assert_eq!(
        agent_choice.passed_over,
        Some(&DisplaySelection::Profile("open".parse().unwrap()))
    );
}

#[test]
fn a_config_written_in_place_on_a_model_is_taken_whole() {
    let (plan, _, _) = load(
        "[display]\nlinks = { mode = \"defanged\", allow = [\"https://docs.rs\"] }\n\
         [[model]]\nid = \"m\"\ndisplay = { links = { mode = \"raw\" } }\n",
    )
    .unwrap();
    let resolved = plan.resolve(&model("m"), AGENT_NAME).unwrap();
    assert_eq!(resolved.config.links(), &Links::Raw { deny: Vec::new() });
}

#[test]
fn an_unknown_profile_on_a_model_stops_the_load_and_names_it() {
    let e = load(
        "[display.profiles.news]\nlinks = { mode = \"raw\" }\n\
         [[model]]\nid = \"claude-haiku-5-5\"\ndisplay = \"nwes\"\n",
    )
    .expect_err("refused");
    let e = format!("{e:#}");
    assert!(e.contains("[[model]] claude-haiku-5-5"), "{e}");
    assert!(e.contains("`nwes`"), "{e}");
    assert!(e.contains("the profiles are `news`"), "{e}");
}

#[test]
fn an_unknown_profile_for_an_agent_stops_the_load_and_names_it() {
    let e = load("[agent_display]\nbeta-critic = \"news\"\n").expect_err("refused");
    let e = format!("{e:#}");
    assert!(e.contains("[agent_display] beta-critic"), "{e}");
    assert!(e.contains("no profile is defined"), "{e}");
}

#[test]
fn a_bad_rule_or_a_misspelt_key_is_refused_with_what_is_wrong() {
    // A wildcard over a whole top-level domain
    let e = load("[display]\nlinks = { mode = \"defanged\", allow = [\"https://*.com\"] }\n")
        .expect_err("refused");
    let e = format!("{e:#}");
    assert!(e.contains("*.com"), "{e}");
    // In a profile, the error names the profile
    let e = load(
        "[display.profiles.news]\nlinks = { mode = \"defanged\", allow = [\"ftp://x.org\"] }\n",
    )
    .expect_err("refused");
    let e = format!("{e:#}");
    assert!(e.contains("profile `news`"), "{e}");
    // A typo must not leave the policy at its default
    let e = load("[display]\nlink = { mode = \"raw\" }\n").expect_err("refused");
    assert!(format!("{e:#}").contains("link"), "{e:#}");
    // Nor a profile name agentkit reserves
    assert!(load("[[model]]\nid = \"m\"\ndisplay = \"Profiles\"\n").is_err());
    // An agent is named by its name, not its id
    let e = load("[agent_display]\n\"9726e530-7df8-4f5a-8705-1164fd092fdf\" = \"default\"\n")
        .expect_err("refused");
    let e = format!("{e:#}");
    assert!(
        e.contains("is an agent id; agents are chosen by name"),
        "{e}"
    );
    // A key that is not a name
    let e = load("[agent_display]\n\"a b\" = \"default\"\n").expect_err("refused");
    assert!(
        format!("{e:#}").contains("[agent_display]: \"a b\""),
        "{e:#}"
    );
}

#[test]
fn an_operator_list_on_a_tool_is_used_as_written_and_the_policy_is_not() {
    let (plan, mut config, web) = load(
        "[display]\nlinks = { mode = \"defanged\", allow = [\"https://*.wikipedia.org\"] }\n\
         [seed.web_fetch]\nallowed_domains = [\"rust-lang.org\"]\n\
         [seed.web_search]\n",
    )
    .unwrap();
    assert_eq!(web.fetch, WebSource::Operator);
    assert_eq!(web.search, WebSource::Policy);
    let done = plan
        .apply(
            &mut config,
            web,
            &anthropic(),
            &model(HAIKU),
            agent(),
            AGENT_NAME,
        )
        .unwrap();
    assert_eq!(done.fetch, ToolOutcome::Operator);
    let fetch = config.web_fetch.as_ref().unwrap();
    assert_eq!(strings(&fetch.allowed_domains), Some(vec!["rust-lang.org"]));
    assert_eq!(fetch.blocked_domains, None);
    // The other tool takes the policy's
    assert_eq!(done.search, ToolOutcome::Offered);
    let search = config.web_search.as_ref().unwrap();
    assert_eq!(
        strings(&search.allowed_domains),
        Some(vec!["wikipedia.org"])
    );
    // A blocked list written counts too
    let (_, _, web) = load("[seed.web_search]\nblocked_domains = [\"x.example\"]\n").unwrap();
    assert_eq!(web.search, WebSource::Operator);
    assert_eq!(web.fetch, WebSource::Off);
}

#[test]
fn the_policy_offer_sets_each_enabled_tools_lists() {
    let (plan, base, web) = load(&format!(
        "{MIXED}\n[seed.web_fetch]\nmax_uses = 3\n[seed.web_search]\n"
    ))
    .unwrap();
    // The default's allow list, widened as web fetch takes it
    let mut config = base.clone();
    let done = plan
        .apply(
            &mut config,
            web,
            &anthropic(),
            &model(OLD_HAIKU),
            other_agent(),
            OTHER_NAME,
        )
        .unwrap();
    assert_eq!(done.fetch, ToolOutcome::Offered);
    let fetch = config.web_fetch.as_ref().unwrap();
    assert_eq!(
        strings(&fetch.allowed_domains),
        Some(vec!["wikipedia.org", "www.bbc.com"])
    );
    assert_eq!(
        fetch.max_uses,
        Some(3),
        "the rest of the tool is the operator's"
    );
    // Web search keeps the path as a prefix
    let search = config.web_search.as_ref().unwrap();
    assert_eq!(
        strings(&search.allowed_domains),
        Some(vec!["wikipedia.org", "www.bbc.com/news"])
    );
    assert_eq!(
        config.display,
        *plan.resolve(&model(OLD_HAIKU), OTHER_NAME).unwrap().config
    );
    // A raw profile's deny rule is a blocked list
    let mut config = base.clone();
    plan.apply(
        &mut config,
        web,
        &anthropic(),
        &model(HAIKU),
        other_agent(),
        OTHER_NAME,
    )
    .unwrap();
    let fetch = config.web_fetch.as_ref().unwrap();
    assert_eq!(fetch.allowed_domains, None);
    assert_eq!(strings(&fetch.blocked_domains), Some(vec!["evil.example"]));
    // agentkit's default (no allow rule) leaves the tools unfiltered
    let (plan, mut config, web) = load("[seed.web_fetch]\n[seed.web_search]\n").unwrap();
    let done = plan
        .apply(
            &mut config,
            web,
            &anthropic(),
            &model(HAIKU),
            agent(),
            AGENT_NAME,
        )
        .unwrap();
    assert_eq!(done.fetch, ToolOutcome::Offered);
    let fetch = config.web_fetch.as_ref().unwrap();
    assert_eq!(
        (&fetch.allowed_domains, &fetch.blocked_domains),
        (&None, &None)
    );
}

#[test]
fn a_policy_that_cannot_be_sent_withholds_the_tool_from_the_session() {
    let (plan, base, web) = load(&format!(
        "{MIXED}\n[[model]]\nid = \"local\"\ndisplay = \"local\"\n[seed.web_fetch]\n[seed.web_search]\n"
    ))
    .unwrap();
    let mut config = base.clone();
    let done = plan
        .apply(
            &mut config,
            web,
            &anthropic(),
            &model("local"),
            other_agent(),
            OTHER_NAME,
        )
        .unwrap();
    assert_eq!(
        done,
        SessionDisplay {
            fetch: ToolOutcome::Withheld,
            search: ToolOutcome::Withheld
        }
    );
    assert!(config.web_fetch.is_none());
    assert!(config.web_search.is_none());
    // Only that session: the run's config is untouched, and another
    // model's session keeps its tools
    assert!(base.web_fetch.is_some());
    let mut config = base.clone();
    plan.apply(
        &mut config,
        web,
        &anthropic(),
        &model(OLD_HAIKU),
        other_agent(),
        OTHER_NAME,
    )
    .unwrap();
    assert!(config.web_fetch.is_some());
}

#[test]
fn a_tool_the_operator_did_not_enable_stays_off() {
    let (plan, mut config, web) = load(MIXED).unwrap();
    assert_eq!(
        web,
        WebSources {
            fetch: WebSource::Off,
            search: WebSource::Off
        }
    );
    let done = plan
        .apply(
            &mut config,
            web,
            &anthropic(),
            &model(OLD_HAIKU),
            agent(),
            AGENT_NAME,
        )
        .unwrap();
    assert_eq!(done.fetch, ToolOutcome::Off);
    assert_eq!(done.search, ToolOutcome::Off);
    assert!(config.web_fetch.is_none() && config.web_search.is_none());
}

/// The bridge drives the beta agents through the same per-session code as
/// any endpoint (`run`'s reactor loop calls [`DisplayPlan::apply`] for
/// every routed agent, before the endpoint kind matters): a bridge config
/// with the fleet's `[display]` gives its agent the fleet's display
#[test]
fn a_bridge_session_is_shown_what_a_fleet_session_is() {
    let display = "[display]\nlinks = { mode = \"defanged\", allow = [\"https://docs.rs\"] }\n";
    let fleet = run_config(display);
    let bridge: crate::RunConfig = toml::from_str(&format!(
        "{display}\n[[model]]\nid = \"{}\"\ncontext_window = 262144\n{BRIDGE_REACTOR}",
        crate::bridge::MODEL_ID
    ))
    .unwrap();
    let plan =
        |c: &crate::RunConfig| crate::display_plan(c, &crate::model_specs(c).unwrap()).unwrap();
    let (fleet_plan, bridge_plan) = (plan(&fleet), plan(&bridge));
    let mut fleet_config = SeedConfig::default();
    let mut bridge_config = SeedConfig::default();
    fleet_plan
        .apply(
            &mut fleet_config,
            fleet.seed.web_sources().unwrap(),
            &anthropic(),
            &model(HAIKU),
            agent(),
            AGENT_NAME,
        )
        .unwrap();
    bridge_plan
        .apply(
            &mut bridge_config,
            bridge.seed.web_sources().unwrap(),
            &EndpointTools::of(
                "bridge://claude-code",
                &agora_agentkit::reactor::Inference::quirks(&crate::bridge::bridge().0),
            ),
            &crate::bridge::model_info().id,
            agent(),
            AGENT_NAME,
        )
        .unwrap();
    assert_eq!(bridge_config.display, fleet_config.display);
    assert_eq!(bridge_config.display, allow(&["https://docs.rs"]));
}

#[test]
fn the_report_says_what_each_choice_shows_and_what_the_tools_get() {
    let (plan, _, web) = load(&format!(
        "{MIXED}\n[[model]]\nid = \"local\"\ndisplay = \"local\"\n\
         [seed.web_fetch]\nblocked_domains = [\"x.example\"]\n[seed.web_search]\n"
    ))
    .unwrap();
    let report = plan.report(web, Detail::Full, ENDPOINTS).to_string();
    // The default, and each model's and agent's choice
    assert!(
        report.contains("== display: the default (every agent and model not listed below)"),
        "{report}"
    );
    assert!(
        report.contains("== display: [[model]] claude-haiku-5-5\n"),
        "{report}"
    );
    assert!(report.contains("== display: [[model]] local\n"), "{report}");
    assert!(
        report
            .contains("== display: [agent_display] beta-critic (over its model's choice, if any)"),
        "{report}"
    );
    // A model without a choice is not listed: it is on the default
    assert!(!report.contains("claude-haiku-4-5"), "{report}");
    // agentkit's description of each, every rule listed
    assert!(
        report.contains("   links: defanged, but for 2 allowed\n"),
        "{report}"
    );
    assert!(
        report.contains("     allow https://*.wikipedia.org\n"),
        "{report}"
    );
    assert!(
        report.contains("   display: profile `open`, chosen by the model\n"),
        "{report}"
    );
    assert!(
        report.contains("     deny https://evil.example\n"),
        "{report}"
    );
    // Web search's lists from the policy, with what is widened
    assert!(
        report.contains("   web search: allowed_domains, 2\n"),
        "{report}"
    );
    assert!(
        report.contains("widened: https://*.wikipedia.org as wikipedia.org"),
        "{report}"
    );
    // Web fetch's are the operator's: the policy's is shown as not applied
    assert!(
        report.contains(
            "   web fetch: the operator's lists in [seed.web_fetch] are used as written; the \
             display policy's below is NOT applied\n"
        ),
        "{report}"
    );
    assert!(
        report.contains("(not applied) web fetch: allowed_domains, 2"),
        "{report}"
    );
    // Withheld, and why, for the profile that cannot be sent
    assert!(
        report.contains("   web search: withheld: no allow rule can be sent"),
        "{report}"
    );
    assert!(report.contains("not sent: http://localhost"), "{report}");
}

/// What a summary must keep of the full report: every line that says
/// something was widened, not sent, withheld or overridden, and what each
/// entry also reaches. It drops only the lines listing a rule or an entry
fn summary_keeps_every_change(full: &str, summary: &str) {
    let kept: std::collections::HashSet<&str> = summary.lines().collect();
    for line in summary.lines() {
        assert!(
            full.lines().any(|l| l == line),
            "not in the full report: {line}"
        );
    }
    for line in full.lines() {
        let change = [
            "widened:",
            "not sent:",
            "withheld",
            "NOT applied",
            "each also",
            "== display",
            "display:",
            "links:",
        ]
        .iter()
        .any(|k| line.contains(k))
            || line.trim_start().starts_with("web ");
        assert!(
            !change || kept.contains(line),
            "dropped from the summary: {line}"
        );
    }
}

#[test]
fn the_summary_lists_counts_and_every_change_but_not_the_lists() {
    let (plan, _, web) = load(&format!(
        "{MIXED}\n[[model]]\nid = \"local\"\ndisplay = \"local\"\n\
         [seed.web_fetch]\nblocked_domains = [\"x.example\"]\n[seed.web_search]\n"
    ))
    .unwrap();
    let full = plan.report(web, Detail::Full, ENDPOINTS).to_string();
    let summary = plan.report(web, Detail::Summary, ENDPOINTS).to_string();
    summary_keeps_every_change(&full, &summary);
    // The counts and the changes
    assert!(
        summary.contains("   links: defanged, but for 2 allowed\n"),
        "{summary}"
    );
    assert!(
        summary.contains("   web search: allowed_domains, 2\n"),
        "{summary}"
    );
    assert!(
        summary.contains("widened: https://*.wikipedia.org as wikipedia.org"),
        "{summary}"
    );
    assert!(
        summary.contains("each also reaches its subdomains"),
        "{summary}"
    );
    assert!(summary.contains("NOT applied"), "{summary}");
    assert!(
        summary.contains("web search: withheld: no allow rule can be sent"),
        "{summary}"
    );
    assert!(summary.contains("not sent: http://localhost"), "{summary}");
    assert!(
        summary.contains("   display: profile `open`, chosen by the model\n"),
        "{summary}"
    );
    // Not the rules or the entries themselves
    assert!(
        !summary.contains("allow https://www.bbc.com/news"),
        "{summary}"
    );
    assert!(!summary.contains("deny https://evil.example"), "{summary}");
    assert!(!summary.contains("\n     www.bbc.com/news\n"), "{summary}");
    assert!(full.contains("\n     www.bbc.com/news\n"), "{full}");
}

#[test]
fn the_report_says_when_a_tool_is_off() {
    let (plan, _, web) = load("").unwrap();
    let report = plan.report(web, Detail::Summary, ENDPOINTS).to_string();
    assert!(report.starts_with("== display: every agent\n"), "{report}");
    assert!(
        report.contains("   web fetch: off ([seed.web_fetch] is absent)\n"),
        "{report}"
    );
    assert!(
        report.contains("   web search: off ([seed.web_search] is absent)\n"),
        "{report}"
    );
}

/// The fleet's policy as proposed for the live configs
/// (`examples/display.toml`): every rule loads, and every one reaches the
/// web tools' allow lists, none withheld
#[test]
fn the_fleets_proposed_display_table_loads() {
    let src = include_str!("../../examples/display.toml");
    let (plan, _, web) = load(&format!("{src}\n[seed.web_fetch]\n[seed.web_search]\n")).unwrap();
    // A run's print is a summary: the counts and the 14 widened rules, not
    // 209 rules and 417 entries
    let full = plan.report(web, Detail::Full, ENDPOINTS).to_string();
    let summary = plan.report(web, Detail::Summary, ENDPOINTS).to_string();
    summary_keeps_every_change(&full, &summary);
    assert!(summary.lines().count() < 30, "{summary}");
    assert_eq!(summary.matches("widened:").count(), 14, "{summary}");
    assert!(
        summary.contains("   web fetch: allowed_domains, 208\n"),
        "{summary}"
    );
    assert!(
        summary.contains("   web search: allowed_domains, 209\n"),
        "{summary}"
    );
    assert!(full.lines().count() > 600);
    let resolved = plan.resolve(&model(HAIKU), AGENT_NAME).unwrap();
    let Links::Defanged {
        allow,
        allow_authors,
    } = resolved.config.links()
    else {
        panic!("defanged");
    };
    assert_eq!(allow.len(), 209);
    let authors: Vec<&str> = allow_authors.iter().map(|a| a.as_ref()).collect();
    assert_eq!(authors, ["claude-ai", "claude-opus", "steward"]);
    // The summary names each author, and says the tools are not widened by them
    for name in &authors {
        assert!(
            summary.contains(&format!("     author {name}\n")),
            "{summary}"
        );
    }
    assert!(summary.contains("and in 3 authors' own text"), "{summary}");
    assert_eq!(
        summary
            .matches("3 authors' own text is shown with its links as written; the tool still reaches only the domains above")
            .count(),
        2,
        "{summary}"
    );
    let rules: Vec<LinkRule> = allow.clone();
    assert!(
        rules
            .iter()
            .any(|r| r.to_string() == "https://*.wikipedia.org")
    );
    for domains in [
        resolved.config.links().fetch_domains(),
        resolved.config.links().search_domains(),
    ] {
        assert!(
            matches!(domains.offer(), ToolOffer::Offer { allowed_domains: Some(ref l), .. } if !l.is_empty()),
            "{domains}"
        );
    }
}

/// An operator's list over a policy that would withhold the tool is said
/// apart in the log fields, not as plain `operator_lists`
#[test]
fn an_operator_list_over_a_withholding_policy_is_said_apart() {
    let local = allow(&["http://localhost"]);
    let fetch = local.links().fetch_domains();
    assert_eq!(fetch.offer(), ToolOffer::Withhold);
    assert_eq!(
        describe(&fetch, WebSource::Operator),
        "operator_lists_over_withheld"
    );
    assert_eq!(describe(&fetch, WebSource::Policy), "withheld");
    let wiki = allow(&["https://*.wikipedia.org"]).links().fetch_domains();
    assert_eq!(describe(&wiki, WebSource::Operator), "operator_lists");
    assert_eq!(describe(&wiki, WebSource::Policy), "allowed_domains");
    assert_eq!(describe(&wiki, WebSource::Off), "off");
}

/// An empty list written on a tool is refused at load: its meaning to the
/// API is undocumented, and it would switch the policy off for that tool
#[test]
fn an_empty_written_list_is_refused_at_load() {
    for (table, list) in [
        ("web_fetch", "allowed_domains"),
        ("web_fetch", "blocked_domains"),
        ("web_search", "allowed_domains"),
        ("web_search", "blocked_domains"),
    ] {
        let e = load(&format!("[seed.{table}]\n{list} = []\n")).expect_err("refused");
        let e = format!("{e:#}");
        assert!(
            e.contains(&format!("[seed.{table}]: `{list} = []` is refused")),
            "{e}"
        );
    }
    // A list with something in it is the operator's
    let (_, _, web) = load("[seed.web_fetch]\nblocked_domains = [\"x.example\"]\n").unwrap();
    assert_eq!(web.fetch, WebSource::Operator);
}

/// A session on an endpoint that runs no server tools (blallama, Ollama,
/// the bridge): its endpoint drops the tools, so nothing is withheld or
/// filtered here and no `web_tool_withheld` is said for it
#[test]
fn an_endpoint_without_server_tools_is_not_said_to_lose_them() {
    let (plan, base, web) = load(&format!(
        "{MIXED}\n[[model]]\nid = \"local\"\ndisplay = \"local\"\n[seed.web_fetch]\n[seed.web_search]\n"
    ))
    .unwrap();
    let mut config = base.clone();
    let done = plan
        .apply(
            &mut config,
            web,
            &local(),
            &model("local"),
            other_agent(),
            OTHER_NAME,
        )
        .unwrap();
    assert_eq!(
        done,
        SessionDisplay {
            fetch: ToolOutcome::NotRun,
            search: ToolOutcome::NotRun
        }
    );
    // Left as configured, for agentkit to drop at the endpoint
    assert_eq!(config.web_fetch, base.web_fetch);
    // The same session on Anthropic is withheld
    let mut config = base.clone();
    let done = plan
        .apply(
            &mut config,
            web,
            &anthropic(),
            &model("local"),
            other_agent(),
            OTHER_NAME,
        )
        .unwrap();
    assert_eq!(done.fetch, ToolOutcome::Withheld);
}

#[test]
fn the_report_says_which_endpoints_run_no_server_tools() {
    let (plan, _, web) = load("[seed.web_fetch]\n").unwrap();
    let mixed = [anthropic(), local()];
    let report = plan.report(web, Detail::Summary, &mixed).to_string();
    assert!(
        report.starts_with(
            "== web tools: the web fetch and web search lines below apply only to sessions on \
             endpoints that run server tools; these run none (or not both), and their sessions \
             get no such tool: blallama://h:1\n"
        ),
        "{report}"
    );
    let report = plan
        .report(web, Detail::Summary, &[anthropic()])
        .to_string();
    assert!(
        report.starts_with("== web tools: every endpoint of this run runs server tools"),
        "{report}"
    );
    // No web line when no tool is enabled
    let (plan, _, web) = load("").unwrap();
    let report = plan.report(web, Detail::Summary, &mixed).to_string();
    assert!(!report.contains("== web tools"), "{report}");
}

/// `run` makes every session's config with `crate::session_config`: the
/// model's budgets, the display and web tools, the cadence. Dropping the
/// display step (or any other) fails here
#[test]
fn a_sessions_config_has_its_budgets_display_web_tools_and_cadence() {
    let config: crate::RunConfig = toml::from_str(&format!(
        "[display]\nlinks = {{ mode = \"defanged\", allow = [\"https://docs.rs\"] }}\n\
         [display.profiles.open]\nlinks = {{ mode = \"raw\", deny = [\"https://evil.example\"] }}\n\
         [[model]]\nid = \"{HAIKU}\"\ndisplay = \"open\"\nact_max_tokens = 20000\n\
         switch_context_window = 300000\ncontext_window = 200000\n\
         [seed.web_fetch]\n{REACTOR}"
    ))
    .unwrap();
    let specs = crate::model_specs(&config).unwrap();
    let plan = crate::display_plan(&config, &specs).unwrap();
    let base = config
        .seed
        .to_config(std::path::Path::new("/nonexistent"), false)
        .unwrap();
    let catalog = crate::models::Catalog::new(&specs, &[crate::models::tests::info(HAIKU)]);
    let entry = catalog.get(&model(HAIKU));
    let endpoint = anthropic();
    let web = config.seed.web_sources().unwrap();
    let inputs = |cadence| crate::SessionInputs {
        entry,
        display: &plan,
        web,
        endpoint: &endpoint,
        cadence,
    };
    let made =
        crate::session_config(&base, inputs(None), &model(HAIKU), agent(), AGENT_NAME).unwrap();
    // The model's budgets
    assert_eq!(made.act_max_tokens, 20000);
    assert_eq!(made.context_window, 200000);
    // Its display: the model's profile, whole
    assert_eq!(
        made.display.links(),
        &Links::Raw {
            deny: vec!["https://evil.example".parse().unwrap()]
        }
    );
    // Its web tool, from that profile
    let fetch = made.web_fetch.as_ref().unwrap();
    assert_eq!(strings(&fetch.blocked_domains), Some(vec!["evil.example"]));
    // The run's config is untouched
    assert_eq!(base.display, AgentDisplayConfig::default());
    assert!(base.web_fetch.as_ref().unwrap().blocked_domains.is_none());
    // An agent on its own cadence: its rounds and the switch window
    let switched = crate::session_config(
        &base,
        inputs(Some((20, 158_400))),
        &model(HAIKU),
        agent(),
        AGENT_NAME,
    )
    .unwrap();
    assert_eq!(switched.max_rounds, 20);
    assert_eq!(switched.context_window, 300000);
    assert_eq!(switched.display, made.display);
    // A model not in the table: the run's budgets and the default display
    let other = crate::session_config(
        &base,
        crate::SessionInputs {
            entry: None,
            ..inputs(None)
        },
        &model("other"),
        agent(),
        AGENT_NAME,
    )
    .unwrap();
    assert_eq!(other.act_max_tokens, base.act_max_tokens);
    assert_eq!(other.display, allow(&["https://docs.rs"]));
    assert_eq!(
        strings(&other.web_fetch.as_ref().unwrap().allowed_domains),
        Some(vec!["docs.rs"])
    );
}

/// Names are matched exactly, case included: production has `Oracle` and
/// `oracle`, two agents
#[test]
fn an_agent_is_chosen_by_its_exact_name() {
    let (plan, _, _) = load(
        "[display.profiles.open]\nlinks = { mode = \"raw\" }\n[agent_display]\nOracle = \"open\"\n",
    )
    .unwrap();
    let upper = plan.resolve(&model(HAIKU), "Oracle").unwrap();
    assert_eq!(upper.config.links(), &Links::Raw { deny: Vec::new() });
    let lower = plan.resolve(&model(HAIKU), "oracle").unwrap();
    assert_eq!(lower.config, &AgentDisplayConfig::default());
}

/// `domains = "unfiltered"` on both tools
const UNFILTERED: &str = "[seed.web_fetch]\nmax_uses = 3\ndomains = \"unfiltered\"\n\
                          [seed.web_search]\nmax_uses = 2\ndomains = \"unfiltered\"\n";

/// Under `domains = "unfiltered"` every session on every model gets both
/// tools with neither list, whatever its display policy: an allow list, a
/// deny list, one that would withhold the tool, a model's own profile, or
/// none. What the session is shown is the policy's all the same
#[test]
fn unfiltered_sends_no_list_whatever_the_policy() {
    for display in [
        MIXED,
        "[display]\nlinks = { mode = \"defanged\", allow = [\"http://localhost\"] }\n",
        "[display]\nlinks = { mode = \"raw\", deny = [\"https://bad.example\"] }\n",
        "",
    ] {
        let (plan, base, web) = load(&format!("{UNFILTERED}{display}")).unwrap();
        assert_eq!(
            web,
            WebSources {
                fetch: WebSource::Unfiltered,
                search: WebSource::Unfiltered
            }
        );
        for (model_id, name) in [
            (HAIKU, AGENT_NAME),
            (HAIKU, OTHER_NAME),
            (OLD_HAIKU, OTHER_NAME),
            ("local", OTHER_NAME),
        ] {
            let mut config = base.clone();
            let done = plan
                .apply(
                    &mut config,
                    web,
                    &anthropic(),
                    &model(model_id),
                    agent(),
                    name,
                )
                .unwrap();
            assert_eq!(
                done,
                SessionDisplay {
                    fetch: ToolOutcome::Unfiltered,
                    search: ToolOutcome::Unfiltered
                },
                "{display} {model_id} {name}"
            );
            let fetch = config.web_fetch.as_ref().expect("offered");
            let search = config.web_search.as_ref().expect("offered");
            assert_eq!(
                (&fetch.allowed_domains, &fetch.blocked_domains),
                (&None, &None)
            );
            assert_eq!(
                (&search.allowed_domains, &search.blocked_domains),
                (&None, &None)
            );
            assert_eq!((fetch.max_uses, search.max_uses), (Some(3), Some(2)));
            // The display is the policy's, unchanged
            assert_eq!(
                &config.display,
                plan.resolve(&model(model_id), name).unwrap().config
            );
        }
    }
}

/// The switch is per tool: one unfiltered, the other the policy's list
#[test]
fn the_switch_is_per_tool() {
    let (plan, mut config, web) = load(&format!(
        "{MIXED}[seed.web_fetch]\ndomains = \"unfiltered\"\n[seed.web_search]\ndomains = \"policy\"\n"
    ))
    .unwrap();
    assert_eq!(
        web,
        WebSources {
            fetch: WebSource::Unfiltered,
            search: WebSource::Policy
        }
    );
    let done = plan
        .apply(
            &mut config,
            web,
            &anthropic(),
            &model(OLD_HAIKU),
            other_agent(),
            OTHER_NAME,
        )
        .unwrap();
    assert_eq!(done.fetch, ToolOutcome::Unfiltered);
    assert_eq!(done.search, ToolOutcome::Offered);
    assert_eq!(config.web_fetch.unwrap().allowed_domains, None);
    assert_eq!(
        strings(&config.web_search.unwrap().allowed_domains),
        Some(vec!["wikipedia.org", "www.bbc.com/news"])
    );
}

/// `domains = "policy"` is what an absent key is, with or without a list
#[test]
fn policy_is_the_default() {
    for (written, source) in [
        ("", WebSource::Policy),
        ("allowed_domains = [\"a.example\"]\n", WebSource::Operator),
        ("blocked_domains = [\"a.example\"]\n", WebSource::Operator),
    ] {
        let (_, _, absent) = load(&format!("[seed.web_fetch]\n{written}")).unwrap();
        let (_, _, said) = load(&format!(
            "[seed.web_fetch]\ndomains = \"policy\"\n{written}"
        ))
        .unwrap();
        assert_eq!(absent, said);
        assert_eq!(absent.fetch, source);
    }
}

/// A list written beside `domains = "unfiltered"`, or a value that is not
/// one of the two, stops the load naming what is wrong
#[test]
fn unfiltered_with_a_list_or_an_unknown_value_is_refused_at_load() {
    for table in ["web_fetch", "web_search"] {
        for list in ["allowed_domains", "blocked_domains"] {
            let e = load(&format!(
                "[seed.{table}]\ndomains = \"unfiltered\"\n{list} = [\"a.example\"]\n"
            ))
            .expect_err("refused");
            let e = format!("{e:#}");
            assert!(
                e.contains(&format!(
                    "[seed.{table}]: `domains = \"unfiltered\"` sends the tool with no domain \
                     list, and `{list}` is written beside it"
                )),
                "{e}"
            );
        }
        let e = load(&format!("[seed.{table}]\ndomains = \"none\"\n")).expect_err("refused");
        let e = format!("{e:#}");
        assert!(e.contains("unknown variant `none`"), "{e}");
        assert!(e.contains("expected `policy` or `unfiltered`"), "{e}");
    }
}

/// The print says plainly that no list is sent and what the tool can
/// reach; the policy's lines follow marked as not applied, and no warning
/// about a refused list is printed, since none is sent
#[test]
fn the_report_says_no_list_is_sent() {
    let src = include_str!("../../examples/display.toml");
    let (plan, _, web) = load(&format!("{src}\n{UNFILTERED}")).unwrap();
    for detail in [Detail::Summary, Detail::Full] {
        let report = plan.report(web, detail, ENDPOINTS).to_string();
        for (tool, table) in [
            ("web fetch", "[seed.web_fetch]"),
            ("web search", "[seed.web_search]"),
        ] {
            assert!(
                report.contains(&format!(
                    "   {tool}: no domain list is sent (domains = \"unfiltered\" in {table}); the \
                     tool can reach any address in the session's context, whatever \
                     [display.links] allows\n"
                )),
                "{report}"
            );
            assert!(
                report.contains(&format!("     (not applied) {tool}: allowed_domains, 20")),
                "{report}"
            );
        }
        assert!(!report.contains("WARNING"), "{report}");
    }
    assert_eq!(
        describe(
            &allow(&["http://localhost"]).links().fetch_domains(),
            WebSource::Unfiltered
        ),
        "unfiltered_by_operator"
    );
}

/// Where a tool's list is the policy's allow list, the print warns once
/// per tool and scope that a site blocking Anthropic's crawler fails the
/// whole request; not for a deny list, an operator's list, or no list
#[test]
fn a_policy_allow_list_is_warned_of() {
    let warning = |tool: &str, table: &str| {
        format!(
            "   {tool}: WARNING: Anthropic refuses the whole request when an allowed_domains \
             entry is a site that blocks its crawler (2026-11-18: \"The following domains are \
             not accessible to our user agent\"), so a list from the display policy can fail \
             every session of a run on any day; domains = \"unfiltered\" in {table} sends none\n"
        )
    };
    let (plan, _, web) = load(&format!("{MIXED}[seed.web_fetch]\n[seed.web_search]\n")).unwrap();
    let report = plan.report(web, Detail::Summary, ENDPOINTS).to_string();
    // The default and agent `beta-critic` (the default): allow lists; the
    // `open` profile of Haiku 5.5 is a deny list
    assert_eq!(
        report
            .matches(&warning("web fetch", "[seed.web_fetch]"))
            .count(),
        2,
        "{report}"
    );
    assert_eq!(
        report
            .matches(&warning("web search", "[seed.web_search]"))
            .count(),
        2,
        "{report}"
    );
    for src in [
        "[display]\nlinks = { mode = \"raw\", deny = [\"https://bad.example\"] }\n\
         [seed.web_fetch]\n[seed.web_search]\n",
        "[seed.web_fetch]\nallowed_domains = [\"a.example\"]\n",
        "[seed.web_fetch]\n",
    ] {
        let (plan, _, web) = load(src).unwrap();
        let report = plan.report(web, Detail::Full, ENDPOINTS).to_string();
        assert!(!report.contains("WARNING"), "{src}: {report}");
    }
}

/// An operator's entry the docs say the API refuses (a scheme, a wildcard
/// in the domain) or that never matches (a web fetch entry with a path)
/// stops the load, each named; a web search path, with a wildcard in it,
/// is the docs' own example and loads
#[test]
fn an_operators_entry_the_api_would_refuse_stops_the_load() {
    for (table, entry, why) in [
        ("web_fetch", "https://a.example", "has a scheme"),
        ("web_search", "http://a.example/blog", "has a scheme"),
        ("web_fetch", "*.a.example", "has a wildcard in the domain"),
        ("web_search", "ex*.com", "has a wildcard in the domain"),
        (
            "web_fetch",
            "a.example/blog",
            "has a path: web fetch matches the domain alone",
        ),
        ("web_search", "аmazon.com", "is not ASCII"),
        ("web_search", "a.example:8080", "has a port"),
        ("web_fetch", "a .example", "has whitespace"),
        ("web_fetch", "", "is empty"),
        ("web_search", ".a.example", "is not a domain name"),
        ("web_fetch", "localhost", "is not a domain name"),
        ("web_fetch", "10.0.0.1", "is not a domain name"),
        ("web_search", "a_b.example", "is not a domain name"),
    ] {
        for list in ["allowed_domains", "blocked_domains"] {
            let e = load(&format!(
                "[seed.{table}]\n{list} = [\"good.example\", \"{entry}\"]\n"
            ))
            .expect_err(entry);
            let e = format!("{e:#}");
            assert!(
                e.contains(&format!(
                    "[seed.{table}]: {list}: 1 entry the API would refuse"
                )),
                "{e}"
            );
            assert!(e.contains(&format!("{entry:?} {why}")), "{e}");
        }
    }
    for entry in [
        "example.com/blog",
        "example.com/*",
        "example.com/*/articles",
        "Example.COM",
    ] {
        load(&format!(
            "[seed.web_search]\nallowed_domains = [\"{entry}\"]\n"
        ))
        .expect(entry);
    }
}
