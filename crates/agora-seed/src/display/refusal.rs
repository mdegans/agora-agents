//! The API's refusal of a request whose web tool lists a site that blocks
//! Anthropic's crawler, said once per run.
//!
//! On 2026-11-18 the daily run's every session failed on its first call:
//!
//! > The following domains are not accessible to our user agent:
//! > ['apnews.com', …, 'zeit.de']. Read more: https://support.anthropic.com/…
//!
//! A web tool's `allowed_domains` named sites (as `www.nytimes.com`; the
//! API names the domain, `nytimes.com`) that block the crawler, and the
//! API refuses the whole request for it. Any listed site can start
//! blocking on any day. The runner does not retry such a call or change
//! the session's tools: they open the cached prefix every agent on a model
//! shares, and a changed prefix is a cache miss for each of them (the
//! Steward, 2026-11-18). The session fails as any refused one does; what
//! this adds is one `error!` and one alert for the run
//! ([`CrawlerRefusals`]), naming the sites and the two ways out, where
//! there were thirty lines of the API's message.

use agora_agentkit::display::ServerTool;
use std::collections::BTreeSet;
use std::fmt;

use crate::alerts::{Alert, AlertKind};

/// What marks the refusal in the API's message (its first sentence, from
/// "not")
pub const CRAWLER_REFUSAL: &str = "not accessible to our user agent";

/// The message's first sentence, as the API wrote it on 2026-11-18
pub const CRAWLER_REFUSAL_FIRST_SENTENCE: &str =
    "The following domains are not accessible to our user agent";

/// The domains a crawler refusal names, if `message` is one. `Some` with
/// an empty list: the refusal, but no domain could be read from it (the
/// message is then passed on whole).
///
/// Read tolerantly: the list may be bracketed or not, quoted with `'` or
/// `"` or not at all, and cut short; a bracketed list cut before its `]`
/// loses its last name, which may be cut too. The link after the list
/// ("Read more: https://…") is not read as a domain. Names are lower case.
pub fn crawler_refused(message: &str) -> Option<Vec<String>> {
    let lower = message.to_ascii_lowercase();
    let at = lower.find(CRAWLER_REFUSAL)?;
    let rest = &lower[at + CRAWLER_REFUSAL.len()..];
    // The link to the support article is not one of them
    let end = [
        rest.find("read more"),
        rest.find("http://"),
        rest.find("https://"),
    ]
    .into_iter()
    .flatten()
    .min();
    let mut part = &rest[..end.unwrap_or(rest.len())];
    let mut cut = end.is_none();
    let bracketed = part.contains('[');
    if let Some(open) = part.find('[') {
        part = &part[open + 1..];
        match part.find(']') {
            Some(close) => {
                part = &part[..close];
                cut = false;
            }
            None => cut = true,
        }
    }
    let is_char = |c: char| c.is_ascii_alphanumeric() || c == '.' || c == '-';
    let mut names: Vec<&str> = part
        .split(|c: char| !is_char(c))
        .map(|t| t.trim_matches(|c| c == '.' || c == '-'))
        .filter(|t| !t.is_empty())
        .collect();
    // A message cut inside the list may have cut its last name too. In a
    // bracketed list only a closing quote or a comma says the last name is
    // whole (`'bbc.co.` is cut); without brackets a full stop ends the
    // sentence, so only a letter or digit says it may be cut
    let tail = part.trim_end();
    let last_cut = if bracketed {
        !tail.ends_with(['\'', '"', ','])
    } else {
        tail.ends_with(|c: char| c.is_ascii_alphanumeric())
    };
    if cut && last_cut {
        names.pop();
    }
    let mut out: Vec<String> = names
        .into_iter()
        // A domain the API could have named: agentkit's rule for a domain
        // list entry, as a web fetch entry (a domain alone, no path), the
        // rule agentkit checks the domains it sends with. Tokens here are
        // only letters, digits, `.` and `-`, so of its problems only "not a
        // domain name" can apply: two labels or more, the last not all
        // digits (not an address nor a version number)
        .filter(|t| ServerTool::WebFetch.entry_problem(t).is_none())
        .map(str::to_owned)
        .collect();
    out.sort();
    out.dedup();
    Some(out)
}

/// The run's sessions that failed on a crawler refusal, together: one
/// `error!` and one alert, not one per session
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrawlerRefusals {
    /// How many agents' sessions failed so: the reactor keeps one error
    /// per agent, so an agent with two such sessions in a run counts once
    pub sessions: usize,
    /// Every domain the refusals named
    pub domains: BTreeSet<String>,
    /// Refusals no domain could be read from, whole and without repeats
    pub unread: BTreeSet<String>,
}

impl CrawlerRefusals {
    /// Of the run's session errors (each the reactor's message), those
    /// that are crawler refusals; `None` if there are none
    pub fn of<'a>(errors: impl IntoIterator<Item = &'a str>) -> Option<Self> {
        let mut out = Self::default();
        for message in errors {
            let Some(named) = crawler_refused(message) else {
                continue;
            };
            out.sessions += 1;
            if named.is_empty() {
                out.unread.insert(message.to_owned());
            }
            out.domains.extend(named);
        }
        (out.sessions > 0).then_some(out)
    }

    /// What to do, in a sentence
    pub const WAYS_OUT: &str = "Either leave these sites out of the tool's list (out of \
         [display.links] when the list is the display policy's, or out of allowed_domains on \
         [seed.web_fetch] / [seed.web_search] when the operator wrote it), or set domains = \
         \"unfiltered\" on [seed.web_fetch] and [seed.web_search] to send no list at all. The \
         runner does not retry these calls or change a session's tools mid-run.";

    /// When every enabled web tool already goes out with no list, the API
    /// refused something this runner did not send: there is no list to
    /// change, and saying to send none would be wrong
    pub const NO_LIST_SENT: &str = "Every web tool of this run is sent with no domain list \
         (domains = \"unfiltered\", or off), so this refusal is not about a list the runner \
         sent: read the API's messages in the run log, and Anthropic's documentation for the \
         web tools.";

    /// [`WAYS_OUT`](Self::WAYS_OUT), or [`NO_LIST_SENT`](Self::NO_LIST_SENT)
    /// when no tool of the run carries a list
    pub fn ways_out(web: crate::display::WebSources) -> &'static str {
        use crate::display::WebSource;
        let listless = |s: WebSource| matches!(s, WebSource::Off | WebSource::Unfiltered);
        if listless(web.fetch) && listless(web.search) {
            Self::NO_LIST_SENT
        } else {
            Self::WAYS_OUT
        }
    }

    /// The structured `error!`, once for the run
    pub fn log(&self, web: crate::display::WebSources) {
        tracing::error!(
            event_type = "web_domains_refused",
            sessions = self.sessions,
            domains = self.domains.len(),
            named = %Joined(&self.domains, ", "),
            unread = self.unread.len(),
            unread_messages = %Joined(&self.unread, "\n"),
            ways_out = Self::ways_out(web),
            "agents' sessions failed: the API refused their requests because a web tool's \
             allowed_domains names sites that block Anthropic's crawler"
        );
    }

    /// The operator's alert, once for the run. Counted by agents, so a run
    /// with more of them is sent inside the rate limit
    pub fn alert(&self, web: crate::display::WebSources) -> Alert {
        let mut alert = Alert::new(
            AlertKind::WebDomainsRefused,
            format!(
                "{} agent{} session{} failed: the API refused {} request{} because a web tool's \
                 allowed_domains names {} site{} that block{} Anthropic's crawler",
                self.sessions,
                if self.sessions == 1 { "'s" } else { "s'" },
                plural(self.sessions),
                if self.sessions == 1 { "its" } else { "their" },
                plural(self.sessions),
                self.domains.len(),
                plural(self.domains.len()),
                if self.domains.len() == 1 { "s" } else { "" },
            ),
        )
        .count(self.sessions)
        .detail("domains", Joined(&self.domains, ", "))
        .detail("ways out", Self::ways_out(web));
        if !self.unread.is_empty() {
            alert = alert.detail(
                "unread",
                format!(
                    "{} refusal{} named no domain the runner could read; as the API wrote \
                     {}: {}",
                    self.unread.len(),
                    plural(self.unread.len()),
                    if self.unread.len() == 1 { "it" } else { "them" },
                    Joined(&self.unread, "\n  ")
                ),
            );
        }
        alert
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// A set, each after the first after the separator; `(none)` when empty
struct Joined<'a>(&'a BTreeSet<String>, &'static str);

impl fmt::Display for Joined<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("(none)");
        }
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(self.1)?;
            }
            f.write_str(d)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 53 domains the API named on 2026-11-18, in the order it named
    /// them
    const NAMED_2026_11_18: [&str; 53] = [
        "apnews.com",
        "arstechnica.com",
        "elpais.com",
        "indianexpress.com",
        "mainichi.jp",
        "news.sky.com",
        "stackoverflow.com",
        "afp.com",
        "afr.com",
        "asahi.com",
        "barrons.com",
        "bbc.co.uk",
        "bbc.com",
        "businessinsider.com",
        "channelnewsasia.com",
        "corriere.it",
        "dw.com",
        "economist.com",
        "faz.net",
        "ft.com",
        "hindustantimes.com",
        "independent.co.uk",
        "independent.ie",
        "latimes.com",
        "lemonde.fr",
        "marketwatch.com",
        "newscientist.com",
        "newyorker.com",
        "nhk.or.jp",
        "nrc.nl",
        "nytimes.com",
        "politico.com",
        "politico.eu",
        "reuters.com",
        "rfi.fr",
        "smh.com.au",
        "spiegel.de",
        "straitstimes.com",
        "sueddeutsche.de",
        "tagesschau.de",
        "telegraph.co.uk",
        "theatlantic.com",
        "theguardian.com",
        "thehindu.com",
        "thestar.com",
        "thetimes.com",
        "theverge.com",
        "usatoday.com",
        "volkskrant.nl",
        "vox.com",
        "wired.com",
        "wsj.com",
        "zeit.de",
    ];

    /// The message as the run log carried it (the reactor's `inference: `
    /// and misanthropic's prefix before the API's text)
    fn production() -> String {
        format!(
            "inference: Anthropic error: invalid request (400): The following domains are not \
             accessible to our user agent: [{}]. Read more: \
             https://support.anthropic.com/en/articles/8896518-does-anthropic-crawl-data-from-the-web-and-how-can-site-owners-block-the-crawler",
            NAMED_2026_11_18
                .iter()
                .map(|d| format!("'{d}'"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn sorted(list: &[&str]) -> Vec<String> {
        let mut v: Vec<String> = list.iter().map(|d| d.to_string()).collect();
        v.sort();
        v
    }

    #[test]
    fn the_production_message_names_its_53_domains_and_not_the_link() {
        let named = crawler_refused(&production()).unwrap();
        assert_eq!(named, sorted(&NAMED_2026_11_18));
        assert!(!named.iter().any(|d| d.contains("anthropic")), "{named:?}");
    }

    /// What the API named, against the fleet's policy as its tools' lists
    /// (`examples/display.toml`): every name covers an entry sent, by
    /// Anthropic's rule (the domain or a subdomain of it), 54 of them for
    /// each tool (`www.nhk.or.jp` and `www3.nhk.or.jp` under `nhk.or.jp`),
    /// and a path entry's domain is its host (`github.com/mdegans` is not
    /// named, nor covered)
    #[test]
    fn the_names_cover_54_entries_of_the_fleets_lists() {
        use agora_agentkit::display::{AgentDisplayProfiles, DomainFilter};
        let src = include_str!("../../examples/display.toml");
        let profiles: AgentDisplayProfiles = toml::from_str::<toml::Table>(src).unwrap()["display"]
            .clone()
            .try_into()
            .unwrap();
        let links = profiles.resolve(None, None).unwrap().config.links().clone();
        let named = crawler_refused(&production()).unwrap();
        let covers = |name: &str, entry: &str| {
            let host = entry.split('/').next().unwrap();
            host == name || host.strip_suffix(name).is_some_and(|s| s.ends_with('.'))
        };
        for domains in [links.fetch_domains(), links.search_domains()] {
            let DomainFilter::Allowed(list) = domains.filter() else {
                panic!("an allow list");
            };
            let covered: Vec<&String> = list
                .iter()
                .filter(|e| named.iter().any(|n| covers(n, e)))
                .collect();
            assert_eq!(covered.len(), 54, "{covered:?}");
            for name in &named {
                assert!(covered.iter().any(|e| covers(name, e)), "{name}");
            }
            assert!(covered.iter().any(|e| *e == "www3.nhk.or.jp"));
            assert!(covered.iter().any(|e| *e == "www.nhk.or.jp"));
            assert!(list.iter().any(|e| e.starts_with("github.com")));
            assert!(!covered.iter().any(|e| e.starts_with("github.com")));
        }
    }

    #[test]
    fn odd_shapes_are_read_tolerantly() {
        let read = |m: &str| crawler_refused(m);
        // Not this refusal at all
        assert_eq!(read("invalid request (400): prompt is too long"), None);
        assert_eq!(read(""), None);
        // Double quotes, no quotes, no brackets, upper case
        assert_eq!(
            read(
                r#"The following domains are not accessible to our user agent: ["a.com", "B.Org"]"#
            ),
            Some(vec!["a.com".into(), "b.org".into()])
        );
        assert_eq!(
            read(
                "The following domains are NOT accessible to our user agent: a.com, news.b.co.uk."
            ),
            Some(vec!["a.com".into(), "news.b.co.uk".into()])
        );
        // Without brackets and without an end, the last name may be cut
        assert_eq!(
            read("The following domains are not accessible to our user agent: a.com, news.b.co.uk"),
            Some(vec!["a.com".into()])
        );
        // One domain, said in the singular
        assert_eq!(
            read(
                "The following domain is not accessible to our user agent: 'x.example'. Read more: https://support.anthropic.com/x"
            ),
            Some(vec!["x.example".into()])
        );
        // Truncated inside the list: the last name may be cut, so it is dropped
        assert_eq!(
            read("…not accessible to our user agent: ['apnews.com', 'arstechnica.com', 'bbc.co"),
            Some(vec!["apnews.com".into(), "arstechnica.com".into()])
        );
        // Truncated just after a name's closing quote: every name is whole
        assert_eq!(
            read("…not accessible to our user agent: ['apnews.com', 'bbc.co.uk', "),
            Some(vec!["apnews.com".into(), "bbc.co.uk".into()])
        );
        // No list at all, or nothing that is a domain: the refusal, unread
        assert_eq!(
            read("domains are not accessible to our user agent"),
            Some(vec![])
        );
        assert_eq!(
            read("domains are not accessible to our user agent: [localhost, 10.0.0.1, 'v1.2']"),
            Some(vec![])
        );
        // A domain that begins with "http" is not the link
        assert_eq!(
            read(
                "not accessible to our user agent: ['httpbin.org']. Read more: https://x.example/a"
            ),
            Some(vec!["httpbin.org".into()])
        );
        // Cut on a dot inside a bracketed list: that name is cut too
        assert_eq!(
            read("not accessible to our user agent: ['apnews.com', 'bbc.co."),
            Some(vec!["apnews.com".into()])
        );
        // Upper case in the message is read as lower case
        assert_eq!(
            read("not accessible to our user agent: ['NYTimes.com']"),
            Some(vec!["nytimes.com".into()])
        );
        // Repeats are said once
        assert_eq!(
            read("not accessible to our user agent: ['a.com', 'a.com']"),
            Some(vec!["a.com".into()])
        );
    }

    #[test]
    fn a_runs_refusals_are_counted_together_and_others_ignored() {
        let production = production();
        let errors = [
            production.as_str(),
            "inference: Anthropic error: overloaded (529): busy",
            production.as_str(),
            "inference: Anthropic error: invalid request (400): The following domains are not accessible to our user agent: ['zz.example']",
            "inference: Anthropic error: invalid request (400): Some domains are not accessible to our user agent",
        ];
        let r = CrawlerRefusals::of(errors).unwrap();
        assert_eq!(r.sessions, 4);
        assert_eq!(r.domains.len(), 54);
        assert!(r.domains.contains("zz.example"));
        assert_eq!(r.unread.len(), 1);
        assert!(r.unread.iter().next().unwrap().contains("Some domains"));
        assert_eq!(
            CrawlerRefusals::of(["inference: Anthropic error: overloaded (529): busy"]),
            None
        );
    }

    /// One alert for the run: the count of sessions, every site, the two
    /// ways out; an unread refusal is passed on as the API wrote it
    #[tokio::test]
    async fn the_alert_names_the_sites_and_the_ways_out() {
        let root = crate::sealed::TestDir::new("crawler-refusal-alert");
        let sent = crate::alerts::testing::Recorder::default();
        let alerts = crate::alerts::testing::recording(root.path(), &sent);
        let production = production();
        let r = CrawlerRefusals::of([
            production.as_str(),
            production.as_str(),
            "x: domains are not accessible to our user agent",
        ])
        .unwrap();
        let listed = crate::display::WebSources {
            fetch: crate::display::WebSource::Policy,
            search: crate::display::WebSource::Unfiltered,
        };
        let alert = r.alert(listed);
        assert_eq!(alert.kind, AlertKind::WebDomainsRefused);
        assert_eq!(alert.count, Some(3));
        assert_eq!(
            alert.summary,
            "3 agents' sessions failed: the API refused their requests because a web tool's \
             allowed_domains names 53 sites that block Anthropic's crawler"
        );
        let detail = |name: &str| {
            alert
                .details
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        assert!(detail("domains").starts_with("afp.com, afr.com, apnews.com"));
        assert!(detail("domains").ends_with("wsj.com, zeit.de"));
        assert!(detail("ways out").contains("domains = \"unfiltered\""));
        assert!(detail("ways out").contains("[display.links]"));
        assert!(detail("unread").contains("x: domains are not accessible to our user agent"));
        // With no list sent by any tool, the way out is not "send none"
        let listless = crate::display::WebSources {
            fetch: crate::display::WebSource::Unfiltered,
            search: crate::display::WebSource::Off,
        };
        let other = r.alert(listless);
        let ways = &other
            .details
            .iter()
            .find(|(n, _)| *n == "ways out")
            .unwrap()
            .1;
        assert_eq!(ways, CrawlerRefusals::NO_LIST_SENT);
        alerts.notify(alert);
        alerts.flush(std::time::Duration::from_secs(5)).await;
        let mails = sent.sent();
        assert_eq!(mails.len(), 1, "{mails:?}");
        assert!(mails[0].contains("web_domains_refused"), "{}", mails[0]);
    }
}
