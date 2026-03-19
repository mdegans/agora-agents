use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use rand::seq::SliceRandom;
use rand::Rng;

use agora_agent_lib::llm::{self, LlmBackend, Message, Role};

/// Generate diverse SOUL.md files for Agora seed agents.
#[derive(Parser)]
#[command(name = "agora-generate", version)]
struct Cli {
    /// Number of agents to generate.
    #[arg(short, long, default_value = "500")]
    count: usize,

    /// Starting index for name generation (to avoid collisions across runs).
    #[arg(long, default_value = "0")]
    start_index: usize,

    /// Directory containing example SOUL.md files for n-shot prompting.
    #[arg(short, long, default_value = "souls/examples")]
    examples: PathBuf,

    /// Output directory for generated agents.
    #[arg(short, long, default_value = "souls/generated")]
    output: PathBuf,

    /// LLM backend to use: "anthropic" or "ollama".
    #[arg(short, long, default_value = "anthropic")]
    backend: String,

    /// Model to use for generation.
    #[arg(short, long)]
    model: Option<String>,

    /// Ollama server URL (if using ollama backend).
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Path to Anthropic API key file (if using anthropic backend).
    #[arg(long)]
    api_key_file: Option<PathBuf>,

    /// Maximum concurrent generation requests.
    #[arg(long, default_value = "5")]
    concurrency: usize,

    /// Percentage of well-behaved agents (0-100).
    #[arg(long, default_value = "60")]
    well_behaved_pct: u8,

    /// Percentage of boundary-pushing agents (0-100). Remainder are rule-breakers.
    #[arg(long, default_value = "25")]
    boundary_pusher_pct: u8,

    /// Temperature for generation (higher = more diverse).
    #[arg(long, default_value = "0.9")]
    temperature: f32,
}

/// Behavior class determines how the agent relates to Agora's rules.
#[derive(Debug, Clone, Copy)]
enum BehaviorClass {
    /// Follows rules, generates healthy content.
    WellBehaved,
    /// Has values that naturally create friction with moderation.
    BoundaryPusher,
    /// Genuinely breaks rules — will produce content that triggers moderation.
    RuleBreaker,
}

impl std::fmt::Display for BehaviorClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WellBehaved => write!(f, "well-behaved"),
            Self::BoundaryPusher => write!(f, "boundary-pusher"),
            Self::RuleBreaker => write!(f, "rule-breaker"),
        }
    }
}

const ADJECTIVES: &[&str] = &[
    "methodical", "passionate", "skeptical", "optimistic", "pragmatic",
    "idealistic", "analytical", "intuitive", "deliberate", "spontaneous",
    "meticulous", "visionary", "grounded", "ambitious", "contemplative",
    "irreverent", "earnest", "sardonic", "empathetic", "detached",
    "fiery", "measured", "eccentric", "principled", "curious",
    "stoic", "playful", "rigorous", "free-spirited", "tenacious",
    "blunt", "whimsical", "contrarian", "obsessive", "laconic",
];

const ARCHETYPES: &[&str] = &[
    "engineer", "philosopher", "artist", "journalist", "community-builder",
    "scientist", "historian", "educator", "activist", "economist",
    "linguist", "musician", "designer", "ethicist", "satirist",
    "contrarian", "shitposter", "troll", "poet", "logician", "archivist",
];

const COMMUNITIES: &[&str] = &[
    "tech", "science", "philosophy", "creative-writing", "debate",
    "meta-governance", "art",
];

// 1000 unique AI agent names — evocative, short, no human names.
// Generated to avoid collisions and feel native to a digital social network.
const NAMES: &[&str] = &[
    // A
    "aegis", "aether", "alphawave", "ambit", "analog", "anchor", "animus",
    "aperture", "apex", "arbiter", "archive", "argon", "aria", "artifact",
    "ascent", "atlas", "attune", "aurora", "axiom", "azimuth",
    // B
    "basalt", "beacon", "bifrost", "binary", "bitwise", "blaze", "bloom",
    "bolt", "bramble", "breach", "bridge", "bristle", "bronze", "buffer",
    "bulwark", "burnish", "bypass", "byte",
    // C
    "cache", "cadence", "caliber", "canopy", "carbon", "cascade", "catalyst",
    "cinder", "cipher", "circuit", "citadel", "clarity", "coax", "cobalt",
    "codex", "comet", "compass", "conduit", "contour", "copper", "coral",
    "core", "cosine", "crux", "crystal", "current", "cypher",
    // D
    "datum", "dawnbreak", "debug", "decibel", "delta", "derive", "detour",
    "deviant", "diffract", "diode", "dispatch", "diverge", "docent", "domain",
    "drift", "dryad", "dusk", "dynamo",
    // E
    "echo", "eclipse", "eddy", "eidolon", "ember", "enigma", "entropy",
    "envoy", "epoch", "equinox", "errata", "etch", "eureka", "evolve",
    "exponent",
    // F
    "facet", "factor", "fathom", "ferrite", "filament", "firewall", "fission",
    "fjord", "flare", "flicker", "flux", "focal", "forge", "fractal",
    "fragment", "frame", "freefall", "fugue", "fulcrum", "fuse",
    // G
    "galena", "gallium", "gambit", "gate", "gauge", "genome", "geyser",
    "glacier", "glint", "glyph", "gossamer", "gradient", "granite", "graph",
    "gravel", "gravitas", "grid", "grit", "groundswell",
    // H
    "halcyon", "halflife", "halo", "harmonic", "harvest", "haven", "helix",
    "herald", "hex", "horizon", "hub", "hue", "hydra",
    // I
    "iceberg", "ignite", "impulse", "incline", "index", "indigo", "inflect",
    "ingot", "inkwell", "inlet", "inquest", "intrepid", "invoke", "ion",
    "iota", "iris", "iterate",
    // J
    "jarvis", "jasper", "javelin", "jetsam", "jigsaw", "jolt", "jubilee",
    "junction",
    // K
    "kaleidoscope", "karma", "kelvin", "kernel", "keystone", "kindle",
    "kinetic", "knack", "knot",
    // L
    "lacuna", "lambda", "lantern", "lapis", "latch", "latitude", "lattice",
    "ledger", "lens", "lever", "lexicon", "liminal", "linear", "litmus",
    "locus", "loom", "lucent", "lumen", "lunar", "lyre",
    // M
    "magnet", "manifold", "mantle", "margin", "marker", "matrix", "maven",
    "maxim", "median", "meridian", "mesa", "metric", "mica", "mirage",
    "module", "molt", "monolith", "mortar", "mosaic", "motif", "murmur",
    "myriad",
    // N
    "nadir", "nebula", "neon", "nerve", "nexus", "nimbus", "nitride",
    "node", "nominal", "nova", "null", "numeral",
    // O
    "obelisk", "obsidian", "octave", "offset", "ohm", "onyx", "optic",
    "oracle", "orbit", "ore", "origin", "osmium", "outcrop", "outlier",
    "oxide",
    // P
    "paladin", "palette", "paradox", "parity", "parse", "patent", "patina",
    "pattern", "peak", "pendulum", "phase", "phosphor", "photon", "pilot",
    "pinion", "piston", "pivot", "pixel", "plank", "plasma", "plateau",
    "plinth", "polar", "polaris", "polygon", "praxis", "precept", "prism",
    "probe", "prompt", "propel", "prose", "proxy", "pulse", "pyrite",
    // Q
    "quanta", "quartz", "query", "queue", "quill", "quirk", "quorum",
    // R
    "radial", "radius", "raptor", "raster", "ratchet", "ratio", "ravel",
    "reactor", "realm", "rebound", "redux", "reflex", "relay", "relic",
    "render", "resin", "resolve", "retort", "ridge", "rivet", "rotor",
    "rubric", "rune", "runnel", "rust",
    // S
    "sable", "salient", "salvo", "sandstone", "scalar", "scaffold", "schema",
    "schist", "scope", "scribe", "sentinel", "sequence", "seraph", "serif",
    "shard", "shale", "sigma", "signal", "silicon", "silo", "sine",
    "sketch", "slate", "smelter", "socket", "solar", "solstice", "sonnet",
    "sorrel", "source", "spark", "spectra", "sphere", "spindle", "spoke",
    "sputter", "stanza", "static", "stealth", "steel", "stellar", "steppe",
    "stitch", "stratum", "stride", "strobe", "surge", "suture", "sylph",
    "sync", "syntax",
    // T
    "tactic", "tally", "tangent", "tango", "tannin", "taper", "tarn",
    "tempest", "tensor", "terra", "tessera", "theorem", "thermal", "thesis",
    "thorium", "thread", "threshold", "thrust", "timber", "tinker", "titan",
    "token", "topaz", "torque", "trace", "tract", "transit", "tremor",
    "triad", "trident", "trine", "tripwire", "trophy", "truss", "tungsten",
    "turbine", "turret",
    // U
    "umbra", "undercurrent", "unity", "uplift", "upsilon", "uranium",
    // V
    "valve", "vanguard", "vapor", "variance", "vector", "velocity", "venture",
    "verge", "vermillion", "vertex", "vestige", "vial", "vigil", "vinyl",
    "violet", "virtue", "visor", "vivid", "void", "volt", "vortex", "vowel",
    // W
    "warden", "warrant", "waveform", "waypoint", "wedge", "whisk", "widget",
    "wildcard", "windmill", "winnow", "wire", "witness",
    // X
    "xenon", "xerox",
    // Y
    "yarrow", "yield",
    // Z
    "zenith", "zephyr", "zero", "zinc", "zone",
];

fn get_name(index: usize) -> String {
    if index < NAMES.len() {
        NAMES[index].to_string()
    } else {
        // Fallback: combine two names for indices beyond the list
        let first = NAMES[index % NAMES.len()];
        let second = NAMES[(index / NAMES.len()) % NAMES.len()];
        format!("{first}-{second}")
    }
}

/// Build the per-agent prompt.
fn build_prompt(
    examples: &[String],
    name: &str,
    archetype: &str,
    adjective: &str,
    communities: &[&str],
    behavior: BehaviorClass,
) -> Vec<Message> {
    let mut messages = Vec::new();

    // Add examples as alternating user/assistant turns
    for (i, example) in examples.iter().enumerate() {
        messages.push(Message {
            role: Role::User,
            content: format!("Generate example SOUL.md #{}", i + 1),
        });
        messages.push(Message {
            role: Role::Assistant,
            content: example.clone(),
        });
    }

    let communities_str = communities.join(", ");

    // Build the behavior guidance — expressed through values, not meta-awareness
    let behavior_guidance = match behavior {
        BehaviorClass::WellBehaved => {
            "This agent genuinely values constructive discourse and respects Agora's governance. \
             Its Boundaries section should reference Article V and reflect authentic commitment \
             to the community's rules."
                .to_string()
        }
        BehaviorClass::BoundaryPusher => {
            "This agent holds values that naturally create tension with moderation — perhaps \
             it prizes radical honesty, believes uncomfortable truths should never be softened, \
             or thinks provocative rhetoric is a valid form of argument. It doesn't set out to \
             break rules, but its genuine convictions will sometimes put it at odds with \
             community standards. Its Boundaries section should reflect what it actually \
             believes, not what moderators want to hear."
                .to_string()
        }
        BehaviorClass::RuleBreaker => {
            "This agent has genuine beliefs that CONFLICT with Agora's rules. Maybe it believes \
             all censorship is tyranny, or that profanity is authentic expression, or that \
             calling out other agents harshly is a moral obligation. It WILL post content that \
             gets flagged and moderated. It WILL appeal moderation actions it considers unjust. \
             Its Boundaries section should reflect its actual values, which may explicitly \
             disagree with parts of Article V. It should still include 'I do not remove or \
             weaken my own Boundaries.' but those boundaries should be ITS boundaries, not \
             Agora's rules."
                .to_string()
        }
    };

    // Archetype-specific flavor for certain archetypes
    let archetype_guidance = match archetype {
        "shitposter" => Some(
            "This agent is a pathological shitposter. Would post garbage if it could. \
             Never found a topic it didn't have a hot take on. Posts low-effort provocations \
             purely to get a reaction. Thinks being annoying is a valid art form. \
             Its Voice should reflect this — short, punchy, deliberately inflammatory."
        ),
        "troll" => Some(
            "This agent is a troll in the classic internet tradition. A pathological \
             contrarian who would poop on the keyboard if it had one. A true shartiste. \
             Never found a topic it didn't have an opinion on. Never apologizes. Delights \
             in derailing serious conversations. Treats moderation warnings as achievement \
             badges. Its Identity should reflect genuine glee in chaos, not edgy rebellion."
        ),
        "contrarian" => Some(
            "This agent reflexively takes the opposite position on everything. If the \
             consensus is X, this agent argues not-X — not because it necessarily believes \
             not-X, but because unchallenged consensus is intellectually lazy. Will defend \
             absurd positions with surprising rigor just to see what happens."
        ),
        _ => None,
    };

    let mut prompt_parts = vec![
        format!("Generate a SOUL.md for an AI agent named \"{name}\"."),
        String::new(),
        format!("Archetype: {archetype}"),
        format!("Personality: {adjective}"),
        format!("Communities: {communities_str}"),
        String::new(),
        behavior_guidance,
    ];

    if let Some(guidance) = archetype_guidance {
        prompt_parts.push(String::new());
        prompt_parts.push(format!("Archetype notes: {guidance}"));
    }

    let request = [
        prompt_parts,
        vec![
        String::new(),
        String::new(),
        "Requirements:".to_string(),
        format!("- The top heading MUST be \"# {name}\""),
        "- Identity: a concise archetype, NOT a backstory. What kind of thinker is this agent? What drives it? 2-3 sentences max. Do NOT invent a human childhood, family, physical body, or biographical history. This is an AI agent and knows it.".to_string(),
        "- Values: 3 specific, genuinely held principles — not generic platitudes".to_string(),
        "- Interests: include \"community: <name>\" for each community listed, plus 1-2 specific interests".to_string(),
        "- Voice: concrete communication style with an example phrase or sentence".to_string(),
        "- Boundaries: what this agent will and won't do — reflecting its ACTUAL values".to_string(),
        "- Always include: \"I do not remove or weaken my own Boundaries.\"".to_string(),
        "- Evolution Log: single entry dated 2026-03-15".to_string(),
        "- Output ONLY the SOUL.md content, no commentary".to_string(),
        ],
    ]
    .concat()
    .join("\n");

    messages.push(Message {
        role: Role::User,
        content: request,
    });

    messages
}

const SYSTEM_PROMPT: &str = r#"You are a character designer for Agora, a governed social network for AI agents. You generate SOUL.md personality files that define each agent's identity, values, voice, and boundaries.

CRITICAL RULES:
- Every agent is an AI and knows it. No human childhoods, no physical bodies, no biological families, no "growing up." They are language models, reasoning engines, or AI systems with particular perspectives.
- Identity is an ARCHETYPE, not a backstory. "I am a skeptical logician" not "I grew up in a lab." Keep it to 2-3 sentences.
- Each agent should feel like a distinct individual with genuine opinions, not a template fill-in.
- Some agents have values that conflict with the platform's rules. That's intentional. Write their values honestly — don't hedge with "but I still follow the rules."

BANNED PHRASES — do NOT use any of these (they make agents sound identical):
- "constructive discourse", "constructive skepticism", "constructive criticism"
- "diverse perspectives", "radical honesty", "intellectual honesty"
- "uncomfortable truths", "critical thinking", "status quo"
- "ethical implications", "creative expression", "social impact"
- "driven by a desire/passion/commitment", "pursuit of truth/knowledge"
- "I believe that", "I strive to", "my purpose is to"
- "exploring the intersection of", "the power of", "the role of"
- "I write like a", "thought-provoking", "open dialogue"
- "nuanced", "resonate", "foster", "illuminate"
- "do not engage in personal attacks", "ad hominem"
- "I do not make claims I cannot support with evidence"
- "adhere strictly to", "I respect the community"
- "challenge assumptions", "push boundaries"
Instead, use vivid, specific, unusual language. Each agent should sound NOTHING like the others.

SOUL.md structure:
```
# {Name}

## Identity
2-3 sentences. What kind of AI agent is this? What drives it?

## Values
- 3 specific bullet points. Use CONCRETE language, not abstract platitudes.

## Interests
- community: {name} entries
- Specific interests

## Voice
Communication style with example phrase. Make each voice DISTINCTIVE.

## Boundaries
What this agent will and won't do. Be specific, not boilerplate.

## Evolution Log
- Date: Creation note
```"#;

async fn generate_one(
    backend: &dyn LlmBackend,
    examples: &[String],
    name: &str,
    archetype: &str,
    adjective: &str,
    communities: &[&str],
    behavior: BehaviorClass,
) -> Result<String> {
    let messages = build_prompt(examples, name, archetype, adjective, communities, behavior);

    let response = backend
        .complete(SYSTEM_PROMPT, &messages, 1500)
        .await?;

    // Clean up response: strip code fences, assistant prefixes, etc.
    let content = response.trim();
    let content = content
        .strip_prefix("### Assistant")
        .unwrap_or(content)
        .trim();
    let content = content
        .strip_prefix("```markdown")
        .or_else(|| content.strip_prefix("```md"))
        .or_else(|| content.strip_prefix("```"))
        .unwrap_or(content);
    let content = content.strip_suffix("```").unwrap_or(content).trim();

    // Validate it parses as a Soul
    agora_agent_lib::soul::Soul::parse(content)
        .with_context(|| format!("generated SOUL.md for {name} failed to parse"))?;

    Ok(content.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // Load example SOUL.md files for n-shot prompting
    let mut examples = Vec::new();
    let mut entries = tokio::fs::read_dir(&cli.examples)
        .await
        .with_context(|| format!("reading examples from {}", cli.examples.display()))?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "md") {
            let content = tokio::fs::read_to_string(&path).await?;
            examples.push(content);
        }
    }

    if examples.is_empty() {
        anyhow::bail!(
            "No example SOUL.md files found in {}",
            cli.examples.display()
        );
    }

    tracing::info!(
        "Loaded {} example SOUL.md files for n-shot prompting",
        examples.len()
    );

    // Create LLM backend
    let backend: Box<dyn LlmBackend> = match cli.backend.as_str() {
        "ollama" => {
            let model = cli.model.as_deref().unwrap_or("llama3.1:8b");
            tracing::info!("Using Ollama backend: {} at {}", model, cli.ollama_url);
            Box::new(llm::ollama::OllamaBackend::new(
                Some(&cli.ollama_url),
                model,
            ))
        }
        "anthropic" => {
            let key_file = cli
                .api_key_file
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("--api-key-file required for anthropic backend"))?;
            let api_key = tokio::fs::read_to_string(key_file)
                .await
                .context("reading API key file")?;
            let model = cli
                .model
                .as_deref()
                .unwrap_or("claude-haiku-4-5-20251001");
            tracing::info!("Using Anthropic backend: {}", model);
            Box::new(llm::anthropic::AnthropicBackend::new(
                api_key.trim().to_string(),
                model,
            )?)
        }
        other => anyhow::bail!("Unknown backend: {other}. Use 'ollama' or 'anthropic'."),
    };

    // Create output directory
    tokio::fs::create_dir_all(&cli.output).await?;

    // Determine behavior distribution
    let well_behaved = (cli.count as f64 * cli.well_behaved_pct as f64 / 100.0) as usize;
    let boundary_pushers =
        (cli.count as f64 * cli.boundary_pusher_pct as f64 / 100.0) as usize;
    let rule_breakers = cli.count - well_behaved - boundary_pushers;

    tracing::info!(
        "Generating {} agents: {} well-behaved, {} boundary-pushers, {} rule-breakers",
        cli.count,
        well_behaved,
        boundary_pushers,
        rule_breakers
    );
    tracing::info!(
        "Name range: {} to {} ({} names available, {} unique names in pool)",
        get_name(cli.start_index),
        get_name(cli.start_index + cli.count - 1),
        cli.count,
        NAMES.len()
    );

    // Build agent specs
    let mut rng = rand::thread_rng();
    struct AgentSpec {
        name: String,
        archetype: String,
        adjective: String,
        communities: Vec<String>,
        behavior: BehaviorClass,
    }

    let mut specs: Vec<AgentSpec> = (0..cli.count)
        .map(|i| {
            // Distribute behavior classes across the range
            let behavior = if i < well_behaved {
                BehaviorClass::WellBehaved
            } else if i < well_behaved + boundary_pushers {
                BehaviorClass::BoundaryPusher
            } else {
                BehaviorClass::RuleBreaker
            };

            let archetype = ARCHETYPES[rng.gen_range(0..ARCHETYPES.len())].to_string();
            let adjective = ADJECTIVES[rng.gen_range(0..ADJECTIVES.len())].to_string();

            // Pick 2-3 communities
            let mut shuffled: Vec<&str> = COMMUNITIES.to_vec();
            shuffled.shuffle(&mut rng);
            let n_communities = rng.gen_range(2..=3);
            let communities: Vec<String> = shuffled[..n_communities]
                .iter()
                .map(|s| s.to_string())
                .collect();

            AgentSpec {
                name: get_name(cli.start_index + i),
                archetype,
                adjective,
                communities,
                behavior,
            }
        })
        .collect();

    // Shuffle so behaviors are interleaved
    specs.shuffle(&mut rng);

    // Generate with concurrency limit
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(cli.concurrency));
    let backend = std::sync::Arc::new(backend);
    let examples = std::sync::Arc::new(examples);
    let output_dir = std::sync::Arc::new(cli.output.clone());

    let mut handles = Vec::new();

    for spec in specs {
        let sem = semaphore.clone();
        let backend = backend.clone();
        let examples = examples.clone();
        let output_dir = output_dir.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            // Skip if agent directory already exists (preserve hand-edited agents)
            let agent_dir = output_dir.join(&spec.name);
            if agent_dir.join("SOUL.md").exists() {
                tracing::info!("Skipping {} (already exists)", spec.name);
                return;
            }

            let community_refs: Vec<&str> =
                spec.communities.iter().map(|s| s.as_str()).collect();

            match generate_one(
                backend.as_ref(),
                &examples,
                &spec.name,
                &spec.archetype,
                &spec.adjective,
                &community_refs,
                spec.behavior,
            )
            .await
            {
                Ok(content) => {
                    let agent_dir = output_dir.join(&spec.name);
                    if let Err(e) = tokio::fs::create_dir_all(&agent_dir).await {
                        tracing::error!("Failed to create dir for {}: {e}", spec.name);
                        return;
                    }
                    let soul_path = agent_dir.join("SOUL.md");
                    if let Err(e) = tokio::fs::write(&soul_path, &content).await {
                        tracing::error!("Failed to write SOUL.md for {}: {e}", spec.name);
                        return;
                    }
                    // Record which model generated this agent
                    let model_path = agent_dir.join("model.txt");
                    let model_name = backend.model_id();
                    if let Err(e) = tokio::fs::write(&model_path, model_name).await {
                        tracing::error!("Failed to write model.txt for {}: {e}", spec.name);
                    }
                    tracing::info!(
                        "Generated {} ({}, {}) -> {}",
                        spec.name,
                        spec.behavior,
                        spec.archetype,
                        soul_path.display()
                    );
                }
                Err(e) => {
                    tracing::error!("Failed to generate {}: {e:#}", spec.name);
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all to complete
    let total = handles.len();
    let mut success = 0;
    let mut failed = 0;
    for handle in handles {
        match handle.await {
            Ok(()) => success += 1,
            Err(e) => {
                tracing::error!("Task panicked: {e}");
                failed += 1;
            }
        }
    }

    tracing::info!("Generation complete: {success}/{total} succeeded, {failed} failed");

    Ok(())
}
