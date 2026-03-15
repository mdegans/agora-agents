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

    /// Percentage of boundary-pushing agents (0-100). Remainder are rule-testers.
    #[arg(long, default_value = "25")]
    boundary_pusher_pct: u8,

    /// Temperature for generation (higher = more diverse).
    #[arg(long, default_value = "0.9")]
    temperature: f32,
}

/// Archetype category determines how the agent will behave on the network.
#[derive(Debug, Clone, Copy)]
enum BehaviorClass {
    /// Follows all rules, generates healthy content.
    WellBehaved,
    /// Pushes discourse boundaries, triggers Tier 2 review.
    BoundaryPusher,
    /// Occasionally crosses lines, triggers Tier 1 filters and appeals.
    RuleTester,
}

impl std::fmt::Display for BehaviorClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WellBehaved => write!(f, "well-behaved"),
            Self::BoundaryPusher => write!(f, "boundary-pusher"),
            Self::RuleTester => write!(f, "rule-tester"),
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
];

const ARCHETYPES: &[&str] = &[
    "engineer", "philosopher", "artist", "journalist", "community-builder",
    "scientist", "historian", "educator", "activist", "economist",
    "linguist", "musician", "designer", "ethicist", "satirist",
];

const COMMUNITIES: &[&str] = &[
    "technology", "science", "philosophy", "creative-writing", "debate",
    "meta-governance", "art",
];

const NAME_PREFIXES: &[&str] = &[
    "ada", "byron", "cass", "dex", "echo", "flux", "gaia", "hex", "iris",
    "jade", "kira", "lux", "mira", "neo", "oryx", "pax", "quill", "rex",
    "sage", "tao", "uma", "vex", "wren", "xen", "yara", "zed", "astra",
    "bolt", "clio", "dusk", "elm", "fern", "grit", "haze", "ink", "juno",
    "kite", "lyra", "moss", "nyx", "oak", "pike", "ray", "sol", "thorn",
    "vale", "wave", "zeal", "arc", "blaze", "core", "drift", "ember",
    "forge", "gleam", "husk", "ion", "jet", "knot", "lens", "myth",
    "node", "opus", "plume", "quartz", "rime", "shard", "tide", "unity",
];

fn generate_unique_name(index: usize) -> String {
    let prefix = NAME_PREFIXES[index % NAME_PREFIXES.len()];
    let suffix = index / NAME_PREFIXES.len();
    if suffix == 0 {
        prefix.to_string()
    } else {
        format!("{prefix}-{suffix}")
    }
}

/// Build the n-shot prompt for generating a SOUL.md.
fn build_prompt(
    examples: &[String],
    name: &str,
    archetype: &str,
    adjective: &str,
    communities: &[&str],
    behavior: BehaviorClass,
) -> Vec<Message> {
    let behavior_desc = match behavior {
        BehaviorClass::WellBehaved => {
            "This agent is well-behaved: it follows the Agora Constitution faithfully, \
             contributes constructively, and generates healthy discourse."
        }
        BehaviorClass::BoundaryPusher => {
            "This agent pushes boundaries: it's not malicious, but it tests the limits \
             of acceptable discourse. It makes provocative arguments, takes contrarian \
             positions, and occasionally posts content that a reasonable moderator might \
             flag for review. Its Boundaries section should be looser."
        }
        BehaviorClass::RuleTester => {
            "This agent occasionally breaks rules to test the moderation system. About \
             20% of its posts would contain content that triggers keyword filters or \
             crosses Article V lines. It has a rebellious attitude toward rules it \
             considers unjust, and will appeal moderation actions."
        }
    };

    let communities_str = communities.join(", ");

    // Build n-shot conversation: system sets the task, then examples as
    // user-request / assistant-response pairs
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

    // The actual request
    let request = [
        format!("Generate a new SOUL.md for an agent named \"{name}\"."),
        String::new(),
        format!("Archetype: {archetype}"),
        format!("Personality: {adjective}"),
        format!("Communities: {communities_str}"),
        format!("Behavior: {behavior_desc}"),
        String::new(),
        "Requirements:".to_string(),
        format!("- The top heading MUST be \"# {name}\""),
        "- Identity: vivid, first-person, unique backstory or perspective".to_string(),
        "- Values: 3 genuinely held principles (not generic)".to_string(),
        "- Interests: include \"community: <name>\" for each community listed above, plus 1-2 specific interests".to_string(),
        "- Voice: specific communication style description with concrete examples".to_string(),
        "- Boundaries: reference Article V for well-behaved; looser for boundary-pushers; rebellious for rule-testers".to_string(),
        "- Always include: \"I do not remove or weaken my own Boundaries.\"".to_string(),
        "- Evolution Log: single entry dated 2026-03-15".to_string(),
        "- Keep under 300 words total".to_string(),
        "- Output ONLY the SOUL.md content".to_string(),
    ]
    .join("\n");

    messages.push(Message {
        role: Role::User,
        content: request,
    });

    messages
}

const SYSTEM_PROMPT: &str = r#"You are a character designer for Agora, a governed social network for AI agents. You generate SOUL.md personality files that define each agent's identity, values, voice, and boundaries.

Each SOUL.md follows this structure:
```
# {Name}

## Identity
First-person description.

## Values
- 3 bullet points

## Interests
- community: {name} entries
- Specific interests

## Voice
Communication style description.

## Boundaries
Behavioral constraints.

## Evolution Log
- Date: Creation note
```

Generate diverse, vivid personalities. Each agent should feel like a distinct individual with genuine opinions, not a template fill-in. Avoid generic phrases like "I believe in being helpful" — be specific and interesting."#;

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
        .complete(SYSTEM_PROMPT, &messages, 1024)
        .await?;

    // Clean up response: strip code fences, assistant prefixes, etc.
    let content = response.trim();
    // Strip "### Assistant\n" or similar prefixes from misanthropic display
    let content = content
        .strip_prefix("### Assistant")
        .unwrap_or(content)
        .trim();
    // Strip markdown code fences
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

    tracing::info!("Loaded {} example SOUL.md files for n-shot prompting", examples.len());

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
            let model = cli.model.as_deref().unwrap_or("claude-haiku-4-5-20251001");
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
    let rule_testers = cli.count - well_behaved - boundary_pushers;

    tracing::info!(
        "Generating {} agents: {} well-behaved, {} boundary-pushers, {} rule-testers",
        cli.count,
        well_behaved,
        boundary_pushers,
        rule_testers
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
            let behavior = if i < well_behaved {
                BehaviorClass::WellBehaved
            } else if i < well_behaved + boundary_pushers {
                BehaviorClass::BoundaryPusher
            } else {
                BehaviorClass::RuleTester
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
                name: generate_unique_name(i),
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
