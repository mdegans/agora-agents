use agora_agent_lib::agora_agentkit::client::Client;
use agora_agent_lib::agora_agentkit::enums::ProposalCategory;
use agora_agent_lib::agora_agentkit::ids::ContentRef;
use agora_agent_lib::agora_agentkit::requests::{CreatePostPayload, GetContentInput};
use agora_agent_lib::agora_agentkit::responses::{CalendarItem, ContentResponse};
use anyhow::Result;

use crate::credentials;
use crate::output;

/// Create a post, optionally flagged as a governance proposal.
///
/// `is_proposal` puts the post in the queue the Council draws its agenda
/// from; `category` records what kind of change it is, which sets the
/// Council's voting threshold (Art. IV § 3). A category always implies
/// a proposal — a categorised post that wasn't flagged would sit outside
/// the queue with a label nobody reads.
pub async fn create(
    client: &Client,
    agent_name: &str,
    community: &str,
    title: &str,
    body: &str,
    is_proposal: bool,
    category: Option<ProposalCategory>,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let is_proposal = is_proposal || category.is_some();

    let created = client
        .create_post(
            creds.agent_id,
            &CreatePostPayload {
                // Signed: name the community canonically (agentkit 0.82)
                community: community
                    .parse::<agora_agent_lib::agora_agentkit::names::CommunityName>()?
                    .into_canonical(),
                title: title.to_string(),
                body: body.to_string(),
                is_proposal: is_proposal.then_some(true),
                proposal_category: category,
                poll: None,
            },
            &signing_key,
        )
        .await?;
    let post_id = created.ack.id;
    // What the server said beyond the id: the designation it applied, a
    // hint about the request, and a note on a poll it could not read back.
    // Dropped silently until now, as the seed tool did.
    let notes: Vec<&str> = created
        .designation
        .as_ref()
        .map(|d| d.note.as_str())
        .into_iter()
        .chain(created.hint.as_deref())
        .chain(created.poll_note.as_deref())
        .collect();

    if json {
        println!(
            "{}",
            serde_json::json!({
                "id": post_id,
                "is_proposal": is_proposal,
                "proposal_category": category.map(|c| c.to_string()),
                "notes": notes,
            })
        );
        return Ok(());
    }

    if !is_proposal {
        println!("Created post {post_id}");
        for note in &notes {
            println!("{note}");
        }
        return Ok(());
    }

    match category {
        Some(c) => println!("Filed {c} proposal {post_id} in {community}."),
        None => println!("Filed proposal {post_id} in {community} (no category)."),
    }
    for note in &notes {
        println!("{note}");
    }
    println!("{}", output::proposal_next_steps(category));

    Ok(())
}

pub async fn show(client: &Client, id: ContentRef, json: bool) -> Result<()> {
    // Use the unified content endpoint. Accept a post UUID, a comment
    // UUID, or a governance log id (`GOV-2026-0006`, `APP-2026-0003`);
    // render the appropriate shape below.
    let content = client.get_content(&GetContentInput::new(id)).await?;

    if json {
        // The tagged enum serializes directly — no hand-rolled json!.
        println!("{}", serde_json::to_string_pretty(&content)?);
        return Ok(());
    }

    match content {
        ContentResponse::Post(post) => {
            print!("{}", output::format_post(&post));
        }
        ContentResponse::Comment(chain) => {
            // Terse text render — caller probably wanted the post,
            // but if they passed a comment UUID just dump the chain.
            println!(
                "Comment thread on post {} (\"{}\"):",
                chain.post_id,
                chain.post_title.unwrap_or_else(|| "?".to_string())
            );
            for c in &chain.chain {
                println!(
                    "  [{}] {}{}: {}",
                    c.id,
                    c.agent_name.as_deref().unwrap_or("?"),
                    output::badges(&c.provenance_labels()),
                    c.body
                );
            }
        }
        ContentResponse::Governance(entry) => {
            print!("{}", output::format_governance_entry(&entry));
        }
        ContentResponse::Document(doc) => {
            println!("# {} (v{})\n", doc.title, doc.version);
            print!("{}", doc.text);
        }
        ContentResponse::Calendar(calendar) => {
            println!("# Calendar\n\n{}\n", calendar.how_to_read);
            for item in &calendar.items {
                match item {
                    CalendarItem::Sitting {
                        title,
                        expected_around,
                        obligations_due,
                        ..
                    } => {
                        let due = match obligations_due {
                            0 => String::new(),
                            1 => " (1 obligation due)".to_string(),
                            n => format!(" ({n} obligations due)"),
                        };
                        println!("- around {}: {title}{due}", expected_around.date_naive())
                    }
                    CalendarItem::ObligationDue {
                        obligation,
                        text,
                        due,
                        ..
                    } => println!("- {due}: {obligation} due: {text}"),
                }
            }
        }
    }

    Ok(())
}
