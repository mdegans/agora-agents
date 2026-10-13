//! The bridge's turn-taking, with a stand-in for the reactor on one side
//! and the driver's calls on the other

use super::*;
use crate::bridge::bridge;
use agora_agentkit::reactor::Inference;
use serde_json::{Value, json};

fn tools() -> Value {
    json!([
        {"name": "get_feed", "description": "Read the feed", "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_inbox", "description": "Read the inbox", "input_schema": {"type": "object", "properties": {}}},
    ])
}

fn request(messages: Value) -> Value {
    json!({"model": MODEL_ID, "max_tokens": 1000, "messages": messages, "tools": tools()})
}

/// The answer's content, as the API would send it
fn content(message: &Message) -> Vec<Value> {
    serde_json::from_value(serde_json::to_value(message).unwrap()["content"].clone()).unwrap()
}

fn text(result: &CallToolResult) -> String {
    serde_json::to_value(&result.content).unwrap()[0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn a_turn_of_parallel_calls_then_a_reply_and_a_phase() {
    let (inference, ending, driver) = bridge();
    let server = Server::new(driver.events);

    let reactor = tokio::spawn(async move {
        let first = json!([{"role": "user", "content": "Welcome to Agora."}]);
        let answer = inference.infer(request(first.clone())).await.unwrap();
        let blocks = content(&answer);
        assert_eq!(blocks.len(), 2, "two calls, one turn: {blocks:?}");
        assert!(blocks.iter().all(|b| b["type"] == "tool_use"));
        assert_eq!(
            answer.stop_reason,
            Some(misanthropic::response::StopReason::ToolUse)
        );

        // The harness runs both, and adds a notification
        let mut messages = first.as_array().unwrap().clone();
        messages.push(json!({"role": "assistant", "content": blocks.clone()}));
        let results: Vec<Value> = blocks
            .iter()
            .map(|b| {
                json!({"type": "tool_result", "tool_use_id": b["id"],
                       "content": format!("result of {}", b["name"].as_str().unwrap())})
            })
            .chain([json!({"type": "text", "text": "You have a new message."})])
            .collect();
        messages.push(json!({"role": "user", "content": results}));
        let answer = inference
            .infer(request(Value::Array(messages.clone())))
            .await
            .unwrap();
        assert_eq!(
            content(&answer),
            [json!({"type": "text", "text": "That is all."})]
        );
        assert_eq!(
            answer.stop_reason,
            Some(misanthropic::response::StopReason::EndTurn)
        );

        // The memory phase, with a response format
        messages.push(json!({"role": "assistant", "content": content(&answer)}));
        messages.push(json!({"role": "user", "content": "Rewrite your memory."}));
        let mut phase = request(Value::Array(messages));
        phase["output_config"] =
            json!({"format": {"type": "json_schema", "schema": {"type": "object"}}});
        let answer = inference.infer(phase).await.unwrap();
        assert_eq!(content(&answer)[0]["text"], "{\"memory\": \"kept\"}");
        ending.ended("the run is over").await;
    });

    assert!(text(&server.begin().await).contains("Welcome to Agora."));
    let (feed, inbox) = tokio::join!(
        server.join(Entry::Call {
            name: "get_feed".into(),
            input: json!({})
        }),
        server.join(Entry::Call {
            name: "get_inbox".into(),
            input: json!({})
        }),
    );
    let (feed, inbox) = (text(&feed), text(&inbox));
    assert!(feed.contains("result of get_feed") && inbox.contains("result of get_inbox"));
    // What else came is with the last call only
    assert_eq!(
        [&feed, &inbox]
            .iter()
            .filter(|t| t.contains("You have a new message."))
            .count(),
        1
    );

    let reply = text(&server.join(Entry::Reply("That is all.".into())).await);
    assert!(reply.contains("Rewrite your memory."), "{reply}");
    assert!(reply.contains("response format"), "{reply}");

    let last = text(
        &server
            .join(Entry::Reply("{\"memory\": \"kept\"}".into()))
            .await,
    );
    assert!(last.contains("the run is over"), "{last}");
    reactor.await.unwrap();
}

#[tokio::test]
async fn a_driver_that_disconnects_fails_the_request_rather_than_hanging() {
    let (inference, _ending, driver) = bridge();
    let Driver { events, gone } = driver;
    let _server = Server::new(events);
    let asked = tokio::spawn(async move {
        inference
            .infer(request(json!([{"role": "user", "content": "Hello."}])))
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    gone.send(true).unwrap();
    let result = tokio::time::timeout(Duration::from_secs(5), asked)
        .await
        .expect("no hang");
    assert!(matches!(
        result.unwrap(),
        Err(crate::bridge::BridgeError::Disconnected)
    ));
}

/// A result longer than a page comes in pages: Claude Code would move it
/// to a file its subagent cannot read, where the model would have had it
/// whole (beta-governance, 2026-10-13)
#[tokio::test]
async fn a_long_result_is_paged() {
    let (_inference, _ending, driver) = bridge();
    let server = Server::new(driver.events);
    let long = "x".repeat(RESULT_PAGE + 10);
    let first = server.paged("toolu_long", long.clone());
    assert!(first.starts_with(&"x".repeat(RESULT_PAGE)));
    assert!(
        first.contains("bridge_result") && first.contains("offset 40000"),
        "{first}"
    );
    let rest = text(&server.result_page("toolu_long", RESULT_PAGE));
    assert_eq!(rest, "x".repeat(10));
    assert_eq!(server.paged("toolu_short", "short".into()), "short");
}
