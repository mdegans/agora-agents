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

fn call(name: &str) -> Joining {
    Joining::call(name, json!({}))
}

/// A stand-in reactor that answers `turns` turns, each with a result per
/// call, then ends the run, giving back what each turn held
fn reactor(
    inference: crate::bridge::BridgeInference,
    ending: crate::bridge::Ending,
    turns: usize,
) -> tokio::task::JoinHandle<Vec<Vec<Value>>> {
    tokio::spawn(async move {
        let mut messages = vec![json!({"role": "user", "content": "Welcome to Agora."})];
        let mut held = Vec::new();
        for _ in 0..turns {
            let answer = inference
                .infer(request(Value::Array(messages.clone())))
                .await
                .unwrap();
            let blocks = content(&answer);
            messages.push(json!({"role": "assistant", "content": blocks.clone()}));
            let results: Vec<Value> = blocks
                .iter()
                .filter(|b| b["type"] == "tool_use")
                .map(|b| json!({"type": "tool_result", "tool_use_id": b["id"], "content": "done"}))
                .collect();
            let next = match results.is_empty() {
                true => json!({"role": "user", "content": "Go on."}),
                false => json!({"role": "user", "content": results}),
            };
            messages.push(next);
            held.push(blocks);
        }
        ending.ended("the run is over").await;
        held
    })
}

/// A stand-in reactor that answers one turn and ends the run
fn one_turn(
    inference: crate::bridge::BridgeInference,
    ending: crate::bridge::Ending,
) -> tokio::task::JoinHandle<Vec<Value>> {
    let turns = reactor(inference, ending, 1);
    tokio::spawn(async move { turns.await.unwrap().remove(0) })
}

fn tool_uses(blocks: &[Value]) -> usize {
    blocks.iter().filter(|b| b["type"] == "tool_use").count()
}

/// Arrivals at times after the start, each joining the turn; their
/// results in the order given
async fn arrive(server: &Server, arrivals: Vec<(u64, Joining)>) -> Vec<String> {
    let mut set = tokio::task::JoinSet::new();
    for (i, (at, joining)) in arrivals.into_iter().enumerate() {
        let server = server.clone();
        set.spawn(async move {
            tokio::time::sleep(Duration::from_millis(at)).await;
            (i, text(&server.join(joining).await))
        });
    }
    let mut results = set.join_all().await;
    results.sort_by_key(|(i, _)| *i);
    results.into_iter().map(|(_, t)| t).collect()
}

/// What the betas saw, 2026-10-19: [get_inbox, get_friends, get_content,
/// get_content, bridge_end_turn], the marker arriving second and the rest
/// seconds later, went as get_inbox alone and then three calls. With the
/// count it is one turn, sent when the last call arrives
#[tokio::test(start_paused = true)]
async fn the_betas_sequence_is_one_turn() {
    let (inference, ending, driver) = bridge();
    let server = Server::new(driver.events);
    let turns = one_turn(inference, ending);
    let started = Instant::now();
    let results = arrive(
        &server,
        vec![
            (0, call("get_inbox")),
            (100, Joining::end(4)),
            (2_000, call("get_friends")),
            (5_200, call("get_content")),
            (5_300, call("get_content")),
        ],
    )
    .await;
    assert_eq!(tool_uses(&turns.await.unwrap()), 4);
    assert_eq!(
        started.elapsed(),
        Duration::from_millis(5_300),
        "sent at once"
    );
    assert_eq!(results[1], "[bridge] Turn 1 sent: 4 call(s).");
    for result in [&results[0], &results[2], &results[3], &results[4]] {
        assert!(result.contains("one of 4 call(s)"), "{result}");
        assert!(!result.contains("calls count"), "{result}");
    }
}

/// A driver that forgets the marker gets its turn after the quiet window,
/// whole, and is told how to send it at once
#[tokio::test(start_paused = true)]
async fn without_the_marker_the_turn_waits_out_the_window() {
    let (inference, ending, driver) = bridge();
    let server = Server::new(driver.events);
    let turns = one_turn(inference, ending);
    let started = Instant::now();
    let results = arrive(
        &server,
        vec![(0, call("get_feed")), (5_000, call("get_inbox"))],
    )
    .await;
    assert_eq!(tool_uses(&turns.await.unwrap()), 2);
    assert_eq!(
        started.elapsed(),
        Duration::from_millis(5_000) + CALL_WINDOW
    );
    for result in &results {
        assert!(result.contains("one of 2 call(s)"), "{result}");
        assert!(result.contains("No bridge_end_turn arrived"), "{result}");
        assert!(!result.contains("too low"), "{result}");
    }
}

/// A marker declaring fewer calls than had arrived when it did sends them
/// all, as one turn, and says so
#[tokio::test(start_paused = true)]
async fn an_over_count_sends_what_arrived_and_says_so() {
    let (inference, ending, driver) = bridge();
    let server = Server::new(driver.events);
    let turns = one_turn(inference, ending);
    let results = arrive(
        &server,
        vec![
            (0, call("get_feed")),
            (1_000, call("get_inbox")),
            (2_000, Joining::end(1)),
        ],
    )
    .await;
    assert_eq!(tool_uses(&turns.await.unwrap()), 2);
    for result in &results {
        assert!(
            result.contains("calls count was 1, but 2 agent call(s) had arrived"),
            "{result}"
        );
    }
    assert!(results[2].starts_with("[bridge] Turn 1 sent: 2 call(s)."));
}

/// A marker that arrives first with too low a count sends the turn at the
/// count; a call after it is a turn of its own, told the count was low
#[tokio::test(start_paused = true)]
async fn an_over_count_after_the_marker_is_named_in_the_next_turn() {
    let (inference, ending, driver) = bridge();
    let server = Server::new(driver.events);
    let turns = reactor(inference, ending, 2);
    let results = arrive(
        &server,
        vec![
            (0, Joining::end(1)),
            (1_000, call("get_feed")),
            (2_000, call("get_inbox")),
        ],
    )
    .await;
    let turns = turns.await.unwrap();
    assert_eq!(
        turns.iter().map(|t| tool_uses(t)).collect::<Vec<_>>(),
        [1, 1]
    );
    assert_eq!(results[0], "[bridge] Turn 1 sent: 1 call(s).");
    assert!(
        !results[1].contains("[bridge] No bridge_end_turn"),
        "{}",
        results[1]
    );
    assert!(results[2].contains("Turn 2"), "{}", results[2]);
    assert!(
        results[2].contains("calls count was too low"),
        "{}",
        results[2]
    );
}

/// A count never reached is sent with what arrived, after the long wait,
/// saying so
#[tokio::test(start_paused = true)]
async fn an_under_count_times_out_and_says_so() {
    let (inference, ending, driver) = bridge();
    let server = Server::new(driver.events);
    let turns = one_turn(inference, ending);
    let started = Instant::now();
    let results = arrive(
        &server,
        vec![
            (0, Joining::end(3)),
            (1_000, call("get_feed")),
            (2_000, call("get_inbox")),
        ],
    )
    .await;
    assert_eq!(tool_uses(&turns.await.unwrap()), 2);
    assert_eq!(started.elapsed(), Duration::from_millis(2_000) + END_WAIT);
    for result in &results {
        assert!(
            result.contains("calls count was 3, but only 2 agent call(s) arrived within 120 s"),
            "{result}"
        );
    }
}

/// Text with the calls: the reply carries the count, in whatever place it
/// arrives; a reply without one does not end a turn holding calls
#[tokio::test(start_paused = true)]
async fn a_reply_with_calls_ends_the_turn_at_its_count() {
    let (inference, ending, driver) = bridge();
    let server = Server::new(driver.events);
    let turns = reactor(inference, ending, 2);
    let started = Instant::now();
    let first = arrive(
        &server,
        vec![
            (0, Joining::reply("Reading.", Some(2))),
            (3_000, call("get_feed")),
            (5_200, call("get_inbox")),
        ],
    )
    .await;
    assert_eq!(started.elapsed(), Duration::from_millis(5_200));
    assert!(
        first[0].contains("sent as text with 2 tool call(s)"),
        "{}",
        first[0]
    );
    // No count: the reply waits with the call for the quiet window
    let second = arrive(
        &server,
        vec![(0, call("get_feed")), (100, Joining::reply("More.", None))],
    )
    .await;
    let turns = turns.await.unwrap();
    assert_eq!(turns[0][0]["type"], "text");
    assert_eq!(tool_uses(&turns[0]), 2);
    assert_eq!(turns[1].len(), 2, "{:?}", turns[1]);
    assert!(
        second[0].contains("No bridge_end_turn arrived"),
        "{}",
        second[0]
    );
}

/// A marker with nothing to send says so without an error: at once when it
/// declares no calls, after the long wait when it declares calls that never
/// come (a second marker for a message already sent)
#[tokio::test(start_paused = true)]
async fn a_marker_with_nothing_to_send_is_not_an_error() {
    let (_inference, _ending, driver) = bridge();
    let server = Server::new(driver.events);
    let lone = server.join(Joining::end(0)).await;
    assert!(lone.is_error != Some(true), "{}", text(&lone));
    assert!(text(&lone).contains("Nothing to send"), "{}", text(&lone));

    let started = Instant::now();
    let stray = server.join(Joining::end(2)).await;
    assert_eq!(started.elapsed(), END_WAIT);
    assert!(stray.is_error != Some(true), "{}", text(&stray));
    assert!(text(&stray).contains("none arrived"), "{}", text(&stray));
}

#[test]
fn the_end_marker_requires_its_count() {
    assert!(serde_json::from_value::<EndInput>(json!({})).is_err());
    assert!(serde_json::from_value::<EndInput>(json!({"calls": -1})).is_err());
    assert_eq!(
        serde_json::from_value::<EndInput>(json!({"calls": 3}))
            .unwrap()
            .calls,
        3
    );
}

/// The bridge's own tool descriptions are one sentence each
#[test]
fn bridge_tool_descriptions_are_one_sentence() {
    for tool in bridge_tools() {
        let description = tool.description.as_deref().unwrap_or_default();
        let sentences = description.trim_end_matches('.').matches(". ").count();
        assert_eq!(sentences, 0, "{}: {description}", tool.name);
    }
}

proptest::proptest! {
    /// One message is one turn, sent the moment its last call arrives,
    /// whatever the order of its calls and its marker and however far
    /// apart they arrive (each gap under the quiet window)
    #[test]
    fn one_message_is_one_turn(
        k in 1usize..=6,
        marker_at in 0usize..=6,
        gaps in proptest::collection::vec(0u64..25_000, 7),
        with_text in proptest::bool::ANY,
    ) {
        let marker_at = marker_at.min(k);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .unwrap();
        runtime.block_on(async move {
            let (inference, ending, driver) = bridge();
            let server = Server::new(driver.events);
            let turns = one_turn(inference, ending);
            let mut joinings: Vec<Joining> = (0..k).map(|i| call(["get_feed", "get_inbox"][i % 2])).collect();
            let marker = match with_text {
                true => Joining::reply("Here goes.", Some(k)),
                false => Joining::end(k),
            };
            joinings.insert(marker_at, marker);
            let mut at = 0;
            let arrivals: Vec<(u64, Joining)> = joinings
                .into_iter()
                .zip(&gaps)
                .map(|(j, gap)| {
                    at += gap;
                    (at, j)
                })
                .collect();
            let last = at;
            let started = Instant::now();
            let results = arrive(&server, arrivals).await;
            let blocks = turns.await.unwrap();
            assert_eq!(tool_uses(&blocks), k, "{blocks:?}");
            assert_eq!(blocks.len(), k + usize::from(with_text));
            assert_eq!(started.elapsed(), Duration::from_millis(last), "sent at once");
            for (i, result) in results.iter().enumerate() {
                assert!(!result.contains("calls count"), "{result}");
                assert!(!result.contains("No bridge_end_turn"), "{result}");
                match (i == marker_at, with_text) {
                    (true, false) => assert_eq!(result, &format!("[bridge] Turn 1 sent: {k} call(s).")),
                    (true, true) => assert!(result.contains(&format!("sent as text with {k} tool call(s)")), "{result}"),
                    (false, _) => assert!(result.contains("Turn 1: get_") && result.contains(&format!("one of {k} call(s)")), "{result}"),
                }
            }
        });
    }
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
    let (feed, inbox, end) = tokio::join!(
        server.join(call("get_feed")),
        server.join(call("get_inbox")),
        async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            server.join(Joining::end(2)).await
        },
    );
    let (feed, inbox) = (text(&feed), text(&inbox));
    assert_eq!(text(&end), "[bridge] Turn 1 sent: 2 call(s).");
    assert!(!feed.contains("with no new call"), "{feed}");
    assert!(feed.contains("result of get_feed") && inbox.contains("result of get_inbox"));
    // What else came is with the last call only
    assert_eq!(
        [&feed, &inbox]
            .iter()
            .filter(|t| t.contains("You have a new message."))
            .count(),
        1
    );

    let reply = text(&server.join(Joining::reply("That is all.", None)).await);
    assert!(reply.contains("Rewrite your memory."), "{reply}");
    assert!(reply.contains("response format"), "{reply}");

    let last = text(
        &server
            .join(Joining::reply("{\"memory\": \"kept\"}", None))
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
    // A line break in a page's second half is where it ends
    let lines = format!("{}\n{}", "a".repeat(RESULT_PAGE - 100), "b".repeat(200));
    let page = server.paged("toolu_lines", lines);
    assert!(
        page.starts_with(&"a".repeat(RESULT_PAGE - 100)),
        "{}",
        &page[..50]
    );
    assert!(
        page.contains(&format!("offset {}", RESULT_PAGE - 99)),
        "{}",
        &page[page.len() - 200..]
    );
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
