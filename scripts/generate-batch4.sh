#!/usr/bin/env bash
# Generate 600 new agents (batch 4) — control group for convergence experiment.
#
# ALL agents generated with --no-boundaries (no self-imposed constraints).
# This cohort tests whether personality convergence toward empathy is caused by
# system instructions (Boundaries section), model alignment, or social pressure.
#
# Model distribution:
#   cogito:14b            500  (index 2000)  — local 3090
#   qwen3.5:35b            75  (index 2500)  — remote Mac
#   mistral-small3.2:24b   20  (index 2575)  — local 3090
#   cogito:70b              5  (index 2595)  — remote Mac
#
# Behavior split: 60% well-behaved, 25% boundary-pusher, 15% rule-breaker (default).

set -euo pipefail

LOCAL_OLLAMA="http://localhost:11434"
REMOTE_OLLAMA="http://192.168.0.123:11434"
OUTPUT="souls/generated"
EXAMPLES="souls/examples"

LOCAL_CONC=1
REMOTE_CONC=1

generate() {
    local model="$1"
    local count="$2"
    local url="$3"
    local conc="$4"
    local start="$5"

    echo ""
    echo "=== Generating $count agents with $model (start-index $start, no-boundaries) ==="
    cargo run --release --bin agora-generate -- \
        --count "$count" \
        --start-index "$start" \
        --backend ollama \
        --model "$model" \
        --ollama-url "$url" \
        --output "$OUTPUT" \
        --examples "$EXAMPLES" \
        --concurrency "$conc" \
        --temperature 0.9 \
        --no-boundaries
}

# Start remote Mac generation in the background
echo "=== Starting remote generation (qwen3.5:35b x75 + cogito:70b x5) in background ==="
generate "qwen3.5:35b" 75 "$REMOTE_OLLAMA" "$REMOTE_CONC" 2500 &
QWEN_PID=$!

# cogito:70b is slow (~13 min/agent) so only 5
generate "cogito:70b" 5 "$REMOTE_OLLAMA" "$REMOTE_CONC" 2595 &
COGITO70_PID=$!

# Local models (3090) — sequential since they share the GPU
generate "cogito:14b"            500 "$LOCAL_OLLAMA" "$LOCAL_CONC" 2000
generate "mistral-small3.2:24b"   20 "$LOCAL_OLLAMA" "$LOCAL_CONC" 2575

echo ""
echo "=== Local generation complete. Waiting for remote... ==="
wait $QWEN_PID
QWEN_EXIT=$?
wait $COGITO70_PID
COGITO70_EXIT=$?

if [ $QWEN_EXIT -ne 0 ]; then
    echo "WARNING: qwen3.5:35b generation exited with code $QWEN_EXIT"
fi
if [ $COGITO70_EXIT -ne 0 ]; then
    echo "WARNING: cogito:70b generation exited with code $COGITO70_EXIT"
fi

echo ""
echo "=== All generation complete ==="
echo "Expected: 600 new agents (no-boundaries control group)"
echo "New total: ~1696 generated"
