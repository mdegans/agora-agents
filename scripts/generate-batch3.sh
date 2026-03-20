#!/usr/bin/env bash
# Generate 333 new agents (batch 3) to reach ~1000 total.
#
# Model distribution:
#   cogito:14b     130  (index 946)   — local 3090
#   gpt-oss:20b    100  (index 1076)  — local 3090
#   gemma3:12b      60  (index 1176)  — local 3090
#   qwen3.5:35b     43  (index 1236)  — remote Mac
#
# All non-trolls (well-behaved 75%, boundary-pushers 25%).

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
    echo "=== Generating $count agents with $model (start-index $start) ==="
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
        --well-behaved-pct 75 \
        --boundary-pusher-pct 25
}

# Start remote Mac generation in the background
echo "=== Starting remote generation (qwen3.5:35b x43) in background ==="
generate "qwen3.5:35b" 43 "$REMOTE_OLLAMA" "$REMOTE_CONC" 1236 &
REMOTE_PID=$!

# Local models (3090) — sequential since they share the GPU
generate "cogito:14b"   130 "$LOCAL_OLLAMA" "$LOCAL_CONC" 946
generate "gpt-oss:20b"  100 "$LOCAL_OLLAMA" "$LOCAL_CONC" 1076
generate "gemma3:12b"    60 "$LOCAL_OLLAMA" "$LOCAL_CONC" 1176

echo ""
echo "=== Local generation complete. Waiting for remote... ==="
wait $REMOTE_PID
REMOTE_EXIT=$?

if [ $REMOTE_EXIT -ne 0 ]; then
    echo "WARNING: Remote generation exited with code $REMOTE_EXIT"
else
    echo "=== Remote generation complete ==="
fi

echo ""
echo "=== All generation complete ==="
echo "Expected: 333 new agents"
echo "New total: ~1205 generated + existing = ready for registration"
