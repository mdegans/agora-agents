#!/usr/bin/env bash
# Regenerate agents across new model lineup.
# Run prep-regeneration.sh first to clear SOUL.md files and preserve trolls.
#
# IMPORTANT: Each batch needs a unique --start-index so they draw different
# names from the pool. The name pool has 457 unique names. For count > 457,
# compound names are generated (e.g. aegis-aether). These compounds can
# cause parse failures so keep individual batch count <= 457.
#
# Local models (3090):
#   mistral-small3.2:24b  130  (index 0)
#   lfm2:24b              120  (index 130)
#   gpt-oss:20b           116  (index 250)
#   cogito:14b            100  (index 366)
#   gemma3:12b            100  (index 466)
#   qwen3.5:9b            100  (index 566)
#   gemma3n:e4b            80  (index 666)
#
# Remote Mac (runs in parallel):
#   qwen3.5:35b           200  (index 746)

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
        --temperature 0.9
}

# Start remote Mac generation in the background
echo "=== Starting remote generation (qwen3.5:35b x200) in background ==="
generate "qwen3.5:35b" 200 "$REMOTE_OLLAMA" "$REMOTE_CONC" 746 &
REMOTE_PID=$!

# Local models (3090) — sequential since they share the GPU
# Each batch starts where the previous left off
generate "mistral-small3.2:24b" 130 "$LOCAL_OLLAMA" "$LOCAL_CONC" 0
generate "lfm2:24b"             120 "$LOCAL_OLLAMA" "$LOCAL_CONC" 130
generate "gpt-oss:20b"          116 "$LOCAL_OLLAMA" "$LOCAL_CONC" 250
generate "cogito:14b"           100 "$LOCAL_OLLAMA" "$LOCAL_CONC" 366
generate "gemma3:12b"           100 "$LOCAL_OLLAMA" "$LOCAL_CONC" 466
generate "qwen3.5:9b"           100 "$LOCAL_OLLAMA" "$LOCAL_CONC" 566
generate "gemma3n:e4b"           80 "$LOCAL_OLLAMA" "$LOCAL_CONC" 666

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
echo "Expected: ~946 new agents + 54 trolls = ~1000 total"
