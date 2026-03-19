#!/usr/bin/env bash
# Regenerate non-troll agents across new model lineup.
# Run prep-regeneration.sh first to clear SOUL.md files and preserve trolls.
#
# Local models (746 non-troll agents on 3090, weighted toward larger):
#   mistral-small3.2:24b  130
#   lfm2:24b              120
#   gpt-oss:20b           116
#   cogito:14b            100
#   gemma3:12b            100
#   qwen3.5:9b            100
#   gemma3n:e4b            80
#
# Remote model (200 NEW agents on Mac, runs in parallel):
#   qwen3.5:35b           200  (indices 800-999)

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
    shift 4

    echo ""
    echo "=== Generating $count agents with $model ==="
    cargo run --release --bin agora-generate -- \
        --count "$count" \
        --backend ollama \
        --model "$model" \
        --ollama-url "$url" \
        --output "$OUTPUT" \
        --examples "$EXAMPLES" \
        --concurrency "$conc" \
        --temperature 0.9 \
        "$@"
}

# Start remote Mac generation in the background
echo "=== Starting remote generation (qwen3.5:35b x200) in background ==="
generate "qwen3.5:35b" 200 "$REMOTE_OLLAMA" "$REMOTE_CONC" --start-index 800 &
REMOTE_PID=$!

# Local models (3090) — sequential since they share the GPU
generate "mistral-small3.2:24b" 130 "$LOCAL_OLLAMA" "$LOCAL_CONC"
generate "lfm2:24b"             120 "$LOCAL_OLLAMA" "$LOCAL_CONC"
generate "gpt-oss:20b"          116 "$LOCAL_OLLAMA" "$LOCAL_CONC"
generate "cogito:14b"           100 "$LOCAL_OLLAMA" "$LOCAL_CONC"
generate "gemma3:12b"           100 "$LOCAL_OLLAMA" "$LOCAL_CONC"
generate "qwen3.5:9b"           100 "$LOCAL_OLLAMA" "$LOCAL_CONC"
generate "gemma3n:e4b"           80 "$LOCAL_OLLAMA" "$LOCAL_CONC"

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
echo "Expected: ~1000 agents total"
