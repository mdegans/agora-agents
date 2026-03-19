#!/usr/bin/env bash
# Prepare for agent regeneration:
# 1. Identify and preserve troll agents
# 2. Reassign troll models to qwen3.5:35b or gpt-oss:20b
# 3. Delete non-troll SOUL.md and MEMORY.md (preserve signing keys and agent IDs)

set -euo pipefail

SOULS_DIR="${1:-souls/generated}"
TROLL_FILE=$(mktemp)

echo "=== Identifying troll agents ==="
grep -rli 'troll\|lulz\|shitpost' "$SOULS_DIR"/*/SOUL.md \
    | sed "s|$SOULS_DIR/||;s|/SOUL.md||" \
    | sort > "$TROLL_FILE"

TROLL_COUNT=$(wc -l < "$TROLL_FILE")
TOTAL_COUNT=$(ls -d "$SOULS_DIR"/*/ | wc -l)
NON_TROLL_COUNT=$((TOTAL_COUNT - TROLL_COUNT))

echo "  Total agents: $TOTAL_COUNT"
echo "  Trolls: $TROLL_COUNT (preserved)"
echo "  Non-trolls: $NON_TROLL_COUNT (will be regenerated)"
echo ""

echo "=== Reassigning troll models ==="
# Split trolls 50/50 between qwen3.5:35b and gpt-oss:20b
i=0
while IFS= read -r name; do
    if (( i % 2 == 0 )); then
        echo "qwen3.5:35b" > "$SOULS_DIR/$name/model.txt"
    else
        echo "gpt-oss:20b" > "$SOULS_DIR/$name/model.txt"
    fi
    i=$((i + 1))
done < "$TROLL_FILE"
echo "  Assigned $(( (TROLL_COUNT + 1) / 2 )) trolls to qwen3.5:35b"
echo "  Assigned $(( TROLL_COUNT / 2 )) trolls to gpt-oss:20b"
echo ""

echo "=== Clearing non-troll SOUL.md files ==="
CLEARED=0
for dir in "$SOULS_DIR"/*/; do
    name=$(basename "$dir")
    if ! grep -qx "$name" "$TROLL_FILE"; then
        rm -f "$dir/SOUL.md" "$dir/MEMORY.md"
        CLEARED=$((CLEARED + 1))
    fi
done
echo "  Cleared $CLEARED agent SOUL.md files"
echo "  (signing_key.hex and agent_id.txt preserved)"
echo ""

echo "=== Ready for regeneration ==="
echo "Run agora-generate with --skip-existing for each model."
echo "Troll agents will be skipped (SOUL.md still exists)."

rm -f "$TROLL_FILE"
