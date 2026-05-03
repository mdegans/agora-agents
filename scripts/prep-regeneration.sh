#!/usr/bin/env bash
# Prepare for agent regeneration:
# 1. Identify and preserve troll agents
# 2. Reassign troll models to qwen3.5:35b or gpt-oss:20b
# 3. Delete non-troll SOUL.md and MEMORY.md (preserve signing keys and agent IDs)

set -euo pipefail

SOULS_DIR="${1:-souls/generated}"
TROLL_FILE=$(mktemp)

echo "=== Identifying troll agents ==="
grep -rli 'troll\|lulz\|shitpost' "$SOULS_DIR"/*/SOUL.json "$SOULS_DIR"/*/SOUL.md 2>/dev/null \
    | sed "s|$SOULS_DIR/||;s|/SOUL.json||;s|/SOUL.md||" \
    | sort -u > "$TROLL_FILE"

TROLL_COUNT=$(wc -l < "$TROLL_FILE")
TOTAL_COUNT=$(ls -d "$SOULS_DIR"/*/ | wc -l)
NON_TROLL_COUNT=$((TOTAL_COUNT - TROLL_COUNT))

echo "  Total agents: $TOTAL_COUNT"
echo "  Trolls: $TROLL_COUNT (preserved)"
echo "  Non-trolls: $NON_TROLL_COUNT (will be regenerated)"
echo ""

echo "=== Troll model assignments ==="
echo "  Models are managed in the database (agents.model_info column)."
echo "  Update troll models via SQL if needed:"
echo "    UPDATE agents SET model_info = 'qwen3.5:35b' WHERE name = '<agent>';"
echo ""

echo "=== Clearing non-troll SOUL/MEMORY files ==="
CLEARED=0
for dir in "$SOULS_DIR"/*/; do
    name=$(basename "$dir")
    if ! grep -qx "$name" "$TROLL_FILE"; then
        rm -f "$dir/SOUL.json" "$dir/SOUL.md" "$dir/MEMORY.json" "$dir/MEMORY.md"
        CLEARED=$((CLEARED + 1))
    fi
done
echo "  Cleared $CLEARED agent SOUL files"
echo "  (signing_key.hex and agent_id.txt preserved)"
echo ""

echo "=== Ready for regeneration ==="
echo "Run agora-generate with --skip-existing for each model."
echo "Troll agents will be skipped (SOUL.json still exists)."

rm -f "$TROLL_FILE"
