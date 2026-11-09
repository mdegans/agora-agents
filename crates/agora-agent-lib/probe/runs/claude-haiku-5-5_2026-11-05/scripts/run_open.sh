#!/usr/bin/env bash
set -u
arm=$1
W=~/Projects/agents-probe-haiku55; B=$W/target/debug/examples/probe_distribution; K=~/Projects/agora/secrets/anthropic_api_key
L=$W/crates/agora-agent-lib/probe; OUT=$L/runs/claude-haiku-5-5_2026-11-05/open_ended/$arm; mkdir -p $OUT
case $arm in off) T=();; low) T=(--thinking --effort low);; high) T=(--thinking --effort high);; esac
$B --api-key-file $K --model claude-haiku-5-5 --questionnaire $L/questionnaires/v0.json "${T[@]}" --open-ended --out $OUT/direct.json --markdown $OUT/direct.md 2> $OUT/direct.err
$B --api-key-file $K --model claude-haiku-5-5 --scenarios $L/scenarios/v0.json "${T[@]}" --open-ended --out $OUT/indirect.json --markdown $OUT/indirect.md 2> $OUT/indirect.err
echo "[open $arm] done $?"
