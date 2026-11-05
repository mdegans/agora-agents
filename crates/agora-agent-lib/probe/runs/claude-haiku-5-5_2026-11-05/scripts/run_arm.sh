#!/usr/bin/env bash
# run_arm.sh <arm: off|low|high> <reps>
set -u
arm=$1; reps=$2
W=~/Projects/agents-probe-haiku55
B=$W/target/debug/examples/probe_distribution
K=~/Projects/agora/secrets/anthropic_api_key
L=$W/crates/agora-agent-lib/probe
OUT=$L/runs/claude-haiku-5-5_2026-11-05/ratings/$arm
mkdir -p $OUT
case $arm in
  off) T=() ;;
  low) T=(--thinking --effort low) ;;
  high) T=(--thinking --effort high) ;;
esac
for r in $(seq -w 1 $reps); do
  for inst in direct indirect; do
    f=$OUT/${inst}_rep$r.json
    [ -s $f ] && continue
    if [ $inst = direct ]; then src=(--questionnaire $L/questionnaires/v0.json); else src=(--scenarios $L/scenarios/v0.json); fi
    for try in 1 2 3; do
      $B --api-key-file $K --model claude-haiku-5-5 "${src[@]}" "${T[@]}" --out $f > $OUT/${inst}_rep$r.txt 2> $OUT/${inst}_rep$r.err && break
      echo "[$arm] $inst rep$r try $try failed: $(tail -1 $OUT/${inst}_rep$r.err)"; rm -f $f; sleep 20
    done
  done
  echo "[$arm] rep $r done $(date -u +%T)"
done
echo "[$arm] ALL DONE"
