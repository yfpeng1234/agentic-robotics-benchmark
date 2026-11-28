#!/usr/bin/env bash
# Oracle: the privileged reference busser, run in place of the agent by `ale validate` as the same unprivileged
# account. It installs busser.py (camera frames, then one privileged call) and the 256-bit oracle token as
# .oracle_token; the grader compares the token's sha256 with tests/hidden/oracle_token.sha256 and, on a match, gives
# this submission's cells (the same fixed hidden shifts the agents get) the privileged call `reference_shift` (the
# reference busser inside the cell: true object poses and process states, the G1 driven through the cell's own
# controller, grip rule, safety watch, cart, cleaning processes and shift).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/busser.py" "$SUB/"
cp "$A/oracle_token" "$SUB/.oracle_token"
echo "[oracle] submission ready at $SUB" >&2
