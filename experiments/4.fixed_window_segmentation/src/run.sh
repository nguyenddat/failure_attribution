#!/usr/bin/env bash
# Launcher cho pm2 (interpreter: none). Xem ../ecosystem.config.cjs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "$(conda info --base)/envs/rs_segment/bin/python" -u run.py "$@"
