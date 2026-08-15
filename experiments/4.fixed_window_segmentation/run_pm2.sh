#!/usr/bin/env bash
# Chạy experiment 4 qua pm2, no-autorestart, tự dừng khi xong.
set -euo pipefail

APP_NAME="exp4-fixed-window"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"

# conda run tránh cần activate trong shell của pm2 daemon
INTERPRETER="conda"
INTERPRETER_ARGS="run --no-capture-output -n rs_segment python -u"

npx pm2 delete "$APP_NAME" 2>/dev/null || true

npx pm2 start run.py \
  --name "$APP_NAME" \
  --cwd "$SRC_DIR" \
  --no-autorestart \
  --interpreter "$INTERPRETER" \
  --interpreter-args "$INTERPRETER_ARGS" \
  -- "$@"

echo "Started $APP_NAME. Theo dõi: npx pm2 logs $APP_NAME"
