#!/usr/bin/env bash
#
# deploy.sh - the same job as deploy.ps1, for a Mac, a Linux laptop, WSL or
# Git Bash on Windows.
#
#     bash server/deploy.sh ubuntu@192.168.1.50
#     bash server/deploy.sh ubuntu@192.168.1.50 /opt/person-counter
#
# Environment variables:
#     INCLUDE_MODEL=1    also send yolov8n.pt and yolov8n_openvino_model/
#     INCLUDE_CAMERAS=1  also send cameras.json (it holds passwords)
#     SSH_PORT=2222      if the server's SSH is not on 22
#
# What is deliberately NOT sent: server/.env, lines/, cameras.json, .venv,
# .git, __pycache__. The first three are the server's own state and copying
# over them would undo whatever you set up there.

set -euo pipefail

SERVER="${1:-}"
REMOTE_PATH="${2:-~/person-counter}"
SSH_PORT="${SSH_PORT:-22}"

if [ -z "$SERVER" ]; then
  echo "Usage: bash server/deploy.sh user@host [remote-path]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

[ -f config.py ] || { echo "config.py is missing - the server needs it." >&2; exit 1; }

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "==> Staging files"
for file in config.py export_openvino.py main.py requirements.txt README.md; do
  [ -f "$file" ] && cp "$file" "$STAGING/" && echo "    $file"
done

cp -r person_counter "$STAGING/"
cp -r server "$STAGING/"
rm -f "$STAGING/server/.env"
find "$STAGING" -type d -name __pycache__ -prune -exec rm -rf {} +
echo "    person_counter/  server/"

if [ "${INCLUDE_MODEL:-0}" = "1" ]; then
  for item in yolov8n.pt yolov8n_openvino_model; do
    [ -e "$item" ] && cp -r "$item" "$STAGING/" && echo "    $item"
  done
fi
if [ "${INCLUDE_CAMERAS:-0}" = "1" ] && [ -f cameras.json ]; then
  cp cameras.json "$STAGING/" && echo "    cameras.json"
fi

echo "==> Copying to $SERVER:$REMOTE_PATH"
# Piping tar straight into ssh: one connection, no temporary file at either
# end, and file modes preserved.
tar -czf - -C "$STAGING" . \
  | ssh -p "$SSH_PORT" "$SERVER" \
      "mkdir -p $REMOTE_PATH && tar -xzf - -C $REMOTE_PATH && chmod +x $REMOTE_PATH/server/*.sh 2>/dev/null || true"

HOST_ONLY="${SERVER##*@}"
cat <<EOF

==================================================================
 Files are on the server.
==================================================================

 First time - log in and set it up:
     ssh -p $SSH_PORT $SERVER
     cd $REMOTE_PATH
     bash server/install.sh
     nano server/.env        # put your camera address in
     bash server/run.sh

 Then on this laptop, open:
     http://$HOST_ONLY:8000

 After a code change:
     bash server/deploy.sh $SERVER $REMOTE_PATH
     ssh -p $SSH_PORT $SERVER 'sudo systemctl restart person-counter'
==================================================================
EOF
