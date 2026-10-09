#!/usr/bin/env bash
#
# run.sh - start the server in the foreground, printing everything it does.
#
#     bash server/run.sh
#
# This is the version to use while you are setting things up: Ctrl+C stops it,
# and every message - the camera connecting, the model loading, a person
# crossing the line - appears in the terminal where you can read it.
#
# Once it works, install_service.sh runs the same thing in the background,
# started automatically when the server boots.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "No virtual environment at $PROJECT_ROOT/.venv - run 'bash server/install.sh' first." >&2
  exit 1
fi

# cd to the project root as well as letting server_config do it, so that a
# relative path typed into .env (sample.mp4, say) means what you expect.
cd "$PROJECT_ROOT"
exec "$PYTHON" server/stream_server.py "$@"
