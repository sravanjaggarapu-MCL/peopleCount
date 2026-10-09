#!/usr/bin/env bash
#
# install_service.sh - keep the counter running in the background, for good.
#
#     bash server/install_service.sh
#
# Starting it with run.sh ties it to your SSH session: close the terminal and
# it stops. A systemd service instead belongs to the machine - it starts when
# the server boots, restarts itself if it crashes, and keeps a log you can
# read afterwards.
#
# This script writes the unit file with the real paths and username filled in
# (person-counter.service in this folder is the template, there to read), then
# enables and starts it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
SERVICE_NAME="person-counter"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
RUN_AS="${SUDO_USER:-$USER}"

[ -x "$PYTHON" ] || { echo "No .venv - run 'bash server/install.sh' first." >&2; exit 1; }
command -v systemctl >/dev/null 2>&1 || {
  echo "This system does not use systemd. Use Docker (see the README) or run it under tmux/screen." >&2
  exit 1
}

echo "Installing $UNIT_PATH"
echo "  running as:  $RUN_AS"
echo "  working dir: $PROJECT_ROOT"

sudo tee "$UNIT_PATH" >/dev/null <<EOF
[Unit]
Description=Person counter - RTSP person counting with a web view
# Do not start before the network is actually usable: the first thing the
# program does is open an RTSP connection, and a service that starts half a
# second too early just logs a failure and waits out the retry delay.
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_AS
WorkingDirectory=$PROJECT_ROOT
ExecStart=$PYTHON $PROJECT_ROOT/server/stream_server.py

# Restart=always with a 10 second gap: whatever went wrong - the camera
# rebooted, the machine ran out of memory - the right answer is to try again.
# RestartSec stops a program that fails instantly from spinning the CPU.
Restart=always
RestartSec=10

# Python buffers its output when it is not writing to a terminal, so without
# this the log would arrive in silent 8 KB lumps, often minutes late.
Environment=PYTHONUNBUFFERED=1

# Give the camera time to let go of the RTSP session on the way down. Cameras
# allow only a handful of simultaneous connections, and one left hanging makes
# the next start fail for a reason that looks unrelated.
KillSignal=SIGTERM
TimeoutStopSec=20

# Everything it writes goes to the system journal: journalctl -u $SERVICE_NAME
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

sleep 2
sudo systemctl --no-pager status "$SERVICE_NAME" || true

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
PORT="$(grep -E '^PORT=' "$SCRIPT_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2 || true)"

cat <<EOF

==================================================================
 The service is installed and running.
==================================================================
 Watch it:     http://${IP:-<server-ip>}:${PORT:-8000}
 Live log:     sudo journalctl -u $SERVICE_NAME -f
 Stop:         sudo systemctl stop $SERVICE_NAME
 Start:        sudo systemctl start $SERVICE_NAME
 After a code change or an edit to .env:
               sudo systemctl restart $SERVICE_NAME
 Stop it starting at boot:
               sudo systemctl disable $SERVICE_NAME
==================================================================
EOF
