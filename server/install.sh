#!/usr/bin/env bash
#
# install.sh - set up the person counter on a fresh Linux server.
#
# Run it once, from anywhere:
#
#     bash server/install.sh
#
# It is safe to run again: everything it does is checked first, so a second
# run upgrades what needs upgrading and leaves the rest alone.
#
# What it does, in order:
#   1. installs the system packages Python and OpenCV need (needs sudo)
#   2. creates a virtual environment at .venv in the project folder
#   3. installs the Python packages into it
#   4. fetches yolov8n.pt if it is missing
#   5. exports the OpenVINO model if it is missing
#   6. creates server/.env from the example if you have not made one
#
# A "virtual environment" is just a private folder of Python packages. It
# exists so that installing PyTorch here cannot break something else on the
# server that needs a different version, which is a genuinely common way to
# ruin a machine's afternoon.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "Project folder: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
# python3-venv   creating the virtual environment (Debian splits it out)
# python3-pip    installing Python packages
# libgl1 and libglib2.0-0
#                OpenCV links against these even in a program that never opens
#                a window. Without them `import cv2` fails with
#                "libGL.so.1: cannot open shared object file", which says
#                nothing about the real cause.
# ffmpeg         not strictly required (opencv-python bundles its own), but
#                `ffprobe rtsp://...` is the quickest way to prove a camera is
#                reachable when something is wrong.
install_system_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    say "Installing system packages with apt (you may be asked for your password)"
    sudo apt-get update
    # libgl1-mesa-glx on Debian 11 / Ubuntu 20.04 and older; libgl1 after
    # that. Asking for both in one command fails if either is missing, so
    # they are tried in turn.
    sudo apt-get install -y python3 python3-venv python3-pip libglib2.0-0 ffmpeg
    sudo apt-get install -y libgl1 || sudo apt-get install -y libgl1-mesa-glx \
      || warn "Could not install libgl1. If 'import cv2' later complains about libGL, install it by hand."
  elif command -v dnf >/dev/null 2>&1; then
    say "Installing system packages with dnf"
    sudo dnf install -y python3 python3-pip mesa-libGL glib2 ffmpeg-free || \
      sudo dnf install -y python3 python3-pip mesa-libGL glib2
  elif command -v yum >/dev/null 2>&1; then
    say "Installing system packages with yum"
    sudo yum install -y python3 python3-pip mesa-libGL glib2
  else
    warn "Unknown package manager - skipping system packages."
    warn "Make sure python3, python3-venv, pip, libGL and glib are installed."
  fi
}

if [ "${SKIP_SYSTEM_PACKAGES:-0}" = "1" ]; then
  warn "SKIP_SYSTEM_PACKAGES=1 - not touching apt/dnf."
else
  install_system_packages
fi

command -v python3 >/dev/null 2>&1 || die "python3 is not installed."
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
say "Python $PY_VERSION"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "Python 3.10 or newer is required (the project uses the 'int | None' syntax). Found $PY_VERSION."

# ---------------------------------------------------------------------------
# 2 and 3. Virtual environment and Python packages
# ---------------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  say "Creating the virtual environment at $VENV"
  python3 -m venv "$VENV"
else
  say "Virtual environment already exists at $VENV"
fi

say "Installing Python packages (this pulls in PyTorch - several hundred MB, give it a few minutes)"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

say "Checking that OpenCV loads"
"$VENV/bin/python" - <<'PY'
import cv2
print(f"    OpenCV {cv2.__version__} imports cleanly.")
PY

# ---------------------------------------------------------------------------
# 4. The weights
# ---------------------------------------------------------------------------
# yolov8n.pt is in .gitignore and is not copied by deploy.ps1 if you did not
# have it either, so fetch it here. ultralytics downloads it on first use
# anyway; doing it now means the failure, if the server has no route out to
# the internet, happens during setup rather than at 3am.
if [ ! -f "$PROJECT_ROOT/yolov8n.pt" ]; then
  say "Downloading yolov8n.pt (about 6 MB)"
  "$VENV/bin/python" - <<'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt")       # downloads into the working directory
print("    Downloaded.")
PY
else
  say "yolov8n.pt is already here"
fi

# ---------------------------------------------------------------------------
# 5. The OpenVINO model
# ---------------------------------------------------------------------------
# The .xml/.bin pair is compiled for one fixed input size, so it has to be
# built on - or at least for - the size this server will run at. Copying the
# laptop's export up is fine (it is portable across x86 machines); building it
# here is simply the one step that cannot be got wrong.
MODEL_XML="$PROJECT_ROOT/yolov8n_openvino_model/yolov8n.xml"
if [ ! -f "$MODEL_XML" ]; then
  say "Building the OpenVINO model (one minute or so)"
  "$VENV/bin/python" export_openvino.py || warn \
    "The export failed. Set MODEL_BACKEND=pytorch in server/.env to run the .pt weights instead - slower, but it will work."
else
  say "OpenVINO model already built"
fi

# ---------------------------------------------------------------------------
# 6. The settings file
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  say "Creating server/.env from the example"
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  chmod 600 "$SCRIPT_DIR/.env"      # it will hold a camera password
  warn "EDIT server/.env and put your camera's address in CAMERA_URL."
else
  say "server/.env already exists - leaving it alone"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
PORT="$(grep -E '^PORT=' "$SCRIPT_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
PORT="${PORT:-8000}"

cat <<EOF

==================================================================
 Setup finished.
==================================================================

 1. Check your camera address:
        nano $SCRIPT_DIR/.env

 2. Start it in the foreground to see what happens:
        bash $SCRIPT_DIR/run.sh

 3. On your laptop, open:
        http://${IP:-<server-ip>}:${PORT}

 4. Once it works, install it as a service so it starts on boot:
        bash $SCRIPT_DIR/install_service.sh

 The README in $SCRIPT_DIR explains all of this, and what to do when
 a step does not go to plan.
==================================================================
EOF
