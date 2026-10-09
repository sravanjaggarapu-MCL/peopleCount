"""
server_config.py
================

Settings for the headless server, and the three bits of bootstrapping that
have to happen before anything else is imported.

Importing this module (first, always):

  1. puts the project root on sys.path, so `import config` and
     `import person_counter` work no matter which directory the service was
     started from;
  2. changes the working directory to the project root, because almost every
     path in the project is relative to it - lines/, yolov8n_openvino_model/,
     cameras.json - and systemd would otherwise start us in /;
  3. reads the .env file into the environment, then copies the values that
     belong to the existing modules into `config`.

Why settings come from the environment here, when the desktop program reads
them from config.py
-------------------------------------------------------------------------
config.py is edited by hand and holds one camera's credentials. That is right
for a laptop and wrong for a server: the file is part of the project tree, so
every redeploy would overwrite the server's settings with the laptop's.
Environment variables live outside the code, survive a redeploy, and are what
systemd and Docker already know how to supply. The .env file is simply a
convenient way to set them without typing them into a unit file.

Real environment variables always win over the .env file - that is what
os.environ.setdefault below means - so `PORT=9000 python server/stream_server.py`
works for a one-off without editing anything.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent

# ---------------------------------------------------------------------------
# 1 and 2: import path and working directory
# ---------------------------------------------------------------------------
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 3: the .env file
# ---------------------------------------------------------------------------
def _load_env_file(path: Path) -> bool:
    """Read KEY=VALUE lines into os.environ. Returns whether the file existed.

    Deliberately tiny rather than a dependency: the format is a handful of
    lines of plain text, and python-dotenv would be one more thing to install
    on a machine that may have no route to the internet.
    """
    if not path.is_file():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Tolerate "export KEY=value", since people paste shell snippets in.
        if line.startswith("export "):
            line = line[len("export "):].strip()

        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        # Strip one layer of matching quotes: a password containing a '#' or a
        # space has to be quotable.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        # setdefault, not assignment: anything already exported in the real
        # environment (systemd, docker run -e, a shell prefix) wins.
        os.environ.setdefault(key, value)
    return True


def _env_candidates() -> list[Path]:
    explicit = os.environ.get("PERSON_COUNTER_ENV")
    paths = [Path(explicit)] if explicit else []
    paths += [SERVER_DIR / ".env", PROJECT_ROOT / ".env"]
    return paths


ENV_FILE_LOADED: Path | None = None
for _candidate in _env_candidates():
    if _load_env_file(_candidate):
        ENV_FILE_LOADED = _candidate
        break


# ---------------------------------------------------------------------------
# Readers, so a typo in .env fails loudly instead of silently reverting
# ---------------------------------------------------------------------------
def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int) -> int:
    raw = env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a whole number, got {raw!r}")


def env_float(name: str, default: float) -> float:
    raw = env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number, got {raw!r}")


def env_bool(name: str, default: bool) -> bool:
    raw = env_str(name).lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"{name} must be true or false, got {raw!r}")


@dataclass
class Settings:
    """Everything the server reads from the environment, in one place."""

    # --- what to watch -----------------------------------------------------
    # Empty means "the camera in config.py", so a deployment that already has
    # a correct config.py needs no CAMERA_URL at all.
    camera_url: str = field(default_factory=lambda: env_str("CAMERA_URL"))

    # --- where to listen ---------------------------------------------------
    # 0.0.0.0 means "every network interface on this machine", which is what
    # makes the page reachable from your laptop. 127.0.0.1 would mean "only
    # from the server itself" - the right choice when you reach it through an
    # SSH tunnel instead. See the README.
    host: str = field(default_factory=lambda: env_str("BIND_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: env_int("PORT", 8000))

    # --- who may look ------------------------------------------------------
    # A blank user disables the login box entirely. Fine on a private LAN, not
    # fine on anything reachable from the internet.
    auth_user: str = field(default_factory=lambda: env_str("AUTH_USER"))
    auth_password: str = field(default_factory=lambda: env_str("AUTH_PASSWORD"))

    # --- video sent to the browser ----------------------------------------
    # 1-100. 70 looks fine for watching people walk about, at roughly a third
    # the bytes of 95.
    jpeg_quality: int = field(default_factory=lambda: env_int("JPEG_QUALITY", 70))
    # Shrink frames wider than this before encoding. 0 disables it. The
    # counting is unaffected - this happens after all the drawing, on the way
    # out - so the only thing given up is detail on your laptop's screen.
    #
    # The default is not 0, because the cost of full size is easy to
    # underestimate: a 2560x1440 camera at quality 70 and 12 FPS measured 50
    # Mbit/s, which saturates most links and gains nothing you can see at the
    # size the picture is displayed. 960 wide is around 3 Mbit/s. A camera's
    # sub-stream is usually 640x480 already, and is never upscaled.
    stream_max_width: int = field(default_factory=lambda: env_int("STREAM_MAX_WIDTH", 960))

    # --- model -------------------------------------------------------------
    # openvino (fast on x86 CPUs, needs the exported .xml/.bin) or pytorch
    # (slower, but runs the .pt directly and works anywhere).
    model_backend: str = field(
        default_factory=lambda: env_str("MODEL_BACKEND", "openvino").lower())
    # 0 means "leave config.INFERENCE_SIZE alone".
    inference_size: int = field(default_factory=lambda: env_int("INFERENCE_SIZE", 0))
    track_point: str = field(default_factory=lambda: env_str("TRACK_POINT", "").lower())
    # -1 means "leave config.TORCH_THREADS alone".
    torch_threads: int = field(default_factory=lambda: env_int("TORCH_THREADS", -1))

    # --- reconnection ------------------------------------------------------
    # The desktop program ends when the camera does; a server has to keep
    # trying, because nobody is sitting in front of it to restart it.
    retry_delay: float = field(default_factory=lambda: env_float("RETRY_DELAY_SECONDS", 5.0))

    def validate(self):
        if self.model_backend not in ("openvino", "pytorch"):
            raise SystemExit(
                f"MODEL_BACKEND must be 'openvino' or 'pytorch', "
                f"got {self.model_backend!r}")
        if self.track_point and self.track_point not in ("foot", "head"):
            raise SystemExit(
                f"TRACK_POINT must be 'foot' or 'head', got {self.track_point!r}")
        if not 1 <= self.jpeg_quality <= 100:
            raise SystemExit("JPEG_QUALITY must be between 1 and 100")
        if self.auth_user and not self.auth_password:
            raise SystemExit("AUTH_USER is set but AUTH_PASSWORD is empty")


settings = Settings()
settings.validate()


# ---------------------------------------------------------------------------
# Hand the relevant settings to the existing modules
# ---------------------------------------------------------------------------
# Every module in person_counter reads its settings through `config.X` at call
# time rather than copying them at import, exactly as main.py relies on - so
# assigning here reaches all of them.
try:
    import config
except ModuleNotFoundError as error:                       # pragma: no cover
    raise SystemExit(
        f"Could not import config.py from {PROJECT_ROOT}.\n"
        "config.py is in .gitignore because it holds the camera password, so a\n"
        "`git clone` on the server will not have it. Copy it up from the laptop\n"
        "(deploy.ps1 does this for you).\n"
        f"Original error: {error}"
    ) from error

if settings.inference_size:
    config.INFERENCE_SIZE = settings.inference_size
if settings.track_point:
    config.TRACK_POINT = settings.track_point

if settings.model_backend == "pytorch":
    # The .pt weights run through PyTorch itself, so the OpenVINO IR is not
    # needed - and TORCH_THREADS=1, which exists purely to keep torch out of
    # OpenVINO's way, would now be throttling the thing doing the work.
    config.MODEL_PATH = config.PYTORCH_MODEL_PATH
    config.TORCH_THREADS = settings.torch_threads if settings.torch_threads >= 0 else 0
elif settings.torch_threads >= 0:
    config.TORCH_THREADS = settings.torch_threads


def describe() -> list[str]:
    """The settings worth printing at start-up, with no secrets among them."""
    import config as _config

    source = settings.camera_url or _config.build_rtsp_url()
    login = (f"required (user {settings.auth_user})" if settings.auth_user
             else "NOT required - anyone who can reach the port can watch")
    return [
        f"project root   {PROJECT_ROOT}",
        f"env file       {ENV_FILE_LOADED or 'none found - using defaults'}",
        f"camera         {_config.mask_url(source)}",
        f"listening on   http://{settings.host}:{settings.port}",
        f"model          {settings.model_backend}  {_config.MODEL_PATH}  "
        f"imgsz={_config.INFERENCE_SIZE}",
        f"track point    {_config.TRACK_POINT}",
        f"login          {login}",
    ]
