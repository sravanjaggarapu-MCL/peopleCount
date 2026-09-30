"""
rtsp_view.py
============

Opens a live RTSP camera stream and renders it in a resizable desktop window.

Usage
-----
    python rtsp_view.py                  # uses the camera configured below
    python rtsp_view.py "rtsp://..."     # overrides the URL from the command line

Press 'q' or the Esc key inside the video window (or simply close the window)
to stop the stream and exit cleanly.

Requirements
------------
    pip install opencv-python

OpenCV ships with FFmpeg built in, which is what actually speaks the RTSP
protocol under the hood. No extra library is needed.
"""

import os
import sys
from urllib.parse import quote

# ---------------------------------------------------------------------------
# FFmpeg transport setting  (MUST be set BEFORE `import cv2`)
# ---------------------------------------------------------------------------
# RTSP can move its video packets over either UDP or TCP:
#
#   UDP - the default. Faster in theory, but packets that get lost are gone
#         forever, which shows up as grey blocks, tearing, or "stuttering"
#         frames. Many corporate networks and firewalls also block the random
#         high-numbered UDP ports that RTSP negotiates.
#
#   TCP - every packet is acknowledged and retransmitted if lost, so the
#         picture stays clean. It reuses the same port 554 connection, so
#         firewalls are happy. Slightly higher latency, but far more reliable.
#
# OpenCV exposes FFmpeg's options only through this environment variable.
# The syntax is "key;value", with multiple options separated by "|".
# `setdefault` means: only set it if the user has not already exported their
# own value in the shell, so an advanced user can still override us.
#
# IMPORTANT: FFmpeg reads this variable when the cv2 module is loaded, which is
# why this line sits above `import cv2` instead of next to the VideoCapture call.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402  (deliberately imported after the env var above)

# ---------------------------------------------------------------------------
# Camera configuration - edit these to point at a different camera
# ---------------------------------------------------------------------------
USERNAME = "admin"
PASSWORD = "Admin@0192"
HOST = "172.20.100.138"   # camera's IP address on the local network
PORT = 554                # 554 is the standard RTSP port

# The stream path is vendor-specific. This one is the Dahua/Amcrest format:
#   channel=1  -> physical camera input #1 (NVRs have several; a standalone
#                 IP camera almost always uses 1)
#   subtype=0  -> MAIN stream: full resolution (e.g. 1920x1080), high bitrate
#   subtype=1  -> SUB stream:  low resolution (640x480 on this camera), light
#                 on CPU and bandwidth - ideal for previews and for feeding a
#                 person-detection model.
PATH = "/cam/realmonitor?channel=1&subtype=1"

# ---------------------------------------------------------------------------
# Building the RTSP URL
# ---------------------------------------------------------------------------
# The final URL has the shape:
#     rtsp://<user>:<password>@<host>:<port>/<path>
#
# Notice that "@" is the separator between the credentials and the host. Our
# password is "Admin@0192" - it CONTAINS an "@". Pasted in raw, the URL would
# read "rtsp://admin:Admin@0192@172.20.100.138:554/..." and the parser would
# split at the FIRST "@", concluding the host is "0192@172..." - the connection
# then fails with a confusing DNS/connection error.
#
# `quote(..., safe="")` percent-encodes every reserved character, turning
# "Admin@0192" into "Admin%400192", so the URL stays unambiguous. The camera
# decodes it back to the original password. `safe=""` is required because by
# default quote() leaves "/" untouched, which would cause the same class of bug
# for passwords containing a slash.
RTSP_URL = (
    f"rtsp://{quote(USERNAME, safe='')}:{quote(PASSWORD, safe='')}"
    f"@{HOST}:{PORT}{PATH}"
)

# Window title shown in the title bar / taskbar.
WINDOW_NAME = f"RTSP - {HOST}"


def main() -> int:
    """Connect, display frames until the user quits, then clean up.

    Returns a process exit code: 0 on success, 1 if the stream never opened.
    """

    # Allow a one-off override, e.g. to peek at the full-resolution main stream
    # without editing the file:
    #   python rtsp_view.py "rtsp://admin:Admin%400192@172.20.100.138:554/cam/realmonitor?channel=1&subtype=0"
    url = sys.argv[1] if len(sys.argv) > 1 else RTSP_URL

    # Print the URL for debugging, but mask the password so it never ends up in
    # a log file, a screen share, or a pasted terminal transcript.
    print(f"Connecting to {url.replace(quote(PASSWORD, safe=''), '****')}")

    # -----------------------------------------------------------------------
    # Open the stream
    # -----------------------------------------------------------------------
    # cv2.CAP_FFMPEG explicitly selects the FFmpeg backend. Without it OpenCV
    # probes its available backends in order, which on Windows can pick MSMF
    # and fail on RTSP URLs. Being explicit removes that ambiguity.
    #
    # This call performs the full RTSP handshake (DESCRIBE -> SETUP -> PLAY)
    # and blocks until it succeeds or times out, so it may take a second or two.
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    # Keep only the newest frame in the internal queue. Without this, OpenCV
    # buffers frames while your loop is busy; each read() then returns an older
    # frame and the displayed video drifts further and further behind real time.
    # (Not every backend honours this hint, but it costs nothing to ask.)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # The three usual culprits, in order of likelihood.
        print("Could not open the stream. Check that:")
        print(f"  1. This machine can reach the camera:  ping {HOST}")
        print("  2. The username/password are correct (case-sensitive).")
        print("  3. The channel/subtype in the path match this camera model.")
        return 1

    # Facts the camera reports once connected. FPS is often 0 or a nonsense
    # value on RTSP streams because the format header may not carry it - that is
    # normal and harmless here, since we display frames as they arrive rather
    # than pacing them ourselves.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Connected. Stream resolution: {width}x{height}, reported FPS: {fps:.1f}")

    # -----------------------------------------------------------------------
    # Create the display window
    # -----------------------------------------------------------------------
    # WINDOW_NORMAL makes the window user-resizable. The default,
    # WINDOW_AUTOSIZE, locks it to the exact frame size and ignores drags.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)  # a comfortable starting size
    print("Streaming. Press 'q' or Esc in the window to quit.")

    # -----------------------------------------------------------------------
    # Main display loop
    # -----------------------------------------------------------------------
    while True:
        # read() grabs the next frame and decodes it.
        #   ok    -> False when the stream ends or the connection drops
        #   frame -> a NumPy array of shape (height, width, 3) in BGR channel
        #            order (not RGB - an OpenCV quirk worth remembering). This
        #            array is what you would hand to a detection model.
        ok, frame = cap.read()

        if not ok:
            # Typical causes: the camera rebooted, the network hiccuped, or
            # another client took the last available stream slot (cameras cap
            # the number of simultaneous viewers, often at 3-5).
            print("Frame read failed - the stream dropped. Exiting.")
            break

        # Hand the frame to the window. Nothing is actually painted on screen
        # until waitKey() below pumps the GUI event loop.
        cv2.imshow(WINDOW_NAME, frame)

        # waitKey(1) does two essential jobs:
        #   1. Gives 1 ms to OpenCV's GUI event loop so the window can repaint
        #      and stay responsive. Skip it and you get a frozen grey rectangle.
        #   2. Returns the key code pressed during that millisecond, or -1.
        # The "& 0xFF" masks off high bits that some platforms set, leaving the
        # plain ASCII code so the comparison below behaves the same everywhere.
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # 'q' or Esc (ASCII 27)
            print("Quit requested.")
            break

        # Detect the user clicking the window's X button. Without this check the
        # window would vanish but the loop would keep running in the background,
        # holding the camera connection open with no way to stop it.
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            break

    # -----------------------------------------------------------------------
    # Cleanup - always release the network/GUI resources
    # -----------------------------------------------------------------------
    # release() tears down the RTSP session so the camera frees the stream slot
    # for the next client; destroyAllWindows() disposes of the GUI window.
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    # sys.exit propagates main()'s return value as the process exit code, so a
    # script or scheduler calling this file can tell success from failure.
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl+C in the terminal: exit quietly instead of dumping a traceback.
        print("\nInterrupted by user.")
        sys.exit(0)
