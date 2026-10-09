# Dockerfile - the alternative to install.sh.
#
# Build it from the PROJECT ROOT, not from this folder, because the image
# needs person_counter/ and config.py as well as this directory:
#
#     docker build -f server/Dockerfile -t person-counter .
#     docker run -d --name person-counter --network host \
#         --env-file server/.env \
#         -v "$PWD/lines:/app/lines" \
#         --restart unless-stopped person-counter
#
# Or, much shorter, from this folder:  docker compose up -d
#
# Why you might prefer this to install.sh: the image carries its own Python
# and its own copy of every library, so nothing on the server is touched and
# "it works on my machine" becomes "it works on every machine". Why you might
# not: it is about 2 GB, and the first build takes a while.

FROM python:3.11-slim

# libgl1 and libglib2.0-0: OpenCV links against them even headless. ffmpeg:
# not required (opencv bundles its own) but makes it possible to test a camera
# from inside the container with ffprobe, which is worth the few MB when
# something is wrong.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first, and on their own layer: Docker reuses a layer whose
# inputs have not changed, so editing the Python code below rebuilds in
# seconds instead of reinstalling PyTorch every time.
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Then the project itself.
COPY config.py export_openvino.py /app/
COPY person_counter/ /app/person_counter/
COPY server/ /app/server/

# The weights and the exported model, if they are present in the build
# context. They are in .gitignore, so a clone may not have them - hence the
# wildcard, which matches nothing without failing the build. If the IR is
# missing, set MODEL_BACKEND=pytorch and the .pt is used instead (ultralytics
# downloads it on first use).
COPY yolov8n.pt* /app/
COPY yolov8n_openvino_model*/ /app/yolov8n_openvino_model/

# Unbuffered, or the log arrives in silent lumps minutes after the event.
ENV PYTHONUNBUFFERED=1
ENV BIND_HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# /healthz answers 200 while the engine is alive, so Docker can restart a
# container that has wedged rather than one that has merely exited.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

CMD ["python", "server/stream_server.py"]
