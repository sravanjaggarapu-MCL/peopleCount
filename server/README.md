# Running the person counter on a Linux server

This folder contains everything needed to move the project off your laptop and
onto a Linux server, where it runs day and night with nobody logged in — and to
watch it, live, from a browser on your laptop.

It is written for somebody who has not deployed anything to a server before.
Every command is given in full, and every step says what it is for and how to
tell whether it worked.

**Read this first if you read nothing else:**

| | |
|---|---|
| What you type on the server | `bash server/install.sh`, then `bash server/run.sh` |
| What you open on your laptop | `http://<server-ip>:8000` in any browser |
| Where the video is processed | On the server. Your laptop only displays it. |
| What you install on your laptop | Nothing. |

---

## Contents

1. [How this works](#1-how-this-works)
2. [What you need before you start](#2-what-you-need-before-you-start)
3. [Deploying, step by step](#3-deploying-step-by-step)
4. [Using the web page](#4-using-the-web-page)
5. [Making it permanent](#5-making-it-permanent)
6. [Day-to-day operation](#6-day-to-day-operation)
7. [When you cannot reach the page](#7-when-you-cannot-reach-the-page)
8. [Security](#8-security)
9. [Making it faster](#9-making-it-faster)
10. [The Docker route](#10-the-docker-route)
11. [Troubleshooting](#11-troubleshooting)
12. [Every file in this folder](#12-every-file-in-this-folder)
13. [Settings reference](#13-settings-reference)
14. [Glossary](#14-glossary)

---

## 1. How this works

### The problem with running it on a server

On your laptop, `python main.py` opens a window and draws the video into it.
A server has no screen, no mouse and nobody sitting in front of it — so a
window is not merely inconvenient there, it is impossible. The program would
crash the moment it tried to open one.

So the server version does the same work and sends the picture somewhere else:

```
   LINUX SERVER (no screen)                    YOUR LAPTOP
   ─────────────────────────                   ───────────

   RTSP camera  ──► reads frames
                    │
                    ▼
                 YOLOv8 finds people
                    │
                    ▼
                 ByteTrack gives each an ID
                    │
                    ▼
                 counts crossings of your line
                    │
                    ▼
                 draws boxes, trails, the line,          ┌────────────────┐
                 the counts  ──► JPEG ──► HTTP ────────► │  web browser   │
                                                         │  Chrome, Edge, │
                 web page + buttons ◄──── your clicks ◄── │  Firefox, any  │
                                                         └────────────────┘
```

The video reaches your browser as **MJPEG** — one JPEG picture after another
down a connection that never closes. The browser treats it as an image that
keeps changing, which is why nothing has to be installed on your laptop: an
`<img>` tag is all it takes. No plugin, no video player, no VPN client.

### What stays exactly the same

The counting is not reimplemented here. `stream_server.py` imports
`PersonDetector`, `CountPipeline`, `PersonCounter` and `CountingLine` from the
same `person_counter/` package `main.py` uses, and runs them unchanged. The
server and your laptop count identically, and the counting line you draw in the
browser is saved to the same `lines/<camera>.json` file the desktop program
reads.

### What is replaced

| On the laptop | On the server |
|---|---|
| The OpenCV window | The live view in your browser |
| Keys: `q` quit, `t` trails, `r` reset | Buttons on the page |
| Drag the counting line on the video | Drag it on a still frame in the browser |
| The "which camera?" question at start-up | `CAMERA_URL` in `server/.env`, or the camera list on the page |
| The "feet or head?" question | A toggle on the page |
| Stops when you close the terminal | Runs as a service, restarts itself, starts at boot |

---

## 2. What you need before you start

### The server

A Linux machine — a physical box, a virtual machine, a cloud instance, it does
not matter — with:

- **Ubuntu 20.04 / Debian 11 or newer** (or RHEL/Rocky/Fedora; the install
  script handles `apt`, `dnf` and `yum`)
- **Python 3.10 or newer.** The project uses the `int | None` type syntax,
  which does not exist before 3.10. Check with `python3 --version`.
- **An x86-64 CPU** with 2 cores or more. 4 is comfortable. No graphics card
  is needed — the whole project is built around running on a CPU.
- **About 4 GB of free disk space.** PyTorch is large.
- **A network route to the camera.** This is the one people forget: the
  *server* must be able to reach the camera, and whether your laptop can reach
  it tells you nothing about whether the server can.
- **SSH access** — a username, and either a password or a key.

### Your laptop

- Windows 10 or 11 (these instructions), with PowerShell. `ssh` and `scp` are
  built in — nothing to install.
- A browser.

### Things to write down before you start

| | Example | Yours |
|---|---|---|
| Server IP address | `192.168.1.50` | |
| SSH username | `ubuntu` | |
| Camera RTSP address | `rtsp://admin:Admin%400192@172.20.100.138:554/cam/realmonitor?channel=1&subtype=1` | |

> **The camera address.** It is the same one you type at the project's
> "WHICH CAMERA?" prompt, and `cameras.json` in the project folder already has
> every camera you have used. Special characters in the password must be
> percent-encoded: `@` → `%40`, `#` → `%23`, `:` → `%3A`, `/` → `%2F`,
> `!` → `%21`. So the password `Admin@0192` is written `Admin%400192`.

---

## 3. Deploying, step by step

### Step 1 — check you can reach the server

In PowerShell on your laptop:

```powershell
ssh ubuntu@192.168.1.50
```

Use your own username and IP. The first time, it asks whether you trust the
server — type `yes`. When you see a prompt like `ubuntu@server:~$`, you are
logged in. Type `exit` to come back to your laptop.

If this does not work, nothing later will. Fix it first: usually a wrong IP, a
wrong username, or SSH not being enabled on the server.

### Step 2 — check the server can see the camera

Still logged in to the server:

```bash
ping -c 3 172.20.100.138
```

Use your camera's IP. If the packets come back, the server can reach it.

> If `ping` fails but your laptop can reach the camera, the server is on a
> different network or a firewall sits between them. **Stop here and sort that
> out** — no amount of software will fix it, and every later step will look
> broken for this reason.

Type `exit` to return to your laptop.

### Step 3 — copy the project to the server

In PowerShell, **in the project folder** (`C:\Users\...\Documents\person count`):

```powershell
.\server\deploy.ps1 -Server ubuntu@192.168.1.50
```

This packs up the files the server needs, copies them across with `scp`, and
unpacks them into `~/person-counter` on the server. It takes a few seconds.

**If the server has no internet access**, add `-IncludeModel` so the weights
travel with it instead of being downloaded there:

```powershell
.\server\deploy.ps1 -Server ubuntu@192.168.1.50 -IncludeModel
```

Other options: `-Path /opt/person-counter` to put it somewhere else,
`-Port 2222` for a non-standard SSH port. `.\server\deploy.ps1 -?` lists them.

> On a Mac, a Linux laptop, WSL or Git Bash, use `bash server/deploy.sh
> ubuntu@192.168.1.50` instead — it does the same job.

> **What is deliberately not copied:** `server/.env`, `lines/` and
> `cameras.json`. Those are the server's own settings and the counting lines
> you drew there; overwriting them on every deploy would undo your work.

### Step 4 — install everything on the server

```powershell
ssh ubuntu@192.168.1.50
```

then, on the server:

```bash
cd ~/person-counter
bash server/install.sh
```

This takes **five to fifteen minutes**, mostly downloading PyTorch. It will ask
for your password once, to install system packages. It:

1. installs the system packages Python and OpenCV need
2. creates a *virtual environment* — a private folder of Python packages, so
   this project cannot break anything else on the machine
3. installs the Python packages into it
4. downloads `yolov8n.pt` if it is missing
5. builds the OpenVINO model, which is what makes it run at a usable speed
6. creates `server/.env` for you to edit

It is safe to run again at any time.

### Step 5 — tell it which camera to watch

```bash
nano server/.env
```

`nano` is a plain text editor. Find the `CAMERA_URL=` line and put your
camera's address after the `=`, with no spaces around it:

```
CAMERA_URL=rtsp://admin:Admin%400192@172.20.100.138:554/cam/realmonitor?channel=1&subtype=1
```

Save and quit: **Ctrl+O**, **Enter**, **Ctrl+X**.

> Use the camera's **sub-stream** if it has one (often `subtype=1`, or
> `channels/102`). It is typically 640×480 rather than 1920×1080 — plenty for
> counting people, and roughly nine times less work for the CPU.

### Step 6 — start it and watch

```bash
bash server/run.sh
```

You should see:

```
====================================================================
 PERSON COUNTER - server
====================================================================
  project root   /home/ubuntu/person-counter
  env file       /home/ubuntu/person-counter/server/.env
  camera         rtsp://admin:****@172.20.100.138:554/cam/realmonitor?channel=1&subtype=1
  listening on   http://0.0.0.0:8000
  model          openvino  yolov8n_openvino_model  imgsz=480
  track point    foot
  login          NOT required - anyone who can reach the port can watch
====================================================================
Opening rtsp://admin:****@172.20.100.138:554/...
Connected: 640x480
Loading yolov8n_openvino_model/yolov8n.xml ...
Model ready (imgsz=480).
Streaming ... at 640x480
```

Loading the model takes 10–30 seconds the first time. Leave this running.

### Step 7 — open it on your laptop

In your browser:

```
http://192.168.1.50:8000
```

Your server's IP, then `:8000`. You should see the camera, with green boxes
around people and an ID on each.

**Nothing appears?** Jump to [section 7](#7-when-you-cannot-reach-the-page) —
it is almost always the firewall.

### Step 8 — draw the counting line

Nothing is counted until you tell it where the doorway is. The page says so in
an orange banner.

1. Press **Draw / move the line**. The picture freezes on a still frame —
   deliberately, because a line placed on moving video lands on whichever frame
   happened to be showing when you let go.
2. **Click and drag** across the doorway, right the way across it.
3. Look at the **green arrow**. It points the direction that counts as an
   *entry*. If it points the wrong way, press **Flip entry direction**.
4. **Ignored band** is the orange dashed band either side of the line. The
   counter ignores anything inside it, which is what stops somebody standing in
   the doorway from running the numbers up as their detection box jitters.
   Widen it until it comfortably covers that shuffling, and no wider. 12–30
   pixels suits most cameras.
5. Press **Save**.

The counts start at zero and the live view comes back, now with the red line,
the orange band and the green arrow drawn into it.

The line is saved on the server in `lines/`, named after the camera, so it
survives restarts and reboots. Each camera keeps its own.

**That is it — it works.** Press Ctrl+C to stop it, then do
[section 5](#5-making-it-permanent) to make it run without you.

---

## 4. Using the web page

```
┌──────────────────────────────────────────┬──────────────────────┐
│                                          │  COUNTS              │
│                                          │   12    9      3     │
│        live video, with boxes,           │  entry exit  inside  │
│        IDs, trails, the counting         │  [ Reset counts ]    │
│        line and the counts drawn         ├──────────────────────┤
│        into the picture                  │  COUNTING LINE       │
│                                          │  [ Draw / move ]     │
│                                          │  [ Remove the line ] │
│                                          ├──────────────────────┤
│                                          │  COUNT PEOPLE BY     │
│                                          │  [ Feet ] [ Head ]   │
├──────────────────────────────────────────┼──────────────────────┤
│  the green arrow points the way an       │  CAMERA   ▼          │
│  entry goes …                            │  HEALTH              │
│                                          │  RECENT CROSSINGS    │
└──────────────────────────────────────────┴──────────────────────┘
```

| Control | What it does |
|---|---|
| **Entries / Exits / Inside** | Crossings in each direction, and the difference. *Inside* assumes the room was empty when counting started, so it can legitimately go negative — reset the counts when you know the room is empty and it becomes meaningful. |
| **Reset counts** | Zeroes the totals. Does not forget where people currently are, so nobody books a phantom crossing straight afterwards. |
| **Draw / move the line** | The line editor described above. |
| **Remove the line** | Deletes it. The camera keeps running and tracking, but nothing is counted until you draw a new one. |
| **Feet / Head** | Which point on each person is tested against the line. **Feet** suits a camera mounted above head height, where bodies lean into the frame and a head would reach the line too early. **Head** is steadier where the lower body is hidden — crowds, or a counter or desk across the bottom of the picture. Changing it clears everyone's remembered side, so no phantom crossings are booked. |
| **Camera** | The cameras this server has used before. Pick one and press *Switch*. It takes a few seconds; the model stays loaded. Each camera remembers its own counting line. |
| **Health** | See below. |
| **Recent crossings** | The last few events, newest first. |

### Reading the health panel

| Row | What it means | What is normal |
|---|---|---|
| **Status** | `running`, `connecting`, `reconnecting`, `loading model` | `running` |
| **Processing** | Frames per second the model manages | 8–15 on a 4-core CPU at `imgsz=480` |
| **Behind live** | How old the frame being processed is | Under 200 ms |
| **Frames skipped** | Frames the camera sent that the model never saw | **This is normal and is the design.** A live camera sends ~25 FPS and the model handles ~12, so the rest are discarded rather than queued. A frame from two seconds ago is worthless for counting; a current one is what the tracker wants. 40–60% is healthy. |
| **Picture size** | The camera's resolution | 640×480 from a sub-stream |
| **Connected for** | How long this camera connection has lasted | Climbing. Resetting to zero repeatedly means the camera keeps dropping out. |

---

## 5. Making it permanent

Started with `run.sh`, the program is tied to your SSH session: close the
terminal and it stops. To make it belong to the machine instead:

```bash
bash server/install_service.sh
```

That installs it as a **systemd service** — Linux's standard way of running
something in the background. From then on it:

- starts automatically when the server boots
- restarts itself within 10 seconds if it ever crashes
- keeps a log you can read afterwards
- runs whether or not anybody is logged in

```bash
sudo systemctl status person-counter     # is it running?
sudo journalctl -u person-counter -f     # watch the log live (Ctrl+C to stop)
sudo systemctl restart person-counter    # after changing .env or the code
sudo systemctl stop person-counter       # stop it
sudo systemctl start person-counter      # start it
sudo systemctl disable person-counter    # stop it starting at boot
```

`person-counter.service` in this folder is the same unit file as a template,
if you ever want to edit it by hand.

---

## 6. Day-to-day operation

### Is it running?

```bash
sudo systemctl status person-counter
```

Or from your laptop, without logging in — open `http://192.168.1.50:8000/healthz`.
It answers `{"status": "running"}`.

### Read the log

```bash
sudo journalctl -u person-counter -f            # live
sudo journalctl -u person-counter --since today # today's
sudo journalctl -u person-counter -n 200        # the last 200 lines
```

### Change a setting

```bash
nano ~/person-counter/server/.env
sudo systemctl restart person-counter
```

Settings only take effect on restart.

### Push a code change from your laptop

```powershell
.\server\deploy.ps1 -Server ubuntu@192.168.1.50
ssh ubuntu@192.168.1.50 "sudo systemctl restart person-counter"
```

Your `.env` and your counting lines are left alone.

### Run a second camera on the same server

Copy the project into a second folder, give it its own `.env` with a different
`CAMERA_URL` **and a different `PORT`** (say 8001), and install it as a second
service under another name. Two cameras on 4 cores will roughly halve the frame
rate of each.

### Where things are kept on the server

| | |
|---|---|
| The project | `~/person-counter/` |
| Settings | `~/person-counter/server/.env` |
| Counting lines | `~/person-counter/lines/*.json` |
| Remembered cameras | `~/person-counter/cameras.json` |
| Python packages | `~/person-counter/.venv/` |
| The service | `/etc/systemd/system/person-counter.service` |
| The log | the system journal — `journalctl -u person-counter` |

---

## 7. When you cannot reach the page

Work down this list. It is in order of likelihood.

### 1. Is it actually running?

On the server:

```bash
curl -s http://127.0.0.1:8000/healthz
```

- **Answers `{"status": ...}`** → the program is fine, the problem is between
  your laptop and the server. Go to step 2.
- **"Connection refused"** → the program is not running. Check the log.

### 2. Open the firewall

This is the usual answer. Linux servers normally block everything but SSH.

**Ubuntu / Debian:**

```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

**RHEL / Rocky / Fedora:**

```bash
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

**A cloud server** (AWS, Azure, GCP) also has a firewall in the cloud console —
a *security group* or *network security group*. Allow inbound TCP 8000 there
too, from your own address rather than from everywhere.

### 3. Are you using the right IP?

On the server:

```bash
hostname -I
```

The first address is usually the one. A server with several network cards has
several addresses, and only the one on *your* network will do.

### 4. Use an SSH tunnel instead

If you cannot open the firewall — or would rather not — this works without
changing anything on the server. In PowerShell on your laptop:

```powershell
ssh -L 8000:localhost:8000 ubuntu@192.168.1.50
```

Leave that window open, and browse to **`http://localhost:8000`**.

Your laptop's port 8000 is now carried, encrypted, through the SSH connection
to the server's port 8000. Nothing new is exposed to the network, and the
traffic is encrypted end to end. It is the right answer whenever the server is
not on a network you fully trust.

---

## 8. Security

**As shipped, anybody who can reach port 8000 can watch your camera.** There is
no login until you add one. On a private office LAN that may be perfectly
reasonable. Decide deliberately rather than by default.

### Add a username and password

In `server/.env`:

```
AUTH_USER=viewer
AUTH_PASSWORD=something-long-and-not-guessable
```

then `sudo systemctl restart person-counter`. The browser now asks for them.

This is HTTP Basic authentication: the password is *encoded*, not *encrypted*.
Anybody able to watch the traffic between your laptop and the server could read
it. Over a LAN among colleagues that is a reasonable trade. Over the internet
it is not.

### Do not put this straight on the internet

If you need to reach it from outside, in order of preference:

1. **An SSH tunnel** (section 7, step 4) — encrypted, nothing exposed, nothing
   to configure. Best for one or two people.
2. **A VPN** into the office network.
3. **An HTTPS reverse proxy** (nginx or Caddy) in front of it, with a real
   certificate. Then set `BIND_HOST=127.0.0.1` so the port itself is reachable
   only by the proxy.

### Passwords in files

`config.py`, `cameras.json` and `server/.env` all contain camera passwords in
clear text. That is how the project has always worked and it is noted in the
main README too. On the server:

```bash
chmod 600 ~/person-counter/server/.env ~/person-counter/config.py
```

so that only your account can read them. None of these files should ever be
committed to a shared repository — the project's `.gitignore` already excludes
them.

The web page never receives a camera URL. The camera list it shows is
host-and-path only, with the password stripped, precisely so it cannot end up
in a browser history or a screenshot.

---

## 9. Making it faster

Check the **Processing** figure in the health panel first. Below about 6 FPS,
people walking briskly can cross the line between two frames and be missed.

In order of how much they buy you:

| Change | Effect | How |
|---|---|---|
| **Use the camera's sub-stream** | Often 3–5× | A 640×480 stream instead of 1920×1080. Change `CAMERA_URL` — usually `subtype=1` or `channels/102`. |
| **Smaller inference size** | ~2× from 480 → 320 | `INFERENCE_SIZE=320` in `.env`, then re-export: `.venv/bin/python export_openvino.py --imgsz 320`. **Both must match**, or the server stops at start-up and tells you so. Distant people are missed more often. |
| **Use OpenVINO, not PyTorch** | ~1.7× | `MODEL_BACKEND=openvino`. The default, and what `install.sh` sets up. |
| **More CPU cores** | Roughly linear | The model is CPU-bound. |

Things that do **not** help: a graphics card (the OpenVINO path targets the
CPU deliberately — on this project's hardware an integrated GPU measured
*five times slower*); lowering `JPEG_QUALITY` (that is network bandwidth, not
processing).

If the video looks fine but the counts are wrong, speed is not your problem —
re-read the counting-line part of the project's main README.

---

## 10. The Docker route

An alternative to `install.sh`. Docker packages the whole thing — Python,
PyTorch, OpenCV — into an image, so nothing is installed on the server itself
and the result is identical on every machine. The cost is about 2 GB and a
longer first build.

Use it if Docker is already part of how you run things, or if `install.sh`
fails on an unusual distribution. Otherwise `install.sh` is simpler.

```bash
cd ~/person-counter/server
cp .env.example .env
nano .env                  # set CAMERA_URL
docker compose up -d       # build and start
docker compose logs -f     # watch it
docker compose down        # stop
```

The compose file uses `network_mode: host`, which is deliberate: RTSP
negotiates its own ports and some cameras announce an address that only makes
sense on the real network. Sharing the host's network side-steps all of it.
`lines/` and `cameras.json` are mounted from outside the container, so
rebuilding the image does not lose the line you drew.

---

## 11. Troubleshooting

### It will not start

| What you see | What it means | What to do |
|---|---|---|
| `Could not import config.py` | `config.py` did not get copied — it is in `.gitignore`, so a `git clone` on the server will not have it. | Redeploy with `deploy.ps1`, which copies it. |
| `ModuleNotFoundError: No module named 'flask'` | Running the system Python instead of the virtual environment. | Use `bash server/run.sh`, or `.venv/bin/python server/stream_server.py`. |
| `libGL.so.1: cannot open shared object file` | OpenCV's system libraries are missing. | `sudo apt-get install -y libgl1 libglib2.0-0` |
| `yolov8n_openvino_model/yolov8n.xml not found` | The model was never exported. | `.venv/bin/python export_openvino.py`, or set `MODEL_BACKEND=pytorch`. |
| `was exported for imgsz=480, but this run wants imgsz=320` | `.env` and the exported model disagree. | Re-export at that size, or put the size back. |
| `Address already in use` | Something already holds port 8000 — often an older copy of this. | `sudo lsof -i :8000`, stop it, or set a different `PORT`. |
| `Python 3.10 or newer is required` | The distribution ships an older Python. | Install a newer one (`sudo apt install python3.11 python3.11-venv`), or use Docker. |

### The page loads but the video does not

| What you see | What it means | What to do |
|---|---|---|
| "Connecting to the camera …" that never changes | The server cannot open the stream. | `ping` the camera from the server. Check the username, password and stream path. Remember `@` in a password must be `%40`. |
| "Camera unavailable — retrying" | It connected once and lost it. | Often another client took the last stream slot — cameras usually allow only 3–5 at once. Close any VLC or NVR window watching the same stream. |
| The picture freezes but counts keep climbing | The browser's connection to the stream dropped. | Reload the page. If it happens repeatedly, lower `STREAM_MAX_WIDTH` and `JPEG_QUALITY`. |
| Very choppy video | The server is at its limit, or the link is. | Check **Processing** in the health panel. Low → [section 9](#9-making-it-faster). Fine → lower `STREAM_MAX_WIDTH`. |

### The counts are wrong

| What you see | What it means | What to do |
|---|---|---|
| Counts climb with nobody moving | Somebody is standing on the line and their detection box is jittering across it. | Widen the **ignored band**. |
| People crossing are not counted | The line does not span the whole doorway, or they are crossing past its end. A line counts only along the part you drew, not its invisible continuation. | Redraw it right across. |
| Entries and exits are swapped | The green arrow points the wrong way. | Edit the line, press **Flip entry direction**, save. |
| Counted too early or too late | The wrong reference point. | Switch between **Feet** and **Head**. |
| **Inside** is negative | People who were already in the room when counting started were never counted in. | Reset the counts when the room is empty. |

---

## 12. Every file in this folder

| File | What it is | Where it runs |
|---|---|---|
| `README.md` | This document | — |
| `stream_server.py` | The server itself: camera, model, counting, web pages | Server |
| `server_config.py` | Reads `.env`, hands the settings to the project's modules | Server |
| `templates/index.html` | The page you look at — the whole UI, one file, no frameworks | Your browser |
| `requirements.txt` | The Python packages to install | Server |
| `.env.example` | A fully commented template for your settings | Copy to `.env` |
| `install.sh` | One-shot setup: system packages, virtual environment, model | Server, once |
| `run.sh` | Start it in the foreground, for testing | Server |
| `install_service.sh` | Install it as a systemd service | Server, once |
| `person-counter.service` | The systemd unit as a template, for editing by hand | Server |
| `Dockerfile` | Build a container image instead | Server |
| `docker-compose.yml` | Build and run it in one command | Server |
| `deploy.ps1` | Copy the project laptop → server | **Your laptop** (PowerShell) |
| `deploy.sh` | The same, for Mac / Linux / WSL / Git Bash | Your laptop |

And what it uses from the project above it, unchanged: `person_counter/`
(camera, detector, counter, counting line, drawing), `config.py`,
`export_openvino.py`.

### The API, if you want to pull the counts into something else

Every endpoint needs the login if you set one.

| Method and path | What it does |
|---|---|
| `GET /` | The web page |
| `GET /video` | The MJPEG stream |
| `GET /snapshot.jpg` | One frame. `?clean=1` for one with nothing drawn on it |
| `GET /api/stats` | Counts, FPS, status, the line — as JSON |
| `POST /api/reset` | Zero the counts |
| `GET`/`POST`/`DELETE` `/api/line` | Read, set or remove the counting line |
| `POST /api/track-point` | `{"track_point": "foot"}` or `"head"` |
| `GET /api/cameras` | The remembered cameras, without passwords |
| `POST /api/source` | Switch camera: `{"index": 0}` or `{"source": "rtsp://…"}` |
| `GET /healthz` | For monitoring. No login needed. |

To log the counts every minute, for example:

```bash
* * * * * curl -s http://127.0.0.1:8000/api/stats >> /var/log/person-counts.jsonl
```

---

## 13. Settings reference

All of these live in `server/.env`. A real environment variable beats the file,
so `PORT=9000 bash server/run.sh` works for a one-off.

| Setting | Default | What it does |
|---|---|---|
| `CAMERA_URL` | the camera in `config.py` | Which camera to watch |
| `BIND_HOST` | `0.0.0.0` | `0.0.0.0` = reachable from the network; `127.0.0.1` = only through an SSH tunnel or a local proxy |
| `PORT` | `8000` | The port the page is on |
| `AUTH_USER` | *(blank)* | Blank = no login at all |
| `AUTH_PASSWORD` | *(blank)* | |
| `JPEG_QUALITY` | `70` | 1–100. Lower = less bandwidth, blockier picture |
| `STREAM_MAX_WIDTH` | `960` | Shrink wider frames before sending. 0 = don't. Does not affect counting |
| `MODEL_BACKEND` | `openvino` | `openvino` (fast, needs the export) or `pytorch` (slower, works anywhere) |
| `INFERENCE_SIZE` | `480` | Multiple of 32. Must match the exported model |
| `TRACK_POINT` | `foot` | `foot` or `head`. Changeable from the page |
| `TORCH_THREADS` | `1` | Leave alone unless you are measuring something |
| `RETRY_DELAY_SECONDS` | `5` | How long to wait before trying the camera again |

---

## 14. Glossary

**SSH** — a secure way to type commands on another computer. `ssh user@ip` logs
you in; `exit` comes back.

**scp** — copies files over SSH. `deploy.ps1` uses it.

**Virtual environment (`.venv`)** — a private folder of Python packages for one
project, so installing PyTorch here cannot break something else on the machine.
`bash server/run.sh` uses it automatically.

**systemd service** — Linux's way of running a program in the background: it
starts at boot, restarts on crash, and keeps a log.

**journalctl** — reads the log a systemd service writes.

**Firewall** — blocks incoming connections. Why you cannot reach port 8000
until you allow it.

**RTSP** — the protocol IP cameras speak. An address looks like
`rtsp://user:password@ip:554/path`.

**Sub-stream** — a second, smaller video stream most cameras offer alongside
the full-resolution one. Much cheaper to process, and plenty for counting.

**MJPEG** — video as a stream of separate JPEG pictures. Inefficient, but every
browser shows it with no plugin, which is why the video reaches you this way.

**OpenVINO** — Intel's inference runtime. The project converts the YOLO model
into its format because that roughly halves the time each frame takes on a CPU.

**imgsz / inference size** — the size each frame is scaled to before the model
looks at it. Smaller is faster and misses more distant people.

**ByteTrack** — the tracker that gives each person a stable ID from frame to
frame. Without it there is no way to say *this* person crossed the line.

**Dead zone / ignored band** — the strip either side of the counting line that
is ignored, so somebody standing in the doorway cannot run the counts up.
