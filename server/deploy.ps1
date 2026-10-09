<#
.SYNOPSIS
    Copy this project from your Windows laptop to the Linux server.

.DESCRIPTION
    Run this ON THE LAPTOP, from the project folder, in PowerShell:

        .\server\deploy.ps1 -Server ubuntu@192.168.1.50

    It packs the files the server needs into a single archive, copies it over
    with scp, and unpacks it there. Nothing is installed - that is install.sh,
    which you run once on the server afterwards.

    ssh and scp are part of Windows 10 and 11; nothing needs installing here.

    What is NOT copied, and why:
      server/.env    the server's own settings, including which camera it
                     watches. Overwriting it on every deploy would undo
                     whatever you set up there.
      lines/         the counting lines you drew in the browser. They belong
                     to the server's cameras.
      cameras.json   the server's remembered camera list. -IncludeCameras
                     sends yours anyway.
      .venv/, .git/, __pycache__/, *.zip, sample.mp4
                     either rebuilt on the server or simply large.

.PARAMETER Server
    user@host of the Linux server, e.g. ubuntu@192.168.1.50

.PARAMETER Path
    Where to put it on the server. Default ~/person-counter, which needs no
    sudo. /opt/person-counter is the conventional place for a service, but you
    have to create it and give yourself ownership first.

.PARAMETER IncludeModel
    Also send yolov8n.pt and the exported yolov8n_openvino_model/ (about 18 MB
    together). Use this when the server has no route to the internet and
    cannot download the weights itself.

.PARAMETER IncludeCameras
    Also send cameras.json, the remembered camera list. It contains passwords
    in clear text, exactly as it does here.

.PARAMETER Port
    SSH port, if it is not 22.

.EXAMPLE
    .\server\deploy.ps1 -Server ubuntu@192.168.1.50

.EXAMPLE
    .\server\deploy.ps1 -Server admin@10.0.0.9 -Path /opt/person-counter -IncludeModel
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Server,

    [string]$Path = "~/person-counter",

    [switch]$IncludeModel,

    [switch]$IncludeCameras,

    [int]$Port = 22
)

$ErrorActionPreference = "Stop"

function Say  { param($m) Write-Host "==> $m" -ForegroundColor Cyan }
function Warn { param($m) Write-Host "    $m" -ForegroundColor Yellow }
function Die  { param($m) Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# The project root is the folder above this script, however the script was
# invoked - so it works from the project folder, from server/, or by full path.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
Say "Project folder: $ProjectRoot"

foreach ($tool in @("ssh", "scp", "tar")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Die "$tool was not found. On Windows 10/11: Settings > System > Optional features > OpenSSH Client."
    }
}

# ---------------------------------------------------------------------------
# Stage the files
# ---------------------------------------------------------------------------
# Copying into a clean folder first, rather than listing excludes to tar, so
# that what gets sent is a thing you can look at and check before it goes.
$Staging = Join-Path $env:TEMP ("person-counter-deploy-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $Staging -Force | Out-Null

try {
    Say "Staging files"

    $files = @("config.py", "export_openvino.py", "main.py", "requirements.txt", "README.md")
    foreach ($file in $files) {
        if (Test-Path $file) {
            Copy-Item $file -Destination $Staging
            Write-Host "    $file"
        } elseif ($file -eq "config.py") {
            Die "config.py is missing. The server needs it: it holds the camera login and the model paths."
        }
    }

    # person_counter/, without the compiled cache - those are built for this
    # machine's Python and are rebuilt on the server anyway.
    Copy-Item "person_counter" -Destination $Staging -Recurse
    Get-ChildItem (Join-Path $Staging "person_counter") -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    Write-Host "    person_counter/"

    # server/, without .env (the server's own) or the cache.
    Copy-Item "server" -Destination $Staging -Recurse
    $stagedServer = Join-Path $Staging "server"
    Remove-Item (Join-Path $stagedServer ".env") -ErrorAction SilentlyContinue
    Get-ChildItem $stagedServer -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    Write-Host "    server/"

    if ($IncludeModel) {
        foreach ($item in @("yolov8n.pt", "yolov8n_openvino_model")) {
            if (Test-Path $item) {
                Copy-Item $item -Destination $Staging -Recurse
                Write-Host "    $item"
            } else {
                Warn "$item is not here - the server will fetch or build it itself."
            }
        }
    }

    if ($IncludeCameras -and (Test-Path "cameras.json")) {
        Copy-Item "cameras.json" -Destination $Staging
        Write-Host "    cameras.json"
    }

    # -----------------------------------------------------------------------
    # Pack, send, unpack
    # -----------------------------------------------------------------------
    # One archive rather than `scp -r` of a dozen items: it is a single
    # connection, it keeps the file modes, and it is far quicker over a link
    # with any latency, where each small file otherwise costs a round trip.
    $Archive = Join-Path $env:TEMP "person-counter-deploy.tar.gz"
    if (Test-Path $Archive) { Remove-Item $Archive -Force }

    Say "Packing"
    tar -czf $Archive -C $Staging .
    if ($LASTEXITCODE -ne 0) { Die "tar failed." }
    $sizeMB = [math]::Round((Get-Item $Archive).Length / 1MB, 1)
    Write-Host "    $sizeMB MB"

    Say "Copying to $Server`:$Path"
    # The remote temp name includes the PID so two deploys at once cannot
    # trample each other's archive.
    $RemoteTmp = "/tmp/person-counter-deploy-$PID.tar.gz"
    scp -P $Port $Archive "$Server`:$RemoteTmp"
    if ($LASTEXITCODE -ne 0) { Die "scp failed. Check that you can run: ssh -p $Port $Server" }

    Say "Unpacking on the server"
    # Single-quoted here-string: PowerShell must not touch any of this, it is
    # shell script for the far end.
    $remoteScript = @"
set -e
mkdir -p $Path
tar -xzf $RemoteTmp -C $Path
rm -f $RemoteTmp
chmod +x $Path/server/*.sh 2>/dev/null || true
echo "    unpacked into \$(cd $Path && pwd)"
"@
    ssh -p $Port $Server $remoteScript
    if ($LASTEXITCODE -ne 0) { Die "The remote unpack failed." }

    Remove-Item $Archive -Force

    $hostOnly = $Server.Split("@")[-1]
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host " Files are on the server." -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host " FIRST TIME - log in and set it up:"
    Write-Host "     ssh -p $Port $Server"
    Write-Host "     cd $Path"
    Write-Host "     bash server/install.sh"
    Write-Host "     nano server/.env          # put your camera address in"
    Write-Host "     bash server/run.sh"
    Write-Host ""
    Write-Host " Then on this laptop, open:"
    Write-Host "     http://$hostOnly`:8000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host " AFTER A CODE CHANGE - redeploy and restart:"
    Write-Host "     .\server\deploy.ps1 -Server $Server -Path $Path"
    Write-Host "     ssh -p $Port $Server 'sudo systemctl restart person-counter'"
    Write-Host ""
}
finally {
    Remove-Item $Staging -Recurse -Force -ErrorAction SilentlyContinue
}
