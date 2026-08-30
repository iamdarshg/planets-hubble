# Start the local Colab results receiver plus a free cloudflared quick tunnel.
# Results uploaded by the Colab notebook land under artifacts/colab-uploads/uploads.
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$ReceiverPort = 8787
$LogDir = Join-Path $RepoRoot "artifacts/colab-uploads"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Get-NetTCPConnection -LocalPort $ReceiverPort -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $ReceiverPort is already in use; stop the existing receiver first."
}

# 1) Build the prepared real-data bundle the notebook downloads.
& python (Join-Path $PSScriptRoot "colab_bundle.py") | Out-Null
if ($LASTEXITCODE -ne 0) { throw "colab_bundle.py failed" }

# 2) Start the receiver.
$ReceiverLog = Join-Path $LogDir "receiver.log"
$ReceiverErr = Join-Path $LogDir "receiver.err.log"
$receiver = Start-Process -FilePath "python" -ArgumentList @(
    (Join-Path $PSScriptRoot "colab_receiver.py"),
    "--port", "$ReceiverPort",
    "--upload-dir", (Join-Path $RepoRoot "artifacts/colab-uploads/uploads"),
    "--bundle-dir", (Join-Path $RepoRoot "artifacts/colab-uploads/bundles"),
    "--token-file", (Join-Path $PSScriptRoot "colab_receiver.token")
) -WindowStyle Hidden -RedirectStandardOutput $ReceiverLog -RedirectStandardError $ReceiverErr -PassThru
Start-Sleep -Seconds 3
if ($receiver.HasExited) {
    Get-Content $ReceiverLog -Tail 20 -ErrorAction SilentlyContinue
    Get-Content $ReceiverErr -Tail 20 -ErrorAction SilentlyContinue
    throw "receiver exited early"
}
Write-Output ("receiver_pid=" + $receiver.Id)

# 3) Ensure cloudflared is available.
$Cloudflared = Join-Path $PSScriptRoot "cloudflared.exe"
if (-not (Test-Path $Cloudflared)) {
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $Cloudflared
}

# 4) Start the tunnel.
$TunnelLog = Join-Path $LogDir "tunnel.log"
$TunnelErr = Join-Path $LogDir "tunnel.err.log"
$tunnel = Start-Process -FilePath $Cloudflared -ArgumentList @(
    "tunnel", "--url", ("http://127.0.0.1:" + $ReceiverPort), "--no-autoupdate"
) -WindowStyle Hidden -RedirectStandardOutput $TunnelLog -RedirectStandardError $TunnelErr -PassThru
Write-Output ("tunnel_pid=" + $tunnel.Id)

# 5) Wait for the public trycloudflare URL.
$url = $null
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    $content = ""
    if (Test-Path $TunnelLog) { $content += Get-Content $TunnelLog -Raw -ErrorAction SilentlyContinue }
    if (Test-Path $TunnelErr) { $content += Get-Content $TunnelErr -Raw -ErrorAction SilentlyContinue }
    if ($content -match "https://[a-z0-9-]+[.]trycloudflare[.]com") { $url = $Matches[0]; break }
    if ($tunnel.HasExited) { break }
}
if (-not $url) {
    Get-Content $TunnelLog -Tail 40 -ErrorAction SilentlyContinue
    Get-Content $TunnelErr -Tail 40 -ErrorAction SilentlyContinue
    throw "cloudflared tunnel URL not found"
}

# 6) Publish the endpoint file the notebook fetches from GitHub raw.
$TokenFile = Join-Path $PSScriptRoot "colab_receiver.token"
$Token = (Get-Content $TokenFile -Raw).Trim()
$EndpointFile = Join-Path $RepoRoot "colab/upload_endpoint.txt"
$payload = @{ url = $url; token = $Token; started_at = (Get-Date -Format o) } | ConvertTo-Json -Compress
Set-Content -LiteralPath $EndpointFile -Value $payload -Encoding ascii -NoNewline

# 7) Verify health and bundle availability through the tunnel.
$health = Invoke-WebRequest -Uri ($url + "/health") -UseBasicParsing -TimeoutSec 30
Write-Output ("health_status=" + $health.StatusCode)
Write-Output ("health_body=" + $health.Content)
Write-Output ("UPLOAD_ENDPOINT=" + $url)
Write-Output ("ENDPOINT_FILE=" + $EndpointFile)
