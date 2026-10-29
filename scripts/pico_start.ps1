<#
PICO 4 / 4 Ultra + XRoboToolkit 준비 (PC에서 실행)

  1) XRoboToolkit PC 서비스가 꺼져 있으면 실행
  2) 헤드셋이 USB로 연결돼 있으면: adb reverse tcp:63901, 헤드셋 깨우기, XRoboToolkit 앱 실행
  3) 헤드셋 앱에 입력할 PC IP 안내 (USB: 127.0.0.1 / Wi-Fi: 헤드셋에서 닿는 PC 주소)
  4) -Live: 시뮬레이션 서버(sim_server.py --source xrt --viewer) 실행, -Record <session>: A 버튼으로 녹화

    powershell -ExecutionPolicy Bypass -File scripts\pico_start.ps1
    powershell -ExecutionPolicy Bypass -File scripts\pico_start.ps1 -Live -Scene tabletop
    powershell -ExecutionPolicy Bypass -File scripts\pico_start.ps1 -Live -Record sim01 -SubjectHeight 1.72
#>
param(
    [switch]$Live,
    [string]$Scene = "tabletop",
    [string]$Record = "",
    [double]$SubjectHeight = 0,
    [string]$ServiceDir = "$env:USERPROFILE\Downloads\XRoboToolkit-PC-Service.win"
)
$root = Split-Path -Parent $PSScriptRoot
$port = 63901
$pkg = "com.xrobotoolkit.client"
$activity = "$pkg/com.unity3d.player.UnityPlayerActivity"  # the APK has two LAUNCHER activities; this is the app

# 1) PC service
if (Get-Process RoboticsServiceProcess -ErrorAction SilentlyContinue) {
    Write-Host "[service] running"
} elseif (Test-Path "$ServiceDir\RoboticsServiceProcess.exe") {
    Start-Process -FilePath "$ServiceDir\RoboticsServiceProcess.exe" -WorkingDirectory $ServiceDir
    Start-Sleep -Seconds 2
    Write-Host "[service] started from $ServiceDir"
} else {
    Write-Warning "XRoboToolkit PC service not found in $ServiceDir (-ServiceDir <folder>)"
}

# 2) headset over USB
$adb = (Get-Command adb -ErrorAction SilentlyContinue).Source
if (-not $adb) { $adb = "$env:USERPROFILE\Downloads\platform-tools-latest-windows\platform-tools\adb.exe" }
$usb = $false
if (Test-Path $adb) {
    $dev = (& $adb devices) | Select-String -Pattern "`tdevice$"
    if ($dev) {
        $usb = $true
        if (-not ((& $adb reverse --list) -match "tcp:$port tcp:$port")) { & $adb reverse "tcp:$port" "tcp:$port" | Out-Null }
        & $adb shell input keyevent KEYCODE_WAKEUP | Out-Null
        # "intent delivered to currently running instance" (already open) is printed on stderr: not an error
        cmd /c "`"$adb`" shell am start -n $activity >nul 2>&1"
        $model = (& $adb shell getprop ro.product.model).Trim()
        Write-Host "[usb] headset $model : adb reverse tcp:$port ok, XRoboToolkit started"
    } else {
        Write-Host "[usb] no headset on USB (Wi-Fi mode)"
    }
}

# 3) what to type in the headset app
Write-Host ""
Write-Host "Headset app (XRoboToolkit): PC IP ->"
if ($usb) { Write-Host "   127.0.0.1                (USB, recommended)" }
Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq "Up" } | ForEach-Object {
    Write-Host ("   {0,-24} (headset on Wi-Fi: PC address on '{1}'; works if the headset network routes to it)" -f $_.IPv4Address.IPAddress, $_.InterfaceAlias)
}
Write-Host "   then: Motion Tracker = Full body, Send = on. Keep the headset on (it pauses the app when taken off)."
Write-Host ""

# 4) simulation server with the XRoboToolkit stream (README section 5)
if ($Live) {
    $py = Join-Path $root ".venv\Scripts\python.exe"
    $argv = @((Join-Path $root "scripts\sim_server.py"), "--source", "xrt", "--scene", $Scene, "--viewer", "--show-human")
    if ($Record) { $argv += @("--record-dir", (Join-Path $root "data\$Record\raw")) }
    if ($SubjectHeight -gt 0) { $argv += @("--subject-height", "$SubjectHeight") }
    & $py @argv
}
