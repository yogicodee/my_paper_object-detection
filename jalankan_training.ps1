# =============================================================================
#  Menjalankan seluruh pelatihan secara mandiri.
#
#  Skrip ini sengaja dibuat agar TIDAK bergantung pada sesi Claude Code maupun
#  jendela terminal yang membukanya. Pelatihan berlangsung berhari-hari, jadi
#  proses harus bertahan walau sesi ditutup.
#
#  Cara memakai:
#    Klik kanan berkas ini -> "Run with PowerShell"
#  atau dari terminal:
#    powershell -ExecutionPolicy Bypass -File jalankan_training.ps1
#
#  Aman dihentikan kapan saja (Ctrl+C atau tutup jendela). Jalankan lagi untuk
#  melanjutkan: model yang sudah selesai dilewati, yang terpotong dilanjutkan
#  dari checkpoint terakhir.
# =============================================================================

$ErrorActionPreference = "Continue"
$proyek = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $proyek

$vpy = Join-Path $proyek ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Write-Host "Python virtual environment tidak ditemukan di $vpy" -ForegroundColor Red
    Read-Host "Tekan Enter untuk menutup"
    exit 1
}

$logDir = Join-Path $proyek "results"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log    = Join-Path $logDir "training_log.txt"
$status = Join-Path $logDir "training_status.txt"

function Tulis-Status($pesan) {
    $baris = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $pesan
    Add-Content -Path $status -Value $baris -Encoding utf8
    Write-Host $baris -ForegroundColor Cyan
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host " Pelatihan deteksi APD - tiga model, 60 epoch" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Perkiraan total: sekitar 7 hari pada CPU ini."
Write-Host "Jendela ini boleh diminimalkan, tapi JANGAN ditutup"
Write-Host "dan komputer jangan sampai sleep."
Write-Host ""
Write-Host "Log rinci  : $log"
Write-Host "Ringkasan  : $status"
Write-Host ""

Tulis-Status "SESI PELATIHAN DIMULAI"

# Cegah komputer tidur selama pelatihan berlangsung.
# (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
$sig = @"
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
$pwr = $null
try {
    $pwr = Add-Type -MemberDefinition $sig -Name "Power" -Namespace "Win32" -PassThru
    [void]$pwr::SetThreadExecutionState(0x80000001)
    Tulis-Status "Mode tidur otomatis dinonaktifkan selama pelatihan"
} catch {
    Tulis-Status "PERINGATAN: gagal menonaktifkan mode tidur - atur manual lewat Settings"
}

$model = @("yolov8n", "yolo11n", "yolo26n")
$mulai = Get-Date

foreach ($m in $model) {
    Tulis-Status "MULAI $m"
    $t0 = Get-Date

    & $vpy scripts/03_train.py --run $m --resume 2>&1 |
        Tee-Object -FilePath $log -Append

    $durasi = (Get-Date) - $t0
    if ($LASTEXITCODE -eq 0) {
        Tulis-Status ("SELESAI $m dalam {0:hh\:mm\:ss}" -f $durasi)
    } else {
        Tulis-Status "GAGAL $m (kode $LASTEXITCODE) - lanjut ke model berikutnya"
    }
}

$total = (Get-Date) - $mulai
Tulis-Status ("SELURUH PELATIHAN SELESAI dalam {0:d\.hh\:mm\:ss}" -f $total)

# Kembalikan perilaku tidur ke pengaturan normal.
if ($pwr) { try { [void]$pwr::SetThreadExecutionState(0x80000000) } catch { } }

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host " Pelatihan selesai." -ForegroundColor Green
Write-Host " Langkah berikutnya - buka Claude Code lagi," -ForegroundColor Green
Write-Host " lalu minta lanjutkan evaluasi dan analisis." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Read-Host "Tekan Enter untuk menutup jendela"
