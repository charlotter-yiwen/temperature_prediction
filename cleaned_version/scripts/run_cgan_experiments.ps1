# cGAN-CNN Parameter Experiment Runner
# Uses reduced epochs for sweep, then full train on best config

param(
    [int]$SweepEpochs = 500,
    [int]$FinalEpochs = 2000
)

$experiments = @(
    @{ label="baseline (lr=1e-4, bs=8, adv=0.01)";          args="--lr 0.0001 --batch-size 8  --adv-weight 0.01" },
    @{ label="lr=5e-4, bs=8, adv=0.01";                      args="--lr 0.0005 --batch-size 8  --adv-weight 0.01" },
    @{ label="lr=5e-5, bs=8, adv=0.01";                      args="--lr 0.00005 --batch-size 8 --adv-weight 0.01" },
    @{ label="lr=1e-4, bs=16, adv=0.01";                     args="--lr 0.0001 --batch-size 16 --adv-weight 0.01" },
    @{ label="lr=1e-4, bs=4,  adv=0.01";                     args="--lr 0.0001 --batch-size 4  --adv-weight 0.01" },
    @{ label="lr=1e-4, bs=8,  adv=0.001";                    args="--lr 0.0001 --batch-size 8  --adv-weight 0.001" },
    @{ label="lr=1e-4, bs=8,  adv=0.1";                      args="--lr 0.0001 --batch-size 8  --adv-weight 0.1" },
    @{ label="lr=5e-4, bs=16, adv=0.001";                    args="--lr 0.0005 --batch-size 16 --adv-weight 0.001" },
    @{ label="lr=5e-5, bs=4,  adv=0.001";                    args="--lr 0.00005 --batch-size 4 --adv-weight 0.001" }
)

$results = @()

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  cGAN Parameter Sweep (epochs=$SweepEpochs)" -ForegroundColor Cyan
Write-Host "  Total experiments: $($experiments.Count)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$i = 1
foreach ($exp in $experiments) {
    Write-Host "[$i/$($experiments.Count)] $($exp.label)" -ForegroundColor Yellow
    Write-Host "  Command: python models\cgan_cnn.py --epochs $SweepEpochs $($exp.args)"

    $argList = ("--epochs $SweepEpochs " + $exp.args) -split '\s+' | Where-Object { $_ -ne '' }
    $output = python models\cgan_cnn.py $argList 2>&1 | Out-String

    $avgMatch = [regex]::Match($output, 'Average R2 \(finite\):\s*([\d\.]+)')
    $avgR2 = if ($avgMatch.Success) { [double]$avgMatch.Groups[1].Value } else { [double]::NaN }

    $perMatch = [regex]::Match($output, 'Per-sample R2:\s*\[([^\]]+)\]')
    $minR2 = [double]::NaN; $maxR2 = [double]::NaN
    if ($perMatch.Success) {
        $nums = $perMatch.Groups[1].Value -split '\s+' | Where-Object { $_ -match '^-?[\d\.]+$' } | ForEach-Object { [double]$_ }
        if ($nums.Count -gt 0) {
            $minR2 = ($nums | Measure-Object -Minimum).Minimum
            $maxR2 = ($nums | Measure-Object -Maximum).Maximum
        }
    }

    Write-Host "  Avg R2: $avgR2" -ForegroundColor Green
    Write-Host ""

    $results += [PSCustomObject]@{
        No    = $i
        Label = $exp.label
        Args  = $exp.args
        AvgR2 = [math]::Round($avgR2, 6)
        MaxR2 = [math]::Round($maxR2, 6)
        MinR2 = [math]::Round($minR2, 6)
    }
    $i++
}

# Sort by AvgR2 descending
$sorted = $results | Sort-Object AvgR2 -Descending

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  SWEEP RESULTS (sorted by Avg R2)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
$sorted | Format-Table -AutoSize -Property No, AvgR2, MaxR2, MinR2, Label

$best = $sorted[0]
Write-Host "Best sweep config:" -ForegroundColor Green
Write-Host "  Label  : $($best.Label)" -ForegroundColor Green
Write-Host "  Command: python models\cgan_cnn.py --epochs $SweepEpochs $($best.Args)" -ForegroundColor Green
Write-Host "  Avg R2 : $($best.AvgR2)" -ForegroundColor Green

# Write sweep results to log
$logPath = Join-Path (Split-Path $PSScriptRoot -Parent) "training_log.md"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$mdLines = [System.Collections.Generic.List[string]]::new()
$mdLines.Add("")
$mdLines.Add("---")
$mdLines.Add("")
$mdLines.Add("## cGAN Parameter Sweep (epochs=$SweepEpochs) -- $timestamp")
$mdLines.Add("")
$mdLines.Add("| No | Avg R2 | Max R2 | Min R2 | Configuration |")
$mdLines.Add("|---|---|---|---|---|")
foreach ($r in $sorted) {
    $mdLines.Add("| $($r.No) | $($r.AvgR2) | $($r.MaxR2) | $($r.MinR2) | $($r.Label) |")
}
$mdLines.Add("")
$mdLines.Add("**Best (sweep):** ``python cgan_cnn.py $($best.Args)`` -- Avg R2 = $($best.AvgR2)")
$mdLines.Add("")
[System.IO.File]::AppendAllLines($logPath, $mdLines, [System.Text.Encoding]::UTF8)
Write-Host "Sweep results written to training_log.md" -ForegroundColor Cyan

# Full training with best config
Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "  FINAL TRAINING with best config" -ForegroundColor Magenta
Write-Host "  epochs=$FinalEpochs  $($best.Args)" -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta

$finalArgList = ("--epochs $FinalEpochs " + $best.Args) -split '\s+' | Where-Object { $_ -ne '' }
python models\cgan_cnn.py $finalArgList

Write-Host "`nAll done! Check training_log.md for full results." -ForegroundColor Cyan