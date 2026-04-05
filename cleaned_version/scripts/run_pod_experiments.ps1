# POD Parameter Experiment Runner
# Runs all combinations and prints a comparison table at the end

$experiments = @(
    @{ label="baseline (randomized_svd, auto, rank=auto)";   args="" },
    @{ label="svd, auto kernel, rank=auto";                  args="--svd-method svd" },
    @{ label="svd, auto kernel, rank=10";                    args="--svd-method svd --rank 10" },
    @{ label="svd, auto kernel, rank=15";                    args="--svd-method svd --rank 15" },
    @{ label="svd, auto kernel, rank=20";                    args="--svd-method svd --rank 20" },
    @{ label="svd, auto kernel, rank=30";                    args="--svd-method svd --rank 30" },
    @{ label="svd, thin_plate_spline, rank=auto";            args="--svd-method svd --rbf-kernel thin_plate_spline" },
    @{ label="svd, thin_plate_spline, rank=20";              args="--svd-method svd --rank 20 --rbf-kernel thin_plate_spline" },
    @{ label="svd, multiquadric, rank=auto";                 args="--svd-method svd --rbf-kernel multiquadric" },
    @{ label="svd, multiquadric, rank=20";                   args="--svd-method svd --rank 20 --rbf-kernel multiquadric" },
    @{ label="svd, cubic, rank=auto";                        args="--svd-method svd --rbf-kernel cubic" },
    @{ label="randomized_svd, thin_plate_spline, rank=auto"; args="--rbf-kernel thin_plate_spline" },
    @{ label="randomized_svd, multiquadric, rank=auto";      args="--rbf-kernel multiquadric" }
)

$results = @()

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  POD Parameter Experiment Runner" -ForegroundColor Cyan
Write-Host "  Total experiments: $($experiments.Count)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$i = 1
foreach ($exp in $experiments) {
    Write-Host "[$i/$($experiments.Count)] Running: $($exp.label)" -ForegroundColor Yellow
    Write-Host "  Command: python models\pod.py $($exp.args)"

    if ($exp.args -eq "") {
        $output = python models\pod.py 2>&1 | Out-String
    } else {
        $output = python models\pod.py ($exp.args -split ' ') 2>&1 | Out-String
    }

    $avgMatch = [regex]::Match($output, 'Average R2 \(finite\):\s*([\d\.]+)')
    if ($avgMatch.Success) {
        $avgR2 = [double]$avgMatch.Groups[1].Value
    } else {
        $avgR2 = [double]::NaN
    }

    $perMatch = [regex]::Match($output, 'Per-sample R2:\s*\[([^\]]+)\]')
    $minR2 = [double]::NaN
    $maxR2 = [double]::NaN
    if ($perMatch.Success) {
        $nums = $perMatch.Groups[1].Value -split '\s+' | Where-Object { $_ -match '^-?[\d\.]+$' } | ForEach-Object { [double]$_ }
        if ($nums.Count -gt 0) {
            $minR2 = ($nums | Measure-Object -Minimum).Minimum
            $maxR2 = ($nums | Measure-Object -Maximum).Maximum
        }
    }

    Write-Host "  Average R2: $avgR2" -ForegroundColor Green
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

$sorted = $results | Sort-Object AvgR2 -Descending

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  RESULTS COMPARISON (sorted by Avg R2)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$sorted | Format-Table -AutoSize -Property No, AvgR2, MaxR2, MinR2, Label

$best = $sorted[0]
Write-Host "Best configuration:" -ForegroundColor Green
Write-Host "  Label  : $($best.Label)" -ForegroundColor Green
Write-Host "  Command: python models\pod.py $($best.Args)" -ForegroundColor Green
Write-Host "  Avg R2 : $($best.AvgR2)" -ForegroundColor Green
Write-Host "  Max R2 : $($best.MaxR2)" -ForegroundColor Green
Write-Host "  Min R2 : $($best.MinR2)" -ForegroundColor Green

$logPath = Join-Path (Split-Path $PSScriptRoot -Parent) "training_log.md"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$mdLines = [System.Collections.Generic.List[string]]::new()
$mdLines.Add("")
$mdLines.Add("---")
$mdLines.Add("")
$mdLines.Add("## POD Parameter Sweep -- $timestamp")
$mdLines.Add("")
$mdLines.Add("| No | Avg R2 | Max R2 | Min R2 | Configuration |")
$mdLines.Add("|---|---|---|---|---|")
foreach ($r in $sorted) {
    $mdLines.Add("| $($r.No) | $($r.AvgR2) | $($r.MaxR2) | $($r.MinR2) | $($r.Label) |")
}
$mdLines.Add("")
$mdLines.Add("**Best:** ``python pod.py $($best.Args)`` -- Avg R2 = $($best.AvgR2)")
$mdLines.Add("")

[System.IO.File]::AppendAllLines($logPath, $mdLines, [System.Text.Encoding]::UTF8)
Write-Host "`nComparison table appended to training_log.md" -ForegroundColor Cyan