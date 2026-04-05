# Auto-generate multiple sample sets without deleting old files.
# For each sample count:
# 1) run thermal_prediction.py to generate variants in a dedicated folder
# 2) run process_json_to_grid.py --split to create train/test npy files in that folder

param(
    [int[]]$VariantCounts = @(40, 60, 80, 100, 120, 140, 160),
    [double]$TrainRatio = 0.8,
    [int]$GridSize = 200,
    [int]$MaxShift = 50,
    [int]$Xi = 200,
    [int]$Yi = 200,
    [int]$Subsample = 2,
    [int]$CganEpochs = 2000
)

function Get-AvgR2FromOutput($outputText) {
    $match = [regex]::Match($outputText, 'Average R2 \(finite\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    if ($match.Success) {
        return [double]$match.Groups[1].Value
    }
    return [double]::NaN
}

$workspace = Split-Path $PSScriptRoot -Parent
$baseOut = Join-Path $workspace "data\thermal_analysis_output_sets"
if (-not (Test-Path $baseOut)) {
    New-Item -ItemType Directory -Path $baseOut | Out-Null
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Auto Sample Set Generation" -ForegroundColor Cyan
Write-Host "  Counts: $($VariantCounts -join ', ')" -ForegroundColor Cyan
Write-Host "  Train ratio: $TrainRatio" -ForegroundColor Cyan
Write-Host "  POD best  : --svd-method svd --rank 20 --rbf-kernel thin_plate_spline" -ForegroundColor Cyan
Write-Host "  cGAN best : --lr 0.00005 --batch-size 4 --adv-weight 0.001 --epochs $CganEpochs" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

$results = @()
$idx = 1

foreach ($n in $VariantCounts) {
    $setName = "samples_$n"
    $outFolder = Join-Path $baseOut $setName
    if (-not (Test-Path $outFolder)) {
        New-Item -ItemType Directory -Path $outFolder | Out-Null
    }

    Write-Host "[$idx/$($VariantCounts.Count)] Running set: $setName" -ForegroundColor Yellow
    Write-Host "  Output folder: $outFolder"

    # Step 1: Generate variants (JSON + PNG)
    Write-Host "  -> thermal_prediction.py" -ForegroundColor DarkGray
    & python "simulation\thermal_prediction.py" --output $outFolder --grid-size $GridSize --max-shift $MaxShift --variants $n 2>&1 | Out-Null

    # Step 2: Process and split into train/test npy
    Write-Host "  -> process_json_to_grid.py --split" -ForegroundColor DarkGray
    & python "preprocessing\process_json_to_grid.py" $outFolder --split --train-ratio $TrainRatio --xi $Xi --yi $Yi --subsample $Subsample 2>&1 | Out-Null

    # Count generated numeric json files
    $jsonCount = (Get-ChildItem $outFolder -File -Filter "*.json" | Where-Object { $_.BaseName -match '^-?\d+(,-?\d+)*$' } | Measure-Object).Count

    # Verify npy outputs
    $trainParams = Join-Path $outFolder "training data\params_training.npy"
    $trainTemps  = Join-Path $outFolder "training data\temps_training.npy"
    $testParams  = Join-Path $outFolder "test data\params_testing.npy"
    $testTemps   = Join-Path $outFolder "test data\temps_testing.npy"

    $ok = (Test-Path $trainParams) -and (Test-Path $trainTemps) -and (Test-Path $testParams) -and (Test-Path $testTemps)

    $nTrain = [math]::Max(1, [int]($jsonCount * $TrainRatio))
    if ($jsonCount -gt 1) {
        $nTrain = [math]::Min($nTrain, $jsonCount - 1)
    } else {
        $nTrain = $jsonCount
    }
    $nTest = $jsonCount - $nTrain
    if (($nTest -eq 0) -and ($jsonCount -gt 1)) {
        $nTest = 1
        $nTrain = $jsonCount - 1
    }

    $podAvgR2 = [double]::NaN
    $cganAvgR2 = [double]::NaN

    if ($ok) {
        # Step 3: Train/eval POD with best parameters
        Write-Host "  -> pod.py (best params)" -ForegroundColor DarkGray
        $podOutput = & python "models\pod.py" `
            --train-params $trainParams `
            --train-temps $trainTemps `
            --test-params $testParams `
            --test-temps $testTemps `
            --svd-method "svd" `
            --rank "20" `
            --rbf-kernel "thin_plate_spline" 2>&1 | Out-String
        $podAvgR2 = Get-AvgR2FromOutput $podOutput

        # Step 4: Train/eval cGAN with best parameters
        Write-Host "  -> cgan_cnn.py (best params)" -ForegroundColor DarkGray
        $cganOutput = & python "models\cgan_cnn.py" `
            --train-params $trainParams `
            --train-temps $trainTemps `
            --test-params $testParams `
            --test-temps $testTemps `
            --epochs $CganEpochs `
            --lr "0.00005" `
            --batch-size "4" `
            --adv-weight "0.001" 2>&1 | Out-String
        $cganAvgR2 = Get-AvgR2FromOutput $cganOutput
    }

    $results += [PSCustomObject]@{
        SetName    = $setName
        Requested  = $n
        JsonCount  = $jsonCount
        TrainCount = $nTrain
        TestCount  = $nTest
        TrainRatio = $TrainRatio
        NpyReady   = $ok
        POD_AvgR2  = if ([double]::IsNaN($podAvgR2)) { "NaN" } else { [math]::Round($podAvgR2, 6) }
        CGAN_AvgR2 = if ([double]::IsNaN($cganAvgR2)) { "NaN" } else { [math]::Round($cganAvgR2, 6) }
        Folder     = $outFolder
    }

    Write-Host "  Done: json=$jsonCount, train=$nTrain, test=$nTest, npyReady=$ok, POD=$podAvgR2, cGAN=$cganAvgR2" -ForegroundColor Green
    Write-Host ""
    $idx++
}

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
$results | Format-Table -AutoSize -Property SetName, Requested, JsonCount, TrainCount, TestCount, TrainRatio, NpyReady, POD_AvgR2, CGAN_AvgR2

Write-Host ""
Write-Host "All sets finished. No old files were deleted." -ForegroundColor Green
