param(
    [ValidateSet("submodule","vendor")]
    [string]$Mode = "submodule"
)

$Root = git rev-parse --show-toplevel
Set-Location $Root

Write-Host "[INFO] Import mode: $Mode"

$repos = @{
    "ImagenTime" = "https://github.com/azencot-group/ImagenTime.git"
    "T2S" = "https://github.com/WinfredGe/T2S.git"
    "ImagenI2R" = "https://github.com/azencot-group/ImagenI2R.git"
    "Times2D" = "https://github.com/Tims2D/Times2D.git"
    "VisionTS" = "https://github.com/Keytoyze/VisionTS.git"
    "TATO" = "https://github.com/thulab/TATO.git"
    "FlowTS" = "https://github.com/UNITES-Lab/FlowTS.git"
    "TabDiT" = "https://github.com/fabriziogaruti/TabDiT.git"
    "RAE" = "https://github.com/bytetriper/RAE.git"
    "MeanFlow" = "https://github.com/haidog-yaqub/MeanFlow.git"
    "ReMeanFlow" = "https://github.com/Xinxi-Zhang/Re-MeanFlow.git"
}

$targetDir = "reference_models"
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

foreach ($name in $repos.Keys) {
    $url = $repos[$name]
    $path = Join-Path $targetDir $name

    Write-Host "[INFO] Processing $name"

    if ($Mode -eq "submodule") {
        if (-not (Test-Path $path)) {
            git submodule add $url $path
        } else {
            Write-Host "[SKIP] $name already exists"
        }
    }
    elseif ($Mode -eq "vendor") {
        $tmp = Join-Path $env:TEMP $name
        if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }

        git clone --depth 1 $url $tmp
        Remove-Item -Recurse -Force (Join-Path $tmp ".git")

        if (Test-Path $path) { Remove-Item -Recurse -Force $path }
        Copy-Item -Recurse $tmp $path
    }
}

git add .gitmodules reference_models scripts

git commit -m "Import reference models via $Mode mode" | Out-Null
git push

Write-Host "[DONE] Import completed"
