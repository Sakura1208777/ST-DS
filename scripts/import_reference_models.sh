#!/usr/bin/env bash
set -e

MODE=${1:-submodule}
ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$ROOT_DIR"

echo "[INFO] Import mode: $MODE"

declare -A REPOS
REPOS["]=""

# Core models
REPOS[ImagenTime]="https://github.com/azencot-group/ImagenTime.git"
REPOS[T2S]="https://github.com/WinfredGe/T2S.git"
REPOS[ImagenI2R]="https://github.com/azencot-group/ImagenI2R.git"
REPOS[Times2D]="https://github.com/Tims2D/Times2D.git"
REPOS[VisionTS]="https://github.com/Keytoyze/VisionTS.git"
REPOS[TATO]="https://github.com/thulab/TATO.git"
REPOS[FlowTS]="https://github.com/UNITES-Lab/FlowTS.git"
REPOS[TabDiT]="https://github.com/fabriziogaruti/TabDiT.git"
REPOS[RAE]="https://github.com/bytetriper/RAE.git"
REPOS[MeanFlow]="https://github.com/haidog-yaqub/MeanFlow.git"
REPOS[ReMeanFlow]="https://github.com/Xinxi-Zhang/Re-MeanFlow.git"

IMPORT_DIR="reference_models"
mkdir -p "$IMPORT_DIR"

for name in "${!REPOS[@]}"; do
    url=${REPOS[$name]}
    target="$IMPORT_DIR/$name"

    echo "[INFO] Processing $name"

    if [ "$MODE" == "submodule" ]; then
        if [ ! -d "$target" ]; then
            git submodule add "$url" "$target" || echo "[WARN] submodule add failed for $name"
        else
            echo "[SKIP] $name already exists"
        fi
    elif [ "$MODE" == "vendor" ]; then
        tmp_dir="/tmp/$name"
        rm -rf "$tmp_dir"
        git clone --depth 1 "$url" "$tmp_dir"
        rm -rf "$tmp_dir/.git"
        rm -rf "$target"
        cp -r "$tmp_dir" "$target"
    else
        echo "[ERROR] Unknown mode: $MODE"
        exit 1
    fi
done

git add .gitmodules reference_models scripts || true
git commit -m "Import reference models via $MODE mode" || true
git push || true

echo "[DONE] Import completed"
