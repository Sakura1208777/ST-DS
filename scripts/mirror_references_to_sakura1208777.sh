#!/usr/bin/env bash
set -e

OWNER="${1:-Sakura1208777}"
WORKDIR="${2:-/tmp/sakura_reference_mirror}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# This script mirrors upstream reference repositories into the GitHub namespace OWNER.
# It skips repositories that already exist under OWNER.
# Requirements:
#   1. git installed
#   2. GitHub CLI installed and authenticated: gh auth login
#   3. Permission to create repositories under OWNER
#
# Usage:
#   bash scripts/mirror_references_to_sakura1208777.sh Sakura1208777

repos=(
  "ImagenTime https://github.com/azencot-group/ImagenTime.git"
  "T2S https://github.com/WinfredGe/T2S.git"
  "ImagenI2R https://github.com/azencot-group/ImagenI2R.git"
  "Times2D https://github.com/Tims2D/Times2D.git"
  "VisionTS https://github.com/Keytoyze/VisionTS.git"
  "TATO https://github.com/thulab/TATO.git"
  "FlowTS https://github.com/UNITES-Lab/FlowTS.git"
  "TabDiT https://github.com/fabriziogaruti/TabDiT.git"
  "RAE https://github.com/bytetriper/RAE.git"
  "MeanFlow https://github.com/haidog-yaqub/MeanFlow.git"
  "Re-MeanFlow https://github.com/Xinxi-Zhang/Re-MeanFlow.git"
)

for item in "${repos[@]}"; do
  name=$(echo "$item" | awk '{print $1}')
  url=$(echo "$item" | awk '{print $2}')

  echo "============================================================"
  echo "[INFO] Checking $OWNER/$name"

  if gh repo view "$OWNER/$name" >/dev/null 2>&1; then
    echo "[SKIP] $OWNER/$name already exists."
    continue
  fi

  echo "[INFO] Creating $OWNER/$name"
  gh repo create "$OWNER/$name" --public --description "Mirrored reference implementation for ImagenTime improvement: $name"

  rm -rf "$name.git"
  git clone --mirror "$url" "$name.git"
  cd "$name.git"

  echo "[INFO] Pushing mirror to https://github.com/$OWNER/$name.git"
  git push --mirror "https://github.com/$OWNER/$name.git"

  cd "$WORKDIR"
  rm -rf "$name.git"
done

echo "[DONE] All missing repositories have been mirrored to $OWNER."
