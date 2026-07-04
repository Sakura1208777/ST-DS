# Reference model repositories for ImagenTime improvement

This directory records and imports the external reference implementations for the planned ImagenTime improvement work.

Target problems:

1. Adaptive time-series length to image / latent representation.
2. Preserving temporal relationships after TS2Image conversion.
3. Replacing the older, slower diffusion backbone with newer flow / DiT / one-step generation methods.

## Recommended storage mode

Use `submodule` mode first. It keeps the repository light and preserves the exact upstream source location.

Use `vendor` mode only if you explicitly need the full source code copied into this repository. Vendor mode can make the repository large and may introduce license obligations from each upstream project.

## How to import locally

From the root of this repository:

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts/import_reference_models.ps1 -Mode submodule
```

To copy full source trees instead of adding submodules:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/import_reference_models.ps1 -Mode vendor
```

### Linux / macOS / Git Bash

```bash
bash scripts/import_reference_models.sh submodule
```

To copy full source trees:

```bash
bash scripts/import_reference_models.sh vendor
```

## Repository list

| Group | Model / method | Repository | Main use |
|---|---|---|---|
| Baseline | ImagenTime | https://github.com/azencot-group/ImagenTime.git | Original TS2Image + image diffusion baseline |
| Adaptive length / latent | T2S | https://github.com/WinfredGe/T2S.git | Length-adaptive VAE, Flow Matching, DiT for arbitrary-length time series |
| Mask-aware imaging | ImagenI2R | https://github.com/azencot-group/ImagenI2R.git | Completion + mask-aware diffusion for irregular time series |
| 1D to 2D representation | Times2D | https://github.com/Tims2D/Times2D.git | Multi-period decomposition and derivative heatmaps |
| Visual time-series representation | VisionTS | https://github.com/Keytoyze/VisionTS.git | Time series as visual reconstruction / visual foundation backbone |
| Adaptive transform selection | TATO | https://github.com/thulab/TATO.git | Adaptive transformation optimization for time-series foundation models |
| Efficient generation | FlowTS | https://github.com/UNITES-Lab/FlowTS.git | Rectified Flow for time-series generation |
| Variable-length DiT | TabDiT | https://github.com/fabriziogaruti/TabDiT.git | DiT-style generation for variable-length tabular time series |
| Modern DiT latent | RAE | https://github.com/bytetriper/RAE.git | Representation Autoencoder + DiT reference |
| One-step generation | MeanFlow | https://github.com/haidog-yaqub/MeanFlow.git | MeanFlow one-step generation reference |
| One-step flow upgrade | Re-MeanFlow | https://github.com/Xinxi-Zhang/Re-MeanFlow.git | Rectified MeanFlow / one-step flow reference |

## Suggested reading order

1. `ImagenTime`: understand the original `img_transformations`, `model`, and sampler design.
2. `T2S`: study length-adaptive VAE and Flow Matching + DiT.
3. `Times2D` and `ImagenI2R`: design adaptive TS2Image, derivative heatmaps, mask/meta maps.
4. `VisionTS` and `TATO`: design visual time-series representation and adaptive transform selection.
5. `FlowTS`, `TabDiT`, `RAE`, `MeanFlow`, `Re-MeanFlow`: design efficient Flow/DiT/one-step generation backbone.

## Notes

- Do not directly mix these repositories into the main model code. Keep them under `reference_models/` and copy only the needed modules into a clean implementation branch.
- Check each upstream license before vendoring code into this repository.
- Prefer adding submodules for research reference; prefer vendoring only when modifying a small subset of files with attribution.
