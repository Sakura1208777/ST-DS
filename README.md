# ImagenTime / ST-DS ImagenTime

## English Version

## News

- (Sep 2025) Our papers, "A Diffusion Model for Regular Time Series Generation from Irregular Data" and "Time Series Generation Under Data Scarcity: A Unified Generative Modeling Approach", were accepted to NeurIPS 2025.
- (Sep 2025) We released ImagenI2R code: https://github.com/azencot-group/ImagenI2R
- (July 2025) We released ImagenFew code: https://github.com/azencot-group/ImagenFew
- (May 2025) We announced our new model, ImagenFew. Technical report: https://arxiv.org/pdf/2505.20446
- (September 2024) The ImagenTime paper was accepted to NeurIPS 2024.
- (November 2024) Conditional benchmarking is available for all datasets.

## Utilizing Image Transforms and Diffusion Models for Generative Modeling of Short and Long Time Series

Paper: https://arxiv.org/abs/2410.19538

![TS2IMG samples](visuals/ts2img.png)

## Overview

This project presents a generative modeling approach for time series data by transforming sequences into images. The method supports both short and long sequences and covers unconditional generation, interpolation, and extrapolation.

By using invertible transforms such as delay embedding and the short-time Fourier transform, the framework can process time series with different sequence lengths efficiently. ST-DS ImagenTime extends this codebase with additional spatiotemporal and dynamic-structure training schemes.

The code and benchmarks can be used to develop, evaluate, and compare new time-series generation methods.

## Setup

Download and set up the repository:

```bash
git clone https://github.com/azencot-group/ImagenTime.git
cd ImagenTime
```

Create the Conda environment:

```bash
conda env create -f requirements.yaml
conda activate ImagenTime
```

## Data

Download the prepared dataset package from:

```text
https://drive.google.com/drive/folders/11PXAj0RYei5MyXJVasikmYnEDK6V8awt?usp=share_link
```

Unzip the downloaded files into the empty `data` directory. The datasets in the package are already preprocessed according to the benchmark protocols.

Short datasets:

- Unconditional generation: Energy, MuJoCo, Stocks, Sine.
- Conditional generation: ETTh1, ETTh2, ETTm1, ETTm2.

Long datasets:

- Unconditional generation: FRED-MD, NN5 Daily, Temp Rain.
- Conditional generation: Physionet, USHCN.

Ultra-long datasets:

- Unconditional generation: Traffic, KDD-Cup.
- Conditional generation: Traffic, KDD-Cup.

If you use these datasets, please cite the sources referenced in the paper.

## Usage

Main scripts:

- `run_unconditional.py`: unconditional generation.
- `run_visualization.py`: visualization and metric evaluation for trained runs.
- `run_conditional.py`: conditional generation.

Configuration files are provided under `configs`.

### Unconditional Training and Evaluation

Basic command:

```bash
python run_unconditional.py --config ./configs/unconditional/<desired_dataset>.yaml
```

Run ST-DS ImagenTime with the 500-epoch F3 preset:

```bash
python run_unconditional.py --config ./configs/unconditional/<desired_dataset>.yaml --train_budget f3
```

Resume an interrupted unconditional run:

```bash
python run_unconditional.py --config ./configs/unconditional/<desired_dataset>.yaml --train_budget f3 --resume true
```

Baseline and ablation examples:

```bash
python run_unconditional.py --config ./configs/unconditional/<desired_dataset>.yaml --train_budget f3 --use_st_adapter false --use_ds_train false
python run_unconditional.py --config ./configs/unconditional/<desired_dataset>.yaml --train_budget f3 --use_st_adapter true --use_ds_train false
python run_unconditional.py --config ./configs/unconditional/<desired_dataset>.yaml --train_budget f3 --use_st_adapter false --use_ds_train true
```

### Conditional Training and Evaluation

```bash
python run_conditional.py --config ./configs/conditional/<interpolation_or_extrapolation>/<desired_dataset>.yaml
```

### Visualization and Metric Evaluation

The visualization script expects a trained model. Run training first, then evaluate:

```bash
python run_visualization.py --config ./configs/unconditional/<desired_dataset>.yaml
```

Evaluate a specific trained result by specifying both `--train_budget` and `--run_id`. The script restores the best checkpoint for that run when available and falls back to the latest checkpoint if the best checkpoint is missing.

```bash
python run_visualization.py --config ./configs/unconditional/<desired_dataset>.yaml --train_budget <scheme_name> --run_id <run_id>
```

Example:

```bash
python run_visualization.py --config ./configs/unconditional/energy.yaml --train_budget f3 --run_id 20260531-000001
```

If you intentionally evaluate an older checkpoint whose saved model keys differ from the current code, allow partial loading:

```bash
python run_visualization.py --config ./configs/unconditional/energy.yaml --train_budget f3 --run_id 20260531-000001 --allow_partial_resume true
```

## BibTeX

```bibtex
@article{naiman2024utilizing,
  title={Utilizing image transforms and diffusion models for generative modeling of short and long time series},
  author={Naiman, Ilan and Berman, Nimrod and Pemper, Itai and Arbiv, Idan and Fadlon, Gal and Azencot, Omri},
  journal={Advances in Neural Information Systems},
  volume={37},
  pages={121699--121730},
  year={2024}
}
```

---

# 中文版

## 最新消息

- 2025 年 9 月，论文 "A Diffusion Model for Regular Time Series Generation from Irregular Data" 和 "Time Series Generation Under Data Scarcity: A Unified Generative Modeling Approach" 被 NeurIPS 2025 接收。
- 2025 年 9 月，ImagenI2R 代码发布：https://github.com/azencot-group/ImagenI2R
- 2025 年 7 月，ImagenFew 代码发布：https://github.com/azencot-group/ImagenFew
- 2025 年 5 月，发布 ImagenFew 技术报告：https://arxiv.org/pdf/2505.20446
- 2024 年 9 月，ImagenTime 论文被 NeurIPS 2024 接收。
- 2024 年 11 月，所有数据集均支持条件生成 benchmark。

## 面向短序列和长序列生成建模的图像变换扩散模型

论文地址：https://arxiv.org/abs/2410.19538

![TS2IMG samples](visuals/ts2img.png)

## 项目概述

本项目通过将时间序列转换为图像来进行生成建模。该方法同时支持短序列和长序列，并覆盖无条件生成、插值和外推等任务。

模型使用 delay embedding 和短时傅里叶变换等可逆变换，将不同长度的时间序列统一到图像建模框架中。ST-DS ImagenTime 在原 ImagenTime 基础上加入了时空结构和动态结构相关的训练方案。

本代码库可以用于开发、评估和对比新的时间序列生成方法。

## 环境配置

下载仓库：

```bash
git clone https://github.com/azencot-group/ImagenTime.git
cd ImagenTime
```

创建 Conda 环境：

```bash
conda env create -f requirements.yaml
conda activate ImagenTime
```

## 数据集

数据集下载地址：

```text
https://drive.google.com/drive/folders/11PXAj0RYei5MyXJVasikmYnEDK6V8awt?usp=share_link
```

下载后将数据解压到空的 `data` 目录中。压缩包中的数据已经按照 benchmark 协议完成预处理。

短序列数据集：

- 无条件生成：Energy、MuJoCo、Stocks、Sine。
- 条件生成：ETTh1、ETTh2、ETTm1、ETTm2。

长序列数据集：

- 无条件生成：FRED-MD、NN5 Daily、Temp Rain。
- 条件生成：Physionet、USHCN。

超长序列数据集：

- 无条件生成：Traffic、KDD-Cup。
- 条件生成：Traffic、KDD-Cup。

如果使用这些数据集，请引用论文中对应的数据来源。

## 使用方法

主要脚本：

- `run_unconditional.py`：无条件生成训练与评估。
- `run_visualization.py`：对已训练结果进行可视化和指标评估。
- `run_conditional.py`：条件生成训练与评估。

配置文件位于 `configs` 目录。

### 无条件生成训练与评估

基础命令：

```bash
python run_unconditional.py --config ./configs/unconditional/<数据集>.yaml
```

使用 ST-DS ImagenTime 的 500 轮 F3 方案：

```bash
python run_unconditional.py --config ./configs/unconditional/<数据集>.yaml --train_budget f3
```

恢复中断的无条件生成训练：

```bash
python run_unconditional.py --config ./configs/unconditional/<数据集>.yaml --train_budget f3 --resume true
```

baseline 和消融实验示例：

```bash
python run_unconditional.py --config ./configs/unconditional/<数据集>.yaml --train_budget f3 --use_st_adapter false --use_ds_train false
python run_unconditional.py --config ./configs/unconditional/<数据集>.yaml --train_budget f3 --use_st_adapter true --use_ds_train false
python run_unconditional.py --config ./configs/unconditional/<数据集>.yaml --train_budget f3 --use_st_adapter false --use_ds_train true
```

### 条件生成训练与评估

```bash
python run_conditional.py --config ./configs/conditional/<interpolation_or_extrapolation>/<数据集>.yaml
```

### 可视化与指标评估

可视化脚本需要先有训练好的模型。训练完成后运行：

```bash
python run_visualization.py --config ./configs/unconditional/<数据集>.yaml
```

如果需要评估某一次指定的训练结果，请同时指定 `--train_budget` 和 `--run_id`。脚本会优先恢复该 run 下的 best checkpoint；如果 best checkpoint 不存在，则回退到 latest checkpoint。

```bash
python run_visualization.py --config ./configs/unconditional/<数据集>.yaml --train_budget <方案名> --run_id <运行编号>
```

示例：

```bash
python run_visualization.py --config ./configs/unconditional/energy.yaml --train_budget f3 --run_id 20260531-000001
```

如果补评较早的 checkpoint，且保存的模型参数键和当前代码不完全一致，可以显式允许部分加载：

```bash
python run_visualization.py --config ./configs/unconditional/energy.yaml --train_budget f3 --run_id 20260531-000001 --allow_partial_resume true
```

## 引用

```bibtex
@article{naiman2024utilizing,
  title={Utilizing image transforms and diffusion models for generative modeling of short and long time series},
  author={Naiman, Ilan and Berman, Nimrod and Pemper, Itai and Arbiv, Idan and Fadlon, Gal and Azencot, Omri},
  journal={Advances in Neural Information Systems},
  volume={37},
  pages={121699--121730},
  year={2024}
}
```
