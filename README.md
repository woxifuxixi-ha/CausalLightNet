# CausalLight
## 环境与安装

见 [环境说明](docs/ENVIRONMENT.md)。从本目录运行命令：

```bash
conda create -n causallight python=3.10 -y
conda activate causallight
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python main.py --help
python -m unittest test_leadtime_ablation.py test_density_ablation.py
```

## 训练

以下为 Bash 命令；将 `/path/to/h5_events` 替换为实际目录。

```bash
for mode in full no_causal no_spatial no_focal; do
  folder="$mode"
  if [ "$mode" = "full" ]; then folder="FULL"; fi
  python main.py --mode "$mode" --data-dir /path/to/h5_events \
    --output-dir "XR/$folder" --device cuda:0 --no-case-vis || break
done
```

默认历史闪电12帧，其他模态6帧，输出6帧，128×128；Adam、lr=1e-4、batch=4、50 epochs。
已有有效权重时无需重训。每个模型需保留验证集 maxETS 权重及 epoch 元数据文件。

## 共享推理与实验评价

```bash
python evaluate_leadtime_ablation.py --data-dir /path/to/h5_events \
  --weights-root XR --device cuda:0 --batch-size 4 --workers 4 \
  --include-empty-targets --save-predictions --output-dir results/shared
```

输出逐提前量 ETS/POD/FAR、三面板曲线图、样本清单与四版本配对概率缓存。
GPU 选择示例：命令前加 `CUDA_VISIBLE_DEVICES=1`，程序内仍用 `--device cuda:0`。
Windows 初次运行建议 `--workers 0`。输出目录必须为空。

密度实验要求训练集确定分界；运行训练集定界工具后复用上述预测：

```bash
python fit_density_boundaries.py --data-dir /path/to/h5_events --output density_boundaries.json
python evaluate_density_ablation.py --prediction-dir results/shared \
  --boundaries-json density_boundaries.json --output-dir results/density
```

完整定义、零闪电组处理、选模规则和限制见 [评价协议](docs/EVALUATION.md)。

