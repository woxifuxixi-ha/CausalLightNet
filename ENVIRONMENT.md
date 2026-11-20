# 环境要求

## 参考环境

已有服务器评估记录：Ubuntu、PyTorch 2.1.2+cu118、NumPy 1.26.4，NVIDIA V100S 32 GB。
这不是已验证的最小硬件要求；batch size 与是否保存概率会影响显存、内存和磁盘使用。
建议新环境使用 Python 3.10。此 Python 建议版本不等同于历史运行版本记录。
requirements.txt 对辅助包采用版本范围，未声称是完整历史环境锁文件。

README 中的 PyTorch 安装命令选择 CUDA 11.8 构建；驱动需兼容所安装的 PyTorch CUDA 运行时。
CPU 调试可安装 CPU 构建并传 `--device cpu`：

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

不需要编译项目内 CUDA 扩展。本整理版明确采用原网络的普通 Conv2d 备用模块，
保留 `dcn_tm.conv.*` 参数名称。含真实 DCN 参数的其他权重不保证兼容，必须严格加载验证。

## 兼容性

- 评估加载器支持没有 `torch.serialization.safe_globals` 的旧版 PyTorch；此时常规反序列化仅用于可信权重。
- main3 原来的 ReduceLROnPlateau `verbose=True` 已去除，调度逻辑不变。
- 固定随机种子的设置仅存在于新增共享推理入口；原训练入口未固定种子，不保证逐次训练完全复现。
- DataLoader 默认训练使用4个 worker；共享评估可通过 `--workers` 设置，Windows 建议先用0。
- 测试仅验证局部逻辑与运行兼容性，不等同于在新环境完成50轮训练。

## 输出存储

共享预测保存四模型 float32 概率和标签；1140个128×128×6窗口的未压缩数组约2.1 GiB，
实际 NPZ 大小取决于压缩率。推理文件和模型权重默认在 .gitignore 中排除。
