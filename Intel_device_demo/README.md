# Intel Device Demo

本文件夹主要辅助开发者 在 Intel 设备上加速部署 ChatGLM3-6B 模型。

## 1. 硬件准备
本文件夹中的设备支持列表包括：
- Intel CPU 系列, 包含个人CPU 和 服务器 / 工作站 CPU
- Intel Arc 独立显卡系列，包括 Arc A770 等显卡。
- Intel CPU 核显系列
- 其他理论支持 OpenVINO 加速的Intel 工具套件。

## 2. 文件目录
- IPEX_llm_xxx_demo: IPEX-LLM 是一个为Intel XPU(Xeon/Core/Flex/Arc/PVC)打造的低精度轻量级大语言模型库，在Intel平台上具有广泛的模型支持、最低的延迟和最小的内存占用，实现加速模型部署示例。
- OpenVINO_demo: 使用 Intel OpenVINO 推理加速框架，实现加速模型部署示例。
- Pytorch_demo (暂未推出) : 使用 Intel Pytorch Extension 实现在 Pytorch 环境上开发（适用于 Intel Arc 系列 GPU）

