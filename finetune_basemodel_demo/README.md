# ChatGLM3-6B-base 微调示例

本目录提供 ChatGLM3-6B-base 模型的微调示例，包括Lora 微调。格式上，请使用`alpaca`数据集。
例如
```bash

{"context": "hello", "target": "hi,I am ChatGLM3"}

```

+ 如果将模型下载到了本地，本文和代码中的 `THUDM/chatglm3-6b-base` 字段均应替换为相应地址以从本地加载模型。

+ 请注意，运行本脚本，你还需要安装本目录下的 `requirements.txt` 中的所有内容。

+ 此Readme还在完善阶段，将在后续中不断更新。
