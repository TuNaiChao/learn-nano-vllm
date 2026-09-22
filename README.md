<p align="center">
<img width="300" src="assets/logo.png">
</p>

<h1 align="center">Nano-vLLM · 从零入门（学习增强版）</h1>

<p align="center">
一个从零构建的轻量级 vLLM 实现，配套一套面向零基础的中文深度学习文档。
</p>

<p align="center">
<a href="https://github.com/GeeeekExplorer/nano-vllm" target="_blank">上游项目 GeeeekExplorer/nano-vllm</a> · <a href="./docs/README.md">📘 学习文档入口</a> · <a href="./LICENSE">MIT License</a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

---

## 📖 这个仓库是什么

本仓库是 **[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** 的 fork。原项目由 [Xingkai Yu](https://github.com/GeeeekExplorer) 用约 **1,200 行 Python** 从零复刻了 vLLM 的核心推理流程。

本 fork **完整保留了原项目的全部代码与功能**，并在此基础上新增了 `docs/` 目录——一套面向零基础、配套源码逐行精讲的中文学习文档，帮助你从"这是什么"一路学到"能亲手改代码"。

> 如果你只是想使用推理引擎，建议直接使用[上游项目](https://github.com/GeeeekExplorer/nano-vllm)；如果你想**读懂它、改它、搞懂 LLM 推理引擎的底层原理**，本仓库的学习文档正是为此而生。
>
> 📌 学习文档基于上游提交 `bb823b3`（chunked prefill 版，2026-04-26）撰写；若上游继续更新，一切以源码为准。

---

## 🆕 本仓库相比上游的增量

| 内容 | 说明 | 入口 |
|---|---|---|
| 📘 **中文学习文档**（共 10 篇） | 零基础友好：大量类比 + 可视化图 + 硬核数学推导 + 逐行代码精讲 | [docs/README.md](./docs/README.md) |
| ⚙️ 推理引擎代码 | 原项目代码，**未做改动** | [nanovllm/](./nanovllm/) |

学习文档涵盖从导论、全局架构、AI Infra 理论（RoPE / KV Cache / PagedAttention / Flash Attention / Tensor Parallelism / CUDA Graph 等）、Python 语法清单，到逐文件代码精讲与动手实践，并提供「系统精读 / 快速了解 / 面试速查」三条学习路径。详见 [📘 学习文档总览](./docs/README.md)。

---

## 🙏 致谢

本仓库的推理引擎代码全部来自上游项目 **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**，原作者 **Xingkai Yu**，在此致以诚挚感谢。如果这个项目对你有帮助，欢迎前往 [原项目仓库](https://github.com/GeeeekExplorer/nano-vllm) 点个 ⭐。

---

以下内容来自上游原项目 README，保留原样以方便使用。

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

---

## 📄 License

本项目遵循 [MIT License](./LICENSE)。

- 推理引擎代码（`nanovllm/`、`example.py`、`bench.py` 等）版权归上游原作者 **Xingkai Yu** 所有。
- 本仓库新增的学习文档（`docs/`）同样基于 MIT 协议开源，欢迎学习与传播。

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
