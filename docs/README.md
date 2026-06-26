# 📘 nano-vllm 从零入门 · 学习文档

> 一套面向**零基础**的 nano-vllm（轻量级 LLM 推理引擎）深度学习文档。用**大量类比 + 可视化图 + 硬核数学推导 + 逐行代码精讲**，带你从"这是什么"一路学到"能亲手改代码"。
>
> 配套项目：[nano-vllm](../README.md)（~1200 行复刻 vLLM 核心的推理引擎）

---

## 🗺️ 系列地图

```
第0部分  导论 ──── 搞清楚是什么、跑起来
   │
第1部分  全局视角 ── 建立地图（先见森林）
   │
第2部分  AI Infra 理论（重头戏，数学密集）
   ├─ 2a 模型本体（RoPE/RMSNorm/SwiGLU/Attention）
   ├─ 2b 引擎机制（KV cache/PagedAttention/Prefix Cache/调度）
   └─ 2c 加速采样（Flash Attn/Tensor Parallelism/CUDA Graph/采样）
   │
第3部分  Python 语法清单 ── 读懂代码所需的进阶语法
   │
第4部分  逐文件精讲（理论落地代码）
   ├─ 04a 入口与引擎层
   ├─ 04b 执行层与模型层（model_runner + qwen3）
   └─ 04c 算子与工具层
   │
第5部分  动手实践 ──── 跑、看、改
```

---

## 📑 完整目录

| 文档 | 内容 | 核心知识点 | 对应代码 | 难度 |
|---|---|---|---|---|
| **[00 导论](00-导论.md)** | 项目定位、前置知识、跑通环境、阅读路线 | 依赖解读（torch/triton/flash-attn/xxhash/safetensors） | 全局 | ⭐ |
| **[01 全局视角](01-全局视角.md)** | 三层架构、请求生命周期、核心名词字典 | Prefill/Decode、KV Cache、PagedAttention、Continuous Batching（直觉+数学） | 全局 | ⭐⭐ |
| **[02a 模型本体理论](02a-模型本体理论.md)** | 模型本体每项技术的数学原理 | **RoPE 复数推导**、RMSNorm、SwiGLU、GQA、q/k_norm | [qwen3.py](../nanovllm/models/qwen3.py)、[layers/](../nanovllm/layers/) | ⭐⭐⭐⭐ |
| **[02b 引擎核心机制](02b-引擎核心机制.md)** | 引擎层的数据结构与算法 | KV cache 形态、PagedAttention 三件套、**链式哈希 Prefix Cache**、调度三连 | [block_manager.py](../nanovllm/engine/block_manager.py)、[scheduler.py](../nanovllm/engine/scheduler.py) | ⭐⭐⭐⭐ |
| **[02c 加速并行与采样](02c-加速并行与采样.md)** | 加速黑科技的数学 | **Flash Attention IO 分析**、**Tensor Parallelism 通信量推导**、**CUDA Graph 开销模型**、Gumbel-max、显存预算 | [attention.py](../nanovllm/layers/attention.py)、[linear.py](../nanovllm/layers/linear.py)、[model_runner.py](../nanovllm/engine/model_runner.py)、[sampler.py](../nanovllm/layers/sampler.py) | ⭐⭐⭐⭐⭐ |
| **[03 Python 语法清单](03-Python语法清单.md)** | 读懂项目所需的 Python 进阶语法 | dataclass/slots、魔术方法、`__getstate__`、lru_cache、多进程、反射、for-else | 散落各文件 | ⭐⭐⭐ |
| **[04a 入口与引擎层](04a-入口与引擎层.md)** | 逐函数精讲接口与引擎 | Config、Sequence、BlockManager、Scheduler、LLMEngine | [engine/](../nanovllm/engine/)、入口 | ⭐⭐⭐ |
| **[04b 执行与模型层](04b-执行层与模型层.md)** | 拆最复杂的 model_runner | slot_mapping 跨块计算、KV cache 分配、CUDA Graph 录制、多进程通信 | [model_runner.py](../nanovllm/engine/model_runner.py)、[qwen3.py](../nanovllm/models/qwen3.py) | ⭐⭐⭐⭐⭐ |
| **[04c 算子与工具层](04c-算子与工具层.md)** | 算子实现与工具 | Triton kernel、各种并行 Linear 的切分、VocabEmbedding mask、权重加载 | [layers/](../nanovllm/layers/)、[utils/](../nanovllm/utils/) | ⭐⭐⭐⭐ |
| **[05 动手实践](05-动手实践.md)** | 跑、看、改 | 加日志观察调度、改参数实验、**给采样器加 top-k** | 实操 | ⭐⭐⭐ |

---

## 🎯 三条学习路径

### 路径 A：系统精读（推荐，约 1–2 周）

按编号顺序读：`00 → 01 → 02a → 02b → 02c → 03 → 04a → 04b → 04c → 05`。
配合源码：每读完一篇理论（第 2 部分），对照读相应代码（第 4 部分）。

### 路径 B：快速了解（半天）

只读 `00 → 01 → 05`。建立全局印象 + 动手跑，知道 nano-vllm 在干嘛。

### 路径 C：面试速查

主攻理论三篇：`01 → 02a → 02b → 02c`。每篇的公式推导（RoPE、算术强度、TP 通信量、Gumbel-max）都是高频考点。

---

## 📋 前置要求

- **会 Python**（函数、类、列表字典）。进阶语法不会没关系，第 3 部分全讲。
- **大致知道 Transformer 是个模型结构**。细节不会没关系，第 2a 部分从零讲。
- **一台 NVIDIA GPU**（≥6GB 显存，跑 Qwen3-0.6B）。环境配置见 [00 导论](00-导论.md)。

---

## 💡 怎么用这套文档

1. **边读边跑**：每篇遇到代码，打开对应文件对照看。文件引用都是可点击的相对链接。
2. **先建地图**：第 1 部分那张"总地图"是全系列的锚点，迷路了就回去看。
3. **理论配代码**：第 2 部分（理论）和第 4 部分（代码）是配对的，交叉着读。
4. **动手才算懂**：第 5 部分的"加日志看调度"和"加 top-k"强烈建议亲手做。

---

## 🏷️ 风格说明

- 🍳 **类比打底**：每个抽象概念先有生活类比（餐厅、虚拟内存、闹钟竞争）。
- 🗺️ **可视化**：大量 ASCII/Mermaid 图、表格、时序图。
- 🧮 **硬核数学**：公式推导 + Roofline 量化判据 + 真实数值代入。
- 🔗 **代码对应**：每个知识点标注对应文件和函数，可点击跳转。
- 📍 **进度感**：每篇末尾有"你在哪、下一步去哪"。

---

> 如果这套文档对你有帮助，欢迎在原项目 [nano-vllm](../README.md) 点个 ⭐。
> 发现错误或想补充内容，欢迎反馈。
