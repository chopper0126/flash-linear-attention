# 学习计划：flash-linear-attention（MLA / KDA / GLA / GDN 专项）

> 项目路径：`/home/y00889327/flash-linear-attention`
> 学习周期：1–2 周，每天 2–3 小时
> 硬件：Linux / CPU + 8× Ascend 910B2C NPU（CANN 9.0.1）

## 0. 环境现状（两个坑）

| 项目 | 状态 |
|---|---|
| torch / triton | 2.9.0+cpu / 3.6.0 ✅ |
| NPU | 8× Ascend 910B2C，CANN 9.0.1，torch_npu 2.9.0 ✅ |
| **triton-ascend** | ⚠️ 3.2.0 装过但被后来装的纯 triton 3.6.0 **覆盖了**（`triton/backends/ascend` 已消失），需要重装 |
| **fla 本体** | 未安装，需 `pip install -e .` |

**两个关键认知：**

1. fla 的 kernel 全是 **Triton** 写的（`tl.dot` 等），**CPU 上跑不了真 kernel**。CPU 能跑的是 `naive.py` 参考实现（纯 PyTorch）——这恰好是学数学公式的最佳入口。真 kernel 要在 NPU 上跑。
2. 测试即教程：`tests/ops/test_*.py` 的每个用例都是「随机生成 q/k/v → naive 参考实现 vs Triton kernel → 数值对比」，**这就是"直观输入输出"**，不需要自己造数据。

### Day 0 环境修复（半天）

```bash
cd /home/y00889327/flash-linear-attention
pip install -e .                      # 装 fla
pip install triton-ascend==3.2.0      # 修复被覆盖的 ascend 补丁（会整体替换 triton 包）
# CPU 冒烟（naive 路径）
pytest tests/ops/test_gla.py -x -q
# NPU 冒烟（真 kernel，A2 芯片官方支持）
ASCEND_RT_VISIBLE_DEVICES=0 pytest tests/ops/test_gla.py -x -q
```

## 1. 路线总览（由简到难，公式递进）

| 天数 | 主题 | 为什么这个顺序 |
|---|---|---|
| 1–2 | **GLA**（线性注意力范式总览） | 最简形式：乘性 decay + 状态递推，chunk 并行的教科书 |
| 3–4 | **DeltaNet** + WY 表示 | Delta rule 是 GDN/KDA 的数学核心，WY 表示是所有快速 kernel 的骨架 |
| 5–6 | **GDN**（Gated DeltaNet） | Delta rule + gate + 归一化，Qwen3-Next 在用 |
| 7–8 | **KDA**（Kimi Delta Attention） | 当前最复杂：二维状态 + 块内/块间两套更新 |
| 9–10 | **MLA**（DeepSeek-V2） | 思路完全不同：缓存压缩派 vs 状态递推派，放在最后做对照 |
| 11–12 | 选做进阶 | DeltaFormer / GDN-2 / RWKV7 / NPU benchmark，按兴趣挑 |
| 13–14 | 综合产出 | 手写一个 op 的实现并对比 kernel 输出，或做变体对比报告 |

7 天紧凑版：砍掉 Day 11–12 和 Day 13 的进阶项。

## 2. 每个 op 的统一学习套路（5 步）

以 GDN 为例（`fla/ops/gated_delta_rule/`）：

1. **读论文公式**（10 分钟）：GDN 论文 2412.06464，只看 state update 那页。
2. **读 `naive.py`**（30 分钟）：纯 PyTorch 逐 token 循环，公式长什么样代码就长什么样，这是全项目最好懂的文件。
3. **跑现成对比**：改 `tests/ops/test_gdn.py` 里的 `B/T/H/D`，跑 `test_naive_chunk`，亲眼确认 naive 和 chunk kernel 输出一致（`assert_close` 前打印两边的输出，看数值长啥样）。
4. **读 kernel 数学注释**：`chunk_fwd.py` / `wy_fast.py` 头部都有分块推导注释，配合 naive 对照读。
5. **NPU 上跑真 kernel**：`pytest` + `benchmarks/ops` 里的 benchmark 脚本（`python benchmarks/ops/benchmark_gated_delta_rule.py` 这类）。

## 3. 文件地图（4 个目标 op）

```
fla/ops/gla/            # 入口：naive.py → chunk_fwd.py → fused_recurrent.py
fla/ops/delta_rule/     # WY 表示在这里首次出现：naive → chunk → wy_fast
fla/ops/gated_delta_rule/
  naive.py              # 公式参考实现 ← 先读这个
  chunk_fwd.py          # 分块并行 forward（块间状态传递 + 块内并行）
  wy_fast.py            # WY 表示 kernel（最快的路径）
  fused_recurrent.py    # 流式推理用
  backends/             # flash_qla.py（FlashQLA）、triton_ascend/（NPU 专用）
fla/ops/kda/
  naive.py chunk_fwd.py chunk_intra.py wy_fast.py chunk_bwd.py
  # chunk_intra.py 是 KDA 独有：块内逐 token 的 delta 更新
fla/layers/mla.py       # MLA 在 fla 里是 layer 级实现（PyTorch），不是 triton kernel
fla/models/mla/         # MLAForCausalLM，可直接跑 generation
tests/ops/test_{gla,delta,gdn,kda}.py   # 现成的"输入→输出"对照
```

## 4. 每天的关键产出（验收标准）

- **Day 1–2（GLA）**：能口述 chunk 并行三件套的差异；写出 naive vs chunk 输出对比打印；跑通 CPU 冒烟测试。
- **Day 3–4（DeltaNet）**：理解 WY 表示解决什么问题（把 N 次序列化的 delta 更新变成一次可并行的矩阵乘积）；能画出 WY 矩阵的形状。
- **Day 5–6（GDN）**：能解释 `beta`、`normalize`、`gate` 三个组件的各自作用；在 NPU 上跑通 `test_gdn.py`。
- **Day 7–8（KDA）**：能画出 (c_k, c_kv) 二维状态 vs GDN 一维状态的差别；理解 `chunk_intra` 与块间传递的关系；NPU 跑通 `test_kda.py`。
- **Day 9–10（MLA）**：能手推 低秩 KV 压缩 → RoPE 解耦 → absorbed attention 三步变换；用 `fla/models/mla` 跑一个 10-token 的 generation demo，打印每步缓存。
- **Day 13–14（综合）**：推荐做一张对比表——状态维度 / 更新规则（乘性 decay vs delta rule）/ 归一化 / 并行策略 / 推理成本，覆盖 GLA、DeltaNet、GDN、KDA、MLA、RWKV7。

## 5. 项目机制速查（学习时遇到不慌）

- **三种计算范式**贯穿所有 op：`naive recurrent`（教学用）/ `chunk parallel`（训练用，O(T) 状态传递 + 块内并行）/ `fused recurrent`（推理用，流式）。每个 op 都实现这三套，且测试保证它们数值一致——这是这个项目最漂亮的设计。
- **`backends/` 目录**：同一算法多平台实现（triton_ascend 目录是 NPU 专用 kernel），`FLA_DISABLE_BACKEND_DISPATCH=1` 可强制走默认 Triton 路径（ENVs.md 有说明）。
- **`fla/models/`**：`modeling_*.py` 是可直接加载 HuggingFace 权重的完整模型（KDA、MLA 都有），跑 generation 看真实模型输出是很好的"直观输入输出"。
- **环境变量速查**：完整列表见项目根目录 `ENVs.md`（如 `FLA_USE_TMA`、`FLA_CACHE_RESULTS`、`FLA_TILELANG` 等）。

## 参考论文

| 模型 | 论文 |
|---|---|
| GLA | Gated Linear Attention Transformers with Hardware-Efficient Training (2312.06635) |
| DeltaNet | Parallelizing Linear Transformers with Delta Rule over Sequence Length (2406.06484) |
| GDN | Gated Delta Networks: Improving Mamba2 with Delta Rule (2412.06464) |
| KDA | Kimi Delta Attention (2510.26692) |
| MLA | DeepSeek-V2: A Strong, Economical, and Efficient MoE LM (2405.04434) |
| DeltaFormer | (2505.19488) |
| RWKV7 | RWKV-7 "Goose" (2503.14456) |
