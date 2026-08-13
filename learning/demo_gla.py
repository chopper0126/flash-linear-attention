"""
GLA (Gated Linear Attention) 直观演示
======================================
核心数学（naive.py 里那 33 行）：

    h_t = exp(g_t) * h_{t-1} + k_t (x) v_t     # 状态矩阵 [K, V]：先衰减旧状态，再写入新外积
    o_t = q_t^T h_t                            # 读取：query 与状态做内积

三种实现数学等价，只是并行策略不同：
    naive_recurrent_gla   逐 token 循环（教学参考，最直观）
    chunk_gla             分块并行（训练用）
    fused_recurrent_gla   流式循环 kernel（推理用）

运行（venv 激活后，在仓库根目录）：
    python learning/demo_gla.py

也可配合源码阅读：
    fla/ops/gla/naive.py            ← 33 行，全项目最好懂的参考实现
    fla/ops/gla/chunk.py            ← 分块并行（fwd_A / fwd_h / fwd_o 三阶段）
    tests/ops/test_gla.py           ← 官方"naive vs kernel"数值对照测试
"""
import torch
import torch.nn.functional as F

from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.gla.naive import naive_recurrent_gla

torch.manual_seed(42)

# 故意选很小的形状，方便你把中间结果打印出来盯着看
B, T, H, K, V = 2, 6, 2, 8, 8
device = 'cpu'

q = torch.randn(B, T, H, K, device=device)
k = torch.randn(B, T, H, K, device=device)
v = torch.randn(B, T, H, V, device=device)
# 遗忘门 g 取 logsigmoid(randn)：g < 0，exp(g) ∈ (0,1)，是"衰减率"
# 这正是 GLA 里 decay 的来源：不是参数化的标量，而是逐 (B,T,H,K) 可学习的门
g = F.logsigmoid(torch.randn(B, T, H, K, device=device))

print("=" * 60)
print("输入形状: q/k/g: [B={}, T={}, H={}, K={}]   v: [B, T, H, V={}]".format(B, T, H, K, V))
print("遗忘门 g 的取值范围: [{:.3f}, {:.3f}]  → exp(g) 衰减率 ∈ [{:.3f}, {:.3f}]".format(
    g.min().item(), g.max().item(), g.exp().min().item(), g.exp().max().item()))
print("=" * 60)

# ---------- 1. 三种实现，同一份输入，输出必须一致 ----------
o_naive, h_naive = naive_recurrent_gla(q.clone(), k.clone(), v.clone(), g.clone(),
                                       output_final_state=True)
o_chunk, h_chunk = chunk_gla(q.clone(), k.clone(), v.clone(), g.clone(),
                             output_final_state=True)
o_fused, h_fused = fused_recurrent_gla(q.clone(), k.clone(), v.clone(), g.clone(),
                                       output_final_state=True)

print("\n[1] 三种实现输出一致性（误差越小越一致）:")
print("    max|o_naive - o_chunk| = {:.3e}".format((o_naive - o_chunk).abs().max().item()))
print("    max|o_naive - o_fused| = {:.3e}".format((o_naive - o_fused).abs().max().item()))
print("    max|h_naive - h_chunk| = {:.3e}".format((h_naive - h_chunk).abs().max().item()))
print("    max|h_naive - h_fused| = {:.3e}".format((h_naive - h_fused).abs().max().item()))

# ---------- 2. 状态矩阵 h 的演化（照着 naive.py 的循环手写一遍） ----------
print("\n[2] 状态 h 的演化过程 (batch=0, head=0)：")
h = torch.zeros(H, K, V)
with torch.no_grad():
    for t in range(T):
        # 第 33-39 行：先衰减旧状态，再写入新的 rank-1 外积 k_t (x) v_t
        h = h * g[0, t, :, :, None].exp() + k[0, t, :, :, None] * v[0, t, :, None, :]
        # 第 40 行：o_t = q_t 与 h 做内积（attention 意义上的"读取"）
        o_t = (q[0, t, :, :, None] * h).sum(-2)
        print("    t={}: |h|_fro={:8.3f}   o[0,{},0,:2] = {}".format(
            t, h.norm().item(), t, o_t[0, :2].tolist()))

# ---------- 3. 手算验证第一个 token ----------
print("\n[3] 手算验证 (batch=0, head=0, t=0)：")
print("    o_0 = q_0^T h_0, 其中 h_0 = k_0 (x) v_0（第 0 个 token 没有历史可衰减）")
o_hand = q[0, 0, 0] @ (torch.outer(k[0, 0, 0], v[0, 0, 0]))
print("    手算 o_0[:3] = {}".format(o_hand[:3].tolist()))
print("    实现 o_0[:3] = {}".format(o_naive[0, 0, 0, :3].tolist()))
print("    一致 ✓" if torch.allclose(o_hand, o_naive[0, 0, 0], atol=1e-6) else "    不一致 ✗")

# ---------- 4. 状态传递：把 chunk 的中间状态喂给下一个 chunk ----------
print("\n[4] chunk 分段 = 状态接力：")
print("    把 T=6 切成 3 段 (BT=2)，每段把 final state 传给下一段：")
with torch.no_grad():
    states = [torch.zeros(B, H, K, V)]
    o_all = []
    for s in range(0, T, 2):
        seg = slice(s, s + 2)
        o_seg, h_seg = chunk_gla(q[:, seg].clone(), k[:, seg].clone(), v[:, seg].clone(),
                                 g[:, seg].clone(), initial_state=states[-1],
                                 output_final_state=True)
        states.append(h_seg)
        o_all.append(o_seg)
    o_stitched = torch.cat(o_all, dim=1)
    print("    分段拼接 o vs 一次性 chunk: max diff = {:.3e}".format(
        (o_stitched - o_chunk).abs().max().item()))
    print("    分段末态 vs 一次性末态:     max diff = {:.3e}".format(
        (states[-1] - h_chunk).abs().max().item()))
print("\n→ 这就是 chunk 并行能工作的原因：状态是"接力棒"，段内可并行、段间顺序传。")
