# LLM 算法实习面试准备文档

> 基于 SGLang 推理框架 + slime 后训练框架的深度学习与面试准备指南
> 面向：LLM & Agent 应用 / 后训练方向的 MS 在读学生

---

## 目录

- [第一部分：项目概述与技术栈](#第一部分项目概述与技术栈)
- [第二部分：SGLang 推理框架深度解析](#第二部分sglang-推理框架深度解析)
- [第三部分：slime 后训练框架深度解析](#第三部分slime-后训练框架深度解析)
- [第四部分：核心技术原理与面试考察点](#第四部分核心技术原理与面试考察点)
- [第五部分：Agent 系统与多轮交互](#第五部分agent-系统与多轮交互)
- [第六部分：高频面试问题与参考答案](#第六部分高频面试问题与参考答案)
- [第七部分：项目经验介绍模板](#第七部分项目经验介绍模板)
- [附录：关键代码位置索引](#附录关键代码位置索引)

---

## 第一部分：项目概述与技术栈

### 1.1 SGLang 项目定位

SGLang 是由 LMSYS 组织维护的**高性能 LLM 推理服务框架**，已在超过 40 万张 GPU 上运行，日处理数万亿 token。

**核心能力**：
- 高性能推理引擎（连续批处理、RadixAttention、CUDA Graph）
- 前端 DSL（领域特定语言）用于编程式组合 LLM 调用
- 200+ 模型支持（Llama、Qwen、DeepSeek、Gemma 等）
- 多硬件平台支持（NVIDIA、AMD、Intel、Google TPU、Apple MLX、Ascend NPU）

**关键代码路径**：
```
/data/home/yizhou/sglang/
├── python/sglang/
│   ├── lang/           # 前端 DSL
│   └── srt/            # 推理运行时
│       ├── managers/   # 调度器、分词器
│       ├── mem_cache/  # 内存管理、RadixCache
│       ├── model_executor/  # 模型执行器
│       ├── models/     # 200+ 模型定义
│       └── layers/     # 注意力层、MoE 层
└── sgl-kernel/         # 自定义 CUDA 算子
```

### 1.2 slime 项目定位

slime 是清华大学 THUDM 团队开发的 **RL 后训练框架**，核心设计是将 Megatron 训练与 SGLang 推理引擎深度集成。

**核心能力**：
- GRPO/PPO/GSPO/CISPO/REINFORCE++ 等多种 RL 算法
- 灵活的 Agent RL 支持（17 个钩子点）
- 生产级分布式训练（Ray + Megatron）
- 已验证于 GLM-5、Qwen3、DeepSeek-V3/R1 等前沿模型

**关键代码路径**：
```
/data/home/yizhou/slime/
├── slime/
│   ├── backends/megatron_utils/  # 训练后端（loss.py 核心）
│   ├── backends/sglang_utils/    # 推理后端
│   ├── rollout/                  # Rollout 生成
│   ├── ray/                      # 分布式调度
│   └── utils/ppo_utils.py        # RL 算法核心
└── examples/                     # Agent 示例
```

### 1.3 两者关系

```
┌─────────────────────────────────────────────────────────────┐
│                        slime                                │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │   Training   │────>│   Rollout    │────>│ Data Buffer │ │
│  │  (Megatron)  │<────│  (SGLang)    │<────│             │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**SGLang 是 slime 唯一的推理后端**，slime 通过原生参数透传机制直接使用 SGLang 的所有能力。

---

## 第二部分：SGLang 推理框架深度解析

### 2.1 整体架构

SGLang 采用**多进程、分层解耦**架构：

```
用户请求 (HTTP/OpenAI API/SGLang DSL)
        │
        ▼
┌─────────────────────────────────────┐
│  TokenizerManager (主进程)          │  ← Tokenization, 多模态处理
└──────────────┬──────────────────────┘
               │  ZMQ IPC
               ▼
┌─────────────────────────────────────┐
│  Scheduler (核心调度器)              │  ← 请求调度, 批处理策略
│  ├── RadixCache (前缀缓存)          │
│  ├── SchedulePolicy (调度策略)      │
│  └── MemoryPool (内存管理)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ModelRunner (GPU 执行)             │  ← 模型前向计算
│  ├── ForwardBatch                   │
│  ├── RadixAttention                 │
│  └── CUDA Graph Runner              │
└──────────────┬──────────────────────┘
               │  ZMQ IPC
               ▼
┌─────────────────────────────────────┐
│  DetokenizerManager                 │  ← 增量 Detokenization
└─────────────────────────────────────┘
```

**面试考察点**：
- Q: 为什么使用多进程而不是多线程？
- A: 避免 Python GIL 限制，实现真正的并行。TokenizerManager、DetokenizerManager、Scheduler 运行在独立进程，通过 ZMQ 通信。

### 2.2 RadixAttention（核心技术）

**文件**：`python/sglang/srt/mem_cache/radix_cache.py`

RadixAttention 是 SGLang 最核心的创新，使用 **Radix Tree（基数树）** 管理 KV Cache。

**核心思想**：
- 将 token 序列作为 Radix Tree 的 key
- 每个树节点存储对应的 KV Cache 索引
- 共享前缀的多个请求可以复用同一 KV Cache
- 使用 LRU 策略进行缓存淘汰

**数据结构**：
```python
class TreeNode:
    children: Dict       # 子节点
    parent: TreeNode     # 父节点
    key: RadixKey        # token 序列
    value: Tensor        # GPU KV cache 索引
    lock_ref: int        # 引用计数（防止正在使用的节点被淘汰）
    last_access_time     # LRU 时间戳
    host_value: Tensor   # CPU 端备份（HiCache L2）
```

**工作流程**：
1. 新请求到达 → `match_prefix()` 查找已缓存的最长公共前缀
2. 匹配到的前缀对应的 KV Cache 直接复用，无需重新计算
3. 新生成的 KV Cache 通过 `insert()` 插入 Radix Tree
4. 内存不足时通过 `evict()` 按 LRU 驱逐

**面试深挖点**：
- Q: RadixAttention 相比简单的 hash-based prefix caching 有什么优势？
- A: 
  1. 支持更灵活的前缀匹配（不要求完全匹配）
  2. 支持前缀共享（多个请求可以共享同一前缀的 KV Cache）
  3. 支持分层缓存（HiCache：GPU → CPU → 分布式存储）
  4. 支持 LRU 淘汰策略

- Q: Radix Tree 的时间复杂度？
- A: 匹配 O(最长前缀)，插入 O(1)（已知位置），淘汰 O(1)（LRU）

### 2.3 调度策略

**文件**：`python/sglang/srt/managers/schedule_policy.py`

| 策略 | 类型 | 说明 |
|------|------|------|
| LPM | Cache-Aware | 最长前缀匹配，优先调度缓存命中率高的请求 |
| DFS-Weight | Cache-Aware | 基于 RadixTree 深度优先搜索的加权排序 |
| FCFS | Cache-Agnostic | 先来先服务 |
| LOF | Cache-Agnostic | 最长输出优先 |

**面试深挖点**：
- Q: 为什么需要 Cache-Aware 策略？
- A: 在多轮对话和 system prompt 复用场景下，Cache-Aware 策略可以最大化 prefix cache 命中率，减少重复计算。

### 2.4 连续批处理（Continuous Batching）

**文件**：`python/sglang/srt/managers/scheduler.py`

**核心事件循环**：
```python
def event_loop_normal(self):
    while True:
        recv_reqs = self.request_receiver.recv_requests()  # 1. 接收请求
        self.process_input_requests(recv_reqs)              # 2. 处理输入
        batch = self.get_next_batch_to_run()                # 3. 获取下一个 batch
        if batch:
            result = self.run_batch(batch)                  # 4. 执行前向
            self.process_batch_result(batch, result)        # 5. 处理结果
```

**关键机制**：
1. **Prefill-Decode 分离调度**：新请求先 prefill，然后合并到 running batch 做 decode
2. **Chunked Prefill**：长 prompt 分块处理，避免阻塞 decode 请求
3. **Overlap Scheduling**：CPU 调度和 GPU 计算重叠，GPU 始终忙碌
4. **Retraction**：内存不足时回退部分请求到等待队列

**面试深挖点**：
- Q: 什么是 Chunked Prefill？为什么需要它？
- A: 长 prompt 的 prefill 阶段会占用大量 GPU 时间，阻塞正在进行的 decode 请求。Chunked Prefill 将长 prompt 分成多个 chunk，每个 chunk 与 decode 请求混合执行，避免 decode 请求被饿死。

### 2.5 内存管理

**文件**：`python/sglang/srt/mem_cache/memory_pool.py`

**两级内存池设计**：
```
ReqToTokenPool: 请求 → token 位置映射
        │
        ▼
TokenToKVPoolAllocator: token → KV cache 物理索引分配
        │
        ▼
KVCache Pool: 物理 KV cache 存储（GPU 显存）
```

**优化技术**：
- **Paged Attention**：以 page 为单位管理 KV Cache，减少碎片化
- **HiCache**：分层缓存（GPU → CPU → 分布式存储）

### 2.6 推测解码（Speculative Decoding）

**目录**：`python/sglang/srt/speculative/`

| 方法 | 说明 |
|------|------|
| EAGLE-2/3 | 基于特征的 draft + 树验证，最高 2.3x 加速 |
| MTP | 模型内置多 token 预测头 |
| NGRAM | 基于 n-gram 缓存的无模型 draft |

**面试深挖点**：
- Q: 推测解码的原理是什么？
- A: 使用小模型（或轻量级方法）快速生成多个候选 token，然后大模型并行验证这些候选 token。如果候选 token 被接受，就一次生成多个 token，提高吞吐量。

### 2.7 PD 分离（Prefill/Decode Disaggregation）

**目录**：`python/sglang/srt/disaggregation/`

将 Prefill 和 Decode 阶段分离到不同实例：
- **Prefill 实例**：计算密集，处理完整输入
- **Decode 实例**：内存密集，逐 token 生成
- 通过 Mooncake 或 NIXL 等 RDMA 引擎传输 KV Cache

**面试深挖点**：
- Q: 为什么需要 PD 分离？
- A: 
  1. Prefill 和 Decode 的资源需求不同（计算密集 vs 内存密集）
  2. 避免 Prefill 打断 Decode 的延迟
  3. 支持独立扩缩 Prefill/Decode 资源

---

## 第三部分：slime 后训练框架深度解析

### 3.1 RL 后训练闭环

```
Data Sampling → Weight Update → Data Sampling → ...
```

**关键参数约束**：
```
(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)
```

### 3.2 RL 算法实现

**文件**：`slime/utils/ppo_utils.py` 和 `slime/backends/megatron_utils/loss.py`

#### GRPO（Group Relative Policy Optimization）

**核心公式**：
```python
# 对每个 sample，advantage 就是该 sample 的标量 reward
A_i(t) = R_i    (对所有 token t)
```

**代码实现**：
```python
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

**特点**：不需要 Critic 模型，计算效率高

#### PPO + GAE

**核心公式**：
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
Return_t = A_t + V(s_t)
```

**代码实现**：
```python
for t in reversed(range(response_len)):
    nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
    delta = full_rewards[t] + gamma * nextvalues - full_values[t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    advantages_reversed.append(lastgaelam)
```

#### Policy Loss 的 Clipping 机制

**公式**：
```
L_PG = max(-r(θ)·A, -clip(r(θ), 1-ε, 1+ε_high)·A)
```

**代码实现**：
```python
ratio = (-ppo_kl).exp()  # ratio = exp(new_logp - old_logp)
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
```

**面试深挖点**：
- Q: PPO 的 clipping 机制有什么作用？
- A: 限制策略更新幅度，防止策略偏离太远，稳定训练。取两者最大值是悲观估计，确保不会过度优化。

#### CISPO（Clipped IS-Weighted Stop-Gradient）

来自 MiniMax-M1 论文：
```python
ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
pg_losses = -ratio_truncated.detach() * advantages * log_probs
```

**与 PPO 的区别**：IS ratio 在 stop-gradient 下裁剪，梯度通过 log_probs 流动，被裁剪的 token 仍然贡献梯度。

### 3.3 KL 散度控制

**文件**：`slime/utils/ppo_utils.py`

支持三种 KL 估计器：

| 类型 | 公式 | 特点 |
|------|------|------|
| k1 | KL = log(π/π_ref) | 一阶近似，可能为负 |
| k2 | KL = (log(π/π_ref))^2 / 2 | 二阶近似，始终非负 |
| k3 (low_var_kl) | KL = π_ref/π - 1 - log(π_ref/π) | 非负、无偏、低方差 |

**KL 的两种使用方式**：
1. **作为 reward 惩罚**：`rewards[t] -= kl_coef * KL_t`
2. **作为额外 loss 项**：`loss = loss + kl_loss_coef * kl`

### 3.4 权重同步机制

**目录**：`slime/backends/megatron_utils/update_weight/`

| 模式 | 说明 |
|------|------|
| full + nccl | 标准全量广播（默认） |
| full + disk | 写完整 HF checkpoint 到磁盘 |
| delta + nccl | 仅同步变更字节的 NCCL 广播 |
| delta + disk | 稀疏增量写入共享文件系统 |

**Delta Weight Sync 原理**：
1. **Diff**：逐字节比较当前权重与快照
2. **Encode**：编码变化的 (position, value) 对
3. **传输**：通过 NCCL 或磁盘
4. **应用**：接收端 NaN-masked overwrite

**面试深挖点**：
- Q: 为什么需要 Delta Weight Sync？
- A: 全量同步成本与模型大小线性相关，但 RL 步之间只有少量权重变化（~3% 密度）。Delta sync 只传输差异部分，大幅降低通信成本。

### 3.5 训练模式

#### 同步训练
```python
for rollout_id in range(args.num_rollout):
    rollout_data_ref = rollout_manager.generate.remote(rollout_id)
    actor_model.async_train(rollout_id, rollout_data_ref)
    actor_model.update_weights()
```

#### 异步训练
```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # 提前开始下一轮 rollout
    rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
    # 训练当前数据
    actor_model.async_train(rollout_id, rollout_data_curr_ref)
```

**面试深挖点**：
- Q: 异步训练的优势和挑战？
- A:
  - 优势：重叠训练和推理，提高吞吐量
  - 挑战：权重同步时机、数据一致性、训练稳定性

---

## 第四部分：核心技术原理与面试考察点

### 4.1 连续批处理 vs 静态批处理

| 特性 | 静态批处理 | 连续批处理 |
|------|-----------|-----------|
| 请求到达 | 等待 batch 满 | 随到随处理 |
| 请求完成 | 等待整个 batch 完成 | 立即释放资源 |
| GPU 利用率 | 低（有空闲） | 高（始终忙碌） |
| 延迟 | 高 | 低 |

**面试深挖点**：
- Q: 连续批处理如何实现？
- A: 
  1. 维护 waiting queue 和 running batch
  2. 每个 iteration 决定哪些请求进入 prefill/decode
  3. 请求完成后立即从 running batch 中移除
  4. 新请求可以随时加入 waiting queue

### 4.2 Prefix Caching 的工程价值

**场景**：多轮对话、system prompt 复用、Agent 工作流

**价值**：
- 减少重复计算（相同前缀只需计算一次）
- 降低首 token 延迟（TTFT）
- 提高吞吐量

**面试深挖点**：
- Q: Prefix Caching 在 Agent 场景下有什么特殊价值？
- A: Agent 的多轮交互中，system prompt 和历史对话是共享的。Prefix Caching 可以避免每轮都重新计算这些共享部分，大幅提高效率。

### 4.3 RL 后训练的核心挑战

#### 挑战 1：信用分配（Credit Assignment）

**问题**：多轮交互中，如何分配奖励到每一步？

**解决方案**：
1. **稀疏奖励**：只在最终结果给予奖励，依赖 RL 算法自动分配
2. **密集奖励**：每一步都给予奖励（如工具调用成功/失败）
3. **GAE**：使用 Critic 模型估计每一步的价值
4. **TIS**：通过重要性采样修正 off-policy 数据

#### 挑战 2：Loss Masking

**问题**：Agent 轨迹包含模型生成和工具返回两部分，工具返回不应参与 loss 计算。

**实现**：
```python
# 模型生成的 token → loss_mask = 1
# 工具返回的 token → loss_mask = 0
loss_masks += [0] * len(tool_tokens)
```

**面试深挖点**：
- Q: 为什么需要 Loss Masking？
- A: 工具返回的内容不是模型生成的，如果参与 loss 计算，会让模型学习"复制"工具输出，而不是学习如何正确使用工具。

#### 挑战 3：Off-Policy 问题

**问题**：使用旧策略生成的数据训练新策略，存在分布差异。

**解决方案**：
1. **TIS（Truncated Importance Sampling）**：裁剪 importance ratio
2. **OPSM（Off-Policy Sequence Masking）**：序列级 KL 超阈值时 mask 掉
3. **KL 惩罚**：限制策略偏离参考模型

#### 挑战 4：训练稳定性

**问题**：RL 训练容易不稳定，策略可能崩溃。

**解决方案**：
1. **PPO Clipping**：限制策略更新幅度
2. **KL 惩罚**：防止策略偏离太远
3. **Dual-Clip PPO**：对负 advantage 额外裁剪
4. **Routing Replay**：MoE 模型的路由稳定性

### 4.4 并行策略对比

| 策略 | 分割位置 | 适用场景 | 通信开销 |
|------|---------|---------|---------|
| TP（张量并行） | 单层内部 | 大模型、单节点 | 高（每层通信） |
| PP（流水线并行） | 层间 | 超大模型、多节点 | 低（仅 bubble） |
| CP（上下文并行） | 序列维度 | 长序列 | 中等 |
| EP（专家并行） | MoE 专家 | MoE 模型 | 高（token dispatch） |
| DP（数据并行） | 数据维度 | 通用 | 低（梯度同步） |

**面试深挖点**：
- Q: 如何选择并行策略？
- A: 
  1. 单节点内优先 TP（通信带宽高）
  2. 多节点间使用 PP（通信量小）
  3. MoE 模型使用 EP
  4. 长序列使用 CP
  5. 数据量大时使用 DP

### 4.5 显存优化技术

| 技术 | 说明 | 代码位置 |
|------|------|---------|
| Gradient Checkpointing | 用计算换内存 | Megatron 原生支持 |
| CPU Offload | 将模型卸载到 CPU | `--offload-train` |
| Paged Attention | 减少 KV Cache 碎片 | SGLang 原生支持 |
| FP8 量化 | 降低精度换内存 | SGLang 原生支持 |
| Chunked Prefill | 分块处理长 prompt | SGLang 原生支持 |

---

## 第五部分：Agent 系统与多轮交互

### 5.1 Agent 训练架构

```
┌─────────────────────────────────────────────────┐
│                 Agent RL 工作流                   │
│                                                   │
│  ┌──────────────┐     ┌───────────────────────┐  │
│  │ Protocol      │     │ Custom Generate Func  │  │
│  │ Adapters      │     │ (--custom-generate-   │  │
│  │ (Anthropic/   │────>│  function-path)       │  │
│  │  OpenAI)      │     │                       │  │
│  └──────────────┘     └───────────────────────┘  │
│          │                      │                 │
│          v                      v                 │
│  ┌──────────────┐     ┌───────────────────────┐  │
│  │ SGLang        │     │ Custom RM             │  │
│  │ Session        │     │ (--custom-rm-path)    │  │
│  │ Routing        │     │                       │  │
│  └──────────────┘     └───────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 5.2 自定义生成函数

```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    for turn in range(max_turns):
        # 1. 模型生成动作
        model_output = await call_sglang(prompt + full_response, ...)
        
        # 2. 解析并执行动作
        action, content = parse_action(model_output)
        
        # 3. 获取观察结果
        if action == "search":
            tool_output = await search(content)
            # 4. 设置 loss_mask
            loss_masks += [0] * len(tool_tokens)
        
        # 5. 终止条件
        if action == "answer":
            break
    
    sample.loss_mask = loss_masks
    return sample
```

### 5.3 Session Routing

**问题**：Agent 的多轮交互需要将请求路由到同一 worker，复用 prefix cache。

**解决方案**：使用 consistent hashing 路由
```bash
--router-policy consistent_hashing
```

### 5.4 Fan-out 训练

**概念**：一次 rollout 拆分为多个可训练段（如 subagent 轨迹、主 agent 继续、上下文压缩前后段），所有段共享同一个 rollout_id。

**代码示例**：
```python
async def generate(args, sample, sampling_params):
    # 主 agent 生成
    main_output = await call_sglang(main_prompt)
    
    # 子 agent 生成
    sub_output = await call_sglang(sub_prompt)
    
    # 返回多个可训练段
    return [main_sample, sub_sample]
```

### 5.5 异步训练与长尾处理

**问题**：Agent 生成时间长且不均匀，同步训练会浪费资源。

**解决方案**：
1. **异步训练**：提前开始下一轮 rollout
2. **完全异步**：后台持续消费 data buffer，不等待最慢样本
3. **Partial Rollout**：缓存部分生成的样本，避免计算浪费

---

## 第六部分：高频面试问题与参考答案

### Q1: 什么是 RL 后训练？为什么需要它？

**参考答案**：
RL 后训练是在预训练/微调之后，使用强化学习进一步优化模型的过程。它可以：
1. 学习人类偏好（如 RLHF）
2. 提高推理能力（如 DeepSeek-R1）
3. 学习使用工具（如 Agent 训练）
4. 适应特定任务（如代码生成、数学推理）

### Q2: 解释 GRPO 和 PPO 的区别

**参考答案**：

| 方面 | GRPO | PPO |
|------|------|-----|
| Critic 模型 | 不需要 | 需要 |
| Advantage 计算 | 组内相对优势 | GAE（时序差分） |
| 计算效率 | 高 | 低 |
| 适用场景 | 稀疏奖励、组内比较 | 密集奖励、连续控制 |

**GRPO 公式**：`A_i = (r_i - mean(r_group)) / std(r_group)`
**PPO 公式**：`A_t = δ_t + (γλ) * A_{t+1}`，其中 `δ_t = r_t + γ * V(s_{t+1}) - V(s_t)`

### Q3: 为什么需要 Prefix Caching？

**参考答案**：
在多轮对话、system prompt 复用、Agent 工作流等场景下，多个请求共享相同的前缀。Prefix Caching 可以：
1. 避免重复计算相同前缀的 KV Cache
2. 降低首 token 延迟（TTFT）
3. 提高吞吐量
4. 节省 GPU 显存

### Q4: 如何处理 Agent 训练中的信用分配问题？

**参考答案**：
信用分配是 Agent 训练的核心挑战，有几种方法：
1. **稀疏奖励**：只在最终结果给予奖励，依赖 RL 算法自动分配
2. **密集奖励**：每一步都给予奖励（如工具调用成功/失败）
3. **GAE**：使用 Critic 模型估计每一步的价值
4. **TIS**：通过重要性采样修正 off-policy 数据
5. **OPSM**：序列级 KL 超阈值时 mask 掉，防止负优势样本造成过大梯度

### Q5: 解释训练-推理分离 vs 共置的权衡

**参考答案**：

| 方面 | 分离模式 | 共置模式 |
|------|----------|----------|
| 资源利用率 | 较低（需要两套 GPU） | 较高（共享 GPU） |
| 通信成本 | 高（需要权重同步） | 低（本地访问） |
| 显存管理 | 简单 | 复杂（需要卸载） |
| 适用场景 | 大规模训练 | 资源受限 |

### Q6: 解释 Loss Masking 在 Agent 训练中的作用

**参考答案**：
在 Agent 训练中，轨迹包含模型生成和工具返回两部分：
- 模型生成的 token：应该参与 loss 计算（loss_mask = 1）
- 工具返回的 token：不应该参与 loss 计算（loss_mask = 0）

原因：工具返回的内容不是模型生成的，如果参与 loss，会让模型学习"复制"工具输出，而不是学习如何正确使用工具。

### Q7: 如何设计一个可扩展的 RL 训练框架？

**参考答案**：
关键设计原则：
1. **模块化**：训练、推理、数据管理分离
2. **可扩展性**：通过自定义接口支持新功能（如 slime 的 17 个钩子点）
3. **正确性优先**：显式数据流，易于调试
4. **性能优化**：异步执行、权重同步优化（Delta Weight Sync）
5. **生产级**：容错机制、检查点、监控

### Q8: 解释动态采样的作用和实现

**参考答案**：
**作用**：提高数据多样性，避免同质化数据

**实现**：
1. 采样比实际需要更多的 prompts
2. 每个 prompt 生成多个响应
3. 应用过滤函数（如检查奖励标准差 > 0）
4. 丢弃不满足条件的样本
5. 如果过滤太严格，自动触发新一轮采样

### Q9: 推测解码的原理是什么？

**参考答案**：
使用小模型（或轻量级方法）快速生成多个候选 token，然后大模型并行验证这些候选 token。如果候选 token 被接受，就一次生成多个 token，提高吞吐量。

**关键点**：
1. Draft 模型要足够快
2. 验证过程可以并行
3. 接受率越高，加速效果越好
4. 保证输出分布与原模型一致

### Q10: 如何处理长序列 Agent 轨迹？

**参考答案**：
几种方法：
1. **上下文压缩**：定期压缩历史上下文
2. **分段训练**：将长轨迹分成多个段分别训练
3. **滑动窗口**：只保留最近的 N 轮交互
4. **高效注意力**：使用 Flash Attention 等优化
5. **Chunked Prefill**：分块处理长 prompt

---

## 第七部分：项目经验介绍模板

### 7.1 项目背景（1分钟）

"我深入研究了 SGLang 推理框架和 slime 后训练框架。SGLang 是一个高性能 LLM 推理服务框架，已在超过 40 万张 GPU 上运行。slime 是一个 RL 后训练框架，将 Megatron 训练与 SGLang 推理深度集成，用于训练 GLM-5、Qwen3、DeepSeek-V3/R1 等前沿模型。"

### 7.2 技术亮点（2分钟）

"这两个项目有几个关键技术亮点：

1. **SGLang 的 RadixAttention**：使用 Radix Tree 管理 KV Cache，实现高效的前缀共享和缓存复用。在多轮对话和 Agent 场景下，可以大幅减少重复计算。

2. **slime 的 RL 算法实现**：支持 GRPO、PPO、GSPO、CISPO 等多种算法，特别是 CISPO 的 stop-gradient 裁剪机制，使得被裁剪的 token 仍然贡献梯度。

3. **训练-推理协同设计**：slime 通过原生参数透传机制直接使用 SGLang 的所有能力，包括 Prefix Caching、Session Routing、PD 分离等。

4. **灵活的 Agent RL 支持**：通过 17 个钩子点支持任意 Agent 工作流，包括工具调用、沙盒、多 Agent、搜索、编码等。"

### 7.3 个人贡献/学习（1分钟）

"通过学习这两个项目，我深入理解了：
1. LLM 推理的核心优化技术（连续批处理、Prefix Caching、推测解码）
2. RL 后训练的完整流程和算法实现（GRPO/PPO、KL 控制、信用分配）
3. Agent 系统的训练方法（Loss Masking、Session Routing、Fan-out 训练）
4. 大规模分布式系统的设计权衡（训练-推理解耦 vs 共置、全量 vs 增量权重同步）"

### 7.4 技术细节准备

**如果被问到具体实现**：
- **RadixAttention**：可以解释 Radix Tree 的数据结构和 LRU 淘汰策略
- **GRPO/PPO**：可以解释 advantage 计算、clipping 机制、KL 控制
- **权重同步**：可以解释 Delta Weight Sync 的原理
- **Agent 训练**：可以解释 Loss Masking 和 Session Routing

---

## 附录：关键代码位置索引

### SGLang 核心代码

| 模块 | 文件路径 | 关键类/函数 |
|------|----------|-------------|
| 引擎入口 | `python/sglang/srt/entrypoints/engine.py` | `Engine` |
| 调度器 | `python/sglang/srt/managers/scheduler.py` | `Scheduler`, `event_loop_normal`, `get_next_batch_to_run` |
| 调度策略 | `python/sglang/srt/managers/schedule_policy.py` | `SchedulePolicy`, `PrefillAdder` |
| Radix Cache | `python/sglang/srt/mem_cache/radix_cache.py` | `RadixCache`, `TreeNode` |
| 内存池 | `python/sglang/srt/mem_cache/memory_pool.py` | `ReqToTokenPool`, `KVCache` |
| 模型执行器 | `python/sglang/srt/model_executor/model_runner.py` | `ModelRunner` |
| Forward 信息 | `python/sglang/srt/model_executor/forward_batch_info.py` | `ForwardBatch`, `ForwardMode` |
| 注意力层 | `python/sglang/srt/layers/radix_attention.py` | `RadixAttention` |
| 推测解码 | `python/sglang/srt/speculative/` | EAGLE、MTP、NGRAM |
| PD 分离 | `python/sglang/srt/disaggregation/` | Prefill/Decode 分离 |

### slime 核心代码

| 模块 | 文件路径 | 关键类/函数 |
|------|----------|-------------|
| RL 算法 | `slime/utils/ppo_utils.py` | `get_grpo_returns`, `get_advantages_and_returns` |
| 损失函数 | `slime/backends/megatron_utils/loss.py` | `policy_loss_function`, `value_loss_function` |
| Rollout | `slime/rollout/sglang_rollout.py` | `generate_rollout`, `generate_and_rm` |
| 权重同步 | `slime/backends/megatron_utils/update_weight/` | Delta Weight Sync |
| Agent 适配 | `slime/agent/adapters/` | `AnthropicAdapter`, `OpenAIAdapter` |
| 参数定义 | `slime/utils/arguments.py` | 所有超参数定义 |

### 关键配置示例

```bash
# 基础 GRPO 训练
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.001
--eps-clip 0.2
--eps-clip-high 0.28

# Agent 训练
--custom-generate-function-path my_agent.generate
--custom-rm-path my_agent.reward_func
--rollout-max-response-len 8192

# 大规模训练
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--context-parallel-size 2
--expert-model-parallel-size 4

# 异步训练
--update-weights-interval 5
--colocate
--offload-train
```

---

**最后更新**：2026-06-26

**适用场景**：LLM 算法实习面试准备，重点关注推理优化、RL 后训练、Agent 系统、分布式训练等方向
