# 学习增量记录：SGLang + slime 复现过程中的新知识点

> 本文档记录在 8 卡 4090 复现 SGLang + slime 过程中，对比前两轮分析增量新学习到的点
> 按周次组织，每周更新

---

## 目录

- [Week 1: 环境搭建 + SGLang 基础](#week-1-环境搭建--sglang-基础)
- [Week 2: SGLang 核心特性](#week-2-slang-核心特性)
- [Week 3: slime 基础训练](#week-3-slime-基础训练)
- [Week 4: RL 算法深入](#week-4-rl-算法深入)
- [Week 5: Agent 训练](#week-5-agent-训练)
- [Week 6: 端到端项目](#week-6-端到端项目)

---

## Week 1: 环境搭建 + SGLang 基础

### 1.1 SGLang 的 GPU 显存分档机制

**前两轮认知**：知道 SGLang 有 `--mem-fraction-static` 参数控制显存分配

**新增理解**：
- SGLang 在 `server_args.py:3233-3259` 中实现了自动分档逻辑
- RTX 4090 (24GB) 被识别为 20-35GB 档位，自动设置：
  - `chunked_prefill_size = 2048`（而非更大的 4096/8192）
  - `cuda_graph_max_bs = 24`（tp_size < 4 时）
- 这种自动适配避免了手动调参的复杂性

**代码位置**：
```python
# python/sglang/srt/server_args.py:3233-3259
if available_gpu_mem < 20_000:  # < 20GB
    chunked_prefill_size = 2048
    cuda_graph_max_bs = 8
elif available_gpu_mem < 35_000:  # 20-35GB (4090)
    chunked_prefill_size = 2048
    cuda_graph_max_bs = 24 if tp_size < 4 else 80
elif available_gpu_mem < 60_000:  # 35-60GB
    chunked_prefill_size = 4096
    cuda_graph_max_bs = 32 if tp_size < 4 else 160
else:  # >= 60GB
    chunked_prefill_size = 8192
    cuda_graph_max_bs = 128 if tp_size < 4 else 256
```

### 1.2 FlashAttention-3 在 4090 上的支持

**前两轮认知**：知道 SGLang 支持多种注意力后端

**新增理解**：
- RTX 4090 (SM89, Ada Lovelace) 完全支持 FlashAttention-3
- SGLang 代码中明确注释：A100/A*0/L20/L40/L40s/4090 可以使用 fa3
- FA3 相比 FA2 在 4090 上有约 10-20% 的性能提升

**代码位置**：
```python
# python/sglang/jit_kernel/flash_attention_v3.py:92
# "That means if you use A100/A*0/L20/L40/L40s/4090 you can use fa3."
```

### 1.3 bench_one_batch 的 dummy weights 模式

**前两轮认知**：知道 SGLang 有 benchmark 工具

**新增理解**：
- `bench_one_batch` 支持 `--load-format dummy` 模式
- 使用随机初始化的权重进行延迟测试，无需下载模型
- 适合快速验证系统是否正常工作，以及性能基准测试

**使用方法**：
```bash
python -m sglang.benchmark.one_batch \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --load-format dummy
```

### 1.4 SGLang 的多进程架构细节

**前两轮认知**：知道 SGLang 使用多进程架构

**新增理解**：
- 三个核心进程通过 ZMQ IPC 通信：
  - TokenizerManager：处理 tokenization 和多模态输入
  - Scheduler：核心调度器，管理请求队列和批处理
  - DetokenizerManager：处理 detokenization 和流式输出
- 每个 Scheduler 进程对应一个 TP rank
- DataParallelController 在多 DP 场景下进行负载均衡

**通信流程**：
```
TokenizerManager --[ZMQ]--> Scheduler --[ZMQ]--> DetokenizerManager
      ↑                         ↑                      ↓
      └─────────────────────────┴──────────────────────┘
```

### 1.5 RadixCache 的 RadixKey 设计

**前两轮认知**：知道 RadixCache 使用 Radix Tree 管理 KV Cache

**新增理解**：
- RadixKey 支持 `extra_key` 命名空间隔离（如 LoRA ID、cache_salt）
- 支持 `is_bigram` 模式用于 EAGLE 推测解码
- 支持 `limit` 避免 O(n) 拷贝的虚拟切片
- `page_aligned()` 确保 key 长度对齐到 page_size

**代码位置**：
```python
# python/sglang/srt/mem_cache/radix_cache.py:57-200
class RadixKey:
    def __init__(self, token_ids, extra_key=None, is_bigram=False):
        self.token_ids = token_ids
        self.extra_key = extra_key  # 命名空间隔离
        self.is_bigram = is_bigram  # EAGLE 推测解码
```

### 1.6 指数搜索实现前缀匹配

**前两轮认知**：知道 RadixCache 使用前缀匹配

**新增理解**：
- 使用指数搜索（exponential search）加速前缀匹配
- 避免逐 token 的 Python 循环，对长共享前缀效果显著
- 算法复杂度：O(log(最长前缀))

**代码逻辑**：
```python
# 指数搜索 first diverging token：倍增窗口 + 二分窗口
lo = 0; step = 1
while lo < n:
    hi = lo + step if lo + step < n else n
    if t0[lo:hi] != t1[lo:hi]:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if t0[lo:mid] == t1[lo:mid]: lo = mid
            else: hi = mid
        matched_tokens = lo
        break
    lo = hi; step *= 2
```

---

## Week 2: SGLang 核心特性

### 2.1 Overlap Scheduling 的实现细节

**前两轮认知**：知道 SGLang 支持 CPU 调度与 GPU 计算重叠

**新增理解**：
- 使用 `result_queue` 双缓冲实现重叠
- 当第 N 个 batch 在 GPU 上执行时，CPU 同时处理第 N-1 个 batch 的结果
- 通过 WAR Barrier (`_apply_war_barrier`) 保证共享缓冲区的读写安全
- 对连续 prefill batch 可选择性禁用重叠以改善 TTFT

**代码位置**：
```python
# python/sglang/srt/managers/scheduler.py:1548-1605
def event_loop_overlap(self):
    result_queue = deque()
    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run()
        if batch:
            batch_result = self.run_batch(batch)  # GPU 计算
            result_queue.append((batch, batch_result))
        if self.last_batch:
            tmp_batch, tmp_result = result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)  # CPU 处理
```

### 2.2 LPM 调度策略的自适应降级

**前两轮认知**：知道 SGLang 支持多种调度策略

**新增理解**：
- LPM (Longest Prefix Match) 优先调度前缀匹配最长的请求
- 当 waiting_queue > 128 时，LPM 自动退化为 FCFS
- 避免高开销的前缀匹配，平衡缓存命中率和调度开销

**代码位置**：
```python
# python/sglang/srt/managers/schedule_policy.py:223
if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
    return CacheAgnosticPolicy.FCFS
```

### 2.3 In-Batch Prefix Caching

**前两轮认知**：知道 Prefix Caching 可以复用前缀

**新增理解**：
- 对前缀命中较短的请求，额外检查 waiting_queue 中是否有共享前缀
- 如果有则临时降低其优先级，让先调度的请求建立缓存
- 后续请求可以命中这个缓存，提高整体缓存命中率

**代码位置**：
```python
# python/sglang/srt/managers/schedule_policy.py:247-293
def _calc_in_batch_prefix_caching_priority(self, waiting_queue):
    # 检查 waiting_queue 中是否有共享前缀
    # 如果有，临时降低优先级
```

### 2.4 PrefillAdder 的 Token 预算管理

**前两轮认知**：知道 SGLang 支持 Chunked Prefill

**新增理解**：
- PrefillAdder 管理精细化的 token 预算：
  - `rem_total_tokens`：全局可用 + 可驱逐 token
  - `rem_chunk_tokens`：chunked prefill 的剩余 token 预算
  - `rem_input_tokens`：剩余输入 token 预算
  - `new_token_ratio`：预估新生成 token 比率，防止过度预留
- 对 Hybrid SWA 模型有独立的 SWA 预算管理

**代码位置**：
```python
# python/sglang/srt/managers/schedule_policy.py:425
class PrefillAdder:
    def __init__(self, ...):
        self.rem_total_tokens = ...
        self.rem_chunk_tokens = ...
        self.rem_input_tokens = ...
        self.new_token_ratio = ...
```

### 2.5 HiRadixCache 的多级缓存

**前两轮认知**：知道 RadixCache 管理 KV Cache

**新增理解**：
- HiRadixCache 实现了 GPU-CPU-Storage 三级缓存：
  - GPU 内存（热数据）
  - Host 内存（温数据，可配置 ratio）
  - 外部存储后端（冷数据，如 Redis/本地文件）
- 支持异步 prefetch 和 write-through
- 适合 prefix 多样性高的生产环境

**代码位置**：
```python
# python/sglang/srt/mem_cache/hiradix_cache.py:1744
class HiRadixCache:
    def __init__(self, ...):
        self.gpu_cache = ...  # L1: GPU 内存
        self.host_cache = ...  # L2: CPU 内存
        self.storage_backend = ...  # L3: 外部存储
```

### 2.6 EAGLE 推测解码的树构建

**前两轮认知**：知道推测解码使用 draft-verify 机制

**新增理解**：
- EAGLE 使用树状 draft + 并行验证
- 树构建使用 `build_tree_kernel_efficient` 内核
- 支持三种 tree mask 模式：FULL_MASK, QLEN_ONLY, QLEN_ONLY_BITPACKING
- topk 控制树的宽度，num_steps 控制树的深度

**代码位置**：
```python
# python/sglang/srt/speculative/eagle_utils.py:113
def build_tree_kernel_efficient(draft_tokens, bonus_tokens, ...):
    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()
    # 使用 sgl_kernel.build_tree_kernel_efficient 构建树
```

### 2.7 Prometheus 指标体系

**前两轮认知**：知道 SGLang 有监控指标

**新增理解**：
- 完整的 Prometheus 指标体系，包括：
  - `sglang:num_running_reqs`：正在运行的请求数
  - `sglang:num_queue_reqs`：排队等待的请求数
  - `sglang:gen_throughput`：生成吞吐量（token/s）
  - `sglang:token_usage`：KV 缓存利用率
  - `sglang:cache_hit_rate`：前缀缓存命中率
  - `sglang:time_to_first_token_seconds`：TTFT 分布
  - `sglang:e2e_request_latency_seconds`：端到端延迟分布
- 支持 MFU（Model FLOPs Utilization）指标

**代码位置**：
```python
# python/sglang/srt/observability/metrics_collector.py
class MetricsCollector:
    def __init__(self):
        self.num_running_reqs = Gauge(...)
        self.num_queue_reqs = Gauge(...)
        self.gen_throughput = Gauge(...)
        self.token_usage = Gauge(...)
        self.cache_hit_rate = Gauge(...)
```

---

## Week 3: slime 基础训练

### 3.1 slime 的 Megatron 参数透传机制

**前两轮认知**：知道 slime 使用 Megatron 作为训练后端

**新增理解**：
- slime 直接透传所有 Megatron 参数，不需要额外的包装层
- 通过 `source scripts/models/qwen3-4B.sh` 加载模型配置
- 这种设计保持了与 Megatron 的完全兼容性

**示例**：
```bash
# 直接使用 Megatron 参数
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--sequence-parallel
--recompute-granularity full
```

### 3.2 HF -> Megatron torch_dist 权重转换

**前两轮认知**：知道需要权重格式转换

**新增理解**：
- 使用 `tools/convert_hf_to_torch_dist.py` 进行转换
- torch_dist 是 Megatron 的分布式 checkpoint 格式
- 支持 TP/PP 下的权重切分
- 转换后的权重可以被 Megatron 直接加载

**命令**：
```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen2.5-0.5B-Instruct \
    --save /root/models/Qwen2.5-0.5B-Instruct_torch_dist
```

### 3.3 Colocate 模式的资源管理

**前两轮认知**：知道 slime 支持训练-推理共置

**新增理解**：
- Colocate 模式下，训练和推理共享同一组 GPU
- 需要通过 `--sglang-mem-fraction-static` 控制 SGLang 的显存占比
- 为 Megatron 训练预留足够的显存
- 4090 建议设为 0.55-0.7

**配置示例**：
```bash
--colocate
--sglang-mem-fraction-static 0.7
--sglang-cuda-graph-max-bs 16
```

### 3.4 Ray 调度的基本概念

**前两轮认知**：知道 slime 使用 Ray 进行调度

**新增理解**：
- Ray 是一个分布式计算框架
- 使用 Placement Group 管理 GPU 资源
- 支持 Ray Actor（分布式计算单元）和 Remote Function（远程函数调用）
- 通过 `ray start --head` 启动 Ray 集群

**启动命令**：
```bash
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
  -- python3 train.py ...
```

### 3.5 GRPO 算法的参数设置

**前两轮认知**：知道 GRPO 是一种 RL 算法

**新增理解**：
- GRPO 的关键参数：
  - `--advantage-estimator grpo`：选择 GRPO 算法
  - `--n-samples-per-prompt 4`：每个 prompt 生成 4 个响应
  - `--eps-clip 0.2`：PPO clipping 的下界
  - `--eps-clip-high 0.28`：PPO clipping 的上界（可不对称）
  - `--kl-loss-coef 0.001`：KL 散度 loss 系数
  - `--kl-loss-type low_var_kl`：使用低方差 KL 估计器

**配置示例**：
```bash
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.001
--kl-loss-type low_var_kl
--eps-clip 0.2
--eps-clip-high 0.28
```

### 3.6 Deepscaler 奖励函数

**前两轮认知**：知道 slime 支持自定义奖励函数

**新增理解**：
- `--rm-type deepscaler` 使用 Deepscaler 的数学答案验证器
- 自动提取模型输出中的数学答案
- 与 ground truth 进行比对，返回 0 或 1 的奖励
- 适合数学推理任务

**使用方法**：
```bash
--rm-type deepscaler
--label-key label  # ground truth 的 key
```

---

## Week 4: RL 算法深入

### 4.1 PPO vs GRPO 的核心区别

**前两轮认知**：知道 PPO 需要 Critic 模型，GRPO 不需要

**新增理解**：
- **PPO**：
  - 需要 Critic 模型估计价值函数 V(s)
  - 使用 GAE (Generalized Advantage Estimation) 计算 advantage
  - 公式：A_t = δ_t + (γλ) * A_{t+1}，其中 δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
  - 计算成本高，但更稳定

- **GRPO**：
  - 不需要 Critic 模型
  - 使用组内相对优势
  - 公式：A_i = (r_i - mean(r_group)) / std(r_group)
  - 计算效率高，但需要更多采样

### 4.2 REINFORCE++ 的实现细节

**前两轮认知**：知道 slime 支持 REINFORCE++ 算法

**新增理解**：
- REINFORCE++ 是带折扣回报的 REINFORCE
- 构建 token-level rewards：r_t = -kl_coef * KL_t
- 在最后一个 token 上加上标量 reward
- 计算折扣回报：G_t = r_t + γ * G_{t+1}
- 支持 Advantage 标准化

**公式**：
```python
# 构建 token-level rewards
token_level_rewards = -kl_coef * masked_kl
token_level_rewards[last_idx] += rewards[i]

# 计算折扣回报
for t in reversed(range(...)):
    running_return = token_level_rewards[t] + gamma * running_return
    returns_for_seq[t] = running_return
```

### 4.3 KL 散度的三种估计器

**前两轮认知**：知道 KL 散度用于限制策略偏离

**新增理解**：
- **k1**：KL = log(π/π_ref)，一阶近似，可能为负
- **k2**：KL = (log(π/π_ref))^2 / 2，二阶近似，始终非负
- **k3 (low_var_kl)**：KL = π_ref/π - 1 - log(π_ref/π)，非负、无偏、低方差

**代码实现**：
```python
# python/sglang/utils/ppo_utils.py:11-51
if kl_type == "k1":
    kl = log_ratio
elif kl_type == "k2":
    kl = log_ratio ** 2 / 2
elif kl_type == "k3" or kl_type == "low_var_kl":
    kl = torch.exp(-log_ratio) - 1 - (-log_ratio)
```

### 4.4 CISPO 算法的 stop-gradient 裁剪

**前两轮认知**：知道 slime 支持 CISPO 算法

**新增理解**：
- CISPO 来自 MiniMax-M1 论文
- IS ratio 在 stop-gradient 下裁剪
- 梯度通过 log_probs 流动
- 被裁剪的 token 仍然贡献梯度

**代码实现**：
```python
# python/sglang/utils/ppo_utils.py:151-171
ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
pg_losses = -ratio_truncated.detach() * advantages * log_probs
```

### 4.5 Dual-Clip PPO 的实现

**前两轮认知**：知道 slime 支持 Dual-Clip PPO

**新增理解**：
- 对负 advantage 额外施加上界裁剪
- 防止负 advantage 被 ratio 倍放大
- `eps_clip_c` 必须 > 1.0

**代码实现**：
```python
# python/sglang/utils/ppo_utils.py:138-144
pg_losses3 = -eps_clip_c * advantages
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

### 4.6 OPSM (Off-Policy Sequence Masking)

**前两轮认知**：知道 slime 支持 KL 控制

**新增理解**：
- OPSM 用于处理 off-policy 数据
- 当序列级 KL 超过阈值 `opsm_delta` 且 advantage 为负时，mask 掉整个序列
- 防止 off-policy 负优势样本造成过大梯度

**代码实现**：
```python
# python/sglang/utils/ppo_utils.py:54-92
seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
```

### 4.7 动态采样的实现

**前两轮认知**：知道 slime 支持动态采样

**新增理解**：
- 动态采样是 DAPO 风格的采样策略
- 采样比实际需要更多的 prompts
- 应用过滤函数（如检查奖励标准差 > 0）
- 丢弃不满足条件的样本
- 如果过滤太严格，自动触发新一轮采样

**配置示例**：
```bash
--over-sampling-batch-size 8
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

---

## Week 5: Agent 训练

### 5.1 自定义生成函数的接口设计

**前两轮认知**：知道 slime 支持自定义生成函数

**新增理解**：
- 自定义生成函数需要遵循特定的接口：
  - 输入：`args`, `sample`, `sampling_params`
  - 输出：`sample`（包含 `loss_mask`）
- 支持异步实现（`async def`）
- 可以返回 `list[Sample]` 实现 Fan-out 训练

**代码示例**：
```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    for turn in range(max_turns):
        model_output = await call_sglang(prompt + full_response)
        action, content = parse_action(model_output)
        
        if action == "search":
            tool_output = await search(content)
            loss_masks += [0] * len(tool_tokens)
        
        if action == "answer":
            break
    
    sample.loss_mask = loss_masks
    return sample
```

### 5.2 Loss Masking 的实现细节

**前两轮认知**：知道 Agent 训练需要 Loss Masking

**新增理解**：
- Loss Masking 通过 `Sample.loss_mask` 字段实现
- 在 `loss.py` 中，通过 `loss_mask` 对每个 token 的 loss 进行加权：
  ```python
  loss = (loss * loss_mask).sum() / loss_mask.sum()
  ```
- 工具返回的 token 的 `loss_mask = 0`，不参与 loss 计算
- 模型生成的 token 的 `loss_mask = 1`，参与 loss 计算

### 5.3 Session Routing 的实现

**前两轮认知**：知道 Session Routing 可以复用 prefix cache

**新增理解**：
- 使用 consistent hashing 路由
- 通过 `session_id` 或 `X-SMG-Routing-Key` header 实现
- 将同一 session 的请求路由到同一 worker
- 可以最大化 prefix cache 命中率

**配置示例**：
```bash
--router-policy consistent_hashing
```

### 5.4 Fan-out 训练的实现

**前两轮认知**：知道 slime 支持 Agent 训练

**新增理解**：
- 自定义 generate 函数可以返回 `list[Sample]`
- 允许一次 rollout 拆分为多个可训练段
- 所有段共享同一个 `rollout_id`
- 适合复杂的 Agent 工作流（如子 agent 调用）

**代码示例**：
```python
async def generate(args, sample, sampling_params):
    # 主 agent 生成
    main_output = await call_sglang(main_prompt)
    main_sample = Sample(...)
    
    # 子 agent 生成
    sub_output = await call_sglang(sub_prompt)
    sub_sample = Sample(...)
    
    # 返回多个可训练段
    return [main_sample, sub_sample]
```

---

## Week 6: 端到端项目

### 6.1 性能调优三板斧

**前两轮认知**：知道 SGLang 有多种性能优化技术

**新增理解**：
1. **最大化 batch size**：
   - 确保 `#queue-req` 保持 100-2000
   - `token usage > 0.9`

2. **KV 缓存最大化**：
   - `--mem-fraction-static` 逐步调高直到 OOM 前回退
   - 检查日志中 `available_gpu_mem` 值，5-8GB 为理想范围

3. **CUDA Graph 覆盖**：
   - 适当增大 `--cuda-graph-max-bs`
   - 4090 默认值为 24

### 6.2 监控指标的关键阈值

**前两轮认知**：知道 SGLang 有监控指标

**新增理解**：
- 关键告警指标：
  - `sglang:num_retracted_reqs_total` 增速 → 调度过于激进
  - `sglang:token_usage` 持续 > 0.95 → 接近 OOM 风险
  - `sglang:time_to_first_token_seconds` P99 过大 → prefill 瓶颈
  - `sglang:cache_hit_rate` 过低 → 考虑 `--schedule-policy lpm`

### 6.3 slime 的 Trace Viewer

**前两轮认知**：知道 slime 有调试工具

**新增理解**：
- slime 提供了 Trace Viewer 工具
- 可以可视化每个 sample 的生成过程
- 支持 PD 分解（Prefill/Decode）的虚拟 lane 展示
- 支持过滤、排序、展开/折叠、时间缩放等交互操作

**使用方法**：
```bash
# 保存 rollout 调试数据
python train.py --save-debug-rollout-data /tmp/rollout_0.pt

# 生成时间线
python tools/trace_timeline_viewer.py /tmp/rollout_0.pt
# 生成 .trace_timeline_viewer.html，可在浏览器中查看
```

### 6.4 确定性训练的配置

**前两轮认知**：知道 slime 支持可复现训练

**新增理解**：
- slime 结合 SGLang 的确定性推理和 Megatron-LM 的确定性模式
- 提供比特级实验复现
- 需要设置多个环境变量和配置

**配置示例**：
```bash
--sglang-enable-deterministic-inference
--sglang-attention-backend flashinfer
--deterministic-mode

# 环境变量
NCCL_ALGO=Ring
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 6.5 故障容错机制

**前两轮认知**：知道 slime 有容错机制

**新增理解**：
- slime 提供完整的故障容错机制：
  - 健康检查：定期向所有 SGLang 服务器发送心跳
  - 超时检测：超时后停止不健康的服务器
  - 自动恢复：当前 rollout 完成后重启并更新参数
- 适合长时间运行的训练任务

**配置示例**：
```bash
--use-fault-tolerance
--rollout-health-check-first-wait 600  # MoE 首次编译等待
--rollout-health-check-interval 10    # 心跳间隔
--rollout-health-check-timeout 5      # 心跳超时
```

---

## 持续更新

本文档将随着复现过程的推进持续更新，记录新学到的知识点和经验教训。

---

**最后更新**：2026-06-26（Week 0：计划制定阶段）
**下次更新**：Week 1 完成后
