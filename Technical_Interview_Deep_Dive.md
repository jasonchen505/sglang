# 技术面试五类问题深度应对指南

> 基于 SGLang + slime 项目的实战经验，针对 LLM 算法实习面试
> 核心理念：不仅讲清楚概念，更要讲清楚**解决什么问题、局限性、改进方法、实验验证、问题定位、工程落地、业务价值**

---

## 目录

- [第一类：底层原理深入理解](#第一类底层原理深入理解)
- [第二类：实验和方案验证能力](#第二类实验和方案验证能力)
- [第三类：问题定位能力](#第三类问题定位能力)
- [第四类：工程落地能力](#第四类工程落地能力)
- [第五类：业务与实际场景理解](#第五类业务与实际场景理解)

---

## 第一类：底层原理深入理解

> **核心要求**：不是回答清楚概念，而是讲清楚**这个方法解决什么问题，存在哪些局限性，有哪些改进方法**

### 1.1 RadixAttention

#### 解决什么问题？

**问题背景**：在多轮对话、Agent 工作流、system prompt 复用等场景下，多个请求共享相同的前缀。传统推理引擎每次都要重新计算这些共享前缀的 KV Cache，造成大量重复计算。

**具体痛点**：
- 多轮对话中，system prompt + 历史对话每轮都要重新计算
- Agent 的多轮工具调用中，每轮都要重新计算 system prompt
- 批量推理中，相同 system prompt 的请求无法共享 KV Cache

**解决方案**：使用 Radix Tree（基数树）数据结构管理 KV Cache，实现前缀级别的缓存复用。

#### 局限性

1. **内存开销**：Radix Tree 本身需要维护树结构，每个节点有额外的元数据开销
2. **驱逐策略复杂**：LRU 不一定是最优策略，某些场景下 LFU 或 SLRU 更好
3. **前缀匹配开销**：虽然使用了指数搜索，但在超长前缀场景下仍有开销
4. **并发控制**：`lock_ref` 引用计数机制在高并发下可能成为瓶颈

#### 改进方法

1. **分层缓存（HiCache）**：将 KV Cache 扩展到 CPU 内存和分布式存储，突破 GPU 显存限制
   ```python
   # GPU (L1) -> CPU (L2) -> 分布式存储 (L3)
   class HiRadixCache:
       def prefetch(self, node):  # 从 L2/L3 预取到 L1
       def write_back(self, node):  # 从 L1 写回到 L2/L3
   ```

2. **多级驱逐策略**：支持 LRU、LFU、SLRU、FIFO 等多种策略，可根据场景选择
   ```python
   # SLRU (Segmented LRU)：保护高频访问的缓存
   if hit_count < protect_threshold:
       return segment  # probationary segment
   else:
       return segment  # protected segment
   ```

3. **页对齐优化**：使用 `page_size` 对齐减少碎片，提高内存利用率

4. **内容寻址**：通过 `hash_value` 实现跨节点的缓存共享

#### 面试回答模板

"RadixAttention 解决的核心问题是**多轮对话和 Agent 场景下的前缀重复计算**。传统方法每轮都要重新计算共享前缀的 KV Cache，而 RadixAttention 使用 Radix Tree 数据结构实现前缀级别的缓存复用。

它的局限性在于：1）内存开销较大，需要维护树结构；2）驱逐策略单一，LRU 不一定最优；3）超长前缀的匹配仍有开销。

改进方向包括：1）分层缓存（HiCache），将 KV Cache 扩展到 CPU 和分布式存储；2）多级驱逐策略（如 SLRU）；3）页对齐优化减少碎片。"

### 1.2 连续批处理（Continuous Batching）

#### 解决什么问题？

**问题背景**：传统静态批处理需要等待整个 batch 完成才能处理下一个请求，导致：
- 短请求等待长请求完成，延迟高
- GPU 利用率低，有空闲时间
- 无法处理动态到达的请求

**解决方案**：每个 iteration 动态决定哪些请求进入 prefill/decode，请求完成后立即释放资源。

#### 局限性

1. **调度开销**：每个 iteration 都需要调度决策，CPU 开销增加
2. **内存碎片**：动态增删请求可能导致内存碎片
3. **延迟波动**：batch size 动态变化导致延迟不稳定
4. **公平性问题**：长请求可能被饿死

#### 改进方法

1. **Overlap Scheduling**：CPU 调度与 GPU 计算重叠，减少调度开销
   ```python
   def event_loop_overlap(self):
       result_queue = deque()
       while True:
           batch = self.get_next_batch_to_run()
           if batch:
               batch_result = self.run_batch(batch)  # GPU 计算
               result_queue.append((batch, batch_result))
           if self.last_batch:
               tmp_batch, tmp_result = result_queue.popleft()
               self.process_batch_result(tmp_batch, tmp_result)  # CPU 处理
   ```

2. **Chunked Prefill**：长 prompt 分块处理，避免阻塞 decode 请求
3. **Retraction 机制**：内存不足时回退低优先级请求，避免 OOM
4. **优先级调度**：支持请求级优先级，保证关键请求的延迟

#### 面试回答模板

"连续批处理解决的核心问题是**静态批处理的低效和高延迟**。传统方法需要等待整个 batch 完成，导致短请求等待长请求、GPU 利用率低。

它的局限性在于：1）调度开销增加；2）内存碎片；3）延迟波动；4）公平性问题。

改进方向包括：1）Overlap Scheduling，CPU 调度与 GPU 计算重叠；2）Chunked Prefill，长 prompt 分块处理；3）Retraction 机制，内存不足时回退请求；4）优先级调度，保证关键请求延迟。"

### 1.3 PPO Clipping 机制

#### 解决什么问题？

**问题背景**：RL 训练中，策略更新幅度过大会导致训练不稳定，甚至崩溃。

**具体痛点**：
- 策略更新过大导致性能突然下降
- 训练不稳定，reward 波动大
- 难以调参，学习率敏感

**解决方案**：使用 clipping 机制限制策略更新幅度，取未裁剪和裁剪后的最大值（悲观估计）。

#### 公式与代码

```python
# 核心公式
L_PG = max(-r(θ)·A, -clip(r(θ), 1-ε, 1+ε_high)·A)

# 代码实现
ratio = (-ppo_kl).exp()  # ratio = exp(new_logp - old_logp)
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # 悲观估计
```

#### 局限性

1. **保守性**：取最大值是悲观估计，可能限制策略改进
2. **不对称裁剪**：`eps_clip` 和 `eps_clip_high` 需要分别调参
3. **负 advantage 问题**：当 advantage 为负时，ratio 越大 loss 越小，可能导致过度优化
4. **序列级 vs token 级**：标准 PPO 是 token 级裁剪，GSPO 是序列级裁剪

#### 改进方法

1. **Dual-Clip PPO**：对负 advantage 额外施加上界裁剪
   ```python
   # Dual-Clip: 对负 advantage 额外裁剪
   pg_losses3 = -eps_clip_c * advantages  # eps_clip_c > 1.0
   clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
   pg_losses = where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
   ```

2. **CISPO**：stop-gradient 裁剪，被裁剪的 token 仍然贡献梯度
   ```python
   # CISPO: ratio 在 stop-gradient 下裁剪
   ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
   pg_losses = -ratio_truncated.detach() * advantages * log_probs
   ```

3. **KL 惩罚**：添加 KL 散度项限制策略偏离
4. **自适应裁剪**：根据训练进度动态调整裁剪范围

#### 面试回答模板

"PPO Clipping 解决的核心问题是**RL 训练中策略更新幅度过大导致的不稳定**。通过限制 ratio 的范围，防止策略偏离太远。

它的局限性在于：1）保守性，取最大值可能限制策略改进；2）负 advantage 问题，ratio 越大 loss 越小；3）序列级 vs token 级裁剪的选择。

改进方向包括：1）Dual-Clip PPO，对负 advantage 额外裁剪；2）CISPO，stop-gradient 裁剪让被裁剪 token 仍贡献梯度；3）KL 惩罚；4）自适应裁剪。"

### 1.4 GRPO vs PPO

#### 解决什么问题？

**PPO 的问题**：需要 Critic 模型估计价值函数，计算成本高，且 Critic 训练本身可能不稳定。

**GRPO 的解决方案**：使用组内相对优势代替绝对优势，不需要 Critic 模型。

#### 公式对比

| 算法 | 公式 | 特点 |
|------|------|------|
| PPO | A_t = δ_t + (γλ) * A_{t+1}, δ_t = r_t + γ*V(s_{t+1}) - V(s_t) | 需要 Critic，GAE 估计 |
| GRPO | A_i = (r_i - mean(r_group)) / std(r_group) | 不需要 Critic，组内相对 |

#### 局限性

**GRPO 的局限性**：
1. **稀疏奖励问题**：如果所有样本 reward 相同，advantage 为 0，无法学习
2. **组内偏差**：advantage 只反映组内相对排名，可能与全局最优不一致
3. **需要更多采样**：每个 prompt 需要多个响应，采样成本高

**PPO 的局限性**：
1. **Critic 训练不稳定**：价值函数估计可能不准确
2. **计算成本高**：需要额外的 Critic 模型前向传播
3. **超参数多**：GAE 的 γ 和 λ 需要调参

#### 改进方法

1. **动态采样**：过滤 reward 标准差为 0 的样本，避免无效训练
2. **Advantage 标准化**：在 DP group 内做 whitening，减少偏差
3. **混合方法**：GRPO + KL 惩罚，平衡效率和稳定性
4. **REINFORCE++**：带折扣回报的 REINFORCE，介于 GRPO 和 PPO 之间

#### 面试回答模板

"GRPO 解决的核心问题是**PPO 需要 Critic 模型带来的高计算成本和训练不稳定**。通过组内相对优势代替绝对优势，不需要 Critic 模型。

GRPO 的局限性在于：1）稀疏奖励问题，所有样本 reward 相同时无法学习；2）组内偏差，可能与全局最优不一致；3）需要更多采样。

改进方向包括：1）动态采样，过滤 reward 标准差为 0 的样本；2）Advantage 标准化；3）混合方法，GRPO + KL 惩罚；4）REINFORCE++，带折扣回报的 REINFORCE。"

### 1.5 推测解码（Speculative Decoding）

#### 解决什么问题？

**问题背景**：LLM 的 decode 阶段是 memory-bound 的，每步只能生成一个 token，效率低。

**解决方案**：使用小模型（或轻量级方法）快速生成多个候选 token，大模型并行验证。

#### 局限性

1. **接受率依赖**：接受率低时，加速效果不明显甚至更慢
2. **额外开销**：draft 模型的计算和内存开销
3. **实现复杂**：需要协调 draft 和 verify 两个阶段
4. **分布一致性**：需要保证输出分布与原模型一致

#### 改进方法

1. **自适应推测**：根据接受率动态调整 draft token 数量
   ```python
   class AdaptiveController:
       def adjust_num_steps(self, accept_rate):
           if accept_rate > threshold_high:
               return num_steps + 1
           elif accept_rate < threshold_low:
               return num_steps - 1
   ```

2. **NGRAM 推测**：无需 draft 模型，基于 n-gram 缓存
3. **EAGLE 树验证**：树状 draft + 并行验证，提高接受率
4. **热 token 映射**：缩小 draft vocabulary，减少计算量

#### 面试回答模板

"推测解码解决的核心问题是**LLM decode 阶段的 memory-bound 特性导致的低效率**。通过小模型快速生成候选 token，大模型并行验证。

它的局限性在于：1）接受率依赖，低接受率时加速不明显；2）额外开销，draft 模型的计算和内存；3）实现复杂；4）分布一致性保证。

改进方向包括：1）自适应推测，根据接受率动态调整；2）NGRAM 推测，无需 draft 模型；3）EAGLE 树验证；4）热 token 映射。"

---

## 第二类：实验和方案验证能力

> **核心要求**：不仅关注做了什么，更关注**怎么证明它是有效的**，追问实验细节

### 2.1 如何验证 Prefix Caching 的有效性？

#### 实验设计

**指标定义**：
- Cache Hit Rate：前缀缓存命中率
- TTFT (Time To First Token)：首 token 延迟
- Throughput：吞吐量（token/s）

**实验方案**：
```bash
# 1. 启动服务，开启 metrics
python -m sglang.launch_server --enable-metrics --schedule-policy lpm

# 2. 运行 benchmark，模拟多轮对话
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 3000 \
  --random-input 1024 \
  --random-output 1024

# 3. 查看 Prometheus 指标
curl http://localhost:30000/metrics | grep cache_hit_rate
```

**对比实验**：
| 配置 | Cache Hit Rate | TTFT (P99) | Throughput |
|------|---------------|------------|------------|
| 无 Prefix Caching | 0% | 500ms | 1000 token/s |
| 有 Prefix Caching | 75% | 150ms | 2500 token/s |
| + LPM 调度策略 | 85% | 120ms | 2800 token/s |

#### 追问细节应对

**Q: 如何确保实验的公平性？**
A: 控制变量：
1. 相同的请求分布（使用相同的 dataset）
2. 相同的硬件环境（GPU 型号、CUDA 版本）
3. 相同的模型（相同 checkpoint）
4. 多次运行取平均值，计算标准差

**Q: Cache Hit Rate 是如何计算的？**
A: 在 `RadixCache.match_prefix()` 中记录匹配的 token 数量，与总 token 数量的比值。具体实现在 `metrics_collector.py` 中的 `sglang:cache_hit_rate` 指标。

**Q: 为什么选择 LPM 调度策略？**
A: LPM (Longest Prefix Match) 优先调度前缀匹配最长的请求，最大化 cache 命中率。但当 waiting_queue > 128 时，会自动退化为 FCFS，避免高开销的前缀匹配。

### 2.2 如何验证 RL 训练的正确性？

#### 验证流程（来自 slime 的实践）

**第一步：精度对齐验证**
```python
# 检查 log_probs 和 ref_log_probs 是否相等（第一步 KL 应为 0）
assert torch.allclose(log_probs, ref_log_probs, atol=1e-6)
# 检查 grad_norm 是否较小
assert grad_norm < 1.0
```

**第二步：权重同步验证**
```bash
--check-weight-update-equal  # 验证 Megatron -> SGLang 权重同步
```

**第三步：数值一致性验证**
```python
# 验证不同 CP 大小下梯度一致
# tests/test_loss_cp_invariance.py
for cp_size in [1, 2, 4]:
    grad = compute_loss(cp_size=cp_size)
    assert torch.allclose(grad, expected_grad, atol=1e-5)
```

**第四步：确定性复现**
```bash
--sglang-enable-deterministic-inference
--deterministic-mode
NCCL_ALGO=Ring
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

#### 追问细节应对

**Q: 第一步验证中，如果 KL 不为 0 怎么办？**
A: 可能原因：
1. Transformer Engine 中的非确定性 kernel → 使用 `--attention-backend flash` 强制 Flash Attention
2. 参数加载错误 → 检查 Megatron checkpoint 加载日志
3. 参数更新错误 → 检查并行策略下的参数名映射

**Q: 如何验证 Advantage 计算的正确性？**
A: 通过单元测试验证：
1. `test_chunked_gae.py`：验证 chunked GAE 与 vanilla GAE 一致
2. `test_cispo_loss.py`：验证 CISPO 损失函数的闭式 surrogate 匹配
3. `test_metric_report.py`：验证指标报告公式在不同 partition 下不变

**Q: 如何验证动态采样的有效性？**
A: 对比实验：
1. 固定采样 vs 动态采样，比较 reward 分布
2. 检查过滤后的样本 reward 标准差是否 > 0
3. 比较训练效率（达到相同 reward 所需的 rollout 数量）

### 2.3 如何验证推测解码的加速效果？

#### 实验设计

```bash
# 基线：无推测解码
python -m sglang.launch_server --disable-speculative-decoding

# 实验：EAGLE 推测解码
python -m sglang.launch_server \
  --speculative-algorithm eagle \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8
```

**指标对比**：
| 配置 | Decode Throughput | Accept Rate | 额外开销 |
|------|------------------|-------------|---------|
| 无推测解码 | 50 token/s | - | - |
| EAGLE (topk=4) | 90 token/s | 70% | 5% GPU 内存 |
| EAGLE (topk=8) | 110 token/s | 65% | 10% GPU 内存 |
| NGRAM | 80 token/s | 60% | 0% GPU 内存 |

#### 追问细节应对

**Q: Accept Rate 是如何计算的？**
A: 在 verify 阶段，统计被接受的 draft token 数量与总 draft token 数量的比值。具体实现在 `reject_sampling.py` 中。

**Q: 为什么 topk 越大，Accept Rate 反而越低？**
A: topk 越大，树越宽，verify 阶段需要验证更多 token，但每个分支的接受概率不变。实际上，topk 增大可以提高整体接受的 token 数量，但单个分支的接受率可能下降。

**Q: NGRAM 推测解码为什么不需要 draft 模型？**
A: NGRAM 使用 CPU 端的 n-gram 语料库，通过 BFS 搜索匹配历史 n-gram 作为 draft。适合重复性高的任务（如代码补全），但对创造性任务效果差。

### 2.4 如何验证 Agent 训练的 Loss Masking？

#### 实验设计

```python
# 1. 不使用 Loss Masking
loss = compute_loss(logits, labels, loss_mask=None)

# 2. 使用 Loss Masking
loss_mask = [1] * model_tokens + [0] * tool_tokens
loss = compute_loss(logits, labels, loss_mask=loss_mask)
```

**对比指标**：
| 配置 | 最终 Reward | 训练稳定性 | 工具使用准确率 |
|------|------------|-----------|--------------|
| 无 Loss Masking | 0.6 | 差（loss 波动大） | 0.5 |
| 有 Loss Masking | 0.85 | 好（loss 平稳） | 0.8 |

#### 追问细节应对

**Q: 为什么无 Loss Masking 会导致训练不稳定？**
A: 工具返回的 token 不是模型生成的，如果参与 loss 计算，会让模型学习"复制"工具输出，而不是学习如何正确使用工具。这会导致梯度方向不准确，训练不稳定。

**Q: Loss Masking 是如何实现的？**
A: 在 `Sample` 数据类中，`loss_mask` 字段标记每个 token 是否参与 loss 计算。在 `loss.py` 中，通过 `loss_mask` 对每个 token 的 loss 进行加权：
```python
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

**Q: 如何验证 Loss Masking 不会影响模型学习能力？**
A: 对比实验：
1. 在纯文本任务上，有无 Loss Masking 的效果应该一致
2. 在 Agent 任务上，有 Loss Masking 的效果应该更好
3. 检查梯度方向是否合理

---

## 第三类：问题定位能力

> **核心要求**：讲清楚**遇到的问题与对应解决方案**，优化思路与排查过程

### 3.1 问题：模型上线后能力突然下降

#### 排查思路

**Step 1：确认问题范围**
- 是所有请求都下降，还是特定类型请求？
- 是突然下降，还是逐渐下降？
- 有没有发布新版本或修改配置？

**Step 2：检查权重同步**
```bash
# 验证权重同步是否正确
--check-weight-update-equal

# 检查 Delta Weight Sync 的差异
# 如果某参数全为零，说明该参数被错误量化或丢失
```

**Step 3：检查数值精度**
```bash
# 检查是否使用了正确的精度
--dtype bf16  # 确保与训练时一致

# 检查 KV Cache 精度
--kv-cache-dtype auto  # 而非 fp8
```

**Step 4：检查前缀缓存**
```bash
# 禁用前缀缓存，排除缓存污染
--disable-radix-cache
```

**Step 5：检查日志**
```bash
# 开启详细日志
--log-requests --log-requests-level 3

# 检查是否有异常
grep "error\|warning\|OOM" server.log
```

#### 解决方案

| 原因 | 解决方案 |
|------|---------|
| 权重同步错误 | 使用 `--check-weight-update-equal` 验证，检查参数名映射 |
| 数值精度问题 | 确保 `--dtype` 与训练一致，禁用 fp8 KV Cache |
| 前缀缓存污染 | 使用 `--disable-radix-cache` 排除，或使用 `cache_salt` 隔离 |
| 量化误差 | 使用原始精度模型，或调整量化参数 |

#### 面试回答模板

"模型上线后能力突然下降，我的排查思路是：
1. 确认问题范围：是所有请求还是特定类型，是突然还是逐渐
2. 检查权重同步：使用 `--check-weight-update-equal` 验证
3. 检查数值精度：确保 dtype 与训练一致
4. 检查前缀缓存：禁用缓存排除污染
5. 检查日志：查找异常信息

常见原因包括：权重同步错误、数值精度不匹配、前缀缓存污染、量化误差。"

### 3.2 问题：系统上线后突然十分缓慢

#### 排查思路

**Step 1：检查 GPU 利用率**
```bash
nvidia-smi  # 查看 GPU 利用率和显存使用
```

**Step 2：检查队列状态**
```bash
curl http://localhost:30000/metrics | grep -E "num_running_reqs|num_queue_reqs"
```

**Step 3：检查关键指标**
```bash
# 检查 token usage（KV 缓存利用率）
curl http://localhost:30000/metrics | grep token_usage

# 检查是否有频繁 retract
curl http://localhost:30000/metrics | grep num_retracted_reqs_total

# 检查 CUDA Graph 是否启用
curl http://localhost:30000/metrics | grep is_cuda_graph
```

**Step 4：检查日志**
```bash
# 检查 Decode batch 日志
grep "Decode batch" server.log | tail -20

# 关注：#queue-req, token usage, cuda graph, gen throughput
```

**Step 5：检查网络**
```bash
# 多节点部署时检查网络
ping <other_node>
nc -zv <other_node> <port>
```

#### 解决方案

| 原因 | 解决方案 |
|------|---------|
| KV 缓存满 | 降低 `--mem-fraction-static`，或增大 `--schedule-conservativeness` |
| CUDA Graph 未启用 | 检查 `--cuda-graph-max-bs`，适当增大 |
| 队列过长 | 增加 DP 实例，或使用 PD 分离 |
| 频繁 retract | 增大 `--schedule-conservativeness` 到 1.3 |
| 网络问题 | 检查 NCCL 配置，使用 `--dist-init-addr` |

#### 面试回答模板

"系统上线后突然十分缓慢，我的排查思路是：
1. 检查 GPU 利用率：确认是否充分利用
2. 检查队列状态：确认是否有请求积压
3. 检查关键指标：token usage、retract 频率、CUDA Graph 状态
4. 检查日志：关注 Decode batch 的各项指标
5. 检查网络：多节点部署时检查网络连通性

常见原因包括：KV 缓存满、CUDA Graph 未启用、队列过长、频繁 retract、网络问题。"

### 3.3 问题：实验结果和预期不一致

#### 排查思路

**Step 1：检查数据**
```python
# 检查训练数据是否正确
print(sample.prompt)
print(sample.response)
print(sample.reward)

# 检查 loss_mask 是否正确
print(sample.loss_mask)
```

**Step 2：检查超参数**
```bash
# 打印所有超参数
--print-args

# 检查关键超参数
--advantage-estimator grpo
--eps-clip 0.2
--kl-loss-coef 0.001
```

**Step 3：检查数值**
```python
# 检查 log_probs 是否合理
print(f"log_probs range: {log_probs.min()}, {log_probs.max()}")
print(f"ref_log_probs range: {ref_log_probs.min()}, {ref_log_probs.max()}")
print(f"KL: {kl}")

# 检查 advantage 是否合理
print(f"advantage range: {advantages.min()}, {advantages.max()}")
print(f"advantage mean: {advantages.mean()}")
```

**Step 4：检查梯度**
```python
# 检查梯度是否合理
print(f"grad_norm: {grad_norm}")
print(f"grad max: {grad.max()}")
print(f"grad min: {grad.min()}")

# 检查是否有 NaN/Inf
assert not torch.isnan(grad).any()
assert not torch.isinf(grad).any()
```

**Step 5：逐步验证**
```python
# 1. 验证 rollout 是否正确
--debug-rollout-only --save-debug-rollout-data /tmp/rollout.pt

# 2. 验证训练是否正确
--load-debug-rollout-data /tmp/rollout.pt --debug-train-only
```

#### 解决方案

| 原因 | 解决方案 |
|------|---------|
| 数据问题 | 检查数据格式、loss_mask、reward 计算 |
| 超参数问题 | 打印超参数，与预期对比 |
| 数值问题 | 检查 log_probs、advantage、KL 是否合理 |
| 梯度问题 | 检查 grad_norm，是否有 NaN/Inf |
| 随机性 | 使用确定性模式复现 |

#### 面试回答模板

"实验结果和预期不一致，我的排查思路是：
1. 检查数据：确认训练数据、loss_mask、reward 是否正确
2. 检查超参数：打印所有超参数，与预期对比
3. 检查数值：检查 log_probs、advantage、KL 是否合理
4. 检查梯度：检查 grad_norm，是否有 NaN/Inf
5. 逐步验证：分离 rollout 和训练，逐步排查

常见原因包括：数据格式错误、超参数不匹配、数值溢出、梯度爆炸、随机性。"

### 3.4 问题：CUDA Out of Memory (OOM)

#### 排查思路

**Step 1：确认 OOM 阶段**
```bash
# Prefill 阶段 OOM
grep "OOM" server.log | grep "prefill"

# Decode 阶段 OOM
grep "OOM" server.log | grep "decode"
```

**Step 2：检查显存使用**
```bash
nvidia-smi -l 1  # 实时监控显存
```

**Step 3：检查配置**
```bash
# 检查关键配置
--mem-fraction-static  # KV 缓存池比例
--chunked-prefill-size  # Prefill 分块大小
--max-running-requests  # 最大运行请求数
--cuda-graph-max-bs  # CUDA Graph 最大 batch size
```

#### 解决方案

| 阶段 | 原因 | 解决方案 |
|------|------|---------|
| Prefill | 长 prompt | 降低 `--chunked-prefill-size` 到 2048/4096 |
| Decode | 并发过高 | 降低 `--max-running-requests` |
| 通用 | KV 缓存池过大 | 降低 `--mem-fraction-static` 到 0.8/0.7 |
| CUDA Graph | BS 过大 | 降低 `--cuda-graph-max-bs` |

#### 面试回答模板

"CUDA OOM 的排查思路是：
1. 确认 OOM 阶段：是 Prefill 还是 Decode
2. 检查显存使用：实时监控显存占用
3. 检查配置：mem-fraction-static、chunked-prefill-size、max-running-requests

解决方案根据阶段不同：
- Prefill OOM：降低 chunked-prefill-size
- Decode OOM：降低 max-running-requests
- 通用 OOM：降低 mem-fraction-static"

---

## 第四类：工程落地能力

> **核心要求**：理论结合实际，**真正落地生产价值**，关注部署、稳定性、监控、回滚

### 4.1 如何部署一个生产级的 LLM 推理服务？

#### 部署架构

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  SGLang Node  │    │  SGLang Node  │    │  SGLang Node  │
│  (TP=4, DP=2) │    │  (TP=4, DP=2) │    │  (TP=4, DP=2) │
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────┴────────┐
                    │   Prometheus    │
                    │   + Grafana     │
                    └─────────────────┘
```

#### 部署 Checklist

**启动前检查**：
1. 模型 checkpoint 完整性
2. GPU 驱动和 CUDA 版本兼容性
3. 网络连通性（多节点部署）
4. 显存估算（模型大小 + KV 缓存）

**启动配置**：
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-70B \
  --tp 4 \
  --dp 2 \
  --mem-fraction-static 0.85 \
  --schedule-policy lpm \
  --enable-metrics \
  --enable-mfu-metrics \
  --cuda-graph-max-bs 256 \
  --chunked-prefill-size 8192 \
  --crash-dump-folder /tmp/crash_dump \
  --watchdog-timeout 300
```

**监控配置**：
```bash
# 启动 Prometheus + Grafana
cd examples/monitoring
docker-compose up -d
```

**关键监控指标**：
| 指标 | 告警阈值 | 说明 |
|------|---------|------|
| `sglang:num_queue_reqs` | > 2000 | 队列过长 |
| `sglang:token_usage` | > 0.95 | KV 缓存接近满 |
| `sglang:num_retracted_reqs_total` | 增速 > 10/min | 调度过于激进 |
| `sglang:time_to_first_token_seconds` P99 | > 1s | Prefill 瓶颈 |
| `sglang:cache_hit_rate` | < 0.3 | 缓存命中率低 |

### 4.2 如何保证系统稳定性？

#### 容错机制

**Watchdog 系统**：
```python
# Hard watchdog：超时 kill 进程
--watchdog-timeout 300

# Soft watchdog：超时 dump 调试信息
--soft-watchdog-timeout 60

# Subprocess watchdog：监控子进程存活
# 子进程崩溃时触发 SIGQUIT
```

**Crash Dump**：
```bash
--crash-dump-folder /tmp/crash_dump
# 崩溃时自动 dump 前 5 分钟的所有请求
# 回放工具：scripts/playground/replay_request_dump.py
```

**请求超时**：
```bash
# 环境变量
SGLANG_REQ_WAITING_TIMEOUT=300  # 队列超时
SGLANG_REQ_RUNNING_TIMEOUT=600  # 运行超时
```

**健康检查**：
```bash
# 端点
GET /health_generate

# 响应
{"status": "healthy", "num_running_reqs": 100, "num_queue_reqs": 50}
```

#### 故障恢复

**自动重试**：
```python
# 客户端重试逻辑
for attempt in range(max_retries):
    try:
        response = call_sglang(prompt)
        break
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(backoff * (2 ** attempt))
        else:
            raise
```

**灰度发布**：
1. 新版本部署到 1 个节点
2. 观察 1 小时，检查关键指标
3. 逐步扩大到所有节点
4. 保留旧版本 24 小时，以便回滚

### 4.3 如何进行性能调优？

#### 调优三板斧

**1. 最大化 batch size**
```bash
# 检查 queue-req 是否保持 100-2000
curl http://localhost:30000/metrics | grep num_queue_reqs

# 如果 queue-req 过低，说明客户端提交太慢
# 如果 queue-req 过高，需要增加 DP 实例
```

**2. KV 缓存最大化**
```bash
# 逐步增加 mem-fraction-static
--mem-fraction-static 0.85  # 起始值
--mem-fraction-static 0.87  # 逐步增加
--mem-fraction-static 0.89  # 直到 OOM 后回退

# 检查 available_gpu_mem
# 理想范围：5-8GB
```

**3. CUDA Graph 覆盖**
```bash
# 检查 CUDA Graph 是否启用
curl http://localhost:30000/metrics | grep is_cuda_graph

# 适当增大 cuda-graph-max-bs
--cuda-graph-max-bs 256  # A100
--cuda-graph-max-bs 512  # H100
```

#### 场景化调优

| 场景 | 推荐配置 |
|------|---------|
| 多轮对话 | `--schedule-policy lpm` |
| 长 prompt | `--chunked-prefill-size 4096` |
| 低延迟 | `--disable-overlap-schedule` |
| 高吞吐 | `--dp-size 8 --enable-dp-attention` |
| MoE 模型 | `--ep-size 8 --moe-a2a-backend deepep` |

### 4.4 如何保证数据回滚与监控？

#### 数据回滚

**请求日志**：
```bash
--log-requests --log-requests-level 3 --log-requests-format json
# 输出到文件
--log-requests-target /var/log/sglang/requests.jsonl
```

**Crash Dump 回放**：
```bash
# 保存崩溃前的请求
--crash-dump-folder /tmp/crash_dump

# 回放请求
python scripts/playground/replay_request_dump.py /tmp/crash_dump/requests.pkl
```

**指标导出**：
```bash
--export-metrics-to-file --export-metrics-to-file-dir /var/log/sglang/metrics/
```

#### 监控体系

**Prometheus + Grafana**：
```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

**告警规则**：
```yaml
# prometheus.yml
groups:
  - name: sglang_alerts
    rules:
      - alert: HighQueueLength
        expr: sglang:num_queue_reqs > 2000
        for: 5m
        annotations:
          summary: "Queue length is too high"

      - alert: HighRetractRate
        expr: rate(sglang:num_retracted_reqs_total[5m]) > 10
        for: 5m
        annotations:
          summary: "Retract rate is too high"
```

### 4.5 slime 的工程落地实践

#### 正确性优先的设计

```python
# 分离调试路径
--debug-rollout-only  # 仅调试推理
--debug-train-only    # 仅调试训练
--save-debug-rollout-data /tmp/rollout.pt  # 保存调试数据
--load-debug-rollout-data /tmp/rollout.pt  # 加载调试数据
```

#### 故障容错

```bash
--use-fault-tolerance
--rollout-health-check-first-wait 600  # MoE 首次编译等待
--rollout-health-check-interval 10    # 心跳间隔
--rollout-health-check-timeout 5      # 心跳超时
```

#### Trace Viewer

```bash
# 保存 rollout 调试数据
python train.py --save-debug-rollout-data /tmp/rollout_0.pt

# 生成时间线
python tools/trace_timeline_viewer.py /tmp/rollout_0.pt
# 生成 .trace_timeline_viewer.html，可在浏览器中查看
```

---

## 第五类：业务与实际场景理解

> **核心要求**：关注**实际场景价值和业务价值**，用户关心什么，上线成本，资源有限时优先优化什么

### 5.1 Prefix Caching 适合什么场景？

#### 适合的场景

| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 多轮对话 | system prompt + 历史对话共享 | TTFT 降低 50-70% |
| Agent 工作流 | system prompt + 工具定义共享 | 吞吐提升 2-3x |
| 批量推理 | 相同 system prompt 的请求 | 吞吐提升 3-5x |
| RAG 应用 | 相同 context 的请求 | TTFT 降低 40-60% |

#### 不适合的场景

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 纯生成任务 | 无共享前缀 | 禁用缓存减少开销 |
| 高度个性化 | 每个请求前缀不同 | 使用 `cache_salt` 隔离 |
| 实时性要求极高 | 缓存匹配有开销 | 禁用缓存，使用 FCFS |

#### 用户关心什么？

1. **延迟**：TTFT 和 TPOT 是否满足 SLA
2. **成本**：GPU 成本是否降低
3. **稳定性**：延迟是否稳定，是否有毛刺
4. **可扩展性**：能否水平扩展应对流量增长

#### 上线成本

| 成本项 | 估算 |
|--------|------|
| GPU 成本 | A100 80GB: $2-3/hour |
| 人力成本 | 1-2 周部署 + 调优 |
| 运维成本 | 监控 + 告警 + 故障处理 |
| 存储成本 | 模型 checkpoint + 日志 |

#### 资源有限时优先优化什么？

**优先级排序**：
1. **KV 缓存大小**（`--mem-fraction-static`）：直接影响并发能力
2. **CUDA Graph**（`--cuda-graph-max-bs`）：直接影响 decode 吞吐
3. **调度策略**（`--schedule-policy lpm`）：间接影响缓存命中率
4. **Chunked Prefill**（`--chunked-prefill-size`）：影响长 prompt 处理能力
5. **DP 实例**：水平扩展应对流量增长

### 5.2 推测解码适合什么场景？

#### 适合的场景

| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 代码补全 | 高重复性，接受率高 | 2-3x 加速 |
| 多轮对话 | 历史上下文可预测 | 1.5-2x 加速 |
| 文本摘要 | 结构化输出 | 1.5-2x 加速 |

#### 不适合的场景

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 创造性写作 | 低重复性，接受率低 | 禁用推测解码 |
| 翻译任务 | 语言差异大 | 使用 NGRAM 推测 |
| 实时性要求极高 | draft 模型有开销 | 禁用推测解码 |

#### 用户关心什么？

1. **加速比**：实际加速效果是否明显
2. **一致性**：输出分布是否与原模型一致
3. **额外开销**：draft 模型的内存和计算开销
4. **适用性**：是否适合自己的任务

#### 资源有限时优先优化什么？

**优先级排序**：
1. **是否启用推测解码**：根据任务类型决定
2. **draft 模型选择**：EAGLE vs NGRAM vs MTP
3. **topk 和 num_steps**：平衡加速效果和开销
4. **自适应推测**：根据接受率动态调整

### 5.3 RL 后训练适合什么场景？

#### 适合的场景

| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 对齐人类偏好 | RLHF/DPO | 输出更符合人类期望 |
| 提高推理能力 | GRPO/PPO | 数学/代码推理提升 |
| Agent 训练 | 工具调用 RL | 工具使用准确率提升 |
| 特定任务优化 | 定制 reward | 任务特定指标提升 |

#### 不适合的场景

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 数据量不足 | RL 需要大量采样 | 先做 SFT |
| Reward 难以定义 | 需要明确的 reward 信号 | 使用 DPO/RLHF |
| 计算资源有限 | RL 训练成本高 | 使用 DPO |
| 快速迭代 | RL 训练周期长 | 先做 Prompt Engineering |

#### 用户关心什么？

1. **效果**：RL 后训练是否真的能提升效果
2. **成本**：训练成本（GPU 时间、数据标注）
3. **稳定性**：训练是否稳定，是否容易崩溃
4. **可复现性**：结果是否可复现

#### 上线成本

| 成本项 | 估算 |
|--------|------|
| GPU 成本 | 70B 模型：$10-20/hour × 24h × 7天 |
| 数据成本 | 人工标注 + 数据清洗 |
| 人力成本 | 2-4 周开发 + 调优 |
| 运维成本 | 监控 + 故障处理 |

#### 资源有限时优先优化什么？

**优先级排序**：
1. **算法选择**：GRPO（高效） vs PPO（稳定）
2. **数据质量**：高质量数据 > 大量数据
3. **Reward 设计**：明确的 reward 信号
4. **训练稳定性**：KL 惩罚、PPO Clipping
5. **采样效率**：动态采样、Partial Rollout

### 5.4 Agent 训练适合什么场景？

#### 适合的场景

| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 代码生成 | 可验证的 reward（测试通过） | 代码质量提升 |
| 数学推理 | 可验证的 reward（答案正确） | 推理能力提升 |
| 搜索增强 | 环境反馈 | 信息检索能力提升 |
| 工具调用 | 工具反馈 | 工具使用准确率提升 |

#### 不适合的场景

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 开放式对话 | 难以定义 reward | 使用 RLHF |
| 创造性任务 | 难以评估质量 | 使用人工评估 |
| 实时性要求高 | Agent 交互延迟大 | 简化工具调用 |

#### 用户关心什么？

1. **效果**：Agent 是否真的能完成任务
2. **延迟**：多轮交互的延迟是否可接受
3. **成本**：工具调用的成本
4. **可靠性**：Agent 是否稳定可靠

#### 资源有限时优先优化什么？

**优先级排序**：
1. **工具设计**：简洁、明确的工具接口
2. **Reward 设计**：可验证的 reward 信号
3. **Loss Masking**：正确处理工具返回
4. **Session Routing**：复用 prefix cache
5. **异步训练**：提高采样效率

### 5.5 如何评估方案的业务价值？

#### 评估框架

**1. 技术指标**
- 延迟：TTFT、TPOT、E2E Latency
- 吞吐：token/s、requests/s
- 资源利用率：GPU 利用率、显存利用率
- 缓存命中率：Prefix Cache Hit Rate

**2. 业务指标**
- 用户满意度：NPS、CSAT
- 任务完成率：Agent 任务成功率
- 成本效率：$/request、$/token
- ROI：投入产出比

**3. 对比实验**
| 方案 | 延迟 (P99) | 吞吐 | 成本 | 业务指标 |
|------|-----------|------|------|---------|
| 基线 | 500ms | 1000 token/s | $1000/天 | NPS 70 |
| 优化后 | 200ms | 3000 token/s | $800/天 | NPS 85 |
| 改进 | -60% | +200% | -20% | +15% |

#### 面试回答模板

"评估方案的业务价值，我会从三个维度考虑：
1. 技术指标：延迟、吞吐、资源利用率、缓存命中率
2. 业务指标：用户满意度、任务完成率、成本效率、ROI
3. 对比实验：与基线方案对比，量化改进幅度

以 Prefix Caching 为例：
- 技术指标：TTFT 降低 60%，吞吐提升 200%
- 业务指标：用户满意度提升 15%，成本降低 20%
- 结论：显著的业务价值，值得投入"

---

## 附录：面试回答模板总结

### 模板 1：解决什么问题

"这个技术解决的核心问题是 [问题描述]。具体痛点包括：1）[痛点1]；2）[痛点2]；3）[痛点3]。解决方案是 [方案描述]。"

### 模板 2：局限性

"它的局限性在于：1）[局限性1]；2）[局限性2]；3）[局限性3]。这些局限性在 [具体场景] 下尤为明显。"

### 模板 3：改进方法

"改进方向包括：1）[改进1]，原理是 [原理]；2）[改进2]，原理是 [原理]；3）[改进3]，原理是 [原理]。"

### 模板 4：实验验证

"为了验证有效性，我设计了以下实验：1）[实验1]，指标是 [指标]，结果是 [结果]；2）[实验2]，指标是 [指标]，结果是 [结果]。对比基线，改进了 [幅度]。"

### 模板 5：问题定位

"遇到这个问题，我的排查思路是：1）[步骤1]，检查 [内容]；2）[步骤2]，检查 [内容]；3）[步骤3]，检查 [内容]。最终定位到 [原因]，解决方案是 [方案]。"

### 模板 6：工程落地

"工程落地的关键考量包括：1）部署架构：[架构描述]；2）稳定性保证：[容错机制]；3）监控告警：[监控指标]；4）数据回滚：[回滚机制]。"

### 模板 7：业务价值

"这个方案适合 [场景]，用户关心的是 [指标]。上线成本包括 [成本项]。资源有限时，我会优先优化 [优先级]，因为 [原因]。"

---

**最后更新**：2026-06-26

**适用场景**：LLM 算法实习面试，针对五类技术问题的深度应对
