# SGLang + slime 8卡4090完整复现计划

> 基于实际硬件资源（8×RTX 4090 24GB）制定的分阶段学习与复现方案
> 目标：完整掌握 SGLang 推理框架 + slime 后训练框架的核心技术

---

## 硬件资源评估

### RTX 4090 规格
| 参数 | 数值 |
|------|------|
| 显存 | 24GB GDDR6X |
| CUDA 核心 | 16384 |
| 内存带宽 | 1008 GB/s |
| 架构 | Ada Lovelace (SM89) |
| NVLink | 不支持（PCIe 4.0） |
| FP16 算力 | 82.6 TFLOPS |

### 资源限制与适配
- **单卡显存**：24GB，可运行 8B 模型（FP16 约 16GB）或 4B 模型（充足空间）
- **无 NVLink**：TP 通信走 PCIe，建议 TP≤2
- **SGLang 自动适配**：chunked_prefill_size=2048，cuda_graph_max_bs=24
- **总显存**：8×24GB = 192GB，可支持 4B 模型的 Colocate 训练

---

## 复现总览（6周计划）

```
Week 1: 环境搭建 + SGLang 基础功能验证
Week 2: SGLang 核心特性深入 + 性能基准测试
Week 3: slime 基础训练流程（小模型验证）
Week 4: slime RL 算法深入 + 实验对比
Week 5: Agent 训练 + 高级特性
Week 6: 端到端项目 + 总结复盘
```

---

## Week 1: 环境搭建 + SGLang 基础功能验证

### 目标
- 搭建完整的开发环境
- 验证 SGLang 基础推理功能
- 理解 SGLang 的核心架构

### Day 1-2: 环境搭建

#### 1.1 Conda 环境
```bash
# 创建环境
conda create -n sglang python=3.12 -y
conda activate sglang

# 安装 SGLang（从源码）
cd /data/home/yizhou/sglang
pip install -e "python[all]"

# 验证安装
python -c "import sglang; print(sglang.__version__)"
```

#### 1.2 Docker 环境（备选）
```bash
# 使用官方 Docker
docker pull lmsysorg/sglang:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v /data/home/yizhou:/workspace \
  -it lmsysorg/sglang:latest /bin/bash
```

#### 1.3 下载模型
```bash
# 小模型（快速验证）
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir /root/models/Llama-3.2-1B

# 中等模型（标准测试）
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /root/models/Llama-3.1-8B

# 推理蒸馏模型（推理基准）
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /root/models/DeepSeek-R1-Distill-Qwen-1.5B
```

### Day 3-4: SGLang 基础功能验证

#### 1.4 最简验证（bench_one_batch）
```bash
# 无需下载模型，使用 dummy weights
python -m sglang.benchmark.one_batch \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
  --correct

# 学习点：
# - 理解 SGLang 的启动流程
# - 理解 ForwardMode（EXTEND/DECODE）
# - 理解 dummy weights 的作用
```

#### 1.5 启动推理服务器
```bash
# 单卡 1B 模型
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 --port 30000

# 测试请求
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'

# 学习点：
# - 理解 HTTP Server 架构
# - 理解 OpenAI API 兼容性
# - 理解请求处理流程
```

#### 1.6 使用 Engine API
```python
import sglang as sgl

# 离线批处理
llm = sgl.Engine(model_path="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(
    ["Hello, my name is", "The capital of France is"],
    {"temperature": 0.8, "top_p": 0.95}
)
print(outputs)
llm.shutdown()

# 学习点：
# - 理解 Engine vs HTTP Server 的区别
# - 理解批处理推理
# - 理解采样参数
```

### Day 5-7: 理解核心架构

#### 1.7 阅读核心代码
```python
# 阅读顺序：
# 1. 入口点：python/sglang/srt/entrypoints/engine.py
# 2. 调度器：python/sglang/srt/managers/scheduler.py（重点）
# 3. 内存管理：python/sglang/srt/mem_cache/radix_cache.py
# 4. 模型执行：python/sglang/srt/model_executor/model_runner.py

# 学习点：
# - 多进程架构（TokenizerManager、Scheduler、DetokenizerManager）
# - ZMQ IPC 通信
# - 连续批处理实现
# - RadixCache 数据结构
```

#### 1.8 画架构图
```
用户请求 → TokenizerManager → Scheduler → ModelRunner → DetokenizerManager
                ↓                   ↓            ↓              ↓
            ZMQ IPC            RadixCache    CUDA Graph    ZMQ IPC
                              SchedulePolicy  ForwardBatch
```

### Week 1 验收标准
- [ ] SGLang 环境安装成功
- [ ] 1B 模型推理正常
- [ ] 理解 SGLang 的多进程架构
- [ ] 理解 RadixCache 的基本原理
- [ ] 能解释连续批处理的工作流程

---

## Week 2: SGLang 核心特性深入 + 性能基准测试

### 目标
- 深入理解 Prefix Caching
- 验证推测解码效果
- 运行性能基准测试

### Day 1-2: Prefix Caching 验证

#### 2.1 对比实验设计
```bash
# 实验组：开启 Prefix Caching（默认）
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.85 \
  --schedule-policy lpm

# 对照组：关闭 Prefix Caching
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30001 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache

# 基准测试
python3 -m sglang.bench_serving \
  --backend sglang --port 30000 \
  --dataset-name random --num-prompts 200 \
  --random-input-len 1024 --random-output-len 256 \
  --output-file results_with_cache.jsonl

python3 -m sglang.bench_serving \
  --backend sglang --port 30001 \
  --dataset-name random --num-prompts 200 \
  --random-input-len 1024 --random-output-len 256 \
  --output-file results_without_cache.jsonl

# 学习点：
# - Cache Hit Rate 的计算方法
# - TTFT 的变化
# - 吞吐量的变化
```

#### 2.2 多轮对话场景验证
```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, system_prompt, questions):
    s += sgl.system(system_prompt)
    for q in questions:
        s += sgl.user(q)
        s += sgl.assistant(sgl.gen("answer", max_tokens=256))

# 构造共享 system_prompt 的多轮对话
# 观察第二次及之后的请求是否命中缓存

# 学习点：
# - 前缀匹配的实现
# - Radix Tree 的构建过程
# - 缓存淘汰策略
```

### Day 3-4: 推测解码验证

#### 2.3 EAGLE 推测解码
```bash
# 启动带推测解码的服务器
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm eagle \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --host 0.0.0.0 --port 30000

# 对比测试
python3 -m sglang.bench_serving \
  --backend sglang --port 30000 \
  --dataset-name random --num-prompts 100 \
  --random-input-len 512 --random-output-len 256 \
  --output-file results_eagle.jsonl

# 学习点：
# - Draft-Verify 机制
# - Accept Rate 的计算
# - Tree Verification 的实现
```

#### 2.4 NGRAM 推测解码
```bash
# 无需 draft 模型的推测解码
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm ngram \
  --speculative-num-steps 3

# 学习点：
# - N-gram 匹配的原理
# - 与 EAGLE 的对比
# - 适用场景分析
```

### Day 5-7: 性能基准测试

#### 2.5 完整性能测试
```bash
# 测试不同 batch size 的延迟
python3 -m sglang.benchmark.one_batch \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --batch-size 1 4 8 16 24 \
  --input-len 256 512 1024 \
  --output-len 128 256

# 测试不同并发数的吞吐
for concurrency in 1 4 8 16 32 64; do
  python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random --num-prompts 100 \
    --random-input-len 512 --random-output-len 128 \
    --max-concurrency $concurrency \
    --output-file results_concurrency_${concurrency}.jsonl
done

# 学习点：
# - 延迟 vs 吞吐的权衡
# - CUDA Graph 的加速效果
# - 显存与并发的关系
```

#### 2.6 监控指标分析
```bash
# 启动时开启 metrics
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-metrics --enable-mfu-metrics

# 查看指标
curl http://localhost:30000/metrics | grep -E "cache_hit_rate|token_usage|gen_throughput"

# 学习点：
# - Prometheus 指标体系
# - 关键性能指标的含义
# - 如何根据指标调优
```

### Week 2 验收标准
- [ ] 能解释 Prefix Caching 的工作原理和效果
- [ ] 能解释推测解码的加速原理
- [ ] 理解不同配置对性能的影响
- [ ] 能根据监控指标进行调优
- [ ] 完成至少 3 组对比实验

---

## Week 3: slime 基础训练流程（小模型验证）

### 目标
- 搭建 slime 训练环境
- 完成 Qwen2.5-0.5B 的 GRPO 训练
- 理解 RL 后训练的完整流程

### Day 1-2: slime 环境搭建

#### 3.1 环境准备
```bash
# 克隆 slime
cd /data/home/yizhou
git clone https://github.com/THUDM/slime.git
cd slime

# 使用 Docker（推荐）
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data/home/yizhou:/workspace \
  -it slimerl/slime:latest /bin/bash

# 或 Conda 安装
bash build_conda.sh
```

#### 3.2 下载模型和数据集
```bash
# 模型
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /root/models/Qwen2.5-0.5B-Instruct

# 数据集
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/datasets/dapo-math-17k

huggingface-cli download --repo-type dataset zhuzilin/gsm8k \
  --local-dir /root/datasets/gsm8k
```

#### 3.3 权重格式转换
```bash
cd /root/slime

# 加载模型配置
source scripts/models/qwen2.5-0.5B.sh

# HF -> Megatron torch_dist
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen2.5-0.5B-Instruct \
    --save /root/models/Qwen2.5-0.5B-Instruct_torch_dist

# 学习点：
# - Megatron 的权重格式
# - TP/PP 下的权重切分
# - torch_dist 格式的优势
```

### Day 3-4: 基础训练验证

#### 3.4 快速 Smoke Test
```bash
# 使用 CI 测试脚本（4卡）
python tests/test_qwen2.5_0.5B_short.py

# 或手动运行（4卡）
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
  -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 4 --colocate \
  --hf-checkpoint /root/models/Qwen2.5-0.5B-Instruct \
  --ref-load /root/models/Qwen2.5-0.5B-Instruct \
  --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt --label-key label --apply-chat-template \
  --rollout-shuffle --rm-type deepscaler \
  --num-rollout 3 --rollout-batch-size 4 --n-samples-per-prompt 4 \
  --rollout-max-response-len 8192 --rollout-temperature 0.8 \
  --global-batch-size 16 --balance-data \
  --tensor-model-parallel-size 1 --sequence-parallel \
  --use-dynamic-batch-size --max-tokens-per-gpu 9216 \
  --advantage-estimator grpo --use-kl-loss \
  --kl-loss-coef 0.00 --kl-loss-type low_var_kl \
  --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.7 --sglang-cuda-graph-max-bs 16 \
  --attention-backend flash

# 学习点：
# - Ray 调度的基本概念
# - Colocate 模式的资源管理
# - GRPO 算法的参数设置
# - 训练循环的完整流程
```

#### 3.5 观察训练日志
```bash
# 关注以下指标：
# 1. reward：奖励值是否上升
# 2. kl：KL 散度是否合理（第一步应为 0）
# 3. grad_norm：梯度范数是否稳定
# 4. loss：损失是否下降

# 学习点：
# - 如何判断训练是否正常
# - 常见的训练问题及排查方法
# - 超参数的影响
```

### Day 5-7: 理解训练流程

#### 3.6 阅读核心代码
```python
# 阅读顺序：
# 1. 训练入口：slime/train.py
# 2. Rollout 生成：slime/rollout/sglang_rollout.py
# 3. RL 算法：slime/utils/ppo_utils.py
# 4. 损失函数：slime/backends/megatron_utils/loss.py

# 学习点：
# - 训练-推理协同的实现
# - Rollout 生成的异步机制
# - GRPO/PPO 的实现细节
# - Loss Masking 的作用
```

#### 3.7 画数据流图
```
Prompt → SGLang Rollout → Reward → Advantage → Loss → Gradient → Weight Update
    ↓                                                              ↓
Data Buffer ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← Weight Sync
```

### Week 3 验收标准
- [ ] slime 环境搭建成功
- [ ] 完成 Qwen2.5-0.5B 的 3 步 GRPO 训练
- [ ] 理解 RL 后训练的完整流程
- [ ] 能解释 GRPO 的实现细节
- [ ] 理解 Colocate 模式的资源管理

---

## Week 4: slime RL 算法深入 + 实验对比

### 目标
- 对比不同 RL 算法（GRPO vs PPO vs REINFORCE++）
- 验证 KL 散度控制的效果
- 理解动态采样和 Partial Rollout

### Day 1-2: RL 算法对比

#### 4.1 GRPO 实验
```bash
# GRPO 训练
--advantage-estimator grpo
--use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
--eps-clip 0.2 --eps-clip-high 0.28
```

#### 4.2 PPO 实验
```bash
# PPO 训练（需要 Critic 模型）
--advantage-estimator ppo
--eps-clip 0.2
```

#### 4.3 REINFORCE++ 实验
```bash
# REINFORCE++ 训练
--advantage-estimator reinforce_plus_plus
--normalize-advantages
```

#### 4.4 对比分析
```python
# 对比指标：
# 1. 训练速度（rollouts/hour）
# 2. 收敛速度（达到相同 reward 所需的 rollout 数）
# 3. 最终 reward
# 4. 训练稳定性（reward 方差）

# 学习点：
# - 各算法的优缺点
# - 适用场景分析
# - 超参数的影响
```

### Day 3-4: KL 散度控制

#### 4.5 KL 作为 Reward 惩罚
```bash
--kl-coef 0.01  # KL 惩罚系数
```

#### 4.6 KL 作为 Loss 项
```bash
--use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
```

#### 4.7 对比实验
```bash
# 无 KL 控制
--kl-coef 0 --kl-loss-coef 0

# KL 作为 Reward
--kl-coef 0.01 --kl-loss-coef 0

# KL 作为 Loss
--kl-coef 0 --kl-loss-coef 0.001

# 学习点：
# - KL 散度的作用
# - 不同 KL 控制方式的效果
# - 如何选择合适的 KL 系数
```

### Day 5-7: 高级特性验证

#### 4.8 动态采样
```bash
--over-sampling-batch-size 8
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

# 学习点：
# - 动态采样的作用
# - 过滤函数的设计
# - 对训练效果的影响
```

#### 4.9 Partial Rollout
```bash
--partial-rollout
--buffer-filter-path slime.rollout.filter_hub.buffer_filters.pop_first

# 学习点：
# - Partial Rollout 的作用
# - 与动态采样的配合
# - 资源利用效率的提升
```

### Week 4 验收标准
- [ ] 完成至少 2 种 RL 算法的对比实验
- [ ] 理解 KL 散度的作用和控制方式
- [ ] 理解动态采样的原理和效果
- [ ] 能解释不同算法的适用场景

---

## Week 5: Agent 训练 + 高级特性

### 目标
- 实现一个简单的 Agent 训练
- 验证 Loss Masking 的效果
- 理解 Session Routing

### Day 1-2: Agent 训练基础

#### 5.1 自定义生成函数
```python
# slime/examples/search-r1/generate_with_search.py
async def generate(args, sample, sampling_params):
    for turn in range(max_turns):
        # 模型生成
        model_output = await call_sglang(prompt + full_response)
        
        # 解析动作
        action, content = parse_action(model_output)
        
        # 执行工具
        if action == "search":
            tool_output = await search(content)
            loss_masks += [0] * len(tool_tokens)
        
        if action == "answer":
            break
    
    sample.loss_mask = loss_masks
    return sample
```

#### 5.2 运行 Agent 训练
```bash
--custom-generate-function-path my_agent.generate
--custom-rm-path my_agent.reward_func
--rollout-max-response-len 8192

# 学习点：
# - Agent 训练的流程
# - 工具调用的实现
# - Loss Masking 的作用
```

### Day 3-4: Loss Masking 验证

#### 5.3 对比实验
```python
# 无 Loss Masking
loss_mask = None  # 所有 token 参与 loss

# 有 Loss Masking
loss_mask = [1] * model_tokens + [0] * tool_tokens

# 对比指标：
# 1. 训练稳定性
# 2. 工具使用准确率
# 3. 最终 reward

# 学习点：
# - 为什么需要 Loss Masking
# - Loss Masking 的实现方式
# - 对训练效果的影响
```

### Day 5-7: Session Routing

#### 5.4 验证 Session Routing
```bash
# 启动带 Session Routing 的服务
--router-policy consistent_hashing

# 学习点：
# - Session Routing 的作用
# - Consistent Hashing 的原理
# - 对 Prefix Cache 命中率的影响
```

### Week 5 验收标准
- [ ] 实现一个简单的 Agent 训练
- [ ] 理解 Loss Masking 的作用和实现
- [ ] 理解 Session Routing 的原理
- [ ] 能解释 Agent 训练的挑战和解决方案

---

## Week 6: 端到端项目 + 总结复盘

### 目标
- 完成一个端到端的项目
- 总结学习成果
- 准备面试材料

### Day 1-3: 端到端项目

#### 6.1 项目选择（任选其一）

**选项 A：数学推理 Agent**
- 使用 Qwen3-4B 模型
- 实现搜索增强的数学推理
- 使用 GRPO 训练
- 评估 GSM8K/AIME 准确率

**选项 B：代码生成 Agent**
- 使用 DeepSeek-R1-Distill-Qwen-1.5B 模型
- 实现代码补全和测试
- 使用 GRPO 训练
- 评估 pass@k

**选项 C：多轮对话优化**
- 使用 Llama-3.1-8B 模型
- 优化 Prefix Cache 命中率
- 对比不同调度策略
- 评估延迟和吞吐

### Day 4-5: 性能优化

#### 6.2 调优 Checklist
```bash
# 1. KV 缓存最大化
--mem-fraction-static 0.85  # 逐步增加到 OOM 前回退

# 2. CUDA Graph 优化
--cuda-graph-max-bs 24  # 4090 默认值

# 3. 调度策略优化
--schedule-policy lpm  # 多轮对话场景

# 4. 量化优化（可选）
--quantization fp8  # Hopper/Blackwell GPU
--kv-cache-dtype fp8_e4m3  # KV 缓存压缩
```

### Day 6-7: 总结复盘

#### 6.3 学习总结
- 整理学习笔记
- 总结关键技术点
- 准备面试回答

#### 6.4 面试准备
- 项目经验介绍
- 技术问题回答
- 实验细节描述

### Week 6 验收标准
- [ ] 完成端到端项目
- [ ] 整理完整的学习笔记
- [ ] 准备面试材料
- [ ] 能清晰地介绍项目经验

---

## 附录：关键资源索引

### SGLang 资源
| 资源 | 路径/URL |
|------|---------|
| 源码 | `/data/home/yizhou/sglang/` |
| 文档 | `/data/home/yizhou/sglang/docs/` |
| 测试 | `/data/home/yizhou/sglang/test/` |
| Benchmark | `/data/home/yizhou/sglang/benchmark/` |
| Examples | `/data/home/yizhou/sglang/examples/` |

### slime 资源
| 资源 | 路径/URL |
|------|---------|
| 源码 | `/data/home/yizhou/slime/` |
| 文档 | `/data/home/yizhou/slime/docs/` |
| 测试 | `/data/home/yizhou/slime/tests/` |
| Scripts | `/data/home/yizhou/slime/scripts/` |
| Examples | `/data/home/yizhou/slime/examples/` |

### 模型和数据集
| 类型 | 路径 |
|------|------|
| Qwen2.5-0.5B | `/root/models/Qwen2.5-0.5B-Instruct/` |
| Qwen3-4B | `/root/Qwen3-4B/` |
| Llama-3.1-8B | `/root/models/Llama-3.1-8B-Instruct/` |
| dapo-math-17k | `/root/datasets/dapo-math-17k/` |
| gsm8k | `/root/datasets/gsm8k/` |

### 面试准备文档
| 文档 | 路径 |
|------|------|
| 面试准备总览 | `/data/home/yizhou/sglang/LLM_Interview_Preparation.md` |
| 技术面试深挖 | `/data/home/yizhou/sglang/Technical_Interview_Deep_Dive.md` |

---

**最后更新**：2026-06-26
**适用硬件**：8×RTX 4090 24GB
**预计工期**：6周（每周 20-30 小时）
