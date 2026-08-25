# verl V1 Trainer 架构学习笔记

## 分析基准

本笔记基于下面这个 commit 分析：

```text
commit: bc6fb966cf969574ed6049bf751439bc2c85c3b3
title:  [BREAKING][trainer, cfg] chore: enable V1 trainer by default (#6823)
author: Begunner <97716296+Begunner@users.noreply.github.com>
date:   2026-06-24 16:51:50 +0800
```

这个 commit 的核心变化是：

```text
The V1 PPO trainer (`verl/trainer/ppo/v1`) is used by default.
```

注意：这份文档讨论的是该 commit 附近源码中的 V1 trainer 设计和实现状态。V1 trainer 还在快速演进，尤其是 async、hybrid rollout、partial rollout、ReplayBuffer 策略、多模态数据传输这些部分，后续 commit 可能会修复或重构当前分析里提到的问题。继续分析新版本时，应先重新确认相关源码。

这份笔记主要梳理 `verl/trainer/ppo/v1` 的核心逻辑。重点不是逐行解释源码，而是回答几个关键问题：

- V1 trainer 为什么一套代码能同时支持 sync、共卡 async、分离式 async？
- `TransferQueue` 在里面到底负责什么？
- async 模式到底异步在哪里？
- 有样本过滤、失败、staleness 时怎么处理？

## 1. 先抓住一个核心模型

V1 trainer 的主训练逻辑可以抽象成：

```text
提交一批 prompt 去生成
    -> 从 ReplayBuffer/TransferQueue 等到一批可训练样本
    -> 对这批样本做 old_log_prob / ref / value / advantage
    -> update critic / actor
    -> 同步或周期同步 rollout 权重
```

代码上最关键的是 `PPOTrainer.step()`：

```python
self._add_batch_to_generate()

batch, off_policy_metrics = self.replay_buffer.sample(
    global_steps=self.global_steps,
    partition_id="train",
    batch_size=self.config.data.train_batch_size,
)

# 后面继续 reward / log_prob / value / advantage / update actor / update critic
```

这段主流程在 sync、colocate async、separate async 里是共用的。

不同 mode 的差异不是训练主循环变了，而是：

```text
1. _add_batch_to_generate() 提交给哪种 rollout client
2. sample() 等到的数据来自哪里
3. sample 后是否停推理引擎
4. step 结束后如何同步权重、是否恢复 generation
```

所以可以记成：

```text
统一训练循环 + 不同 rollout 生产策略 + TransferQueue/ReplayBuffer 作为同步边界
```



## 2. TransferQueue 是什么

这里说的 queue，指的是 `transfer_queue as tq`，不是普通 Python queue。

它更像一个分布式 KV 存储和状态队列，用来连接 rollout 生产者和 trainer 消费者。

它存两类东西。

### 2.1 Prompt group 状态

每条 prompt 会有一个 `uid`，在 TransferQueue 里注册为 tag-only marker：

```text
uid -> pending / running / finished / failure
```

ReplayBuffer 会扫描这些状态，判断哪些 prompt group 可以被采样。

### 2.2 Trajectory 数据

每条生成出来的 trajectory 用下面的 key：

```text
{uid}_{session_id}_{index}
```

含义是：

```text
uid        原始 prompt 的唯一 id
session_id rollout.n 中第几条 response
index      agent loop 中第几个 output，multi-turn/agent 场景可能不止一个
```

trajectory 里会逐步写入这些字段：

```text
prompts
responses
response_mask
rm_scores
rollout_log_probs
old_log_probs
ref_log_prob
values
advantages
returns
extra_fields
...
```

Trainer 后续每个阶段都是按 key 从 TransferQueue 读需要的字段，再把新字段写回去。

所以 ReplayBuffer 本身不是主要存储，它更像 TransferQueue 之上的采样器。

## 3. TransferQueue 在一次 step 中怎么串起来

TransferQueue 的调用链可以按一条数据的生命周期理解。

### 3.1 初始化

`TaskRunnerV1.run()` 里会初始化 TQ：

```python
tq.init(config.transfer_queue)
...
tq.close()
```

每个 `AgentLoopWorkerTQ` 是 Ray actor，独立进程里也会：

```python
tq.init()
```

为什么需要：

```text
trainer、agent loop worker、rollout 后处理逻辑都需要连接同一个 TransferQueue 后端。
```



### 3.2 Trainer 注册 prompt group

`_add_batch_to_generate()` 先写 prompt marker：

```python
tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps}] * len(batch)
tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)
```

这里写的是 tag-only marker，不是完整 prompt 内容。

为什么需要：

```text
告诉 ReplayBuffer：这些 uid 已经进入 rollout 生产流程
初始状态是 pending
记录它们来自哪个 global_steps
```

prompt 内容仍然直接传给 AgentLoopManager：

```python
self.agent_loop_manager.generate_sequences(batch) 内部的 worker 有序列化开销，但是是发给每个 worker，而且会控制大小
```

所以这里的 `kv_batch_put` 不是传输 prompt，而是注册状态。

### 3.3 AgentLoopWorker 更新 prompt 状态

每条 prompt 开始跑时：

```python
await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
```

成功完成时：

```python
await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
```

失败时：

```python
await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})
```

为什么需要：

```text
ReplayBuffer 不直接 await AgentLoopWorker 的每个后台任务。
它只能通过 TransferQueue 里的状态判断哪些 prompt group 可以采样。
```

在 verl 中，轨迹的状态全部在 tq 中维护，而不是 replaybuffer，replaybuffer 只是一个非常简单的控制面，不直接管理任何数据。tq 可以在任何地方获取到。

### 3.4 AgentLoopWorker 写入 trajectory 数据

每个 agent loop output 会写成 trajectory key：

```text
{uid}_{session_id}_{index}
```

写入调用是：

```python
await tq.async_kv_batch_put(
    keys=keys,
    fields=list_of_dict_to_tensordict(fields),
    tags=tags,
    partition_id="train" if not validate else "val",
)
```

fields 里通常有：

```text
prompts
responses
response_mask
loss_mask
input_ids
position_ids
multi_modal_inputs
rm_scores
rollout_log_probs
extra_fields
```

tags 里通常有：

```text
status
prompt_len
response_len
seq_len
global_steps
min_global_steps
max_global_steps
```

为什么需要：

```text
rollout 结果可能很大，且后续多个训练阶段都要用。
把结果放在 TransferQueue 里，trainer 后面可以按 key 分阶段读取。
```



### 3.5 ReplayBuffer 扫描 metadata

ReplayBuffer 通过：

```python
data = tq.kv_list()
```

扫描所有 metadata，然后区分两类 key：

```text
is_prompt=True:
  这是 prompt marker，用来维护 pending/running/finished/failure

is_prompt=False:
  这是 trajectory key，用来记录 seq_len/global_steps 等 tag
```

为什么需要：

```text
ReplayBuffer 只需要看状态和 tags，就能判断是否够一批、哪些样本更老、哪些样本太 stale。
它不需要在采样阶段拉取完整 tensor fields。
```



### 3.6 ReplayBuffer 选中 uid 并清理 prompt marker

当：

```text
finished prompt group 数 + failure prompt group 数 >= data.train_batch_size
```

ReplayBuffer 会选出一批 prompt uid：

```python
selected_prompt_uids = sampleable_keys[:batch_size]
```

然后清理这些 prompt marker：

```python
tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)
```

为什么需要：

```text
这些 prompt group 已经被选中消费。
清掉 uid marker，避免下次 sample 又选到同一个 prompt group。
```

注意，这里清理的是：

```text
uid
```

不是：

```text
{uid}_{session_id}_{index}
```

trajectory 数据还留在 TransferQueue 里，后续训练阶段还要用。

### 3.7 Trainer 按阶段读写训练中间字段

拿到 `KVBatchMeta(keys, tags)` 后，trainer 后续每个阶段都按 keys 读写 TQ。

如果 reward model 是 colocated，先读：

```python
tq.kv_batch_get(keys=batch.keys, select_fields=["prompts", "responses", "raw_prompt"])
```

算完 reward 写回：

```text
rm_scores
reward_extra_info
```

old log prob 阶段写回：

```text
old_log_probs
entropy
```

ref 阶段写回：

```text
ref_log_prob
```

critic value 阶段写回：

```text
values
```

advantage 阶段会读取：

```text
uid
response_mask
rm_scores
rollout_log_probs
old_log_probs
ref_log_prob
values
```

然后写回：

```text
advantages
returns
token_level_rewards
rollout_is_weights  # 如果启用 rollout correction
```

为什么需要：

```text
同一批 trajectory 的字段不是一次性全部产生的。
rollout、reward、old_log_prob、ref、critic、advantage 都在不同阶段补字段。
TransferQueue 提供了一个按 key 逐步补齐字段的数据层。
```



### 3.8 Metrics / dump 读取 TQ

训练日志、吞吐、长度、reward、advantage 分布等指标也会按 key 读取：

```text
prompts
responses
response_mask
values
advantages
returns
rm_scores
token_level_rewards
num_turns
```

如果配置了 `rollout_data_dir`，也会从 TQ 读取 prompts/responses/rm_scores 来 dump 样本。

### 3.9 Step 结束清理 trajectory 数据

训练、日志和 dump 完成后：

```python
tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
```

为什么需要：

```text
这批 trajectory 已经消费完。
清理它们，避免重复训练，也避免 TransferQueue 存储持续增长。
```



### 3.10 总结成一条链

```text
1. Trainer:
   uid marker = pending

2. AgentLoopWorker:
   uid marker = running

3. AgentLoopWorker:
   trajectory keys = {uid}_{session_id}_{index}
   fields = rollout outputs
   tags = seq_len/global_steps/staleness info

4. AgentLoopWorker:
   uid marker = finished/failure

5. ReplayBuffer:
   kv_list 扫到 finished/failure
   选中 uid
   清理 uid marker
   返回 trajectory keys

6. Trainer:
   按 keys 逐步读写 rm_scores / old_log_probs / ref_log_prob / values / advantages / returns

7. Trainer:
   actor/critic update + metrics

8. Trainer:
   清理 trajectory keys
```

这就是为什么 TransferQueue 是 V1 的关键数据层。它把“一批样本”拆成可追踪、可采样、可逐步补字段的 KV 记录。

## 4. 重要设计原则：数据层和控制层分离

V1 trainer 里一个很重要的设计思想是：

```text
TransferQueue 是数据层
ReplayBuffer 是控制层
Trainer 主循环只依赖 ReplayBuffer 返回一批可训练 keys
```

trajectory 的真实内容不在 ReplayBuffer 里，而是在 TransferQueue 里。ReplayBuffer 主要负责：

```text
1. 扫描 TransferQueue metadata
2. 维护 pending / running / finished / failure 的状态视图
3. 判断当前是否有足够 prompt group 可以训练
4. 决定采哪些 uid / trajectory keys
5. 做 staleness 的 drop / wait 控制
6. 返回 KVBatchMeta(keys, tags)
```

而真正的大字段仍然留在 TransferQueue 中，由 trainer 后续各阶段按需读取和写回：

```text
prompts
responses
rm_scores
old_log_probs
ref_log_prob
values
advantages
returns
```

这个分层很关键：采样策略可以变复杂，但训练主循环和数据通路不用变。

如果以后要支持更复杂的策略，比如：

```text
按 data_source 均衡采样
按 reward 分层采样
按 response length 控制 batch
过滤 failed group 后，从 TransferQueue 已有的完成样本池里继续补足
按 staleness 加权采样
DAPO/Entropy 风格的 group filtering
```

理想扩展点不是改 trainer 主循环，而是自定义 ReplayBuffer。
但要注意，ReplayBuffer 当前没有 train dataloader / data sampler 的句柄，不能自己主动从数据集再拉新 prompt。它能做的“补足”通常是从 TransferQueue 中已经存在、已经完成的 surplus 样本里继续选择；如果 TQ 里没有足够 surplus，它只能继续阻塞等待，或者返回较小 batch 后交给 padding 逻辑补形状。

V1 已经留了入口：

```yaml
trainer:
  v1:
    sampler:
      custom_sampler:
        path: /path/to/my_sampler.py
        name: MyReplayBuffer
      sampler_kwargs: {}
```

自定义类通常继承：

```python
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

class MyReplayBuffer(ReplayBuffer):
    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        ...
        return batch, metrics
```

需要兼容的核心返回值是：

```text
KVBatchMeta(keys, tags), metrics
```

这里也能看出作者的设计取舍：

```text
Trainer 不关心样本是同步生成的、共卡超发来的，还是 standalone rollout 生产出来的。
Trainer 只关心 ReplayBuffer 能不能返回一批可训练的 KVBatchMeta。
```

因此，`replay_buffer.sample()` 是整个系统的统一同步点：

```text
rollout 生产快 -> sample 很快返回
rollout 生产慢 -> sample 阻塞等待
```

这个同步点让同一份 trainer 主循环可以同时覆盖 sync、colocate async、separate async。

### 4.1 为什么不需要再引入一个严格的多进程 queue

这里还有一个更底层的设计前提：V1 不是在一个大 Python 进程里同时跑训练和推理。

真正重的组件本来就已经被 Ray actor / Ray worker / rollout server 隔离开了：

```text
PPOTrainer:
  轻量控制器
  负责 step 编排、提交 prompt、触发训练阶段

AgentLoopWorker:
  Ray actor
  后台跑 agent loop / rollout 逻辑
  调 LLM server
  写 TransferQueue

Actor / Critic / Ref / Reward worker:
  Ray worker / worker group
  真正跑 GPU 前向和训练

LLM server:
  rollout replica / 独立 server
  真正跑 decode
```

所以 trainer 主进程并不是一个同时承担训练、推理、数据搬运的大进程。它更像一个 orchestration 层：

```python
self._add_batch_to_generate()
batch = self.replay_buffer.sample(...)
# 后面继续 reward / old_log_prob / ref / value / advantage / update
```

这也是为什么同一套逻辑能覆盖多种模式。不同模式真正变化的是：

```text
_add_batch_to_generate() 提交出去后，谁在生产 trajectory
sample() 会等当前 batch，还是从已有完成样本池里取
rollout server 是否和 trainer 共卡
sample 后 rollout 是否需要 abort / sleep
权重同步怎么做
```

但 trainer 主循环不用知道太多，因为执行隔离已经由 Ray actor / worker 边界提供了。

因此，TransferQueue 不是简单替代 `multiprocessing.Queue` 的“进程间管道”。它更像 V1 的样本数据层：

```text
1. 状态表
   pending / running / finished / failure

2. 样本索引层
   ReplayBuffer 根据 metadata 选择可训练样本

3. trajectory 存储层
   prompts / responses / log_probs / rewards / advantages 等字段分阶段写入

4. producer-consumer 缓冲区
   rollout 侧生产，trainer 侧消费
```

所以更准确的总结是：

```text
Ray actor / worker 提供执行隔离
TransferQueue 提供样本状态和数据交换
ReplayBuffer 提供采样控制
PPOTrainer 只保留轻量编排
```

这套分层是 V1 能复用同一个 trainer 主循环的关键原因。

## 5. 生产者和消费者的职责边界

V1 里最好按三个角色来理解：

```text
PPOTrainer:
  编排者，拥有 dataloader
  决定什么时候从数据集取 prompt
  调 _add_batch_to_generate() 投递新 prompt
  从 ReplayBuffer 拿一批 KVBatchMeta 后执行训练

AgentLoopManager:
  生产执行器
  接收 PPOTrainer 已经取出来的 prompt batch
  分发给 AgentLoopWorker
  调 LLM server / tool / env 生成 trajectory
  把 trajectory 写入 TransferQueue

ReplayBuffer:
  消费控制器
  扫描 TransferQueue 中已有的 prompt/trajectory metadata
  判断哪些样本可用
  选择哪些 keys 给 trainer 训练
  控制 stale / failed / sampling policy
```

也就是说：

```text
AgentLoopManager 负责“怎么生产”
ReplayBuffer 负责“怎么消费”
PPOTrainer 负责“什么时候投递新生产任务，以及拿到样本后怎么训练”
```

这个边界对理解复杂功能很重要。

### 5.1 只改采样策略：重写 ReplayBuffer

如果目标是：

```text
从 TransferQueue 已有完成样本里挑哪些
跳过 failed group
按 data_source 均衡
按 reward/长度/staleness 加权
从已有 surplus 里补足 batch
```

那应该重写 ReplayBuffer。

因为这些都属于：

```text
已有样本池中的消费策略
```

ReplayBuffer 能看到 TQ 里的 pending/running/finished/failure 和 trajectory tags，可以决定返回哪些 `KVBatchMeta(keys, tags)`。

### 5.2 改 rollout 生产方式：重写 AgentLoopManager

如果目标是：

```text
接入新的 agent 框架
改变 prompt 如何变成 trajectory
使用外部环境/仿真器/工具系统
改变 rollout.n 的动态生成逻辑
改变 trajectory 写入 TransferQueue 的格式
使用不同的 worker pool / 路由 / 限流方式
```

那应该重写 AgentLoopManager 或 AgentLoopWorker。

因为这些都属于：

```text
样本如何被生产出来
```

但默认 AgentLoopManager 不拥有 dataloader，它只消费 trainer 已经传给它的 prompt batch。

### 5.3 过滤后主动拉新数据补足：不是 ReplayBuffer 单独能做

如果目标是：

```text
组级别过滤后，如果有效 group 不够，
就从数据集继续取新 prompt，
再发起 rollout，
直到有效 group 凑满一个 train batch
```

这个功能跨越了两个边界：

```text
ReplayBuffer:
  能判断哪些 group 应该过滤
  能从 TransferQueue 已有 surplus 里继续选

PPOTrainer:
  才拥有 dataloader
  才能继续调用 _add_batch_to_generate()
  才能投递新的 prompt batch
```

因此，当前实现下它不是单靠重写 ReplayBuffer 就能完整实现的。ReplayBuffer 没有 `train_dataloader_it`，也没有 `agent_loop_manager`，不能自己主动生产新样本。

比较合理的实现位置有几种：

```text
方案 A：只从已有 surplus 补
  重写 ReplayBuffer。
  要求 async/warmup/overproduce 让 TransferQueue 中经常有多余完成样本。

方案 B：过滤后主动投递新 prompt
  改 PPOTrainer.step() 或新增 trainer-level producer loop。
  让 trainer 在 ReplayBuffer 返回“不够有效样本”时继续调用 _add_batch_to_generate()。

方案 C：把 producer 做成常驻生产者
  让更上层的 trainer/produder 持续向 AgentLoopManager 投递 prompt。
  ReplayBuffer 只负责从样本池消费。

方案 D：重新设计接口
  给 ReplayBuffer 一个 request_more() callback 或 producer handle。
  但这会打破当前 ReplayBuffer 只做消费控制层的简洁边界。
```

所以从当前设计看，作者更像是希望：

```text
采样和过滤策略 -> ReplayBuffer
生产和 rollout 执行 -> AgentLoopManager
主动从 dataloader 补新 prompt -> PPOTrainer / 更上层 orchestration
```

这也是为什么 V1 现在 `ReplayBuffer` 可以自定义，但构造参数里没有 dataloader/sampler。

## 6. Batch size 要分清楚

V1 里容易混淆的是三个 batch size。

### 6.1 `data.train_batch_size`

这是每个 RL step 的全局 prompt 数。

例如：

```yaml
data.train_batch_size: 1024
```

那么 `_add_batch_to_generate()` 一次从 dataloader 取 1024 条 prompt。

### 6.2 `rollout.n`

这是每个 prompt 生成几条 response。

例如：

```yaml
data.train_batch_size: 1024
actor_rollout_ref.rollout.n: 8
```

通常会得到：

```text
1024 prompt group * 8 responses = 8192 trajectories
```

注意 ReplayBuffer 采样时的 `batch_size=data.train_batch_size`，等的是 1024 个 prompt group，不是 1024 条 trajectory。

### 6.3 PPO mini-batch

Actor / critic update 时还有：

```yaml
actor_rollout_ref.actor.ppo_mini_batch_size
critic.ppo_mini_batch_size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
```

在 V1 里 actor 实际 mini-batch row 数通常是：

```text
actor.ppo_mini_batch_size * rollout.n
```

所以例子：

```yaml
data.train_batch_size: 1024
actor_rollout_ref.actor.ppo_mini_batch_size: 256
rollout.n: 8
```

可以理解为：

```text
每 step 采 1024 个 prompt
生成约 8192 条 trajectory
actor 每个 PPO mini-batch 处理 256 * 8 = 2048 条 trajectory
```



## 7. `_add_batch_to_generate()` 到底做了什么

它不是直接生成样本，而是提交生成任务。

核心步骤：

```text
1. 从 dataloader 取一个 batch，大小是 data.train_batch_size
2. 给每条 prompt 分配 uid
3. 在 TransferQueue 里写入 pending prompt marker
4. 调 agent_loop_manager.generate_sequences(batch)
```

`AgentLoopWorkerTQ` 收到 prompt 后，会在 worker 内部用 `asyncio.create_task()` 创建后台任务。

所以 `_add_batch_to_generate()` 的语义是：

```text
把一批 prompt 投递到 rollout 生产流水线
```

不是：

```text
立即返回一批训练样本
```



## 8. `replay_buffer.sample()` 是统一同步点

这点很关键。

`replay_buffer.sample()` 如果没有足够数据，会卡住：

```python
while not self._has_enough_samples(...):
    time.sleep(self.poll_interval)
    self._sync_metadata_from_transfer_queue()
```

默认每 2 秒轮询一次。

它等的是：

```text
finished prompt group 数 + failure prompt group 数 >= data.train_batch_size
```

所以：

```text
rollout 生产快于 trainer 消费 -> sample() 很快返回
rollout 生产慢于 trainer 消费 -> trainer 卡在 sample()
```

这也是为什么同一份主代码既能表现得像同步，也能表现得像异步。

## 9. Sync 模式

默认配置：

```yaml
trainer.use_v1: true
trainer.v1.trainer_mode: sync
```

宏观流程：

```text
1. 提交当前 step 的 prompt batch
2. sample() 等这批 prompt group 完成
3. sample 后 sleep rollout replicas，释放 rollout 显存/KV cache
4. 训练 actor/critic
5. step 结束后 update_weights，把新 actor 权重同步给 rollout replicas
6. 下一步继续
```

虽然内部用了 AgentLoop 和 TransferQueue，但宏观上还是：

```text
生成一批 -> 训练一批 -> 同步权重 -> 下一批
```

它稳定、简单，最接近老 trainer 的行为。

## 10. Colocate Async 模式

配置：

```yaml
trainer.v1.trainer_mode: colocate_async
trainer.v1.colocate_async.num_warmup_batches: 1
```

注意，colocate async 是共卡 async。它不是训练和推理在同一批 GPU 上真正同时跑。

它的核心思想更像：

```text
共卡条件下的异步 rollout 池
提前/超发 rollout 请求
够一批就拿来训练
没完成的长尾请求可以 abort
训练时 sleep rollout engine
训练后同步权重并 resume generation
```

对比 sync：

```text
sync:
  提交 N 个 prompt
  等 N 个 prompt 全部完成
  训练

colocate async:
  训练前先 warmup 提交一批
  step 里继续提交一批
  队列里可能有超过 N 个 prompt 在跑
  只要完成的 prompt group 达到 N 个就训练
  剩余未完成请求 abort
```

关键 hook：

```text
on_train_begin:
  先提交 num_warmup_batches 个 batch

on_sample_end:
  abort_replicas()
  sleep_replicas()

on_step_end:
  update_weights()
  resume_generation_replicas()
```

所以 colocate async 的收益主要是减少 rollout 长尾等待，尤其适合：

```text
multi-turn agent
tool call
多模态
response 长度差异很大
```

代价是：

```text
可能浪费被 abort 的生成
样本可能轻微 stale
TransferQueue 和请求状态更复杂
```



## 11. Separate Async 模式

配置：

```yaml
trainer.v1.trainer_mode: separate_async
trainer.v1.separate_async.num_warmup_batches: 4
trainer.v1.separate_async.parameter_sync_step: 4
```

这是非共卡分离式 async，是 V1 里最接近真正 producer-consumer 的模式。

它和 `colocate_async` 的最大差别是：

```text
colocate_async:
  rollout 和 trainer 共用同一批 GPU
  sample 够了以后要 abort + sleep rollout engine
  训练时推理基本停掉

separate_async:
  standalone rollout 使用独立资源
  trainer 训练时，standalone rollout 仍然可以继续生成
  两边通过 TransferQueue/ReplayBuffer 解耦
```

所以 separate async 的核心不是“多 warmup 几批”，而是资源和流水线被拆开：

```text
推理流水线:
  PPOTrainer 提交 prompt
  -> AgentLoopManager / AgentLoopWorker
  -> standalone rollout server
  -> trajectory 写入 TransferQueue

训练流水线:
  ReplayBuffer 从 TransferQueue 取已完成样本
  -> old_log_prob / ref / value / advantage
  -> update critic / actor
```



### 11.1 资源结构：为什么叫非共卡

`PPOTrainer._setup()` 会先按普通 V1 初始化一套 colocated/hybrid rollout：

```text
self.llm_server_manager
```

这套 rollout 和 actor/trainer worker 绑定在一起，可以理解为 trainer 这边的 hybrid rollout 资源。

`PPOTrainerSeparateAsync._setup()` 在此基础上额外创建：

```text
self.standalone_server_manager
self.standalone_checkpoint_manager
```

也就是再拉起一套独立 rollout server。它不依赖 trainer 这组 GPU sleep/wake，因此可以在 trainer 更新 actor/critic 时继续 decode。

AgentLoop 用的 LLM client 也被切到 standalone rollout：

```python
def get_llm_client(self):
    return self.standalone_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)
```

所以在 separate async 里：

```text
AgentLoopWorker 发出的 generate 请求
主要打到 standalone rollout server
```

这就是“非共卡”的关键。

### 11.2 训练主循环为什么不用改

Separate async 仍然复用同一个 `PPOTrainer.step()`：

```python
self._add_batch_to_generate()

batch, off_policy_metrics = self.replay_buffer.sample(
    global_steps=self.global_steps,
    partition_id="train",
    batch_size=self.config.data.train_batch_size,
)

# reward / old_log_prob / ref / value / advantage / update
```

这说明主循环没有变成复杂的 async scheduler。它还是：

```text
提交新 prompt
从 ReplayBuffer 取一批已完成样本
训练
```

区别在于：

```text
sync:
  sample() 多半等当前提交的 batch

colocate_async:
  sample() 从共卡超发请求中取先完成的一批

separate_async:
  sample() 从 standalone rollout 持续生产的样本池中取一批
```

也就是说，`sample()` 仍然是 trainer 的同步点。如果队列没样本，trainer 仍然会卡住；如果 standalone rollout 生产足够快，trainer 就可以不等当前 step 刚提交的那批 prompt。

### 11.3 生产者 / 消费者时间线

假设：

```yaml
data.train_batch_size: 1024
num_warmup_batches: 4
parameter_sync_step: 4
```

训练开始前：

```text
on_train_begin()
  提交 batch A
  提交 batch B
  提交 batch C
  提交 batch D
```

这些 prompt 进入 AgentLoopWorker 后，worker 内部创建 asyncio 后台任务，再向 standalone rollout server 发 generate 请求。结果不会直接返回给 trainer，而是逐步写入 TransferQueue。

step 1：

```text
1. _add_batch_to_generate()
   再提交 batch E 给 rollout 生产流水线

2. replay_buffer.sample()
   从 TransferQueue 已完成样本池里取 1024 个 prompt group
   这些样本可能来自 A/B/C/D，不一定来自 E

3. on_sample_end()
   如果 hybrid rollout 当前还在 rollout mode，就切回 trainer mode

4. trainer 训练
   old_log_prob / ref / value / advantage / update critic / update actor

5. standalone rollout 同时继续生成
   A/B/C/D/E 中尚未完成的 prompt 继续在独立 rollout 资源上跑
```

所以它不是：

```text
step 1 生成 E，然后训练 E
```

而是：

```text
step 1 提交 E
step 1 训练 TransferQueue 里已经完成的一批样本
step 1 训练期间 standalone rollout 继续生产后续样本
```

这个差别就是非共卡 async 的核心。

### 11.4 hybrid rollout 在 separate async 里的角色

这里容易漏掉一个点：separate async 不是只有 standalone rollout。代码里实际同时存在两套 rollout server：

```text
1. hybrid rollout server
   来自 PPOTrainer._setup()
   绑定 actor_rollout_wg / trainer 这组资源
   manager 是 self.llm_server_manager

2. standalone rollout server
   来自 PPOTrainerSeparateAsync._setup()
   使用额外 rollout 资源
   manager 是 self.standalone_server_manager
```

所以 separate async 的真实结构更像：

```text
trainer resources:
  actor/critic/ref/reward 训练
  + 一套可切换角色的 hybrid rollout server

standalone rollout resources:
  一直用于后台生成
  + 一套独立 rollout server

中间:
  TransferQueue + ReplayBuffer
```



#### 11.4.1 为什么已经有 standalone，还要 hybrid

因为 trainer 资源在某些阶段不是一直训练。

例如训练刚开始时，还没有足够 trajectory 可以训练。此时如果 trainer 资源闲着，可以临时作为 rollout 资源参与生成，帮 TransferQueue 更快积累 warmup 样本。

代码里这个动作发生在 `PPOTrainerSeparateAsync._setup()`：

```python
self.current_mode = HybridEngineMode.ROLLOUT
self.add_replicas_to_balancer()
```

`add_replicas_to_balancer()` 做的事情不是新建 standalone server，而是把 hybrid rollout server 地址加到 standalone 的全局 load balancer 里：

```python
global_load_balancer = self.standalone_server_manager.global_load_balancer
servers = dict(zip(self.llm_server_manager.server_addresses, self.llm_server_manager.server_handles, strict=True))
ray.get(global_load_balancer.add_servers.remote(servers))
```

这意味着 warmup 阶段请求分发大概是：

```text
AgentLoopWorker
  -> FullyAsyncLLMServerClient
  -> standalone_server_manager.global_load_balancer
  -> standalone rollout servers
     + hybrid rollout servers
```

所以 hybrid 在 separate async 里的第一层作用是：

```text
训练开始前 / 训练资源空闲时，临时加入 rollout 生产池，加速填充 TransferQueue。
```



#### 11.4.2 sample 之后为什么要把 hybrid 摘掉

一旦 `replay_buffer.sample()` 已经拿到可训练 batch，trainer 马上要进入：

```text
reward / old_log_prob / ref / value / advantage / actor update / critic update
```

这时 trainer 这组 GPU 不能继续被 rollout 占着，所以 `on_sample_end()` 会切回 trainer：

```python
def on_sample_end(self):
    if self.current_mode == HybridEngineMode.ROLLOUT:
        self.switch_to_trainer()
```

`switch_to_trainer()` 做四件事：

```text
1. remove_replicas_from_balancer()
   从 standalone load balancer 中摘掉 hybrid rollout server

2. abort_replicas()
   中断 hybrid rollout server 上还没完成的请求

3. sleep_replicas()
   释放/休眠 trainer 侧 rollout engine

4. current_mode = TRAINER
   trainer 资源切回训练用途
```

所以 sample 之后的结构变成：

```text
trainer resources:
  训练 actor/critic
  hybrid rollout 已经从 load balancer 摘掉，并进入 sleep

standalone rollout resources:
  继续后台生成
  继续写 TransferQueue
```

这就是 separate async 比 colocate async 更核心的差异：

```text
colocate_async:
  sample 后 rollout 基本停掉，因为同一批 GPU 要训练

separate_async:
  sample 后 hybrid 停掉，但 standalone rollout 不停
```



#### 11.4.3 当前 hybrid 是“实现了机制”，不是完整调度策略

代码里看起来预留了训练中再次切回 rollout 的钩子：

```python
def on_sample_begin(self):
    if self.current_mode == HybridEngineMode.TRAINER and self.should_switch_to_rollout():
        self.switch_to_rollout()
```

如果 `should_switch_to_rollout()` 返回 True，那么 trainer 资源可以再次切到 rollout mode：

```python
def switch_to_rollout(self):
    self.checkpoint_manager.update_weights(self.global_steps)
    self.checkpoint_manager.resume_generation_replicas()
    self.add_replicas_to_balancer()
    self.current_mode = HybridEngineMode.ROLLOUT
```

但当前实现是：

```python
def should_switch_to_rollout(self):
    # TODO: Implement switch strategy by checking replay buffer and switch overhead
    return False
```

所以当前版本的实际行为更接近：

```text
启动/warmup:
  standalone rollout + hybrid rollout 一起产样本

第一次 sample 结束:
  hybrid rollout 被摘掉，切回 trainer

后续 steady-state:
  standalone rollout 继续生产
  trainer 继续消费和训练
  hybrid 通常不会自动再切回 rollout
```

因此我会把它理解成：

```text
hybrid rollout 的基础设施已经接进 separate async 了；
但“根据 ReplayBuffer 积压、训练空闲时间、切换开销动态调度 hybrid”的策略还没有真正实现。
```

换句话说，当前 separate async 的主要收益来自 standalone rollout 的持续后台生成；hybrid rollout 主要提供 warmup 加速和未来动态调度的扩展点。

### 11.5 权重同步和 staleness

standalone rollout server 不会每步同步最新 actor 权重，而是按：

```yaml
parameter_sync_step
```

周期同步。

这会带来 off-policy/staleness：

```text
rollout policy 可能落后当前 training policy
```

V1 用这些配置控制太旧的样本：

```yaml
trainer.v1.sampler.max_off_policy_threshold: 8
trainer.v1.sampler.max_off_policy_strategy: drop  # or wait
```

指标里也会记录：

```text
training/off_policy/trajectory_spans/*
training/off_policy/trajectory_staleness/*
training/off_policy/trajectory_staleness_worst/*
```

这里的设计取舍是：

```text
同步太频繁:
  rollout 经常停下来等权重，async 收益下降

同步太少:
  rollout 样本更 stale，off-policy 风险变大
```

`parameter_sync_step` 就是在吞吐和样本新鲜度之间做折中。

### 11.6 separate async 的阻塞点

Separate async 不是 trainer 永不等待。它仍然会卡在：

```python
replay_buffer.sample(...)
```

如果：

```text
standalone rollout 生产速度 < trainer 消费速度
```

那么 TransferQueue 里完成样本不够，trainer 仍然会等待。

如果：

```text
standalone rollout 生产速度 >= trainer 消费速度
```

trainer 就大概率直接从样本池取到一批，训练和推理可以形成流水线。

所以 separate async 的核心收益来自：

```text
用独立 rollout 资源提前生产样本
让 trainer 训练时 rollout 不停
尽量让 ReplayBuffer 始终有可消费样本
```



### 11.7 separate async 的限制和代价

当前实现有一些硬约束：

```text
data.train_batch_size == actor_rollout_ref.actor.ppo_mini_batch_size
actor_rollout_ref.rollout.nnodes > 0
actor_rollout_ref.rollout.n_gpus_per_node > 0
rollout.checkpoint_engine.backend != naive
```

如果 reward model 开启，还要求：

```text
reward.reward_model.enable_resource_pool=True
```

原因是 standalone rollout 不会像 colocated rollout 那样频繁停下来释放显存，因此 reward model 不能随便 colocate 到同一池资源上。

主要代价：

```text
1. 需要额外 rollout 资源
2. 样本可能 stale
3. 权重同步更复杂
4. TransferQueue backlog / 清理 / drop 策略更重要
5. partial rollout 可能让单条 trajectory 横跨多个权重版本
```

一句话总结 separate async：

```text
它不是让 trainer 主循环变异步；
而是把 rollout 生产者挪到独立资源上持续生产，
trainer 仍然同步地从 ReplayBuffer 取一批已完成样本来训练。
```



## 12. Partial rollout：被打断的请求怎么续跑

V1 async 里还有一个容易误解的点：`FullyAsyncLLMServerClient` 支持 partial rollout。

先区分两个 LLM client：

```text
LLMServerClient:
  普通 one-shot client
  调一次 rollout server.generate()
  server 返回什么就交给上层
  sync 模式默认使用它

FullyAsyncLLMServerClient:
  async 模式使用
  逻辑上是 LLMServerClient 的 abort-resume 包装
  如果 server 返回 aborted/abort，就保留 partial token 并继续 generate
```

所以两者不是连接的 server 不同，而是对“请求被 abort”这件事的处理语义不同。

它的语义不是：

```text
被打断样本写入 TransferQueue
ReplayBuffer 下次优先采出来
AgentLoopManager 再重新调度
```

而是：

```text
同一个 AgentLoopWorker coroutine 不结束
FullyAsyncLLMServerClient 在 generate() 内部循环续跑
最后完整结果才写入 TransferQueue
```

核心逻辑是：

```python
final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)

while True:
    output = await super().generate(
        request_id=request_id,
        prompt_ids=prompt_ids + final_output.token_ids,
        sampling_params=sampling_params,
        ...
    )

    final_output.token_ids.extend(output.token_ids)

    if output.stop_reason not in ("aborted", "abort"):
        break

    await asyncio.sleep(1)
```

也就是说，如果这次 generation 被 rollout engine abort，它会：

```text
1. 保留已经生成的 token
2. 下一次 generate 的输入变成：原 prompt + 已生成 token
3. 调整 max_tokens / max_new_tokens
4. sleep 1 秒后重试
5. 直到 stop_reason 不再是 aborted/abort
```

所以 partial rollout 的续跑发生在：

```text
FullyAsyncLLMServerClient.generate()
```

不是发生在：

```text
ReplayBuffer / TransferQueue / PPOTrainer
```



### 12.1 LLM server 侧的 abort / resume

不同后端的接口不完全一样，但抽象上是：

```text
abort_all_requests:
  打断当前 in-flight generation requests

resume_generation:
  允许 generation 继续调度/接收

sleep:
  释放 weights / kv_cache 的 GPU memory occupation
```

以 SGLang wrapper 为例：

```python
async def abort_all_requests(self):
    await self.tokenizer_manager.pause_generation(PauseGenerationReqInput(mode="abort"))

async def resume_generation(self):
    await self.tokenizer_manager.continue_generation(ContinueGenerationReqInput())

async def sleep(self):
    await self.tokenizer_manager.release_memory_occupation(...)
```

这里要区分两个概念：

```text
pause_generation / continue_generation:
  控制 generation 调度

release_memory_occupation / resume_memory_occupation:
  控制 weights / kv_cache 显存占用
```



### 12.2 源码能确定什么，什么依赖后端契约

从 verl 源码可以确定：

```text
1. FullyAsyncLLMServerClient 收到 aborted/abort 后会循环重试 generate
2. 重试时输入是原 prompt + 已生成 token
3. 这个 partial 样本不会以半成品形式进入 ReplayBuffer
4. 最终完整 AgentLoopOutput 写入 TransferQueue
5. partial rollout 会记录 min_global_steps / max_global_steps
```

从 verl 源码不能单独完全证明的是：

```text
SGLang 在 pause/sleep 状态下收到新的 generate_request 时，内部一定如何处理。
```

因为这部分逻辑在 SGLang 的 `tokenizer_manager` 内部。verl 依赖后端提供这样的契约：

```text
pause_generation(mode="abort"):
  abort 当前请求，并暂停 generation 调度

continue_generation():
  恢复 generation 调度
```

如果后端在 sleep/pause 状态下直接报错，而不是等待或返回 aborted，那么异常会冒泡到 AgentLoopWorker，最终 prompt group 可能被标成 `failure`。

因此更严谨的说法是：

```text
verl 负责 partial rollout 的 client-side continuation。
后端负责 abort/pause/resume 的具体调度语义。
```



### 12.3 partial rollout 的代价

partial rollout 可能让一条 trajectory 横跨多个 actor 权重版本。

例如：

```text
前 300 token 用 step 10 的 rollout weights
后 700 token 用 step 11 的 rollout weights
```

所以 `FullyAsyncLLMServerClient` 会记录：

```text
min_global_steps
max_global_steps
```

AgentLoop 写入 TransferQueue tag 后，trainer 侧会统计：

```text
training/off_policy/trajectory_spans/*
training/off_policy/trajectory_staleness/*
training/off_policy/trajectory_staleness_worst/*
```

这就是 partial rollout 换吞吐的代价：减少长尾等待，但引入更复杂的 policy staleness。

## 13. Hybrid rollout 是什么

Separate async 里有两个 rollout 概念：

```text
standalone rollout:
  独立推理资源，训练时也持续跑

hybrid rollout:
  trainer 这组资源空闲时也可以临时当 rollout server
```

hybrid 的意思不是模型混合，而是资源角色混合：

```text
同一组资源可以在 TRAINER 和 ROLLOUT 之间切换
```

当前实现里：

```text
初始化时 hybrid 处于 ROLLOUT mode
它会被加到 load balancer，和 standalone rollout 一起生成
sample 后 hybrid 会切回 TRAINER mode
```

但是动态调度还很简单：

```python
def should_switch_to_rollout(self):
    return False
```

所以当前版本不是成熟的动态 hybrid scheduler。

可以理解为：

```text
hybrid 机制实现了
但训练过程中自动来回切的策略还没真正实现
steady-state 主要收益仍来自 standalone rollout
```



## 14. 几种模式对照

```text
模式              核心行为
----------------------------------------------------------------------
sync              当前 batch 生成完，再训练当前 batch

colocate_async    同一批 GPU 上超发 rollout，够一批就停推理，
                  abort 长尾，然后训练

separate_async    rollout 和 trainer 分资源并行；
                  rollout 持续生产，trainer 从 TransferQueue 消费
```

更详细一点：

```text
维度                 sync                  colocate_async              separate_async
------------------------------------------------------------------------------------------------
训练/推理资源        共卡                  共卡                        非共卡为主
训练和推理是否并行    否                    否，主要是请求级异步       是，standalone rollout 可持续跑
是否 warmup           否                    是                          是
sample 会不会卡       会                    会                          会
sample 数据来源       当前 step 为主         超发请求中先完成的样本      队列中已完成样本
sample 后动作         sleep rollout         abort + sleep rollout        hybrid 可切 trainer
权重同步              每 step               每 step + resume generation  每 parameter_sync_step
主要收益              简单稳定              减少 rollout 长尾等待        训练/推理 overlap
主要代价              吞吐受 rollout 卡住    abort 浪费和状态复杂        staleness 和系统复杂度
```



## 15. 样本失败、过滤和补齐

这里要特别注意：V1 当前不是严格的“过滤后自动补真实样本直到有效 batch 满”。

ReplayBuffer 采样逻辑是：

```text
等 finished + failure 的 prompt group 数达到 data.train_batch_size
选最老的 data.train_batch_size 个 prompt uid
收集这些 uid 下的 trajectory keys
```

如果因为 staleness 超阈值而 drop：

```text
被 drop 的 trajectory 会从 TransferQueue 清理
返回剩余 batch
```

这里没有自动从 dataloader 继续补真实样本。

后面 `_balance_batch()` 会调用：

```text
upsample_batch_to_divisible_size()
```

但这是补 synthetic padding sample，只是为了满足 DP / PPO mini-batch 的整除性。

所以：

```text
V1 当前补的是形状/整除性，不是补真实有效样本数。
```

如果过滤很多，实际有效训练样本会变少。

## 16. 多模态场景的注意点

如果 `data.train_batch_size` 很大，比如 8192，多模态场景可能会有明显卡顿或内存压力。

原因是：

```text
1. dataloader 一次取 8192 条 prompt object
2. collate_fn 对非 tensor 字段做 object array
3. agent loop 会把这些 prompt 分发给 workers
4. 每条 prompt 可能进一步加载/处理图片、视频、工具调用上下文
5. rollout.n 会把请求数再放大
```

例如：

```yaml
data.train_batch_size: 8192
rollout.n: 8
```

理论上会产生：

```text
8192 prompt group
65536 trajectories
```

对多模态和 agent loop 来说，这通常会形成很大的请求洪峰。

更实用的策略一般是：

```text
适当降低 data.train_batch_size
用 rollout.n 和更多 step 提升统计量
用 colocate_async/separate_async 减少长尾等待
观察 TransferQueue backlog、staleness、drop 指标
```



## 17. 一句话总结

V1 trainer 的核心不是某个 PPO 公式变了，而是训练系统变成了：

```text
AgentLoop/rollout 作为样本生产者
TransferQueue 作为跨进程样本和状态存储
ReplayBuffer 作为采样与 staleness 控制层
Trainer 作为样本消费者和参数更新者
CheckpointEngineManager 负责把 actor 权重同步给 rollout server
```

理解了这点，就能理解为什么同一段 `step()` 可以跑出三种形态：

```text
sync:
  生产和消费强同步

colocate_async:
  共卡条件下超发生产，够量即消费

separate_async:
  非共卡条件下持续生产、持续消费
```
