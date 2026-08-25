# Rollout 关系图

这份笔记记录 V1 trainer 中 rollout 相关主要对象的关系。重点是先看清楚大对象如何连接，再分别理解每个类负责什么。

## 0. 数量符号

下面的图里使用这些数量符号：

```text
1  = 单个对象
A  = actor_rollout_wg.world_size，训练侧 actor/rollout worker 总数
W  = agent loop worker 数量，来自 rollout.agent.num_workers
R  = rollout replica 数量，通常是 world_size // rollout_world_size
K  = 单个 RolloutReplica 内部的 server actor 数量，通常和该 replica 跨的节点数有关
T  = 一批 batch 中的样本数
N  = 每条 prompt 采样的 rollout 条数，来自 rollout.n 或 val_kwargs.n
M  = 每条 agent loop 可能产出的 output 数量，单轮通常是 1，多轮/agent 场景可能大于 1
```

其中 `K` 可以粗略理解为：

```text
vLLM / SGLang 普通 replica:
  K 通常等于该 replica 跨的节点数。

TRT-LLM replica:
  K 通常是 1。

SGLang PD replica:
  K 由 prefill/decode server 共同组成，逻辑上仍对外暴露 1 个主 server_handle。
```

注意：

```text
sync / colocate_async:
  通常是 1 个 LLMServerManager + R 个 RolloutReplica。

separate_async:
  会有 1 个 hybrid LLMServerManager + 1 个 standalone LLMServerManager。
  AgentLoop 默认使用 standalone_server_manager 的 client。
```

## 1. 顶层流程图

```text
PPOTrainer (1)
│
├── AgentLoopManager (1)
│   │
│   └── AgentLoopWorker[W]
│       │
│       ├── 每个 worker 持有 1 个 llm_client
│       │   └── LLMServerClient / FullyAsyncLLMServerClient
│       │       └── 所有 client 指向同一个 GlobalRequestLoadBalancer
│       │
│       └── 每条 sample 运行 1 个具体 AgentLoop
│           ├── SingleTurnAgentLoop
│           └── ToolAgentLoop
│               │
│               └── server_manager.generate(...)
│                   │
│                   └── 实际调用 llm_client.generate(...)
│
├── LLMServerManager (1; separate_async 中可能有 2 个)
│   │
│   ├── RolloutReplica[R]
│   │   │
│   │   ├── vLLMReplica
│   │   ├── SGLangReplica
│   │   │   └── SGLangPDReplica
│   │   └── TRTLLMReplica
│   │       │
│   │       └── launch_servers()
│   │           │
│   │           └── Server Actor[K per replica]
│   │               ├── server_handle: 1 个主 handle / replica
│   │               └── server_address: 1 个主 address / replica
│   │
│   ├── GlobalRequestLoadBalancer (1 per LLMServerManager)
│   │   │
│   │   └── stores:
│   │       └── server_address[R] -> server_handle[R]
│   │
│   └── get_client()
│       │
│       ├── LLMServerClient (每个 AgentLoopWorker 持有 1 份)
│       └── FullyAsyncLLMServerClient (每个 AgentLoopWorker 持有 1 份)
│
├── CheckpointEngineManager (1; separate_async 中 standalone 还会有 1 个)
│   │
│   ├── actor_wg (1)
│   │   └── 训练侧权重来源
│   │
│   └── RolloutReplica[R]
│       └── 来自 LLMServerManager.get_replicas()
│
├── ReplayBuffer (1)
│   │
│   └── 从 TransferQueue 中采样已经完成的 rollout trajectories
│
└── TransferQueue (1 个逻辑后端，多进程共享访问)
    │
    ├── 保存 prompt group 状态[T]
    │   └── uid -> pending / running / finished / failure
    │
    └── 保存 trajectory 数据[T * N * M]
        └── {uid}_{session_id}_{index}
```

## 2. 请求流程

```text
AgentLoopManager.generate_sequences(batch size = T)
│
├── 切分 batch
│
└── 分发给 AgentLoopWorker[W]
    │
    └── 每个 worker 处理约 T / W 条 sample
        │
        ├── 每条 sample instantiate 1 个 AgentLoop
        │   ├── SingleTurnAgentLoop
        │   └── ToolAgentLoop
        │
        └── AgentLoop.run(...)
            │
            └── llm_client.generate(request_id, prompt_ids, sampling_params)
                │
                ├── GlobalRequestLoadBalancer.acquire_server(request_id)
                │   │
                │   ├── sticky session 命中
                │   │   └── request_id -> 原来的 server_handle
                │   │
                │   └── sticky session 未命中
                │       └── 从 R 个 server_handle 中选择 in-flight 请求数最少的一个
                │
                ├── server_handle.generate.remote(...)
                │   │
                │   └── TokenOutput
                │       ├── token_ids
                │       ├── log_probs
                │       ├── stop_reason
                │       └── extra_fields
                │           ├── global_steps
                │           ├── min_global_steps
                │           └── max_global_steps
                │
                └── GlobalRequestLoadBalancer.release_server(server_id)
```

## 3. 数据写回流程

V1 TQ 版本里，AgentLoopWorker 不是直接把结果返回给 trainer 主循环，而是写入 TransferQueue。

```text
AgentLoopWorkerTQ.generate_sequences(batch size = T_worker)
│
├── 为 batch 中每条 sample 创建 1 个后台 task
│   └── asyncio.create_task(_run_prompt(...)) * T_worker
│
└── _run_prompt(...)
    │
    ├── tq.async_kv_put(uid, status="running")
    │
    ├── 并发运行 N 条 agent loop
    │   └── N = rollout.n 或 val_kwargs.n
    │
    ├── _agent_loop_postprocess(...)
    │   │
    │   └── tq.async_kv_batch_put(...)
    │       └── key = {uid}_{session_id}_{index}
    │           ├── session_id: 0..N-1
    │           └── index: 0..M-1
    │
    ├── 成功:
    │   └── tq.async_kv_put(uid, status="finished")
    │
    └── 失败:
        └── tq.async_kv_put(uid, status="failure")
```

trainer 主循环再通过 ReplayBuffer 从 TransferQueue 里取可训练样本：

```text
PPOTrainer.step()
│
├── _add_batch_to_generate()
│   ├── 向 TransferQueue 注册 prompt marker
│   └── AgentLoopManager.generate_sequences(batch)
│
└── replay_buffer.sample(...)
    └── 从 TransferQueue 等待并采样 finished trajectories
```

## 4. 初始化流程

```text
PPOTrainer._setup()
│
├── 创建 actor_rollout_wg
│   └── 训练侧 actor/rollout worker group，world_size = A
│
├── LLMServerManager.create(...)
│   │
│   ├── 选择 RolloutReplica 子类
│   │   ├── rollout.name == "vllm"   -> vLLMReplica
│   │   ├── rollout.name == "sglang" -> SGLangReplica
│   │   └── rollout.name == "trtllm" -> TRTLLMReplica
│   │
│   ├── 计算 rollout_world_size
│   │   └── TP * DP * PP
│   │
│   ├── 计算 num_replicas
│   │   └── R = world_size // rollout_world_size
│   │
│   ├── 初始化 RolloutReplica[R]
│   │   ├── hybrid:
│   │   │   └── 从 actor_rollout_wg 中切 worker，每个 replica 使用 rollout_world_size 个 worker
│   │   │
│   │   └── standalone:
│   │       └── 新建 rollout resource pool 和 worker group，world_size = nnodes * n_gpus_per_node
│   │
│   ├── 每个 RolloutReplica.launch_servers()
│   │   └── 启动 K 个具体后端 server actor
│   │
│   ├── 收集 server_handles[R] / server_addresses[R]
│   │
│   └── 创建 GlobalRequestLoadBalancer
│       └── 注册 server_address[R] -> server_handle[R]
│
├── CheckpointEngineManager(...)
│   └── 持有 LLMServerManager.get_replicas()
│
└── AgentLoopManager.create(...)
    └── 创建 AgentLoopWorker[W]，并把 llm_client 传进去
```

## 5. 主要对象职责

### PPOTrainer

trainer 总控。它负责创建 actor worker group、LLMServerManager、CheckpointEngineManager、AgentLoopManager、ReplayBuffer，并在训练循环中串起生成、采样、训练、权重同步。

### LLMServerManager

rollout server 的控制面。它负责：

- 根据 rollout config 计算 replica 数量。
- 创建 RolloutReplica。
- 初始化 hybrid 或 standalone rollout server。
- 收集 server handle 和 address。
- 创建 GlobalRequestLoadBalancer。
- 通过 `get_client()` 给 AgentLoopWorker 提供 LLMServerClient。

一句话：`LLMServerManager` 负责把 rollout server 这套服务组装起来。

### RolloutReplica

单个 rollout replica 的抽象。它屏蔽不同推理后端的差异。

常见子类：

- `vLLMReplica`
- `SGLangReplica`
- `SGLangPDReplica`
- `TRTLLMReplica`

它负责启动具体 server actor，并提供生命周期控制方法：

- `launch_servers()`
- `sleep()`
- `wake_up()`
- `abort_all_requests()`
- `resume_generation()`
- `release_kv_cache()`
- `resume_kv_cache()`

一句话：`RolloutReplica` 负责把具体推理后端 server 建起来，并提供控制接口。

### Server Actor

具体后端的 Ray actor，例如 vLLM/SGLang/TRT-LLM HTTP server actor。

它真正执行生成请求：

```text
server_handle.generate.remote(...)
```

并返回 `TokenOutput`。

一句话：Server Actor 是实际跑推理的对象。

### GlobalRequestLoadBalancer

全局请求路由器，是一个 Ray actor。所有 AgentLoopWorker 共享它。

它维护：

```text
server_address -> server_handle
server_address -> in-flight request count
request_id     -> server_address
```

它的两个核心能力：

- sticky session：同一个 `request_id` 尽量路由到同一个 server，利于多轮对话 prefix cache。
- least in-flight：新请求选择当前 in-flight 请求数最少的 server。

一句话：`GlobalRequestLoadBalancer` 负责选择请求应该发到哪个 server。

### LLMServerClient

AgentLoop 侧看到的生成入口。

它不直接持有 RolloutReplica，也不知道 replica 结构。它只持有 `GlobalRequestLoadBalancer` 的 handle。

生成时：

```text
LLMServerClient.generate()
├── acquire_server(request_id)
├── server_handle.generate.remote(...)
└── release_server(server_id)
```

一句话：`LLMServerClient` 负责把一次 generate 请求路由到某个 server actor。

### FullyAsyncLLMServerClient

`LLMServerClient` 的子类，用于 partial rollout / async 训练。

如果 server 返回 `stop_reason == "aborted"` 或 `"abort"`，它会把已经生成的 token 拼回 prompt，然后继续请求，直到正常结束或达到长度限制。

它还会记录：

```text
min_global_steps
max_global_steps
```

因为一条 rollout 可能跨越多个权重版本。

一句话：`FullyAsyncLLMServerClient` 让 rollout 中断和恢复对 AgentLoop 尽量透明。

### AgentLoopManager

AgentLoopWorker 的管理器。它负责创建多个 AgentLoopWorker，并把输入 batch 切分后分发给这些 worker。

一句话：`AgentLoopManager` 负责调度 agent loop worker 并发生成。

### AgentLoopWorker

真正执行 agent loop 的 Ray actor。

它持有：

```text
llm_client
teacher_client
reward_loop_worker_handles
```

每条 sample 会实例化具体 AgentLoop，例如：

- `SingleTurnAgentLoop`
- `ToolAgentLoop`

然后调用：

```text
AgentLoop.run(...)
```

一句话：`AgentLoopWorker` 负责把数据样本变成一次或多次 LLM 调用，并做后处理。

### AgentLoop

具体的交互逻辑。

- `SingleTurnAgentLoop`：单轮生成。
- `ToolAgentLoop`：多轮工具调用。

AgentLoop 内部通过：

```text
server_manager.generate(...)
```

调用 LLM server。这里的 `server_manager` 实际上是 `LLMServerClient`。

一句话：AgentLoop 定义一条样本如何和 LLM 交互。

### CheckpointEngineManager

负责训练侧 actor 权重和 rollout replicas 的权重同步，以及 rollout server 的 sleep、abort、resume。

它持有：

```text
actor_wg
RolloutReplica[]
```

常见操作：

- `update_weights()`
- `sleep_replicas()`
- `abort_replicas()`
- `resume_generation_replicas()`
- `release_kv_cache_replicas()`
- `resume_kv_cache_replicas()`

一句话：`CheckpointEngineManager` 负责让 rollout server 使用最新 actor 权重，并处理 async 中断恢复。

### ReplayBuffer

V1 trainer 里采样 rollout trajectories 的控制面。它不真正存放完整数据，主要从 TransferQueue 里找可用样本。

一句话：`ReplayBuffer` 负责从 TransferQueue 中采样已经完成、符合 staleness 条件的 trajectories。

### TransferQueue

rollout producer 和 trainer consumer 之间的数据桥。

它保存两类信息：

```text
prompt group 状态:
uid -> pending / running / finished / failure

trajectory 数据:
{uid}_{session_id}_{index} -> prompts / responses / rm_scores / logprobs / values / advantages / ...
```

一句话：`TransferQueue` 是 V1 rollout 数据流的共享存储和同步边界。

## 6. 一句话总结

```text
AgentLoopManager 管 worker 并发；
AgentLoopWorker 跑具体 AgentLoop；
LLMServerClient 发生成请求；
GlobalRequestLoadBalancer 选择 server；
RolloutReplica 创建和控制 server；
LLMServerManager 把这些 rollout server 组件组装起来；
TransferQueue 存 rollout 结果；
ReplayBuffer 从 TransferQueue 采样训练数据；
CheckpointEngineManager 同步 actor 权重到 rollout server。
```
