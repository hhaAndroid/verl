# verl SGLang 权重同步后端总览

本文基于 verl 当前代码，整理 rollout backend 为 SGLang 时，训练侧权重如何通过不同 `checkpoint_engine` backend 同步到推理引擎。

文档前半部分先给出各 backend 的全局对比和通用对象关系；后半部分重点展开 `nccl`，并补充 `mooncake`、`nixl`、`kimi_ckpt_engine` 的差异。

## 后端快速总览

verl 的 checkpoint engine backend 主要解决一个问题：

```text
Actor/Trainer 侧权重
  -> 如何同步到 Rollout CheckpointEngineWorker
```

对于 SGLang rollout，大多数非 `naive` backend 的最后一跳都类似：

```text
Rollout CheckpointEngineWorker
  -> ServerAdapter.update_weights()
  -> SGLang update_weights_from_tensor / tensor update
```

所以这些 backend 的主要区别不在“怎么调用 SGLang”，而在：

```text
Actor/Trainer -> Rollout CheckpointEngineWorker
这一段怎么搬权重。
```

总表：

| backend | 核心方式 | 主要适用 | 优点 | 代价/限制 |
|---|---|---|---|---|
| `naive` | 共卡/本地直接更新 | actor 和 rollout colocated | 最简单、最快、少一层传输 | 只适合共卡 |
| `nccl` | actor rank0 到 rollout workers 的 NCCL broadcast，metadata 走 ZMQ | NVIDIA GPU 非共卡，固定集群 | 高性能、逻辑直接 | SGLang 不直接入组；需要 ZMQ metadata；弹性较弱 |
| `hccl` | 类似 `nccl`，但用于 Ascend/HCCL | Ascend NPU 非共卡 | Ascend 对应 NCCL 路径 | 依赖 HCCL 环境 |
| `nixl` | NIXL agent P2P chain/ring read | 异构/弹性/容错倾向的非共卡 | 插件式，可接 UCX/UCCL/Mooncake 等；P2P 灵活 | 协议复杂，依赖 NIXL 环境 |
| `mooncake` | 直接用 Mooncake TransferEngine，链式 P2P | 非共卡，明确想用 Mooncake/RDMA | P2P/RDMA，性能较好 | 当前实现要求单 tensor 能放进 bucket；环境依赖较强 |
| `kimi_ckpt_engine` | actor 多 rank GPU->CPU checkpoint/ParameterServer，rollout 拉取后 broadcast | 非共卡，且本来每次要保存 checkpoint/CPU 参数快照 | 复用 checkpoint 路径，actor 多 rank 分摊参数 | 链路长，多 GPU->CPU；环境复杂 |

按通信形态看：

```text
naive:
  共卡直接更新

nccl / hccl:
  collective broadcast 型

nixl / mooncake:
  P2P chain/ring 型

kimi_ckpt_engine:
  checkpoint / ParameterServer + P2P/H2D + rollout 内 broadcast 型
```

按选型直觉看：

```text
共卡：
  naive

NVIDIA 固定集群，想要简单高性能：
  nccl

Ascend：
  hccl 或 kimi_ckpt_engine，取决于是否 checkpoint-driven

想要更通用的 P2P、弹性/异构潜力：
  nixl

明确想直接用 Mooncake TransferEngine：
  mooncake

本来每次都要保存 checkpoint / CPU 参数快照：
  kimi_ckpt_engine
```

一句话总结：

```text
nccl/hccl 解决的是“一次广播给所有 rollout worker”；
nixl/mooncake 解决的是“沿 P2P 链路传给每个 rollout worker”；
kimi_ckpt_engine 解决的是“权重先成为 checkpoint/ParameterServer 状态，再由 rollout 拉取和广播”；
naive 解决的是“共卡时不需要远程传输”。
```

## 核心结论

verl 这套逻辑和 Xtuner/Slime 常见的 SGLang distributed update 路径不一样。

Xtuner/Slime 典型路径是：

```text
训练 rank0
  -> 调 SGLang /init_weights_update_group
  -> SGLang engine 内部 worker 加入 NCCL group
  -> 调 SGLang /update_weights_from_distributed
  -> 训练 rank0 直接 NCCL broadcast 给 SGLang model runner
```

verl 的 NCCL 路径是：

```text
Actor rank0
  -> verl 自己的 NCCLCheckpointEngine
  -> ray.util.collective NCCL broadcast
  -> Rollout CheckpointEngineWorker 收到权重
  -> ServerAdapter.update_weights()
  -> SGLang update_weights_from_tensor / tensor update
```

也就是说：

- verl 的 NCCL group 不调用 SGLang 的 `/init_weights_update_group`。
- SGLang engine 内部 worker 不直接加入 verl 的 NCCL group。
- NCCL 只负责把真实权重从 actor 侧搬到 rollout 侧的 `CheckpointEngineWorker`。
- 最后一跳从 `CheckpointEngineWorker` 到 SGLang，走 tensor update，HTTP/Ray 请求里主要是 tensor metadata / CUDA IPC handle，真实权重不通过 HTTP body 传输。

## 适用场景

该逻辑适用于：

- rollout backend 是 SGLang。
- rollout checkpoint engine backend 是 `"nccl"`。
- trainer/actor 和 rollout 是非 naive 的 disaggregated 权重同步模式。
- `CheckpointEngineManager.update_weights()` 被触发。

如果 backend 是 `"naive"`，则走共卡/本地直接更新，不走本文的 NCCL checkpoint engine 路径。

## 全局视角：checkpoint engine 不是一个对象，而是一组对象

理解 verl 这套权重同步，最重要的一点是：

```text
训练侧和推理侧不会共享同一个 Python checkpoint_engine 对象。

它们是在不同 Ray actor / 不同进程里，各自初始化出自己的 checkpoint_engine 实例。
CheckpointEngineManager 负责把这些分散的实例编排成同一次通信。
```

也就是说，实际运行时更像这样：

```text
Actor worker group
  actor_wg.workers[0]
    = ActorRolloutRefWorker
        self.checkpoint_engine = NCCLCheckpointEngine(...)

  actor_wg.workers[1]
    = ActorRolloutRefWorker
        self.checkpoint_engine = NCCLCheckpointEngine(...)

  actor_wg.workers[2]
    = ActorRolloutRefWorker
        self.checkpoint_engine = NCCLCheckpointEngine(...)

Rollout worker group
  rollout.workers[0]
    = CheckpointEngineWorker
        self.checkpoint_engine = NCCLCheckpointEngine(...)
        self.server_adapter = SGLang ServerAdapter(...)

  rollout.workers[1]
    = CheckpointEngineWorker
        self.checkpoint_engine = NCCLCheckpointEngine(...)
        self.server_adapter = SGLang ServerAdapter(...)
```

这些 `checkpoint_engine` 实例不是同一个对象，但它们：

- 使用同一个 backend 配置创建，例如 `"nccl"` 或 `"mooncake"`。
- 由同一个 `CheckpointEngineManager` 编排。
- 在同一次 `build_process_group()` 中被分配统一的 `rank/world_size/metadata`。
- 最终加入同一个通信上下文。

所以 verl 保证的不是“对象相同”，而是“通信上下文相同”。

### actor 侧 checkpoint engine 在哪里创建

actor 侧的 checkpoint engine 创建在 `ActorRolloutRefWorker.init_model()` 里。

简化逻辑：

```python
if "actor" in self.role:
    checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
    backend = checkpoint_engine_config.backend
    bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
    engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})

    self.checkpoint_engine = CheckpointEngineRegistry.new(
        backend,
        is_master=(torch.distributed.get_rank() == 0),
        bucket_size=bucket_size,
        **engine_kwargs,
    )
```

如果 backend 是 `"nccl"`，这里创建的是：

```text
NCCLCheckpointEngine
```

如果 backend 是 `"mooncake"`，这里创建的是：

```text
MooncakeCheckpointEngine
```

actor 侧还暴露了统一的转发方法：

```python
def execute_checkpoint_engine(self, method: str, *args, **kwargs):
    return getattr(self.checkpoint_engine, method)(*args, **kwargs)
```

因此 `CheckpointEngineManager` 可以通过 actor worker group 远程调用：

```text
actor checkpoint_engine.prepare()
actor checkpoint_engine.init_process_group(...)
actor checkpoint_engine.finalize()
```

### rollout 侧 checkpoint engine 在哪里创建

rollout 侧的 checkpoint engine 创建在 `CheckpointEngineWorker.__init__()` 里。

简化逻辑：

```python
backend = self.rollout_config.checkpoint_engine.backend
bucket_size = self.rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
engine_kwargs = self.rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})

self.checkpoint_engine = CheckpointEngineRegistry.new(
    backend,
    bucket_size=bucket_size,
    **engine_kwargs,
)
```

同时 rollout 侧还有一个 `server_adapter`：

```text
CheckpointEngineWorker
  self.checkpoint_engine  # 负责从 actor 侧接收权重
  self.server_adapter     # 负责把权重写入 SGLang
```

rollout 侧接收权重时：

```python
weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
await self.server_adapter.update_weights(weights, global_steps=global_steps)
```

所以 rollout 侧的职责分两层：

```text
checkpoint_engine:
  传输层，接收 actor 发来的权重

server_adapter:
  引擎适配层，把收到的权重喂给 SGLang
```

### manager 如何把这些对象编排到一起

`CheckpointEngineManager.build_process_group(rollout)` 是全局编排入口。

它不是创建一个共享对象，而是按以下步骤让所有分散的 checkpoint engine 实例加入同一套通信：

```text
1. 所有 actor checkpoint_engine 执行 prepare()
   所有 rollout checkpoint_engine 执行 prepare()

2. manager 收集每个 prepare() 返回的 metadata

3. backend_cls.build_topology(...)
   根据 actor_wg.world_size 和 rollout.world_size 生成：
     actor 每个 worker 的 rank/world_size/metadata
     rollout 每个 worker 的 rank/world_size/metadata

4. 所有 actor checkpoint_engine 执行 init_process_group(...)
   所有 rollout checkpoint_engine 执行 init_process_group(...)

5. 每个实例用自己的 rank 加入同一个通信上下文
```

对 NCCL backend 来说，最终分配是：

```text
actor worker 0:
  rank = 0

actor worker 1..A-1:
  rank = -1

rollout worker 0..R-1:
  rank = 1..R

world_size:
  R + 1
```

其中 `rank = -1` 表示这个 actor worker 不参与 NCCL 传输，只消费权重 iterator，避免训练侧 collective 卡住。

全局图：

```text
CheckpointEngineManager
  |
  | build_process_group()
  |
  +-- actor_wg.execute_checkpoint_engine("prepare")
  |       |
  |       +-- ActorRolloutRefWorker.checkpoint_engine.prepare()
  |
  +-- rollout.execute_checkpoint_engine("prepare")
  |       |
  |       +-- CheckpointEngineWorker.checkpoint_engine.prepare()
  |
  +-- backend.build_topology(...)
  |       |
  |       +-- 生成 actor/rollout 的 rank/world_size/metadata
  |
  +-- actor_wg.execute_checkpoint_engine("init_process_group", ...)
  |
  +-- rollout.execute_checkpoint_engine("init_process_group", ...)
```

初始化完成后，各个对象的关系才变成：

```text
ActorRolloutRefWorker.checkpoint_engine(rank=0)
  和
CheckpointEngineWorker.checkpoint_engine(rank=1..R)

虽然是不同对象，
但已经加入同一个 checkpoint-engine 通信上下文。
```

后续 `update_weights()` 时，actor 侧调用：

```python
self.checkpoint_engine.send_weights(per_tensor_param)
```

rollout 侧调用：

```python
self.checkpoint_engine.receive_weights()
```

这两个调用发生在不同 Ray actor 中，但因为前面已经完成统一建组，所以能配对通信。

## 参与对象

简化后的主要对象如下：

```text
CheckpointEngineManager
  管理一次权重同步的生命周期：
  abort rollout / release kv_cache / build process group / update / finalize / resume

ActorRolloutRefWorker
  actor 侧 worker，负责从训练 engine 产出 per-tensor 权重

NCCLCheckpointEngine
  verl 内部传输层，负责：
    - 建 ray.util.collective NCCL group
    - 切 tensor chunk
    - 打包 uint8 bucket
    - NCCL broadcast bucket
    - ZMQ 广播 bucket metadata

CheckpointEngineWorker
  rollout 侧 worker，和推理引擎部署在同一组 rollout GPU 上
  负责接收 NCCL bucket，重建出 (name, tensor)

SGLang ServerAdapter
  verl 对 SGLang 的适配层
  负责把 CheckpointEngineWorker 收到的 tensor 喂给 SGLang

SGLang engine / server
  最终把 tensor copy/load 到模型参数中
```

## 总体时序

一次 NCCL 权重同步可以分成两段：

1. actor 侧到 rollout `CheckpointEngineWorker`：真实权重走 verl 内部 NCCL checkpoint engine。
2. rollout `CheckpointEngineWorker` 到 SGLang：请求里传 tensor metadata / IPC handle，SGLang 通过 CUDA IPC 访问本地 GPU tensor 并加载。

总体流程：

```text
CheckpointEngineManager.update_weights()
    |
    | 1. abort_replicas()
    |    中断/暂停 rollout 正在生成的请求
    |
    | 2. 收集所有 RolloutReplica.workers
    |    组成临时 rollout RayWorkerGroup
    |
    | 3. release_kv_cache_replicas()
    |    释放 KV cache，保留 model weights
    |
    | 4. build_process_group(rollout)
    |    建 verl 内部 NCCL group
    |    建 ZMQ metadata 通道
    |
    | 5. 并发执行：
    |      actor_wg.update_weights(...)
    |      rollout.update_weights(...)
    |
    | 6. finalize()
    |    清理 checkpoint engine 通信状态
    |
    | 7. resume_kv_cache_replicas()
    |
    | 8. resume_generation_replicas()
```

第 5 步必须并发，因为 actor 侧在 broadcast，rollout 侧必须同时 receive。

## NCCL 通信组是谁和谁建

这是和 Xtuner/Slime 最大的区别。

verl NCCL group 的成员是：

```text
NCCL rank 0:
  Actor worker 0

NCCL rank 1..R:
  Rollout CheckpointEngineWorker 0..R-1
```

其他 actor worker 不加入 NCCL group：

```text
actor ranks = [0, -1, -1, ...]
```

图示：

```text
Actor worker group                         Rollout side

Actor worker 0
  NCCL rank 0
      |
      | ray.util.collective.broadcast(uint8 bucket)
      v
  +---+----------------+----------------+
  |                    |                |
CheckpointEngineWorker 0   CheckpointEngineWorker 1   ...
NCCL rank 1                NCCL rank 2
  |                    |
  | update_weights_from_tensor / IPC
  v
SGLang engine         SGLang engine
```

这里没有调用 SGLang 的：

```text
/init_weights_update_group
/update_weights_from_distributed
```

verl 使用的是自己的 `NCCLCheckpointEngine.init_process_group()`，内部调用：

```python
ray.util.collective.init_collective_group(
    world_size,
    rank,
    "nccl",
    group_name,
)
```

因此，SGLang 的 model runner 并不是这次 NCCL group 的成员。

## build_process_group 阶段

`CheckpointEngineManager.build_process_group(rollout)` 分三步：

```text
1. 所有 actor worker 和 rollout worker 执行 checkpoint_engine.prepare()

2. NCCLCheckpointEngine.build_topology(...)
   生成 actor/rollout 每个 worker 的 rank、world_size、master_metadata

3. 所有 actor worker 和 rollout worker 执行 checkpoint_engine.init_process_group(...)
```

### prepare

actor rank0 是 master，会做两件事：

```text
1. 分配 send_buf / recv_buf
2. 启动 ZMQ PUB socket，返回 zmq_ip / zmq_port
```

rollout worker 会：

```text
1. 分配 send_buf / recv_buf
2. 等后续 init_process_group 时连接 master 的 ZMQ SUB socket
```

### build_topology

以 actor worker 数为 `A`，rollout worker 数为 `R`：

```text
world_size = R + 1

actor_wg_kwargs:
  rank = [0] + [-1] * (A - 1)
  world_size = [R + 1] * A
  master_metadata = [metadata[0]] * A

rollout_kwargs:
  rank = [1, 2, ..., R]
  world_size = [R + 1] * R
  master_metadata = [metadata[0]] * R
```

只有 actor worker 0 是 NCCL rank 0。其他 actor worker 的 rank 是 `-1`，不会真正参与 broadcast。

### init_process_group

actor rank0 和 rollout ranks 会调用 `ray.util.collective.init_collective_group()`。

rollout ranks 还会连接 actor rank0 的 ZMQ PUB socket：

```text
actor rank0:
  ZMQ PUB

rollout rank > 0:
  ZMQ SUB
```

ZMQ 只传 bucket metadata，不传真实权重数据。

## actor 侧发送权重

actor 侧入口是 `ActorRolloutRefWorker.update_weights()`。

当 `mode != "naive"` 时：

```python
per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
await self.checkpoint_engine.send_weights(
    per_tensor_param,
    global_steps=global_steps,
)
```

`per_tensor_param` 是一个权重 iterator/generator，产出：

```python
(name, tensor)
```

### actor rank0

actor rank0 的 `NCCLCheckpointEngine.send_weights()` 会真正发送。

处理过程：

```text
for each (name, tensor):
  1. tensor.view(torch.uint8)
     把 tensor 视为连续 byte buffer

  2. split_weight_chunks(...)
     如果 tensor 大于 bucket_size，就切成多个 chunk

  3. 把 chunk 写入 send_buf
     send_buf 是固定大小的 uint8 bucket

  4. 记录 TensorMeta
     name
     shape
     dtype
     chunk_offset
     chunk_size
     offset

  5. bucket 满了就启动 BroadcastOperation
     ZMQ 发 metadata
     NCCL 发 send_buf
```

`TensorMeta.offset` 表示当前 chunk 在 bucket 内部的起始位置。接收端必须依赖它才能从大 bucket 中切出具体 tensor chunk。

### 非 actor rank0

其他 actor worker 的 checkpoint engine rank 是 `-1`。

它们不会参与 NCCL 发送，但仍然会消费 `weights` iterator：

```python
if self.rank < 0:
    for name, weight in weights:
        pass
    return
```

这么做是为了让 actor engine 的参数 iterator 正常推进。某些训练引擎在产出完整 tensor 时可能涉及 actor worker 内部 collective，如果非 rank0 完全不迭代，可能导致训练侧卡住。

## bucket 传输：NCCL + ZMQ

一个 bucket 的发送分成两条通道：

```text
控制面：
  ZMQ PUB/SUB
  发送 bucket_meta / is_last

数据面：
  ray.util.collective NCCL broadcast
  发送 uint8 bucket
```

图示：

```text
Actor rank0
  send_buf: uint8 bucket
  bucket_meta:
    {
      "model.layers.0.xxx.weight": TensorMeta(...),
      "model.layers.1.xxx.weight": TensorMeta(...),
      "is_last": false
    }

      |---------------- ZMQ ---------------->
      | bucket_meta
      |
      |---------------- NCCL --------------->
      | send_buf
      v

Rollout CheckpointEngineWorker
  recv_buf: uint8 bucket
  bucket_meta
```

为什么要 ZMQ：

```text
NCCL 只广播 tensor buffer。
它不知道这个 uint8 bucket 里每一段 bytes 对应哪个参数。

ZMQ 负责告诉接收端：
  哪个 name
  什么 shape
  什么 dtype
  在 bucket 的哪个 offset
  chunk_size 是多少
  这是不是最后一个 bucket
```

所以真实权重数据走 NCCL，metadata 走 ZMQ。

## rollout 侧接收权重

rollout 侧入口是 `CheckpointEngineWorker.update_weights()`：

```python
weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
await self.server_adapter.update_weights(
    weights,
    global_steps=global_steps,
)
```

`NCCLCheckpointEngine.receive_weights()` 内部会：

```text
1. 等待 ZMQ metadata
2. 等待 NCCL bucket
3. 根据 TensorMeta.offset / chunk_size 从 bucket 中切出 chunk
4. merge_weight_chunks(...)
5. 重建出原始 (name, tensor)
```

如果一个 tensor 大于 bucket size，它会在发送侧被切成多个 chunk。接收侧会根据：

```text
TensorMeta.chunk_offset
TensorMeta.chunk_size
TensorMeta.shape
TensorMeta.dtype
```

把多个 chunk 重新拼成完整 tensor。

最终 `server_adapter.update_weights()` 看到的是普通的：

```python
(name, tensor)
```

而不是裸 bucket。

## 写入 SGLang

对 SGLang rollout，`server_adapter` 是 `verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter`。

它拿到 `(name, tensor)` iterator 后，会再次按 `update_weights_bucket_megabytes` 聚合：

```python
async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
    await sgl_update_weights(
        engine=self._engine,
        params_batch=params_batch,
        device_mesh_key="infer_tp",
        device_mesh=self.device_mesh,
    )
```

这里的 `sgl_update_weights` 来自 SGLang：

```python
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
```

verl 侧的 HTTP adapter 暴露了：

```text
update_weights_from_tensor
```

它的注释说明：

```text
HTTP server will only post meta data,
and the real weights will be copied directly from GPUs.
```

因此最后一跳可以理解为：

```text
CheckpointEngineWorker
  已经通过 NCCL 收到了真实权重 tensor
  tensor 在 rollout 侧 GPU 上
      |
      | update_weights_from_tensor
      | HTTP/Ray 请求中主要是 serialized tensor metadata / CUDA IPC handle
      v
SGLang server/model runner
  通过 CUDA IPC 打开这些 GPU tensor
  copy/load 到 SGLang 模型权重
```

注意，这一步不是把权重 bytes 全部塞进 HTTP body。请求体里的 `serialized_named_tensors` 更接近“如何找到这些 tensor”的描述和 IPC 句柄。真实数据仍在 GPU 内存中。

## 和 Xtuner/Slime 的对比

### Xtuner/Slime distributed update

```text
训练 rank0
  |
  | POST /init_weights_update_group
  v
SGLang engine 内部 TP workers 加入 NCCL group

训练 rank0
  |
  | POST /update_weights_from_distributed
  | metadata: names / dtypes / shapes / group_name
  |
  | torch.distributed.broadcast(tensor)
  v
SGLang model runner 直接接收 NCCL tensor
```

特点：

- SGLang 直接参与 NCCL group。
- metadata 通过 SGLang endpoint 传。
- tensor 通过 SGLang 的 distributed update 接口接收。

### verl NCCL checkpoint engine

```text
Actor rank0
  |
  | ray.util.collective.broadcast(uint8 bucket)
  | ZMQ bucket_meta
  v
Rollout CheckpointEngineWorker
  |
  | update_weights_from_tensor
  | serialized tensor metadata / CUDA IPC handle
  v
SGLang model runner
```

特点：

- SGLang 不直接加入 verl 的 NCCL group。
- NCCL group 是 actor rank0 和 rollout `CheckpointEngineWorker` 之间的。
- metadata 不走 SGLang endpoint，而是走 verl 自己的 ZMQ。
- 最终写入 SGLang 走 tensor update / CUDA IPC。

## 数据到底走哪里

以 NCCL backend 为例：

```text
Actor rank0 -> Rollout CheckpointEngineWorker
  真实权重数据：
    NCCL broadcast

  bucket metadata：
    ZMQ PUB/SUB

Rollout CheckpointEngineWorker -> SGLang
  真实权重数据：
    GPU tensor 本地访问 / CUDA IPC / GPU copy

  请求数据：
    serialized tensor metadata / IPC handle
```

所以不能简单说“verl 非共卡最终全靠 IPC”。更准确的说法是：

```text
跨 actor/rollout 的远程传输靠 NCCL；
rollout 本地喂给 SGLang 的最后一跳靠 tensor IPC。
```

CUDA IPC 不能跨机器。因此，非共卡/跨节点时，actor 到 rollout 这一大段必须由 NCCL/NIXL/Mooncake 等传输后端完成。

## 接口小结

### verl 内部 NCCL 接口

主要发生在 `NCCLCheckpointEngine`：

```text
prepare()
  分配 buffer，master 启动 ZMQ

init_process_group()
  ray.util.collective.init_collective_group(...)
  rollout ranks 连接 ZMQ

send_weights()
  actor rank0 切 chunk、打 bucket、NCCL broadcast

receive_weights()
  rollout ranks 收 bucket、按 metadata 重建 tensor

finalize()
  清理 collective group / buffer
```

### SGLang 写入接口

verl SGLang adapter 最终使用的是 tensor update 路径：

```text
update_weights_from_tensor
```

它不是 SGLang distributed update 路径：

```text
init_weights_update_group
update_weights_from_distributed
```

## 一句话总结

```text
verl 的 NCCL 权重同步不是“训练 rank0 直接 NCCL 到 SGLang”。

它是：
  actor rank0
    -- NCCL/ZMQ -->
  rollout CheckpointEngineWorker
    -- update_weights_from_tensor / CUDA IPC -->
  SGLang model runner
```

因此，verl 的这套实现比 Xtuner/Slime 多了一层 checkpoint engine 抽象：传输层由 verl 自己实现，SGLang 只负责最后的 tensor 加载。

## 附：Mooncake checkpoint engine 有什么区别

Mooncake backend 和 NCCL backend 的高层分层是一样的：

```text
Actor/Trainer
  -> verl checkpoint engine 传输
  -> Rollout CheckpointEngineWorker
  -> SGLang update_weights_from_tensor / tensor update
```

也就是说，Mooncake 也不是让 SGLang 直接加入权重同步 group。最终写入 SGLang 的最后一跳仍然是：

```text
Rollout CheckpointEngineWorker
  -- serialized tensor metadata / CUDA IPC handle -->
SGLang server/model runner
```

Mooncake 和 NCCL 的主要区别在 checkpoint engine 这一段，也就是：

```text
Actor rank0 -> Rollout CheckpointEngineWorker
```

### 拓扑区别

NCCL backend 是 actor rank0 对所有 rollout worker 做 collective broadcast：

```text
Actor rank0
  |
  | NCCL broadcast
  v
+----------------+----------------+----------------+
|                |                |                |
Rollout rank1   Rollout rank2    Rollout rank3    ...
```

Mooncake backend 更像 ring / chain P2P：

```text
Actor rank0
  |
  | Mooncake TransferEngine read/write
  v
Rollout rank1
  |
  | Mooncake TransferEngine read/write
  v
Rollout rank2
  |
  v
Rollout rank3
  |
  v
...
```

它的 topology 仍然是：

```text
rank 0:
  Actor worker 0

rank 1..R:
  Rollout CheckpointEngineWorker 0..R-1

actor 其他 worker:
  rank = -1
```

所以和 NCCL 一样，SGLang model runner 不在这个传输拓扑里。

### 建组方式区别

NCCL backend 使用：

```python
ray.util.collective.init_collective_group(...)
```

同时单独启动 ZMQ，用来传 bucket metadata。

Mooncake backend 使用：

```python
StatelessProcessGroup.create(...)
```

这个 group 主要用于传控制信息，例如：

```text
session_id
buffer ptr
bucket_meta
ptr
len
is_last
```

真实数据搬运不靠 `torch.distributed.broadcast`，而是靠 Mooncake `TransferEngine`。

### 内存注册

Mooncake 初始化时会创建并注册两类 buffer：

```text
self.buf:
  2 * bucket_size 的 uint8 buffer
  切成两个 bucket 做 ping-pong

self.magic_buf:
  4KB 小 buffer
  用来写 magic bytes，通知前一个 rank 当前 buffer 已消费完
```

初始化时会调用：

```python
self.engine.batch_register_memory(
    [self.buf.data_ptr(), self.magic_buf.data_ptr()],
    [2 * self.bucket_size, 4 * 1024],
)
```

这一步是 Mooncake/RDMA 类传输的关键：要让 transfer engine 知道哪些内存区域可以被远端读写。

### metadata 怎么传

NCCL backend 的 metadata 走 ZMQ：

```text
ZMQ:
  bucket_meta / is_last

NCCL:
  uint8 bucket
```

Mooncake backend 不用 ZMQ。它把 metadata 放在 `StatelessProcessGroup` 的 object message 里传：

```text
store.send_obj(info, next_rank)
store.recv_obj(prev_rank)
```

发送端构造的 `info` 大致是：

```python
{
    "bucket_meta": bucket_meta,
    "ptr": current.data_ptr(),
    "len": offset,
    "is_last": False,
}
```

其中：

- `bucket_meta`：记录 bucket 里有哪些 tensor、shape、dtype、offset。
- `ptr`：当前 rank 本地 bucket buffer 的地址。
- `len`：本次 bucket 实际有效字节数。
- `is_last`：是否最后一个 bucket。

### 真实数据怎么传

NCCL backend：

```text
actor rank0 调 collective.broadcast(send_buf)
所有 rollout ranks 同时收到同一个 bucket
```

Mooncake backend：

```text
rank i 收到前一个 rank 发来的 info
  |
  | info 里有前一个 rank 的 session_id / ptr / len
  v
rank i 调 TransferEngine.transfer_sync_read(...)
  从前一个 rank 的 buffer 读 len 字节到自己的 current buffer
  |
  v
rank i 把 info.ptr 改成自己的 current.data_ptr()
  |
  | store.send_obj(info, rank + 1)
  v
下一个 rank 继续读
```

也就是每个 rank 从前一个 rank 读数据，然后把同一个 bucket 信息继续传给下一个 rank。

简化伪流程：

```text
Actor rank0:
  填充 bucket
  store.send_obj({ptr, len, bucket_meta}, rank1)

Rollout rank1:
  info = store.recv_obj(rank0)
  transfer_sync_read(rank0_session, local_buf, info.ptr, info.len)
  yield tensors from local_buf
  store.send_obj({ptr=local_buf.ptr, len, bucket_meta}, rank2)

Rollout rank2:
  info = store.recv_obj(rank1)
  transfer_sync_read(rank1_session, local_buf, info.ptr, info.len)
  yield tensors from local_buf
  store.send_obj(..., rank3)
```

因此 Mooncake 的数据面不是“一发多收”的 collective broadcast，而是“逐跳 P2P read + 转发 metadata”。

### ack / buffer 复用

Mooncake sender 和 receiver 都使用双 buffer：

```text
buf[0] = self.buf[:bucket_size]
buf[1] = self.buf[bucket_size:]
```

这样当前 bucket 在传输时，发送端可以准备下一个 bucket。

为了避免前一个 rank 过早复用 buffer，receiver 消费完当前 buffer 后会向前一个 rank 的 buffer 写 magic bytes：

```python
transfer_sync_write(
    prev_session_id,
    magic_buf.data_ptr(),
    prev_ptr,
    4,
)
```

前一个 rank 通过 `wait_for_complete(current)` 等这几个 magic bytes 出现，再复用该 buffer。

所以 Mooncake 这里自己实现了一个简单的 buffer 生命周期协议。

### bucket 组织区别

NCCL backend 使用 `split_weight_chunks()`：

```text
如果单个 tensor 大于 bucket_size，可以被切成多个 chunk。
接收侧再 merge_weight_chunks() 拼回完整 tensor。
```

Mooncake backend 当前逻辑更简单：

```text
每个 tensor 必须能放进一个 bucket。
多个小 tensor 可以合并进一个 bucket。
```

代码里会断言：

```python
assert offset + weight.nbytes <= self.bucket_size
```

也就是说，如果某个单 tensor 大于 `bucket_size`，Mooncake backend 不会像 NCCL backend 那样自动切 chunk，而是会报错。实际使用时需要保证 `update_weights_bucket_megabytes` 足够大，至少能容纳最大单个权重 tensor。

另外 Mooncake 发送前会做：

```python
weight = weight.to(self.rollout_dtype)
```

也就是传输前会把权重转成 rollout 期望 dtype。

### 和 NCCL 对比小结

```text
共同点：
  - 都是 verl checkpoint engine 内部传输。
  - 都不是调用 SGLang /init_weights_update_group。
  - SGLang 不直接参与传输 group。
  - 最后都由 ServerAdapter.update_weights() 喂给 SGLang。
  - 最后一跳仍然是 update_weights_from_tensor / CUDA IPC。

NCCL 特点：
  - 数据面：ray.util.collective NCCL broadcast。
  - metadata：ZMQ PUB/SUB。
  - 拓扑：actor rank0 一次 broadcast 到所有 rollout ranks。
  - 支持把超大 tensor 切成多个 chunk。

Mooncake 特点：
  - 数据面：Mooncake TransferEngine P2P read/write。
  - metadata：StatelessProcessGroup send_obj/recv_obj。
  - 拓扑：actor rank0 -> rollout rank1 -> rollout rank2 -> ... 的链式传输。
  - 使用注册内存和 magic bytes 管理 buffer 复用。
  - 当前实现要求单个 tensor 能放进一个 bucket。
```

### Mooncake 总体流程图

```text
Actor rank0
  |
  | 1. get_per_tensor_param()
  | 2. 转 rollout_dtype
  | 3. 打包到本地 registered bucket buffer
  | 4. store.send_obj(bucket_meta + ptr + len, rank1)
  v
Rollout CheckpointEngineWorker rank1
  |
  | 5. transfer_sync_read(actor buffer -> local buffer)
  | 6. 根据 bucket_meta 切出 (name, tensor)
  | 7. store.send_obj(bucket_meta + local ptr + len, rank2)
  | 8. transfer_sync_write magic bytes 给 actor，通知 buffer 可复用
  v
Rollout CheckpointEngineWorker rank2
  |
  | 9. transfer_sync_read(rank1 buffer -> local buffer)
  | 10. 根据 bucket_meta 切出 (name, tensor)
  | 11. 继续传给 rank3
  v
...

每个 rollout rank 本地：
  receive_weights() yield (name, tensor)
      |
      v
  ServerAdapter.update_weights()
      |
      v
  SGLang update_weights_from_tensor / CUDA IPC
```

一句话：

```text
NCCL backend 是 actor rank0 broadcast 给所有 rollout worker；
Mooncake backend 是 actor rank0 把 bucket 放进注册内存，然后 rollout workers 按 rank 链式 P2P 读取和转发。

但二者到 SGLang 的最后一跳没有本质区别，仍然是 tensor update / CUDA IPC。
```

## 附：NIXL 是什么，以及 Ray 里的 NIXL backend

`nixl` 指的是 `ai-dynamo/nixl` 这个库，全称是 NVIDIA Inference Xfer Library。

它的定位不是某个单一传输协议，而是一个面向 AI inference 场景的数据传输抽象层：

```text
NIXL
  |
  +-- 抽象不同 memory 类型
  |     - CPU memory
  |     - GPU memory
  |
  +-- 抽象不同 storage 类型
  |     - file
  |     - block storage
  |     - object store
  |
  +-- 通过 plugin/backend 接不同传输实现
        - UCX
        - UCCL
        - Mooncake
        - POSIX
        - GDS
        - 其他 backend
```

所以可以把它理解成：

```text
NIXL 是一个大集合 / 统一传输抽象层。
Mooncake 可以作为 NIXL 的一个 backend/plugin。
```

这和 verl 里单独的 `mooncake` backend 不是同一层：

```text
verl mooncake backend:
  直接调用 mooncake.engine.TransferEngine

verl nixl backend:
  调 NIXL agent 统一 API
  NIXL 底层再选择 UCX / UCCL / Mooncake / POSIX / GDS 等插件
```

### NIXL 在 verl checkpoint engine 里的位置

在 verl 权重同步里，`NIXLCheckpointEngine` 是 checkpoint engine 的一个 backend：

```text
Actor/Trainer
  -- NIXL P2P transfer -->
Rollout CheckpointEngineWorker
  -- update_weights_from_tensor / CUDA IPC -->
SGLang
```

它和 NCCL、Mooncake 一样，只负责：

```text
Actor/Trainer -> Rollout CheckpointEngineWorker
```

这一段权重 bucket 传输。

它不改变最后一跳：

```text
Rollout CheckpointEngineWorker -> SGLang
```

最后写入 SGLang 仍然走 tensor update / CUDA IPC。

verl 里的 NIXL 逻辑大致是：

```text
1. 每个 checkpoint_engine 实例创建一个 nixl_agent
2. 每个 rank 注册自己的 send_buf / recv_buf
3. manager 把相邻 rank 的 agent metadata 分发好
4. 当前 rank add_remote_agent(prev/next)
5. 当前 rank 从前一个 rank 的 buffer 里 READ 数据
6. 当前 rank 再把同一个 bucket 暴露给下一个 rank READ
```

拓扑类似：

```text
Actor rank0
  |
  | NIXL READ by next rank
  v
Rollout rank1
  |
  | NIXL READ by next rank
  v
Rollout rank2
  |
  v
...
```

控制面和数据面：

```text
控制面：
  ZMQ PUSH/PULL
  传 bucket_meta / remote_descs / notify_key / is_last

数据面：
  NIXL agent READ transfer
  传真实 tensor bucket bytes
```

### NIXL 和 Mooncake 的关系

容易混淆的是：verl 里既有 `nixl` backend，也有 `mooncake` backend。

可以这样区分：

```text
MooncakeCheckpointEngine:
  verl 直接使用 Mooncake TransferEngine
  控制面用 StatelessProcessGroup send_obj/recv_obj
  数据面用 transfer_sync_read / transfer_sync_write

NIXLCheckpointEngine:
  verl 使用 NIXL agent
  NIXL 底层可以选择 UCX / UCCL / Mooncake 等 transport
  控制面用 ZMQ PUSH/PULL
  数据面用 initialize_xfer("READ") + transfer + check_xfer_state
```

所以：

```text
NIXL 是抽象层。
Mooncake 是具体传输实现之一。

但 verl 也提供了一个直接调用 Mooncake 的 checkpoint_engine backend。
```

### Ray 里的 NIXL backend 是什么

Ray 也有一个和 NIXL 相关的功能：Ray Direct Transport，简称 RDT。

Ray RDT 允许 actor method 返回 `torch.Tensor` 时不走普通 Ray CPU object store，而是使用指定 tensor transport 直接传输 tensor。

普通 Ray ObjectRef 传 tensor 时，可能会变成：

```text
GPU tensor
  -> copy 到 CPU
  -> serialize / 放入 Ray object store
  -> 目标 actor 再取出
  -> copy 回 GPU
```

Ray RDT + NIXL 的目标是：

```text
tensor 数据尽量留在原设备上
Ray 只保存/传递 tensor reference
目标 actor 需要 tensor 时，用 NIXL 做 P2P 传输
```

Ray 示例形式类似：

```python
@ray.remote(num_gpus=1)
class MyActor:
    @ray.method(tensor_transport="nixl")
    def random_tensor(self):
        return torch.randn(1000, 1000).cuda()

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)
```

然后：

```python
sender = MyActor.remote()
receiver = MyActor.remote()

tensor_ref = sender.random_tensor.remote()
result_ref = receiver.sum.remote(tensor_ref)
```

这里 `tensor_ref` 的数据传输就可以由 Ray RDT 的 NIXL backend 处理。

Ray 文档里还提到：

```text
RDT 支持的 tensor transport 包括：
  - Gloo
  - NCCL
  - NIXL

NIXL 用于 point-to-point transfer，
可以加速 CPU / NVIDIA GPU tensor 在 Ray actors 之间的传输。
```

### Ray NIXL 和 verl NIXL 的区别

二者底层都可以用 `ai-dynamo/nixl`，但集成层不同。

```text
Ray RDT + NIXL:
  目标：
    加速 Ray actor 之间传 torch.Tensor / ObjectRef

  用户接口：
    @ray.method(tensor_transport="nixl")
    ray.put(..., _tensor_transport="nixl")
    ray.get(..., _tensor_transport="nixl")

  Ray 负责：
    tensor 引用管理
    actor 之间的 tensor transport
    避免普通 object store 的 CPU copy/serialization
```

```text
verl checkpoint_engine + NIXL:
  目标：
    actor 权重同步到 rollout workers

  用户接口：
    rollout.checkpoint_engine.backend = "nixl"

  verl 负责：
    权重 iterator
    bucket 切分
    bucket metadata
    ring/chain 拓扑
    send_weights / receive_weights 协议

  NIXL 负责：
    底层 P2P buffer transfer
```

所以可以总结成：

```text
Ray:
  NIXL 是 Ray tensor ObjectRef 传输后端。

verl:
  NIXL 是 checkpoint_engine 权重同步后端。

两者可能使用同一个 NIXL 库，但上层协议和使用方式不同。
```

### NIXL 是否可以加速 CPU / GPU tensor 传输

可以。NIXL 的目标就是统一并加速不同 memory/storage 之间的 point-to-point data transfer。

在 Ray RDT 里，它可以用于：

```text
CPU tensor actor-to-actor transfer
GPU tensor actor-to-actor transfer
```

在 verl checkpoint engine 里，它可以用于：

```text
actor GPU/CPU buffer
  -> rollout GPU/CPU buffer
```

具体走哪种底层传输，取决于：

```text
NIXL 安装时可用的 backend/plugin
运行环境里的硬件
配置使用的 device
UCX/UCCL/Mooncake 等 backend 的可用性
```

直观理解：

```text
普通 Ray:
  更像通过通用 object store 搬对象

Ray RDT + NIXL:
  更像为 torch.Tensor 开了一条专门的数据通道

verl checkpoint_engine + NIXL:
  更像为权重 bucket 开了一条专门的数据通道
```

## 附：Kimi checkpoint engine 是什么

`kimi_ckpt_engine` 是 verl 里的另一个 checkpoint engine backend。

它不是 Kimi 模型专属逻辑，而是一套来自 `checkpoint_engine` 包的参数服务/权重同步方案：

```python
@CheckpointEngineRegistry.register("kimi_ckpt_engine")
class KIMICheckpointEngine(CheckpointEngine):
    ...
```

README 对它的定位是：

```text
Comm Library:
  MOONCAKE + NCCL/HCCL

Topology:
  p2p + broadcast

Use case:
  Off-policy training
  Actor/rollout disaggregated
  Save checkpoint each time
```

这几个关键词很重要：它不是单纯追求最短路径的 GPU-to-GPU 在线同步，而是更偏向 checkpoint-driven 的同步路径。

### 核心思想

`kimi_ckpt_engine` 的核心流程是：

```text
Actor GPU weights
  -> offload 到 CPU
  -> register_checkpoint()
  -> rollout 侧从 ParameterServer / checkpoint 拉权重
  -> 某个 rollout rank 收到 bucket
  -> rollout group 内 broadcast
  -> 每个 rollout rank yield (name, tensor)
  -> ServerAdapter.update_weights()
  -> SGLang tensor update / CUDA IPC
```

所以它和前面几个 backend 的根本差别是：

```text
NCCL / NIXL / Mooncake:
  actor GPU tensor -> rollout GPU tensor
  更像在线传输后端

Kimi checkpoint engine:
  actor GPU tensor -> CPU checkpoint / ParameterServer -> rollout GPU tensor
  更像 checkpoint-driven 同步后端
```

### actor 侧不是只有 rank0 发送

`kimi_ckpt_engine` 里，actor 侧会让多个 actor rank 分摊参数。

分配方式大致是：

```python
if tensor_idx % world_size == rank_id:
    current_bucket[name] = tensor
```

也就是说：

```text
Actor rank0:
  负责一部分参数

Actor rank1:
  负责另一部分参数

Actor rank2:
  负责另一部分参数

...
```

这和 `nccl/nixl/mooncake` 常见路径不一样：

```text
nccl/nixl/mooncake:
  通常 actor rank0 是主要发送端

kimi_ckpt_engine:
  actor 多 rank 分摊权重注册
```

### actor 侧先 offload 到 CPU

actor 侧 `send_weights()` 会把 GPU tensor offload 到 CPU：

```python
def offload_cpu(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    return name, tensor.to("cpu", non_blocking=True)
```

然后把这些 CPU tensor 注册到 ParameterServer：

```python
self.parameter_server.register_checkpoint(
    self.checkpoint_name,
    named_tensors=named_tensors,
)
```

所以这条路径里，actor 侧明确有一步：

```text
GPU -> CPU
```

这一步是它适合“本来每次都要保存 checkpoint”的原因之一。

如果训练流程本来就要生成 CPU 侧参数快照 / checkpoint，那么这一步可以和 checkpoint 流程复用。

如果你只是想最快在线同步权重，这一步可能反而是额外开销。

### 初始化通信组

`kimi_ckpt_engine` 初始化时会创建一个包含 actor 和 rollout 的全局 process group：

```text
world_size = actor_wg_world_size + rollout_world_size
```

rank 分配是：

```text
actor ranks:
  0 .. actor_wg_world_size - 1

rollout ranks:
  actor_wg_world_size .. actor_wg_world_size + rollout_world_size - 1
```

它还会创建一个只包含 rollout ranks 的子 group：

```python
self.rollout_ranks = list(range(self.actor_wg_world_size, self.world_size))
self.rollout_group = dist.new_group(self.rollout_ranks)
```

这个 `rollout_group` 后面用于 rollout 内部 broadcast。

### rollout 侧怎么收

rollout 侧 `receive_weights()` 会先 gather checkpoint metadata：

```python
self.parameter_server.gather_metas(self.checkpoint_name)
```

然后调用 `ParameterServer.receive_tensor(...)`：

```python
async for name, tensor in self.parameter_server.receive_tensor(
    self.checkpoint_name,
    self.rollout_group,
    self.rollout_ranks,
    self.bucket_size,
):
    yield name, tensor
```

这套 `receive_tensor()` 在 verl 中被 monkey patch 成当前文件里的实现。

核心逻辑是：

```text
1. 根据 ParameterMeta 生成 H2D buckets
2. 每个 bucket 有一个 receiver_rank
3. receiver_rank 从 checkpoint / ParameterServer / p2p store 拉取 bucket
4. receiver_rank 在 rollout_group 内 broadcast 该 bucket
5. 所有 rollout ranks 根据 metadata 从 bucket 中切出 tensor
```

broadcast 代码很直接：

```python
dist.broadcast(self.bucket, src=self.rank, group=self.ranks_group)
```

注意这里的 src 是某个 rollout rank，不是 actor rank0。

### 数据流图

```text
Actor rank0       Actor rank1       Actor rank2
  |                 |                 |
  | 负责一部分参数   | 负责一部分参数   | 负责一部分参数
  | GPU -> CPU      | GPU -> CPU      | GPU -> CPU
  | register        | register        | register
  v                 v                 v
          ParameterServer / checkpoint
                    |
                    | P2P / H2D bucket fetch
                    v
             Rollout receiver rank
                    |
                    | broadcast(bucket, src=receiver_rank, group=rollout_group)
                    v
       +------------+------------+------------+
       |                         |            |
  Rollout rank A            Rollout rank B   ...
       |
       | yield (name, tensor)
       v
  ServerAdapter.update_weights()
       |
       v
  SGLang update_weights_from_tensor / CUDA IPC
```

### 和其他 backend 的区别

```text
NCCL backend:
  actor rank0
    -- NCCL broadcast + ZMQ metadata -->
  all rollout CheckpointEngineWorkers

NIXL backend:
  actor rank0
    -- NIXL chain P2P -->
  rollout rank1 -> rollout rank2 -> ...

Mooncake backend:
  actor rank0
    -- Mooncake TransferEngine chain P2P -->
  rollout rank1 -> rollout rank2 -> ...

Kimi checkpoint engine:
  actor all ranks
    -- GPU -> CPU checkpoint/register -->
  ParameterServer
    -- P2P/H2D -->
  selected rollout receiver rank
    -- rollout_group broadcast -->
  all rollout ranks
```

所以可以概括成：

```text
NCCL/NIXL/Mooncake:
  权重同步主要是在线传输问题。

Kimi checkpoint engine:
  权重同步变成 checkpoint/ParameterServer + rollout 内 broadcast 问题。
```

### 为什么适合“每次都要保存 checkpoint”的场景

如果训练流程本来每次同步都需要保存 checkpoint 或生成 CPU 侧参数快照，那么 `kimi_ckpt_engine` 的路径比较自然：

```text
actor GPU weights
  -> CPU checkpoint / ParameterServer
  -> rollout 拉取
```

这能复用 checkpoint 这一步。

它的优势：

```text
1. actor 多 rank 分摊参数
   不是单 actor rank0 独自负责全部传输。

2. 权重先注册为 CPU checkpoint / ParameterServer 状态
   对 checkpoint-driven 流程有复用价值。

3. rollout 侧形成子通信组
   某个 rollout rank 拉到 bucket 后，在 rollout 内 broadcast。

4. 可以结合 checkpoint_engine[p2p] / Mooncake transfer engine 做 P2P/H2D。
```

代价：

```text
1. 多了一步 GPU -> CPU offload
   如果你不需要 checkpoint，这可能是额外开销。

2. 链路更长
   Actor GPU -> CPU checkpoint -> rollout GPU -> rollout broadcast。

3. 依赖 checkpoint_engine.ps / ParameterServer / p2p store
   心智模型和环境依赖都更复杂。

4. README 标注 elastic 较低
   通信 group 变化时需要 rebuild。
```

所以可以这样选：

```text
只想尽快在线同步权重：
  优先看 nccl / nixl / mooncake。

本来每次都要保存 checkpoint / CPU 参数快照：
  kimi_ckpt_engine 更匹配。
```

### 官方用法

repo 中没有看到完整训练 shell 示例直接启用 `kimi_ckpt_engine`。

能看到的官方用法主要是：

```text
verl/checkpoint_engine/README.md
tests/checkpoint_engine/test_correctness_on_gpu.py
tests/checkpoint_engine/test_correctness_on_npu.py
```

测试里的最小配置是：

```python
checkpoint_engine_config = CheckpointEngineConfig(
    backend="kimi_ckpt_engine",
    engine_kwargs={
        "kimi_ckpt_engine": {
            "rebuild_group": rebuild_group,
        }
    },
)
```

然后照常创建 actor worker group、rollout replicas 和 manager：

```python
actor_wg = create_trainer_worker_group(...)
rollout, replicas = await create_rollout_worker_group(...)

checkpoint_manager = CheckpointEngineManager(
    config=checkpoint_engine_config,
    actor_wg=actor_wg,
    replicas=replicas,
)

await checkpoint_manager.update_weights()
```

如果写成 Hydra/YAML，核心配置大致是：

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      backend: kimi_ckpt_engine
      engine_kwargs:
        kimi_ckpt_engine:
          rebuild_group: false
```

命令行覆盖形式类似：

```bash
actor_rollout_ref.rollout.checkpoint_engine.backend=kimi_ckpt_engine \
actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.kimi_ckpt_engine.rebuild_group=false
```

实际 key 路径要以具体训练入口的 config 为准。

### 依赖和注意事项

README 里提到这个模式需要 P2P feature：

```bash
pip install 'checkpoint-engine[p2p]'
```

并且要求：

```text
checkpoint-engine >= 0.4.0
```

此外，相关测试当前是 skip 状态：

```text
temporary skip since our ci environment is not ready
```

所以这个后端更像特定环境/特定集群下的高级 backend，不像 `nccl/nixl` 那样是更常规的在线同步选项。

一句话：

```text
kimi_ckpt_engine 是“actor 多 rank CPU checkpoint + rollout P2P/H2D 拉取 + rollout 内 broadcast”的混合型权重同步后端。
它特别适合非共卡且本来就要保存 checkpoint / CPU 参数快照的场景。
```
