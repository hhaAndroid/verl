# verl 中 kimi_ckpt_engine 权重同步流程

这篇只讲一件事：

```text
actor ranks 把 named tensors 注册到外部 ParameterServer，
rollout ranks 再通过外部 checkpoint-engine 的 receive 逻辑取回，
最后 verl 把取回的 tensors 交给 SGLang 更新权重。
```

这里的 `ParameterServer` 来自外部包：

```python
from checkpoint_engine.ps import ParameterServer
```

verl 里的入口 backend 是：

```python
@CheckpointEngineRegistry.register("kimi_ckpt_engine")
class KIMICheckpointEngine(CheckpointEngine):
    ...
```

源码位置：

- `verl/checkpoint_engine/kimi_checkpoint_engine.py`
- `verl/checkpoint_engine/base.py`
- `verl/workers/engine_workers.py`
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py`

## 1. 总体流程

一次权重同步可以压缩成下面这张图：

```text
trainer loop
  |
  | checkpoint_manager.update_weights(global_steps)
  v
CheckpointEngineManager
  |
  | build_process_group()
  |   - actor CE prepare / init_process_group
  |   - rollout CE prepare / init_process_group
  v
actor_wg.update_weights(mode="kimi_ckpt_engine")
  |
  | actor.engine.get_per_tensor_param()
  | KIMICheckpointEngine.send_weights()
  v
actor ranks register named_tensors to ParameterServer
  |
  | parameter_server.register_checkpoint(...)
  | parameter_server.gather_metas(...)
  v
external checkpoint-engine metadata / bucket plan
  |
  | rollout KIMICheckpointEngine.receive_weights()
  | parameter_server.receive_tensor(...)
  v
rollout ranks get (name, tensor)
  |
  | CheckpointEngineWorker.update_weights()
  | server_adapter.update_weights(weights)
  v
SGLangRollout.update_weights()
  |
  | sgl_update_weights(...)
  | SGLang update_weights_from_tensor
  v
SGLang workers load new weights
```

verl 本身不让 SGLang 直接调用外部 checkpoint-engine 的 `from_ipc` 接口。  
在这条集成里，外部 checkpoint-engine 主要负责 `actor CE -> rollout CE` 的权重传输；SGLang 最终还是走 verl 已有的 `SGLangRollout.update_weights()`。

## 2. 初始化：所有 actor / rollout CE 加入同一个 ParameterServer world

`KIMICheckpointEngine.init_process_group()` 会给每个 actor CE 和 rollout CE 都创建一个外部 `ParameterServer`：

```python
self.parameter_server = ParameterServer(
    rank=rank,
    world_size=self.world_size,
    auto_pg=False,
    master_addr=master_metadata.dist_ip,
    master_port=master_metadata.dist_port,
)
self.parameter_server.init_process_group()
```

rank 编号规则是：

```text
actor CE ranks:
  0 .. actor_wg_world_size - 1

rollout CE ranks:
  actor_wg_world_size .. actor_wg_world_size + rollout_world_size - 1

ParameterServer world_size:
  actor_wg_world_size + rollout_world_size
```

同时会为 rollout ranks 建一个单独 group：

```python
self.rollout_ranks = list(range(self.actor_wg_world_size, self.world_size))
self.rollout_group = dist.new_group(self.rollout_ranks)
```

这个 `rollout_group` 后面用于 rollout ranks 内部广播 bucket。

## 3. actor 侧：从训练引擎拿 named tensors

actor worker 的入口是 `EngineWorker.update_weights()`。

当 backend 是 `kimi_ckpt_engine` 时，它会走非 `naive` 分支：

```python
per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
metrics = await self.checkpoint_engine.send_weights(
    per_tensor_param,
    global_steps=global_steps,
)
```

这里的 `per_tensor_param` 是一个 generator，产出：

```text
(name, tensor)
```

也就是说，权重来源是当前训练引擎内存，不是 HF disk 文件。

## 4. named tensor 是 full tensor，不是原始 shard

这里容易误解：训练侧每张卡上的模型参数通常是 sharded 的，但 `kimi_ckpt_engine` 传给 `ParameterServer.register_checkpoint(named_tensors=...)` 的 tensor 不是原始 shard，而是训练 engine 临时导出的 full tensor。

以 FSDP / DTensor 路径为例，`get_per_tensor_param()` 里会做类似逻辑：

```python
param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
```

其中 `full_tensor()` 会让所有相关训练 ranks 参与 collective，把当前参数的 shard 重组成完整 tensor。

所以真实语义是：

```text
训练侧原始状态:
  rank0 持有 paramA shard0
  rank1 持有 paramA shard1
  rank2 持有 paramA shard2
  ...

get_per_tensor_param() 导出时:
  所有 ranks 参与 paramA 的 full_tensor/all-gather
  每个 rank 临时得到 paramA full tensor

kimi_ckpt_engine 再决定:
  只有 tensor_idx % actor_world_size == 当前 rank 的 rank
  才把这个 full tensor 放进 named_tensors 并注册
  其它 rank 临时生成后就跳过
```

因此它实现的不是：

```text
rank0 注册 paramA shard0
rank1 注册 paramA shard1
rank2 注册 paramA shard2
```

而是：

```text
rank0 注册完整参数 A、完整参数 E、完整参数 I
rank1 注册完整参数 B、完整参数 F、完整参数 J
rank2 注册完整参数 C、完整参数 G、完整参数 K
```

这样做的好处是 `ParameterServer` 侧看到的是标准 HF-style named tensors，后面 rollout / SGLang 不需要理解训练侧 shard 布局。

代价是：当前实现里过滤发生在 full tensor 生成之后。也就是说，每个 actor rank 都会遍历所有参数，并参与每个参数的 full_tensor/all-gather；只是最后只有一部分参数会被当前 rank 注册到 `ParameterServer`。它节省的是 registered memory 不重复保存整模型，不是节省所有参数 full tensor 导出的 collective 成本。

## 5. actor 侧：每个 rank 自动注册一部分参数

`KIMICheckpointEngine.send_weights()` 里会调用：

```python
ckpt_get_named_tensor_buckets(
    weights,
    self.bucket_size,
    self.actor_wg_world_size,
    self.rank,
    self.rollout_dtype,
)
```

它的分配规则是：

```python
if tensor_idx % world_size == rank_id:
    current_bucket[name] = tensor
```

所以用户不需要告诉每张卡注册哪些参数。  
verl 会按参数遍历顺序自动切分：

```text
actor rank0: tensor_idx % actor_world_size == 0 的参数
actor rank1: tensor_idx % actor_world_size == 1 的参数
actor rank2: tensor_idx % actor_world_size == 2 的参数
...
```

这意味着 128 个 actor ranks 会都调用 `register_checkpoint()`，但每个 rank 注册的是不同参数子集，不是 128 份重复全量。

## 6. actor 侧：注册到外部 ParameterServer

每个 actor rank 会先把自己负责的 GPU tensors offload 到 CPU：

```python
def offload_cpu(name, tensor):
    return name, tensor.to("cpu", non_blocking=True)
```

然后注册到外部 `ParameterServer`：

```python
self.parameter_server.register_checkpoint(
    self.checkpoint_name,
    named_tensors=named_tensors,
)
```

注册完成后会收集全局 metadata：

```python
self.parameter_server.gather_metas(self.checkpoint_name)
dist.barrier()
```

最后 actor 侧会 unregister：

```python
self.parameter_server.unregister_checkpoint(self.checkpoint_name)
```

这里的 `gather_metas()` 很关键：rollout 侧后面需要知道全局参数有哪些、每个参数在哪个 owner rank、如何组成 bucket、每个 bucket 从哪里读。

## 7. rollout 侧：通过 receive_tensor 取回权重

rollout CE 的入口是 `KIMICheckpointEngine.receive_weights()`：

```python
self.parameter_server.gather_metas(self.checkpoint_name)

async for name, tensor in self.parameter_server.receive_tensor(
    self.checkpoint_name,
    self.rollout_group,
    self.rollout_ranks,
    self.bucket_size,
):
    yield name, tensor
```

也就是说，rollout CE 不自己解析 actor 权重，而是直接调用外部 checkpoint-engine 的 receive 逻辑。

verl 在 `kimi_checkpoint_engine.py` 里 monkey patch 了 `ParameterServer.receive_tensor`，其核心过程是：

```text
1. 基于全局 ParameterMeta 生成 H2D bucket 计划。

2. 每个 rollout receiver rank 知道：
   - 自己负责哪些 bucket
   - bucket 的 owner actor rank 是谁
   - bucket 内有哪些参数和 offset

3. receiver rank 从 owner rank 的 registered memory 读取 bucket。

4. receiver rank 在 rollout_group 内 dist.broadcast(bucket, src=receiver_rank)。

5. rollout ranks 从本地 bucket buffer 里按 metadata 切出 tensor。

6. yield name, tensor。
```

所以 rollout 侧最终看到的仍然是普通的：

```text
(name, tensor)
```

## 8. rollout CE 到 SGLang：还是走 update_weights_from_tensor

`CheckpointEngineWorker.update_weights()` 负责把 rollout CE 产出的 generator 交给 rollout adapter：

```python
weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)

await self.server_adapter.update_weights(
    weights,
    global_steps=global_steps,
    wire_format=getattr(self.checkpoint_engine, "wire_format", "named_tensors"),
)
```

对于 SGLang，`server_adapter` 就是 `SGLangRollout`。

`SGLangRollout.update_weights()` 会继续按 bucket 分批：

```python
async for params_batch in get_named_tensor_buckets(
    weights,
    update_weights_bucket_bytes,
):
    await sgl_update_weights(
        engine=self._engine,
        params_batch=params_batch,
        device_mesh_key="infer_tp",
        device_mesh=self.device_mesh,
    )
```

`sgl_update_weights()` 最后会让 TP leader 调 SGLang 的：

```text
update_weights_from_tensor
```

所以 `kimi_ckpt_engine` 这条链路里，SGLang 不感知外部 `ParameterServer`。  
SGLang 只接收 verl rollout adapter 发来的 tensor update 请求。

## 9. 128 卡例子

假设：

- actor 训练侧 128 卡。
- rollout 侧 128 卡。
- SGLang rollout 被组织成若干实例，例如 16 个实例，每个实例 8 卡。

在 `kimi_ckpt_engine` 的 ParameterServer world 里：

```text
actor CE ranks:
  0..127

rollout CE ranks:
  128..255

world_size:
  256
```

actor 侧注册：

```text
actor rank0   注册 tensor_idx % 128 == 0 的参数
actor rank1   注册 tensor_idx % 128 == 1 的参数
...
actor rank127 注册 tensor_idx % 128 == 127 的参数
```

rollout 侧接收：

```text
rollout ranks 128..255 共同调用 receive_tensor()
每个 receiver rank 负责一部分 bucket
receiver rank 从对应 owner actor rank 读取 bucket
receiver rank 在 rollout_group 内 broadcast
所有 rollout ranks 都能按 metadata yield 出本轮需要交给 SGLang 的 tensor
```

最后：

```text
每个 rollout worker 把自己拿到的 tensor generator
交给本地 SGLangRollout.update_weights()
再由 SGLang 的 update_weights_from_tensor 完成模型权重更新。
```

## 10. 这条链路的关键理解

1. `register_checkpoint()` 是 actor ranks 调的。
2. 每个 actor rank 都会调，但每个 rank 注册不同参数子集。
3. 用户不需要手动指定每张卡注册哪些参数，分配规则写在 `ckpt_get_named_tensor_buckets()` 里。
4. 注册的是完整参数 tensor，不是训练 rank 的原始 shard。
5. 完整参数通常由训练 engine 的 `get_per_tensor_param()` 临时 all-gather/export 出来。
6. 注册的是 `named_tensors`，来自训练引擎当前权重，不是提前保存好的 HF 文件。
7. `gather_metas()` 是为了让所有 rank 知道全局参数布局和 bucket 读取计划。
8. rollout ranks 通过 `ParameterServer.receive_tensor()` 把权重取回来。
9. verl 取回后仍然通过 `SGLangRollout.update_weights()` 调 SGLang 的 `update_weights_from_tensor`。
10. 在 verl 当前这条实现里，外部 checkpoint-engine 没有直接驱动 SGLang 的 `from_ipc` 接口；它只是 actor CE 到 rollout CE 的传输层。

## 11. 最短总结

`kimi_ckpt_engine` 在 verl 里的作用就是：

```text
actor training weights
  -> named_tensors
  -> actor ranks 分片 register 到外部 ParameterServer
  -> gather global metas
  -> rollout ranks receive_tensor 取回
  -> yield (name, tensor)
  -> SGLangRollout.update_weights
  -> SGLang update_weights_from_tensor
```

如果只关心 verl 如何用外部 checkpoint-engine，这条链路就是全部重点。
