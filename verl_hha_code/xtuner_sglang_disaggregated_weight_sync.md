# SGLang 非共卡场景权重同步逻辑

本文基于 `xtuner/v1/rl/weight_update/update_weighter.py` 和 `xtuner/v1/rl/weight_update/transport.py`，说明 Xtuner 在 train/rollout 非共卡、rollout backend 为 SGLang 时，训练侧如何把已经准备好的权重同步到 SGLang 推理引擎。

本文只关注同步时刻的接口和数据流。权重如何从 Xtuner 训练模型中 gather、转换成 HF key、切 bucket 等准备过程不在本文范围内。

## 适用场景

该逻辑只在以下条件同时满足时生效：

- `update_rollout_info(..., train_rollout_mode="disaggregated")` 已被调用。
- rollout backend 被识别为 `"sglang"`。
- `update_weights()` 被触发，`WeightIterator` 按 bucket 产出待同步的 `state_dict: dict[str, torch.Tensor]`。

在 `UpdateWeighter.set_train_rollout_mode()` 中，非共卡模式会选择 NCCL transport：

```python
elif mode == "disaggregated":
    self.rollout_info.transport_type = "nccl"
```

随后 `_set_transport()` 会创建 `NCCLWeightTransport`，并在 backend 为 SGLang 时使用 `SGLangNCCLBackendAdapter`。因此，SGLang 非共卡场景不会走共卡场景的 `update_weights_from_tensor` IPC/序列化路径，而是走 SGLang 的分布式权重更新接口：

- `/init_weights_update_group`
- `/update_weights_from_distributed`

## 总体时序

一次非共卡权重同步可以分成两层：

1. 首次同步前，Xtuner 与当前 metadata 中带 URL 的 SGLang engine 建立一个临时 Torch distributed process group。
2. 每个权重 bucket 到来时，Xtuner rank 0 通知 SGLang 准备接收，然后通过该 process group broadcast 扁平化后的权重 tensor。

简化时序如下：

```text
Xtuner all train ranks
    |
    | update_weights()
    | WeightIterator.iter_batch_groups()
    |
    +-- non-head train ranks:
    |       barrier(train_update_sync_group) 后返回
    |
    +-- head train rank, dist rank 0:
            |
            | ensure_nccl_weight_update_group()
            |   POST /init_weights_update_group 到每个 SGLang engine。 这是 sglang 接口
            |   本地创建 rank=0 的 NCCL process group
            |
            | POST /update_weights_from_distributed 到每个 SGLang engine。 这是 sglang 接口
            |   payload 只包含 names / dtypes / shapes / group_name / load_format
            |
            | FlattenedTensorBucket(state_dict)
            | dist.broadcast(flattened_tensor, src=0, group=nccl_weight_update_group) # sglang 内部已经建好组，可以接受
            |
            | 等待所有 HTTP response
            |
        barrier(train_update_sync_group)
```

## 初始化更新组

Xtuner 首次同步时会调用 `ensure_nccl_weight_update_group()`。如果 `group` 已存在，则复用已有 group，不重复初始化。

### Xtuner 侧准备

Xtuner 会收集当前 metadata 中带 URL 的 SGLang engine 信息：

- 来自 `rollout_engine_rank_mesh_array` 的 engine rank 分组。
- 来自 `rollout_server_url_dict` 的 server URL。
- 忽略空 URL，并按 URL 去重。
- 每个 engine 记录为 `(rollout_rank, server_url, engine_gpu_count)`。

随后 Xtuner rank 0：

- 选取 `master_address`，优先使用 `rollout_config.weight_update_host`，未配置时尝试使用 Ray node IP，再回退到 hostname 解析。
- 使用 `master_port`，来自 `rollout_config.weight_update_port`，默认值为 `30000`。
- 生成 `group_name = f"xtuner_NCCL_weight_update_{self.rank}"`。
- 计算 `world_size = sum(engine_gpu_count) + 1`。

这里 `+ 1` 是因为 Xtuner 训练侧 head rank 也作为分布式通信组中的 rank 0，SGLang 的各 TP rank 从 rank 1 开始。

### 调用 SGLang 初始化接口

Xtuner 会并发向每个 SGLang engine URL 发送：

```http
POST {sglang_server_url}/init_weights_update_group
Content-Type: application/json
```

请求体：

```json
{
  "master_address": "x.x.x.x",
  "master_port": 12345,
  "rank_offset": 1,
  "world_size": 9,
  "group_name": "xtuner_NCCL_weight_update_0",
  "backend": "nccl"
}
```

字段含义：

- `master_address` / `master_port`：Xtuner rank 0 创建 process group 使用的 rendezvous 地址。
- `rank_offset`：当前 SGLang engine 在全局更新组里的起始 rank。第一个 engine 通常是 1，后续 engine 累加各自 `engine_gpu_count`。
- `world_size`：Xtuner head rank 加所有 SGLang engine GPU rank 的总数。
- `group_name`：本次权重更新组名称，后续同步 bucket 时必须使用同一个值。
- `backend`：当前代码固定为 `nccl`。

SGLang 侧 `ModelRunner.init_weights_update_group()` 会用：

```text
rank = rank_offset + tp_rank
```

为该 engine 内的每个 TP rank 加入同一个 Torch process group，并保存到：

```python
self._model_update_group[group_name]
```

Xtuner 同时在训练侧创建同一个 group：

```python
_init_external_process_group(
    backend="nccl",
    init_method=f"tcp://{master_address}:{master_port}",
    world_size=world_size,
    rank=0,
    group_name=group_name,
)
```

当所有 SGLang 初始化请求返回成功后，Xtuner 在 `NCCLWeightTransport` 中缓存：

- `group`
- `group_name`
- `engine_urls`
- `executor`

后续 bucket 同步会复用这些对象。

## 同步单个权重 bucket

`NCCLWeightTransport.send(batch)` 是实际同步入口。这里假设 `batch.state_dict` 已经是当前 bucket 内需要发给 SGLang 的权重。

### 训练侧 rank 分工

该函数只让训练侧 `dist.get_rank() == 0` 真正发送权重：

```python
head_rank = 0
if dist.get_rank() != head_rank:
    dist.barrier(group=train_sync_group)
    return
```

其他训练 rank 只在 `train_update_sync_group` 上等待，保证所有训练 worker 在一次 bucket 同步完成后再继续。

### Xtuner 发起 SGLang 接收请求

Xtuner rank 0 会先把 tensor 移到当前设备并保证连续：

```python
tensors = [
    tensor.detach().to(device=DEVICE, non_blocking=True).contiguous()
    for tensor in state_dict.values()
]
```

然后构造 HTTP payload：

```json
{
  "names": [
    "model.layers.0.self_attn.q_proj.weight"
  ],
  "dtypes": [
    "bfloat16"
  ],
  "shapes": [
    [4096, 4096]
  ],
  "group_name": "xtuner_NCCL_weight_update_0",
  "load_format": "flattened_bucket"
}
```

Xtuner 并发向每个 SGLang engine 发送：

```http
POST {sglang_server_url}/update_weights_from_distributed
Content-Type: application/json
```

注意：这个 HTTP 请求不携带真实权重数据，只携带元信息。真实权重通过前面建立的 NCCL process group 传输。

### Xtuner broadcast 真实权重

Xtuner 使用 SGLang 的 `FlattenedTensorBucket` 将当前 bucket 内多个 tensor 打平成一个连续 `uint8` buffer：

```python
flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=list(zip(names, tensors)))
flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()
```

随后通过分布式组广播：

```python
dist.broadcast(flattened_tensor, src=0, group=self.group)
DEVICE_MODULE.synchronize()
```

这里 `src=0` 对应 Xtuner head rank。SGLang 各 TP rank 在自己的 `update_weights_from_distributed()` 调用中作为接收方。

### SGLang 接收并加载权重

SGLang HTTP 层收到 `/update_weights_from_distributed` 后，会将请求传到 tokenizer manager、scheduler、TP worker，最终进入 `ModelRunner.update_weights_from_distributed()`。

当 `load_format == "flattened_bucket"` 时，SGLang 走 bucket 化接收逻辑：

1. 根据 `names`、`dtypes`、`shapes` 在每个 SGLang rank 的设备上创建空 tensor。
2. 用这些空 tensor 构造 `FlattenedTensorBucket`。
3. 获取 bucket 的 `flattened_tensor`。
4. 从同一个 `group_name` 对应的 process group 中执行：

```python
torch.distributed.broadcast(
    flattened_tensor,
    src=0,
    group=self._model_update_group[group_name],
)
```

1. 调用 `bucket.reconstruct_tensors()`，按 metadata 将扁平 buffer 还原成 `(name, tensor)` 列表。
2. 调用：

```python
self.model.load_weights(reconstructed_tensors)
```

完成当前 bucket 的在线加载。

SGLang 接口成功时返回：

```json
{
  "success": true,
  "message": "Succeeded to update parameter online."
}
```

Xtuner 会等待所有 SGLang engine 的 HTTP response，检查 HTTP status 和 `success` 字段。如果任一 engine 失败，当前同步会抛错。

## 接口小结

### `POST /init_weights_update_group`

用途：让每个 SGLang engine 加入由 Xtuner 创建的权重更新 process group。

Xtuner 必传字段：

```json
{
  "master_address": "x.x.x.x",
  "master_port": 12345,
  "rank_offset": 1,
  "world_size": 9,
  "group_name": "xtuner_NCCL_weight_update_0",
  "backend": "nccl"
}
```

返回：

```json
{
  "success": true,
  "message": "Succeeded to initialize custom process group."
}
```

### `POST /update_weights_from_distributed`

用途：通知 SGLang 按给定 metadata 创建接收 buffer，并通过分布式 broadcast 接收真实权重。

Xtuner 当前传入字段：

```json
{
  "names": ["model.layers.0.self_attn.q_proj.weight"],
  "dtypes": ["bfloat16"],
  "shapes": [[4096, 4096]],
  "group_name": "xtuner_NCCL_weight_update_0",
  "load_format": "flattened_bucket"
}
```

SGLang schema 还支持但 Xtuner 当前未显式传入的字段：

```json
{
  "flush_cache": true,
  "abort_all_requests": false,
  "weight_version": null
}
```

返回：

```json
{
  "success": true,
  "message": "Succeeded to update parameter online."
}
```

## 附：SGLang 共卡场景接口

共卡场景下，`set_train_rollout_mode("colocate")` 会选择 IPC transport。backend 为 SGLang 时，Xtuner 使用 `SGLangIPCBackendAdapter`，不会创建外部 NCCL process group，也不会调用 `/init_weights_update_group` / `/update_weights_from_distributed`。

实际请求发给 SGLang 的接口是：

```http
POST {sglang_server_url}/update_weights_from_tensor
Content-Type: application/json
```

Xtuner 当前传入字段：

```json
{
  "serialized_named_tensors": [
    "<serialized tensor payload for tp rank 0>",
    "<serialized tensor payload for tp rank 1>"
  ],
  "flush_cache": false,
  "load_format": "flattened_bucket"
}
```

字段含义：

- `serialized_named_tensors`：每个 rollout TP rank 一份序列化后的权重数据。`rollout_tp == 1` 时也是长度为 1 的 list；`rollout_tp > 1` 时，Xtuner 会先在 `engine_parallel` group 内 gather，再由 head rank 发 HTTP 请求。
- `flush_cache`：Xtuner 当前固定传 `false`。
- `load_format`：如果当前 SGLang 提供 `FlattenedTensorBucket`，Xtuner 会传 `"flattened_bucket"`。

当 `load_format == "flattened_bucket"` 时，`serialized_named_tensors` 中每一项是 SGLang `MultiprocessingSerializer` 序列化后的字符串，反序列化后数据结构大致是：

```python
{
    "flattened_tensor": flattened_tensor,
    "metadata": flattened_tensor_bucket.get_metadata(),
}
```

其中 `flattened_tensor` 是由当前 bucket 的 `(name, tensor)` 打平得到的 tensor，`metadata` 用于在 SGLang 侧还原出原始的权重名、shape、dtype 等信息。

如果当前 SGLang 不支持 `FlattenedTensorBucket`，Xtuner 会退回到直接序列化 `state_dict.items()`，此时请求体不带 `load_format: "flattened_bucket"`。
