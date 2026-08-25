# checkpoint-engine broadcast 模式：128 卡全流程解读

本文只讲外部 checkpoint-engine 的 **broadcast 模式**，目标是把一份 **HF safetensors checkpoint** 从磁盘加载到多组 SGLang 推理实例中。

不讨论：

```text
verl 共卡 naive 同步
verl delta_sharded
checkpoint-engine P2P 恢复
训练侧在线导出 named_tensors
```

广播模式的核心逻辑是： 假设 128 卡每张卡都拿了一部分权重，然后每个卡都会根据显存大小组成 n 个各自的 bucket，经过全局 gather meta 后，所有 rank 都知道了全局信息，权重更新时候会依次从 bucket0 进行同步，一直到结束。假设 bucket0 位于 rank0，那么就是 rank0 作为 src，其余 rank 作为 dst，进行 dist.boardcast 通信，这样所有卡都拿到了 bucket0，然后通过 cuda ipc+zmq 进行同步；假设 bucket1 是在 rank1，则 rank1 作为 src，其余 rank 作为 dst，采用同样过程直到所有 bucket 都同步了一次。可以明显可以有数据冗余，核心原因是 ce 是拓扑无感的，他不会感知推理那边的并行方案，所以他是每个 rank 都通过 cuda ipc 发送一次，sglang worker 自己根据需要取就行，如果本 rank 不需要这个权重，可以跳过.

## 1. 先看 128 卡全局总览

假设：

```text
总 GPU 数：128
推理引擎：SGLang
权重来源：HF safetensors on disk
checkpoint-engine 模式：broadcast
```

以一个具体例子展开：

```text
16 个 SGLang 实例
每个实例 8 卡
每个实例内部是 EP8
```

那么整体布局可以理解成：

```text
128 个 checkpoint-engine rank
128 个 SGLang worker

instance 0:  CE/SGLang rank 0..7
instance 1:  CE/SGLang rank 8..15
instance 2:  CE/SGLang rank 16..23
...
instance 15: CE/SGLang rank 120..127
```

每张卡上都有：

```text
checkpoint-engine rank i
  +
SGLang worker i
```

其中：

```text
checkpoint-engine:
  负责读 HF 权重、注册 pinned memory、组成 bucket、跨 128 卡 broadcast。

SGLang:
  负责拿到 bucket 后调用 model.load_weights()。
  TP/EP/MoE 的切片、expert 归属、跳过不需要的权重，都在 SGLang loader 内部处理。
```

最重要的一句话：

```text
checkpoint-engine broadcast 是“所有 rank 都临时看到完整 HF 权重流”；
SGLang load_weights 再决定每个 rank 实际保留哪些 slice/expert。
```



## 2. 128 卡端到端流程

这一节先把全流程写完整。后面的章节只是把这里每个点拆开解释。

先固定一个例子：

```text
总 GPU 数：128
checkpoint-engine rank 数：128，每张卡 1 个 CE rank
SGLang 实例数：16
每个 SGLang 实例：8 卡
权重来源：HF disk
更新方法：broadcast
```

实例分组是：

```text
SGLang instance0:  worker0..7      对应 CE rank0..7
SGLang instance1:  worker8..15     对应 CE rank8..15
SGLang instance2:  worker16..23    对应 CE rank16..23
...
SGLang instance15: worker120..127  对应 CE rank120..127
```

这里默认 CE rank 和 SGLang worker 是共机器、共 GPU 对应的：

```text
CE rank7 负责和 SGLang worker7 通过 CUDA IPC 交接 GPU buffer
CE rank37 负责和 SGLang worker37 通过 CUDA IPC 交接 GPU buffer
```

注意，这里的“共 GPU 对应”不是说 SGLang 直接从训练 rank 拿 tensor。checkpoint-engine
broadcast 这条链路里，SGLang worker 看到的是本机 CE rank 的 GPU bucket buffer。

### 2.1 启动推理实例

先启动 16 个 SGLang 实例。

每个实例 8 张卡，实例内部有自己的 scheduler / model worker。为了后续能在线灌入权重，
实例通常可以先用 dummy load 或 empty/dummy 权重启动，让模型结构、通信组、worker 进程都
先存在。

此时 SGLang 侧已经具备：

```text
HTTP server
实例内 worker 列表
每个 worker 绑定的 GPU
模型结构和 load_weights() 能力
```

但真实权重还没有通过 checkpoint-engine 灌进去。

### 2.2 启动 128 个 checkpoint-engine rank

checkpoint-engine 通常用 `torchrun` 启动 128 个进程：

```text
CE rank0
CE rank1
...
CE rank127
```

每个 CE rank 有两类职责：

```text
数据面:
  参与 torch.distributed.broadcast 发送实际权重字节
  持有本 rank 注册到 CPU pinned memory 的权重切片
  分配本 rank 的 GPU bucket buffer

控制面:
  绑定自己的 ZMQ socket 
  和本机 SGLang worker 交换 IPC handle、bucket metadata、ack/error
```



### 2.3 128 个 CE rank 分摊读取 HF 权重

HF checkpoint 通常是一组 safetensors 文件和 index：

```text
model.safetensors.index.json
model-00001-of-xxxxx.safetensors
model-00002-of-xxxxx.safetensors
...
```

checkpoint-engine / SGLang update 脚本会把 tensor name 或文件切分给 128 个 CE rank。

例如：

```text
CE rank0   读取 layer0 的一部分 tensor
CE rank1   读取 layer0/layer1 的一部分 tensor
CE rank2   读取 embedding 的一部分 tensor
...
CE rank127 读取最后几个 tensor
```

关键点是：

```text
128 个 CE rank 合起来读一份完整 HF 模型。
不是每个 CE rank 都读一份完整 HF 模型。
```

所以此时全局来看模型是完整的，但单个 CE rank 只持有一部分权重。

### 2.4 每个 CE rank 调 register_checkpoint()

每个 CE rank 对自己负责的权重调用：

```python
ps.register_checkpoint(...)
```

如果输入是 file 类型，checkpoint-engine 会从文件中读取该 rank 负责的 tensor。

如果输入是 named_tensors，则该 rank 传入的就是自己负责的那批 `(name, tensor)`。

注册完成后，每个 rank 的状态大概是：

```text
CE rank0:
  CPU pinned memory: [tensor A, tensor B, tensor C]

CE rank1:
  CPU pinned memory: [tensor D, tensor E]

...

CE rank127:
  CPU pinned memory: [tensor X, tensor Y]
```

这一步只是把“本 rank 拥有的权重”变成 checkpoint-engine 后续可以高效 H2D 的 pinned
memory。它还没有把权重发给 SGLang。

### 2.5 全局 gather_metas，生成所有 rank 都一致的视图

注册后，每个 CE rank 只知道自己有什么 tensor。broadcast 要让所有 128 个 rank 按完全相同
的 bucket 顺序参与 collective，所以必须先汇总 metadata。

`gather_metas()` 做的是全局 all-gather：

```text
CE rank0   上报自己有哪些 tensor、shape、dtype、size、offset
CE rank1   上报自己有哪些 tensor、shape、dtype、size、offset
...
CE rank127 上报自己有哪些 tensor、shape、dtype、size、offset
```

all-gather 后，每个 CE rank 都拿到同一份 `global_metas`：

```text
tensor A 在 CE rank0
tensor B 在 CE rank0
tensor D 在 CE rank1
...
tensor X 在 CE rank127
```

然后每个 rank 本地用这份相同的 `global_metas` 生成相同的 bucket schedule：

```text
bucket0: src=rank0,  items=[A, B]
bucket1: src=rank0,  items=[C]
bucket2: src=rank1,  items=[D, E]
bucket3: src=rank2,  items=[F]
...
bucketN: src=rank127, items=[X, Y]
```

这里有两个关键点：

```text
bucket 只由一个 owner rank 的 tensor 组成，不会跨多个 owner rank 拼一个 bucket。
所有 128 个 CE rank 看到的 bucket 顺序必须完全一致。
```

如果 rank0 认为当前该广播 bucket0，而 rank1 认为当前该广播 bucket2，collective 就会错乱。

### 2.6 每个 CE rank 绑定 ZMQ socket

每个 CE rank 会绑定一个自己的 ZMQ endpoint，例如：

```text
CE rank0: ipc://@checkpoint-engine-rank0.sock
CE rank1: ipc://@checkpoint-engine-rank1.sock
...
CE rank7: ipc://@checkpoint-engine-rank7.sock
...
CE rank127: ipc://@checkpoint-engine-rank127.sock
```

这个 socket 是 CE rank 和本机 SGLang worker 的控制通道，后续要利用 zmq 来协调

它后续会传：

```text
CUDA IPC handle  # 只需要发送一次从而建立 ce rank 和绑定的 sg worker cuda ipc 通道

每个 bucket 的 metadata  # 每个 bucket 都需要发送一次，利用这个信息从 cuda ipc 中恢复真实 tensor shape

ack / error

结束信号
```

它不会传权重 bytes。常规的 ipc 权重同步时候不需要这么麻烦，因为只需要通过 from_tensor 接口发送 tensor cuda ipc 过去，sglang worker 接收后即可，不需要结束信号，或者说发一个特定 flag 过去也可以，不需要专门的 zmq。 checkpointengine 需要是因为实际上他是包括了非共卡权重同步那套 from_dist 和共卡那套 from_tensor 的两部分功能。

```text
每个 bucket 的 metadata 对应到非共卡权重同步就是每次权重同步一个 bucket 前要先调用一次 from_dist 接口来发送 metadata。在 ce 中使用的是 zmq 来实现这个逻辑
```

1. CE rank 先在自己进程里 bind ZMQ endpoint
  例如 ipc://@checkpoint-engine-rank7.sock
  1. CE rank 把这些 endpoint 字符串收集成 zmq_handles
  2. 实例 leader rank 调 SGLang 的 HTTP 接口：
    POST /update_weights_from_ipc   # 调用一次就行。只需要给实例的 rank0 发就行，内部会带整个实例所有 zmq handle
  3. POST body 里带 zmq_handles
  4. SGLang HTTP server 收到后，把请求 fanout 给实例内 worker
  5. 每个 worker 根据自己的 GPU UUID 从 zmq_handles 里选到自己的 endpoint
  6. worker connect 这个 endpoint
  7. 后续 CE rank 和 SGLang worker 就通过这个 ZMQ socket 通信



### 2.7 每个实例只发一次 POST，触发 SGLang worker 连接 ZMQ

checkpoint-engine 根据 `--inference-parallel-size=8` 知道每 8 个 CE rank 对应一个
SGLang 实例。

于是每组只让起始 rank 代表整个实例发一次 HTTP POST：

```text
CE rank0   -> POST 给 SGLang instance0
CE rank8   -> POST 给 SGLang instance1
CE rank16  -> POST 给 SGLang instance2
...
CE rank120 -> POST 给 SGLang instance15
```

以 instance0 为例，POST body 携带的是这个实例内 8 个 worker 的 ZMQ 地址：

```text
{
  "zmq_handles": {
    "GPU-uuid-worker0": "ipc://@checkpoint-engine-rank0.sock",
    "GPU-uuid-worker1": "ipc://@checkpoint-engine-rank1.sock",
    ...
    "GPU-uuid-worker7": "ipc://@checkpoint-engine-rank7.sock"
  }
}
```

SGLang HTTP server 收到后，在实例内 fanout 给 8 个 worker：

```text
worker0 根据自己的 GPU UUID 找到 rank0 socket，然后 connect rank0
worker1 根据自己的 GPU UUID 找到 rank1 socket，然后 connect rank1
...
worker7 根据自己的 GPU UUID 找到 rank7 socket，然后 connect rank7
```

所以 HTTP POST 是实例级触发；ZMQ/CUDA IPC 是 worker 级通信。

### 2.8 CE rank 发送 CUDA IPC handle

每个 CE rank 都会分配自己的 GPU double buffer：

```text
CE rank i:
  buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device="cuda")
```

然后 CE rank i 通过自己的 ZMQ socket，把这个 buffer 的 CUDA IPC handle 发给本机  
SGLang worker i。注意建立好 zqm 通信后，后续就都是走 zmq 了，例如把 ce rank 侧的 cuda ipc handle 发送过去是通过 zmq 发送的。

这一步只发一次，因为 handle 是 buffer 级别的。后面每个 bucket 都复用这块 buffer。

此时每个 SGLang worker 都已经能访问“本机 CE rank 的 GPU buffer”，但 buffer 里当前还没有
本轮 bucket 的有效权重内容。

### 2.9 开始逐 bucket broadcast

从这里开始进入真正的权重数据流。

假设当前 schedule 走到：

```text
bucket_k: src=rank37, items=[layer10.mlp.up_proj.weight, layer10.mlp.down_proj.weight]
```

那么过程是：

```text
1. CE rank37 从自己的 CPU pinned memory 找到这两个 tensor 的 bytes
2. CE rank37 把它们拷到自己的 GPU bucket buffer
3. 128 个 CE rank 一起调用 torch.distributed.broadcast(buffer_b, src=rank37)
4. broadcast 完成后，所有 CE rank 的 GPU bucket buffer 都有同一份 bucket bytes
```

这里真正跨 rank / 跨卡移动的是 `torch.distributed.broadcast`。

CUDA IPC 没有负责把 rank37 的 tensor 发到 rank0..127。CUDA IPC 只负责让每个 SGLang worker
访问自己旁边 CE rank 的本地 GPU buffer。

所以 broadcast 完成后是：

```text
CE rank0   的 GPU buffer 里有 bucket_k bytes
CE rank1   的 GPU buffer 里有 bucket_k bytes
...
CE rank37  的 GPU buffer 里有 bucket_k bytes
...
CE rank127 的 GPU buffer 里有 bucket_k bytes
```



### 2.10 CE 通过 ZMQ 发送当前 bucket metadata

权重 bytes 到达所有 CE rank 的 GPU buffer 后，每个 CE rank 通过自己的 ZMQ socket 通知本机
SGLang worker：

```text
当前 buffer 里有这些 tensor:

layer10.mlp.up_proj.weight:
  dtype=bf16
  shape=[...]
  offset=0
  size=...

layer10.mlp.down_proj.weight:
  dtype=bf16
  shape=[...]
  offset=...
  size=...
```

这个 metadata 的作用是让 worker 知道怎么解释 buffer 里的 bytes。

### 2.11 SGLang worker 从 CUDA IPC buffer 切 tensor view

SGLang worker 之前已经 rebuild 了 CE rank 的 CUDA IPC buffer。

收到 metadata 后，它不会再从 ZMQ 收权重数据，而是直接在 IPC buffer 上切 view：

```text
raw = ipc_buffer[offset : offset + size]
tensor = raw.view(dtype).view(shape)
```

这样得到的是一批 `(name, tensor)`。

这些 tensor 是当前 bucket 的 HF 权重视图，底层数据来自本机 CE rank 的 GPU buffer。

### 2.12 SGLang model.load_weights() 决定真正保留什么

每个 SGLang worker 都会临时看到当前 bucket 的 HF tensor。

但是不同 worker 最终加载的内容不一样：

```text
TP 场景:
  每个 TP rank 从 HF tensor 中切自己负责的 shard

EP / MoE 场景:
  只属于本 rank 的 expert 会被加载
  不属于本 rank 的 expert 会被跳过

普通 replicated 权重:
  多个 rank 可能都需要加载同一份
```

因此，broadcast 层不判断某个 rank 是否需要某个 expert。它把完整 HF 权重流发给所有 rank，
然后由 SGLang 的 `model.load_weights()` 负责切片、跳过或加载。

### 2.13 worker ack，CE 继续下一个 bucket

当前 bucket 处理完后，SGLang worker 通过 ZMQ 给 CE rank 回 ack。

如果加载失败，worker 会通过 ZMQ 回 error，CE 侧可以停止本次 update 并报错。

正常情况下，checkpoint-engine 继续下一个 bucket：

```text
bucket0: src=rank0  -> broadcast -> metadata -> load_weights -> ack
bucket1: src=rank0  -> broadcast -> metadata -> load_weights -> ack
bucket2: src=rank1  -> broadcast -> metadata -> load_weights -> ack
bucket3: src=rank37 -> broadcast -> metadata -> load_weights -> ack
...
bucketN: src=rank127 -> broadcast -> metadata -> load_weights -> ack
```

一个 rank 发完自己负责的 bucket 后，也不会退出。它后面仍然要作为 receiver 参与其他 rank
的 broadcast。

### 2.14 所有 bucket 完成后收尾

所有 bucket 都完成后，CE 通过 ZMQ 发送结束信号。

流程可以理解成：

```text
CE 发 None:
  worker 知道 bucket 流结束，释放 / 关闭 IPC buffer 相关状态

CE 再发 None:
  worker 执行 post_hook
```

SGLang 的 post hook 通常会做 cache flush / 状态刷新，让后续请求使用新权重。

最终：

```text
16 个 SGLang 实例都完成 load_weights()
每个实例内 8 个 worker 都完成 post_hook
128 张卡上的 rollout worker 都 ready
```

一张总图：

```text
阶段 A：SGLang 先启动，等待外部灌权重

  SGLang instance0:  worker0..7
  SGLang instance1:  worker8..15
  ...
  SGLang instance15: worker120..127

  每个 worker 已经有模型结构 / load_weights() / GPU 绑定关系。
  但还没有从 checkpoint-engine 收到真实权重。


阶段 B：checkpoint-engine 注册权重，并准备全局 bucket 计划

  HF safetensors on disk
      |
      v
  128 个 CE rank 分摊读取一份完整 HF checkpoint
      |
      v
  每个 CE rank 调 register_checkpoint()
      |
      v
  每个 CE rank 只持有自己负责的 CPU pinned memory 权重切片
      |
      v
  128 个 CE rank 执行 gather_metas()
      |
      v
  所有 CE rank 都拿到同一份 global_metas
      |
      v
  所有 CE rank 本地生成完全一致的 bucket schedule

  例如：
    bucket0: src=rank0,   items=[A, B]
    bucket1: src=rank1,   items=[C]
    bucket2: src=rank37,  items=[D, E]
    ...
    bucketN: src=rank127, items=[X, Y]


阶段 C：CE 侧创建 ZMQ endpoint，但还没有传权重

  CE rank0   bind ipc://@checkpoint-engine-rank0.sock
  CE rank1   bind ipc://@checkpoint-engine-rank1.sock
  ...
  CE rank7   bind ipc://@checkpoint-engine-rank7.sock
  ...
  CE rank127 bind ipc://@checkpoint-engine-rank127.sock

  这些 endpoint 是后续 CE -> sglang worker 的控制通道。
  它们会传 IPC handle、bucket metadata、None、ack/error。
  它们不传大权重 bytes。


阶段 D：每个推理实例只发一次 HTTP POST，告诉 SGLang 去连哪些 ZMQ

  按 --inference-parallel-size=8 分组：

  CE rank0   -> POST /update_weights_from_ipc -> SGLang instance0
  CE rank8   -> POST /update_weights_from_ipc -> SGLang instance1
  CE rank16  -> POST /update_weights_from_ipc -> SGLang instance2
  ...
  CE rank120 -> POST /update_weights_from_ipc -> SGLang instance15

  以 instance0 为例，POST body 是：

    zmq_handles = {
      GPU-uuid-worker0: ipc://@checkpoint-engine-rank0.sock,
      GPU-uuid-worker1: ipc://@checkpoint-engine-rank1.sock,
      ...
      GPU-uuid-worker7: ipc://@checkpoint-engine-rank7.sock
    }

  HTTP 只负责启动一次实例级 update。
  HTTP 不传权重，也不传 bucket bytes。至此两边就建立了专属的 zmq 通信通道，后续都要利用这个通道


阶段 E：SGLang 内部 fanout，worker 连接自己的 ZMQ endpoint

  SGLang instance0 收到 POST 后：

    worker0 读取自己的 GPU UUID
      -> 找到 ipc://@checkpoint-engine-rank0.sock
      -> connect CE rank0

    worker1 读取自己的 GPU UUID
      -> 找到 ipc://@checkpoint-engine-rank1.sock
      -> connect CE rank1

    ...

    worker7 读取自己的 GPU UUID
      -> 找到 ipc://@checkpoint-engine-rank7.sock
      -> connect CE rank7

  其他 15 个实例同理。


阶段 F：CE 侧分配 GPU double buffer，并通过 ZMQ 发 CUDA IPC handle

  每个 CE rank i：

    buffer_i = torch.empty(bucket_size * 2, dtype=torch.uint8, device="cuda")
    ipc_handle_i = reduce_tensor(buffer_i)

    CE rank i --ZMQ--> SGLang worker i:
      send(ipc_handle_i)

  SGLang worker i 收到 handle 后：

    ipc_buffer_i = rebuild_ipc_tensor(ipc_handle_i)

  到这里，worker 已经能访问本机 CE rank 的 GPU buffer。
  但权重内容还没开始逐 bucket 写进去。这一步相当于 cuda ipc 通道建立好了


阶段 G：进入 bucket 循环，broadcast 负责跨 rank 分发权重 bytes

  for bucket_k in bucket_schedule:

    假设 bucket_k 的 src 是 rank37：

      CE rank37:
        从自己的 CPU pinned memory 取 bucket_k 的权重 bytes
        H2D copy 到自己的 GPU buffer_37

      所有 128 个 CE rank:
        dist.broadcast(buffer_b, src=rank37)

      broadcast 结束后：
        CE rank0   的 GPU buffer_b 里有 bucket_k bytes
        CE rank1   的 GPU buffer_b 里有 bucket_k bytes
        ...
        CE rank37  的 GPU buffer_b 里有 bucket_k bytes
        ...
        CE rank127 的 GPU buffer_b 里有 bucket_k bytes

  这里真正搬运大权重数据的是 dist.broadcast。
  CUDA IPC 不负责把 rank37 的 tensor 分发到 rank0..127。


阶段 H：每个 CE rank 通过 ZMQ 发当前 bucket metadata

  broadcast 完成后，每个 CE rank i 都通过自己的 ZMQ socket 发 metadata：

    CE rank i --ZMQ--> SGLang worker i:
      send([
        {
          name: "layer10.mlp.up_proj.weight",
          dtype: bf16,
          shape: [...],
          offset: 0,
          size: ...
        },
        {
          name: "layer10.mlp.down_proj.weight",
          dtype: bf16,
          shape: [...],
          offset: ...,
          size: ...
        }
      ])

  metadata 告诉 worker：
    当前 IPC buffer 里的哪些 byte range 对应哪些 tensor。


阶段 I：worker 从 CUDA IPC buffer 切 tensor view，并 load_weights()

  SGLang worker i：

    for item in bucket_metadata:
      raw = ipc_buffer_i[item.offset : item.offset + item.size]
      tensor = raw.view(item.dtype).view(item.shape)
      named_tensor = (item.name, tensor)

    model.load_weights(named_tensors)

  TP / EP / MoE 的切片和跳过逻辑发生在 model.load_weights() 里：

    TP rank 只保留自己负责的 shard
    EP rank 只保留自己负责的 expert
    不需要的 tensor 临时看见后就跳过


阶段 J：worker 回 ack，CE 继续下一个 bucket

  SGLang worker i --ZMQ--> CE rank i:
    send(ack)

  如果失败：

  SGLang worker i --ZMQ--> CE rank i:
    send(error)

  正常情况下，所有 rank 进入下一个 bucket：

    bucket0 -> broadcast -> metadata -> load_weights -> ack
    bucket1 -> broadcast -> metadata -> load_weights -> ack
    bucket2 -> broadcast -> metadata -> load_weights -> ack
    ...
    bucketN -> broadcast -> metadata -> load_weights -> ack


阶段 K：全部 bucket 完成，CE 通过 ZMQ 发结束信号

  每个 CE rank i --ZMQ--> SGLang worker i:
    send(None)

  worker:
    释放 / 关闭 IPC buffer 相关状态

  每个 CE rank i --ZMQ--> SGLang worker i:
    send(None)

  worker:
    执行 post_hook
    flush cache / 刷新状态


阶段 L：所有实例 ready

  16 个 SGLang 实例都完成权重加载。
  128 个 SGLang worker 都完成 post_hook。
  后续 rollout 请求使用新权重。
```

为啥需要 dist.boardcast 这一步？有两个原因：

1. 每个卡上只有部分权重，如果要从 rank0 权重发送给 rank9, 那么 cuda ipc 是不够的
2. 而且因为 ce 是独立进程，他需要先在 ce 进程内进行 dist.boardcast 后才能某个 ce rank 得到想要的权重，同时因为在 ce rank0 和 sglang worker0 建立了 cuda ipc 链路，此时自然 sglang worker 侧就可以看到 dist.boardcast 后的数据了。



## 3. `--inference-parallel-size` 到底是什么

命令示例：

```bash
python -m sglang.srt.checkpoint_engine.update \
  --update-method broadcast \
  --checkpoint-path $MODEL_PATH \
  --inference-parallel-size 8
```

`--inference-parallel-size` 表示：

```text
一个 SGLang 实例由多少个 GPU worker 组成
```

它不是 EP size，也不是 TP size。它只是实例内 worker 总数。

例子：

```text
每实例 EP8:
  --inference-parallel-size 8

每实例 TP16:
  --inference-parallel-size 16

每实例 TP2 * EP8:
  --inference-parallel-size 16
```

checkpoint-engine 只用这个参数做分组：

```python
rank = int(os.getenv("RANK"))
src = rank // inference_parallel_size * inference_parallel_size

if rank == src:
    POST /update_weights_from_ipc
    zmq_handles = socket_paths[src : src + inference_parallel_size]
```

如果 `inference_parallel_size=8`，128 卡会分成 16 组：

```text
rank 0..7      -> rank0   发 POST
rank 8..15     -> rank8   发 POST
rank 16..23    -> rank16  发 POST
...
rank 120..127  -> rank120 发 POST
```

如果 `inference_parallel_size=16`，128 卡会分成 8 组：

```text
rank 0..15     -> rank0   发 POST
rank 16..31    -> rank16  发 POST
...
rank 112..127  -> rank112 发 POST
```

每组只发一次 POST。SGLang HTTP server 收到后，再在 SGLang 内部分发给该实例的所有 worker。

## 4. 为什么只需要实例 rank0 发一次 POST

这里的 `rank0` 指的是“每个推理实例内部的入口 rank”，不是全局唯一的 rank0。

以 128 卡、每实例 8 卡为例：

```text
实例0: rank0..7      -> rank0 发一次 POST
实例1: rank8..15     -> rank8 发一次 POST
实例2: rank16..23    -> rank16 发一次 POST
...
实例15: rank120..127 -> rank120 发一次 POST
```

原因是：

```text
POST /update_weights_from_ipc 是实例级控制请求，不是每个 worker 的权重传输请求。
```

它发给的是该 SGLang 实例的 HTTP server。HTTP server 收到请求后，会在 SGLang 内部把这次
更新请求 fanout 给本实例的所有 scheduler / model worker。

POST body 里真正关键的信息是这个实例内所有 worker 对应的 ZMQ socket：

```text
{
  "zmq_handles": {
    "GPU-uuid-of-worker0": "ipc://@checkpoint-engine-rank0.sock",
    "GPU-uuid-of-worker1": "ipc://@checkpoint-engine-rank1.sock",
    ...
    "GPU-uuid-of-worker7": "ipc://@checkpoint-engine-rank7.sock"
  }
}
```

所以一次 POST 已经包含了这个实例内 8 张卡需要连接的所有 checkpoint-engine socket。

这里的：

```text
ipc://@checkpoint-engine-rank7.sock
```

可以理解成后续 ZMQ 通信使用的 socket 地址。更准确地说，它是
checkpoint-engine rank7 绑定出来的 ZMQ endpoint：

```text
CE rank7 bind ipc://@checkpoint-engine-rank7.sock
对应的 SGLang worker connect ipc://@checkpoint-engine-rank7.sock
```

后续不是所有 rank 都往这个地址发送消息，而是 CE rank7 通过这个 socket 和它本机对应的
推理 worker 通信。

ZMQ 里发送的也不是权重 bytes，而是：

```text
CUDA IPC handle
每个 bucket 的 metadata: name / shape / dtype / offset
结束信号 None
worker 回传 ack / error
```

权重本体仍然在 CE rank7 的 GPU buffer 里。推理 worker 是通过 CUDA IPC handle 访问这个
buffer，再根据每个 bucket 的 metadata 从对应 offset 切出 tensor。

所以，即使当前 bucket 的 owner 是 rank0，rank7 也会先通过 `torch.distributed.broadcast`
收到这个 bucket 到自己的 GPU buffer。然后 CE rank7 再通过自己的 ZMQ socket 通知本机 worker：

```text
当前 GPU buffer 里，哪些 offset 对应哪些 tensor
```

后续发生的是：

```text
1. 实例入口 rank 发 POST 给 SGLang HTTP server
2. SGLang HTTP server 把请求分发给实例内所有 worker
3. 每个 worker 读取自己的 GPU UUID
4. 每个 worker 从 zmq_handles 中找到自己的 socket path
5. 每个 worker 独立连接本机对应的 checkpoint-engine rank
6. 每个 worker 独立通过 ZMQ + CUDA IPC 接收 bucket
```

因此，POST 只需要发一次，但真正接收权重的 worker 仍然是实例内所有卡。

换句话说：

```text
HTTP POST:
  一次，实例级，用来通知整个实例开始更新，并告诉所有 worker 去哪里连 CE

ZMQ / CUDA IPC:
  多路，worker 级，每张卡各自连接自己的 CE rank，接收自己的 IPC handle 和 bucket metadata
```

如果每张卡都各自发一次 POST，语义会变成 8 次实例级更新请求，反而可能导致重复 fanout、
重复启动更新，甚至破坏一次 checkpoint-engine update 的同步关系。

这也是 `--inference-parallel-size` 必须让 checkpoint-engine 感知的原因：它要知道哪些
rank 属于同一个推理实例，然后只让每组的起始 rank 代表这个实例发一次 POST。

## 5. 为什么 128 个 rank 都要 register_checkpoint

128 个 checkpoint-engine rank 都会调用：

```python
ps.register_checkpoint(...)
```

但每个 rank 注册的是自己负责的那部分权重。

如果 HF 目录里有：

```text
model.safetensors.index.json
```

SGLang update 脚本通常会按 `weight_map` 的 tensor name 切分：

```text
rank0   读取 part0
rank1   读取 part1
...
rank127 读取 part127
```

然后：

```text
rank0   register part0
rank1   register part1
...
rank127 register part127
```

这样整体是：

```text
128 个 rank 分摊注册 1 份完整 HF 模型
```

不是：

```text
128 个 rank 各注册 1 份完整 HF 模型
```

`register_checkpoint()` 做的事情包括：

```text
读取当前 rank 负责的 tensor
构造 ParameterMeta
分配 / 使用 CPU pinned memory
把 tensor bytes 拷贝到 pinned memory
```

所以如果实测耗时主要在 register，这通常来自：

```text
磁盘读取
safetensors 解析
CPU pinned memory 分配
tensor bytes copy 到 pinned buffer
```



## 6. 为什么需要 gather_metas

每个 rank 本地只知道自己注册了什么：

```text
我有哪几个 tensor
它们的 name / shape / dtype
它们在我的 pinned buffer 里的 offset
```

但 broadcast 是 collective。所有 128 个 rank 必须以相同顺序进入同一个 broadcast。

如果没有全局 metadata，就会出现：

```text
rank0 认为下一次 broadcast src=rank0
rank1 认为下一次 broadcast src=rank1
rank2 认为下一次 broadcast src=rank37
```

这样会直接挂死或数据错位。

所以 `gather_metas()` 会做：

```text
rank0 local meta
rank1 local meta
...
rank127 local meta
        |
        v
所有 rank 都得到 global_metas[0..127]
```

然后所有 rank 用同一份 `global_metas` 生成同一份 bucket schedule。

例子：

```text
rank0 注册的权重组成 2 个 bucket
rank1 注册的权重组成 3 个 bucket
```

`gather_metas()` 后，每张卡都知道这 5 个 bucket：

```text
bucket0: src=rank0
bucket1: src=rank0
bucket2: src=rank1
bucket3: src=rank1
bucket4: src=rank1
```

执行时：

```text
bucket0:
  rank0 作为 src
  所有 rank 接收

bucket1:
  rank0 作为 src
  所有 rank 接收

bucket2:
  rank1 作为 src
  所有 rank 接收
  rank0 此时也变成接收方
```

一个 rank 发完自己的 bucket 后不能退出。它还要继续参与后续所有 broadcast。

## 7. bucket 是什么

bucket 本质是：

```text
一段 flattened bytes
+
这段 bytes 里有哪些 tensor 的 metadata
```

一个 bucket 包含两类信息：

```text
items:
  当前 bucket 里有哪些 tensor
  每个 tensor 的 name / shape / dtype / aligned_size

ranges:
  从 owner rank 的哪个 CPU pinned buffer 的哪个 offset 拷贝多少 bytes
```

bucket 是单个 owner rank 自己组成的。

不会出现：

```text
rank0 和 rank1 一起拼一个 bucket
```

而是：

```text
rank0 的 bucket 只来自 rank0 自己的 CPU pinned memory
rank1 的 bucket 只来自 rank1 自己的 CPU pinned memory
```

假设 rank37 注册了这些 tensor：

```text
expert.8.gate_proj.weight
expert.8.up_proj.weight
expert.9.down_proj.weight
```

那么 rank37 的某个 bucket 可能就是：

```text
bucket_k:
  owner/src = rank37
  items = [
    expert.8.gate_proj.weight,
    expert.8.up_proj.weight,
    expert.9.down_proj.weight,
  ]
```

update 时：

```text
rank37 CPU pinned memory
  -> rank37 GPU bucket
  -> broadcast 给所有 128 个 CE rank
```



## 8. bucket size 怎么决定

外部 checkpoint-engine broadcast 路径里，bucket size 通常自动检测。

它会参考：

```text
所有 rank 当前可用 GPU 显存的最小值
全局最大 tensor size
是否能额外分配 h2d_buffer
PS_MEM_FRACTION
PS_MAX_BUCKET_SIZE_GB
```

常见默认：

```text
PS_MEM_FRACTION=0.9
PS_MAX_BUCKET_SIZE_GB=8
```

bucket size 至少要能放下最大单个 tensor。外部 checkpoint-engine 这条 broadcast 路径通常不把一个 tensor 切成多个 bucket，而是保证 bucket 能容纳最大 tensor。

bucket 的打包方式可以理解成贪心：

```text
当前 bucket 剩余空间够放下下一个 tensor:
  放进去

不够:
  当前 bucket 结束
  新开一个 bucket
```



## 9. broadcast 广播的到底是什么

broadcast 广播的是：

```text
真正的权重 bucket bytes
```

不是 metadata。

以 rank37 的 bucket 为例：

```text
rank37 CPU pinned memory
  -> rank37 GPU bucket buffer
  -> dist.broadcast(buffer_b, src=rank37)
  -> 所有 128 个 CE rank 的 GPU buffer_b 都收到同一份 bytes
```

metadata 走 ZMQ，不走 distributed broadcast。

## 10. ZMQ、CUDA IPC、broadcast 三者怎么配合

三条通道职责不同：

```text
distributed broadcast:
  跨卡传大权重 bytes

ZMQ ipc://:
  本机控制面
  传 CUDA IPC handle、bucket metadata、ack/error、完成信号

CUDA IPC:
  让 SGLang worker 跨进程访问本卡 checkpoint-engine 的 GPU bucket buffer
```

更具体的时序：

```text
开始:
  CE rank i 分配 GPU double buffer
  CE rank i 通过 ZMQ 把 CUDA IPC handle 发给 SGLang worker i
  SGLang worker i rebuild 出这块 GPU buffer

for each bucket:
  owner CE rank 把 bucket bytes 放到 GPU buffer
  distributed broadcast 把 bucket bytes 发到所有 CE rank
  CE rank i 通过 ZMQ 发当前 bucket metadata 给 SGLang worker i
  SGLang worker i 用 metadata 从 CUDA IPC buffer 切 tensor view
  SGLang worker i 调 model.load_weights()
  SGLang worker i 通过 ZMQ ack/error

结束:
  CE 通过 ZMQ 发 None
  SGLang 释放 IPC buffer
  CE 再通过 ZMQ 发 None
  SGLang 执行 post_hook
```

所以：

```text
大权重数据:
  不走 ZMQ，不走 HTTP。

ZMQ:
  会发送很多次，但每次都是小控制消息。
```

如果有 `N` 个 bucket，ZMQ 消息数量是 `O(N)`。

### 10.1 CUDA IPC handle 不是“权重自己的 handle”

这里最容易误解的是 CUDA IPC handle 的含义。

它不是：

```text
每个权重 tensor 一个 CUDA IPC handle
```

而是：

```text
每个 CE rank 本地 GPU bucket buffer 的 CUDA IPC handle
```

也就是说，CE rank 先分配一块可复用的 GPU buffer：

```text
buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device="cuda")
```

然后把这块 buffer 的 CUDA IPC handle 通过 ZMQ 发给本机 SGLang worker。这个 handle 的作用是：

```text
让 SGLang worker 可以跨进程访问 CE rank 的这块 GPU buffer
```

权重 bytes 本身不是通过 CUDA IPC handle “发送”过去的。权重 bytes 是每个 bucket 执行时通过
`torch.distributed.broadcast` 写进这块 buffer 的。

完整关系是：

```text
CUDA IPC handle:
  告诉 worker “你可以访问 CE rank 的这块 GPU buffer”

distributed broadcast:
  把 owner rank 的 bucket bytes 写到所有 CE rank 的本地 GPU buffer

bucket metadata:
  告诉 worker “当前 buffer 的哪些 offset 对应哪些 tensor”
```

所以可以把一次 bucket 更新理解成：

```text
1. worker 早就通过 CUDA IPC handle 看到了 CE rank 的 GPU buffer
2. 当前 bucket 的 owner rank 把权重 bytes broadcast 到所有 CE rank 的这个 buffer
3. CE rank 通过 ZMQ 发 metadata
4. worker 根据 metadata 从 buffer[offset:...] 切出 tensor view
5. worker 调 model.load_weights()
```

因此，CUDA IPC handle 是“共享窗口”的 handle；broadcast 是每次往这个窗口里写新的权重内容；
metadata 是告诉 worker 当前窗口内容怎么解释。

### 10.2 为什么不直接用 IPC tensor 同步，反而还要 broadcast

如果是共卡训练 + 推理，直接 IPC tensor 同步通常更合理。

这种场景下，训练 rank 本来就持有 GPU tensor，推理 worker 也在同一张卡或同一组卡上。最短路径是：

```text
trainer GPU tensor -> inference worker
```

常见的 `update_weights_from_tensor` / CUDA IPC tensor update 就是这个思路。它绕过了
checkpoint-engine 的注册、bucket 计划、全局 broadcast 和额外 ZMQ bucket 流程。

checkpoint-engine broadcast 解决的是另一个问题：

```text
权重来自 HF disk 或 checkpoint cache
128 个 CE rank 分摊读取一份 checkpoint
每个 CE rank 只持有一部分权重
所有推理 worker 最终都要看到完整 HF 权重流
```

假设 `xxx.weight` 只被 CE rank0 读到了。如果没有 broadcast：

```text
CE rank0 可以把自己的 CUDA IPC handle 给本机 worker0
但 worker1..127 不能靠这个 handle 自动拿到 rank0 的权重
```

CUDA IPC 不是跨 rank / 跨节点的数据分发机制。它只解决本机两个进程之间怎么共享一块 GPU
memory，不负责把 rank0 的 tensor 复制到 rank1..127 的本地 GPU buffer。

所以 checkpoint-engine 才需要两层机制：

```text
distributed broadcast:
  负责跨 rank / 跨卡分发权重 bytes

CUDA IPC:
  负责本机 CE 进程到推理 worker 的零拷贝交接
```

这也解释了为什么在共卡常规权重同步下，checkpoint-engine broadcast 通常不会比直接 IPC
tensor 更快。

对比一下：

```text
直接 IPC tensor:
  trainer GPU tensor
  -> inference worker

checkpoint-engine broadcast:
  trainer / HF weight
  -> CE CPU pinned memory
  -> CE GPU bucket buffer
  -> distributed broadcast 到所有 CE rank
  -> ZMQ 发送 metadata
  -> inference worker 通过 CUDA IPC 访问 CE buffer
  -> model.load_weights()
```

所以判断标准可以简化成：

```text
共卡、训练侧已经有 GPU tensor:
  优先直接 IPC / update_weights_from_tensor

从 HF disk 恢复、非共卡、多实例大规模分发:
  checkpoint-engine broadcast 更有价值
```



## 11. 为什么 IPC handle 和 metadata 不合并成一次发送

因为它们粒度不同。

CUDA IPC handle 是 **buffer 级别** 的：

```text
buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device=...)
handle = reduce_tensor(buffer)
```

这个 handle 表示整块 GPU double buffer，因此只需要发一次。

后续 bucket 复用这块 double buffer：

```text
bucket0 -> buffer[0 : bucket_size]
bucket1 -> buffer[bucket_size : 2 * bucket_size]
bucket2 -> buffer[0 : bucket_size]
bucket3 -> buffer[bucket_size : 2 * bucket_size]
```

而 bucket metadata 是 **bucket 级别** 的。每个 bucket 里有哪些 tensor 都不一样：

```text
bucket0:
  layer0.q_proj.weight
  layer0.k_proj.weight

bucket1:
  layer0.v_proj.weight
  layer0.o_proj.weight

bucket2:
  expert.8.up_proj.weight
  expert.9.down_proj.weight
```

所以 metadata 需要逐 bucket 发送。

逐 bucket ZMQ 还有一个作用：backpressure。

因为 double buffer 会被复用，如果 SGLang worker 还没从某个 slot 切 tensor 并完成 `load_weights()`，checkpoint-engine 就覆盖这个 slot，会造成数据错乱。逐 bucket metadata + ack 可以保证 buffer 复用是安全的。

## 12. HTTP 负责什么

HTTP 只负责实例级触发。

每个 SGLang 实例 leader 发一次：

```text
POST /update_weights_from_ipc
```

请求里带这个实例所有 worker 的 ZMQ handles。

对于 `inference_parallel_size=8`：

```text
rank0   -> instance0: handles rank0..7
rank8   -> instance1: handles rank8..15
...
rank120 -> instance15: handles rank120..127
```

SGLang HTTP server 收到后，会在内部通知自己的 EP/TP workers。

每个 worker：

```text
获取自己的 GPU UUID
从 zmq_handles 里找到自己的 ipc:// socket
连接本地 checkpoint-engine rank
接收 CUDA IPC handle 和后续 bucket metadata
```

所以每个实例只需要一次 POST，不需要给每个 worker 单独发 HTTP 请求。

## 13. MoE / EP 下不需要的 expert 权重会怎样

会被临时接收，但不会被保留。

假设：

```text
每个 SGLang 实例 EP8
某个权重是 expert.8.down_proj.weight
```

如果这个权重在 rank37 的 bucket 中，那么 broadcast 时：

```text
rank37 把包含 expert.8.down_proj.weight 的 bucket broadcast 给所有 128 个 CE rank
所有 SGLang worker 都临时看到这个 tensor
```

但在每个 SGLang 实例内部：

```text
负责 expert.8 的 EP rank:
  copy 到本地模型参数

不负责 expert.8 的 EP rank:
  skip / 不 copy / 不保留
```

所以：

```text
传输阶段:
  所有 rank 都收到这个 bucket

模型存储阶段:
  只有需要这个 expert 的 rank 真正保存它
```

这就是 broadcast 模式的 tradeoff。

优点：

```text
实现简单
通信顺序稳定
吞吐高
checkpoint-engine 不需要理解 SGLang 的 TP/EP/MoE 细节
```

缺点：

```text
对 MoE/EP 不省网络流量
不需要某个 expert 的 rank 也会临时接收该 expert 所在 bucket
```



## 14. 为什么不只发给需要该 expert 的 EP rank

理论上可以，但那就不是当前 broadcast 模式了。

如果要只发给需要的 EP rank，checkpoint-engine 需要理解：

```text
SGLang 的 expert placement
每个 expert 当前归哪个 EP rank
MoE 是否有 elastic expert placement
每个模型的 load_weights 规则
每个 tensor 应该发给哪些 rank
每个 bucket 的动态接收集合
```

这样 checkpoint-engine 就从通用传输层变成了模型并行调度器。

当前 broadcast 模式选择保持简单：

```text
所有 worker 都看到完整 HF 权重流
SGLang 自己决定要不要加载、怎么切片、expert 放哪
```



## 15. 再用一个小例子串起来

假设只有 4 个 CE rank，2 个 SGLang 实例，每实例 2 卡。

HF 权重被分摊成：

```text
rank0 注册：A, B
rank1 注册：C, D, E
rank2 注册：F
rank3 注册：G, H
```

`gather_metas()` 后，所有 rank 得到同一份 schedule：

```text
bucket0: src=rank0, items=[A, B]
bucket1: src=rank1, items=[C, D]
bucket2: src=rank1, items=[E]
bucket3: src=rank2, items=[F]
bucket4: src=rank3, items=[G, H]
```

执行：

```text
bucket0:
  rank0 发，rank1/2/3 收
  4 个 SGLang worker 都看到 A/B
  各自 load 自己需要的部分

bucket1:
  rank1 发，rank0/2/3 收
  4 个 SGLang worker 都看到 C/D
  各自 load 自己需要的部分

bucket2:
  rank1 发，rank0/2/3 收

bucket3:
  rank2 发，rank0/1/3 收

bucket4:
  rank3 发，rank0/1/2 收
```

rank0 发完 `bucket0` 后，后面就一直作为 receiver 参与 collective。它不能退出。

## 16. 最后总结

checkpoint-engine broadcast 模式可以概括为：

```text
128 个 CE rank 分摊读取一份 HF checkpoint
每个 rank 把自己负责的权重注册到 CPU pinned memory
gather_metas 生成全局一致的 bucket schedule
每个 bucket 由 owner rank 作为 src broadcast 给全部 128 个 rank
每个 SGLang worker 通过 CUDA IPC 访问本地 CE rank 的 GPU bucket
ZMQ 负责 IPC handle、bucket metadata、ack/error
SGLang model.load_weights 决定每个 TP/EP/MoE rank 实际保留哪些权重
```

最容易混淆的点：

```text
broadcast 传的是权重 bytes，不是 metadata。
metadata 走 ZMQ。
CUDA IPC handle 只发一次，bucket metadata 每个 bucket 都发。
bucket 是单个 owner rank 组成的，不由多个 rank 拼。
所有 rank 都会临时收到每个 bucket。
MoE/EP 下，不需要某个 expert 的 rank 也会临时收到，但会在 SGLang load_weights 阶段跳过。
```



## 17. lmdeploy 如果原生接入 checkpoint-engine from_ipc，要怎么改

这里说的是“方案 2”：让 lmdeploy 像 SGLang 一样原生支持 checkpoint-engine 的
`from_ipc` 更新协议。

也就是说，checkpoint-engine 侧仍然维持原来的模式：

```text
ParameterServer.register_checkpoint()
ParameterServer.gather_metas()
ParameterServer.update(update_method="broadcast")
```

lmdeploy 侧新增一个类似 SGLang 的接口：

```text
POST /update_weights_from_ipc
```

这个接口不直接携带权重 tensor。它只携带 checkpoint-engine 为 lmdeploy 每个 worker
准备好的 ZMQ socket 信息，例如：

```text
{
  "zmq_handles": {
    "<gpu_uuid_0>": "ipc://@checkpoint-engine-...",
    "<gpu_uuid_1>": "ipc://@checkpoint-engine-...",
    ...
  }
}
```

真正的权重数据仍然走 checkpoint-engine broadcast + CUDA IPC。

### 16.1 为什么 lmdeploy 需要改代码

lmdeploy 当前已经有在线更新权重能力，但它不是 checkpoint-engine 的
`from_ipc` 协议。

现有入口是：

```text
lmdeploy/serve/openai/api_server.py
  POST /update_weights
```

它的请求结构是：

```text
lmdeploy/serve/openai/protocol.py
  UpdateParamsRequest
```

现有 PyTorch backend 主要支持两类数据：

```text
serialized_named_tensors
flattened_bucket
```

也就是 HTTP request 里直接带序列化后的 tensor 信息，或者带 flatten 后的 bucket。
这和 checkpoint-engine 的设计不同。

checkpoint-engine 的 `from_ipc` 是：

```text
HTTP 只负责通知 lmdeploy：去哪些 ZMQ socket 收权重
ZMQ 负责发送 CUDA IPC handle 和每个 bucket 的 metadata
权重 bytes 通过 torch.distributed.broadcast 进入 CE 的 GPU buffer
lmdeploy worker 通过 CUDA IPC 访问本机 CE GPU buffer
lmdeploy worker 把 bucket 里的 tensor 切出来后调用 model.load_weights()
```

所以，如果要让 lmdeploy 原生接 checkpoint-engine，就需要在 lmdeploy 内部补齐这个协议。

### 16.2 最小改动目标：先只支持 PyTorch backend

优先建议只接 lmdeploy PyTorch backend。

原因是 PyTorch backend 已经有非常接近的最后一步：

```text
lmdeploy/pytorch/engine/model_agent/agent.py
  BaseModelAgent.update_params()
  BaseModelAgent.update_weights_from_distributed()
```

这些逻辑已经会做：

```text
接收一批 (name, tensor)
ModelWeightLoader._rename_weights_iterator(...)
model.load_weights(iter(weights))
finished=True 时遍历模块 mod.update_weights()
torch.cuda.synchronize()
必要时 reset_graph_runner()
torch.cuda.empty_cache()
```

checkpoint-engine from_ipc 接进来以后，最终也应该复用这条加载路径。

不建议第一步就接 Turbomind backend。Turbomind 也有：

```text
lmdeploy/turbomind/turbomind.py
  update_params()
```

但它的权重导出、转换、engine 创建和底层 TurboMind runtime 绑定更深，不是简单把
`(name, tensor)` 传给 `load_weights()` 就能完成。

### 16.3 lmdeploy 需要新增的接口

需要新增一个请求结构，例如：

```python
class UpdateWeightsFromIpcRequest(BaseModel):
    zmq_handles: dict[str, str]
    finished: bool = True
```

其中：

```text
zmq_handles:
  key 是 GPU UUID
  value 是 checkpoint-engine 暴露出来的 ZMQ socket path

finished:
  是否是最后一轮更新
  True 时需要触发 fused module / MoE module / graph runner 的 finalize 逻辑
```

然后在 API server 新增：

```text
POST /update_weights_from_ipc
```

这个 endpoint 的行为应该和 SGLang 类似：

```text
接收一次 POST
把 zmq_handles fanout 给本 lmdeploy 实例内的所有 worker
每个 worker 根据自己的 GPU UUID 选择属于自己的 ZMQ socket
每个 worker 独立连接 checkpoint-engine
每个 worker 独立接收 IPC handle 和 bucket metadata
```

注意：这里仍然是“一个 lmdeploy 实例一个 POST”，不是每张卡都 POST。

以 128 卡、16 个 lmdeploy 实例、每实例 8 卡为例：

```text
checkpoint-engine 一共有 128 个 rank
lmdeploy 一共有 16 个 HTTP server / 实例入口
每个实例收到 1 次 POST /update_weights_from_ipc
每次 POST 携带这个实例内部 8 个 worker 的 zmq_handles
实例内部 fanout 给 8 个 worker
8 个 worker 各自连接自己的 CE rank
```



### 16.4 worker 侧需要实现什么

每个 lmdeploy worker 需要实现一个 checkpoint-engine worker adapter。

这个 adapter 的职责和 SGLang 的
`SGLangCheckpointEngineWorkerExtensionImpl` 类似：

```text
1. 获取当前 worker 的 device UUID
2. 从 zmq_handles 里找到自己的 socket path
3. 连接 checkpoint-engine 的 ZMQ socket
4. 接收 CUDA IPC handle
5. rebuild IPC tensor
6. 循环接收每个 bucket 的 metadata
7. 从 IPC tensor 对应 offset/shape/dtype 切出 named tensors
8. 调用 lmdeploy 的 model.load_weights()
9. 每个 bucket 加载完后给 checkpoint-engine ack
10. 所有 bucket 结束后执行 post_hook
```

其中第 8 步要复用 lmdeploy 现有权重加载逻辑：

```text
ModelWeightLoader._rename_weights_iterator(weights, model)
model.load_weights(iter(renamed_weights))
```

第 10 步要复用 `finished=True` 时已有的 finalize 逻辑：

```text
for _, mod in model.named_modules():
    if hasattr(mod, "update_weights"):
        mod.update_weights()

torch.cuda.synchronize()
reset_graph_runner()
torch.cuda.empty_cache()
```

对 MoE / quant / fused linear 来说，这一步很重要，因为很多 lmdeploy 模块的
`update_weights()` 会重新整理后端 kernel 使用的权重布局。

### 16.5 和现有 `/update_weights` 的关系

新的 `/update_weights_from_ipc` 不应该替代现有 `/update_weights`。

两者适用场景不同：

```text
/update_weights:
  HTTP request 直接带 serialized_named_tensors 或 flattened_bucket
  适合简单在线更新、少量权重、已有 tensor producer 场景

/update_weights_from_ipc:
  HTTP request 只带 zmq_handles
  权重数据走 checkpoint-engine broadcast + CUDA IPC
  适合 128 卡这类大规模 rollout 权重同步
```

可以把它理解成：

```text
/update_weights 是 lmdeploy 自己的数据传输协议
/update_weights_from_ipc 是 checkpoint-engine 的数据传输协议适配层
```

最终两条路径应该在 worker 内部汇合到同一套 `load_weights()` 和 finalize 逻辑。

### 16.6 128 卡下的 lmdeploy 接入流程

假设：

```text
总 GPU 数：128
lmdeploy 实例数：16
每个实例：8 卡
checkpoint-engine rank 数：128
权重来源：HF disk
更新模式：broadcast
```

流程是：

```text
1. 128 个 CE rank 启动
2. 每个 CE rank 注册自己负责的 HF 权重文件或 tensor shard
3. CE 执行 gather_metas，得到全局 bucket schedule
4. CE 为每个 rank 绑定一个 ZMQ socket
5. 对每个 lmdeploy 实例，选该实例的入口 rank 发一次 POST /update_weights_from_ipc
6. POST body 里包含这个实例 8 个 worker 的 zmq_handles
7. lmdeploy HTTP server 把请求 fanout 到本实例 8 个 worker
8. 每个 worker 按自己的 GPU UUID 找 socket 并连接 CE
9. CE 给每个 worker 发送 CUDA IPC handle
10. CE 按 bucket schedule 执行 torch.distributed.broadcast
11. 每个 worker 通过 CUDA IPC 看到本地 CE rank 的 GPU buffer
12. CE 每个 bucket 通过 ZMQ 发 metadata
13. worker 根据 metadata 切出 named tensors
14. worker 调用 lmdeploy model.load_weights()
15. bucket 加载完成后 worker 给 CE ack
16. 全部 bucket 完成后，worker 执行 mod.update_weights() / reset_graph_runner()
17. lmdeploy wakeup kv_cache，恢复服务
```

这个流程里，HTTP 仍然只负责“启动一次更新任务”。它不承载权重数据。

### 16.7 需要特别注意的点

第一，lmdeploy worker 必须能拿到稳定的 GPU UUID。

checkpoint-engine 的 `zmq_handles` 是按 GPU UUID 匹配 worker 的。如果只靠 local rank
匹配，遇到 `CUDA_VISIBLE_DEVICES` 重排时容易连错 socket。

第二，worker 侧要串行化权重更新。

lmdeploy 现在 PyTorch engine 已经有 `_weights_update_lock`，新的 from_ipc 路径也应该复用
这个锁，避免同一实例同时执行两次权重更新。

第三，更新前后要配合 sleep/wakeup。

比较稳的顺序是：

```text
POST /sleep?tags=["weights","kv_cache"]&level=2
POST /wakeup?tags=["weights"]
POST /update_weights_from_ipc
POST /wakeup?tags=["kv_cache"]
```

如果只是在 rollout 间隙更新，也至少要保证没有正在 decode 的请求还在使用旧权重和旧
CUDA graph。

第四，MoE/quant/fused module 必须执行 finalize。

只把 tensor copy 到参数里不一定够。lmdeploy 里很多模块还有后端 kernel 专用的 packed
layout，需要在最后调用模块自己的 `update_weights()`。

第五，checkpoint-engine 的 `from_ipc` worker adapter 最好独立成一个小模块。

不要把 CE 的 ZMQ 协议直接揉进 HTTP server。比较清楚的分层是：

```text
api_server.py:
  只定义 endpoint 和 request schema

engine.py / executor:
  只负责把请求 fanout 到 worker，并加锁

model_agent:
  负责创建 lmdeploy checkpoint-engine adapter

checkpoint_engine_adapter.py:
  负责 ZMQ / CUDA IPC / bucket metadata / ack / error
```

这样 lmdeploy 自己的 `/update_weights`、`/update_weights_from_distributed` 和新的
`/update_weights_from_ipc` 可以共存。

### 16.8 最小实现清单

最小可用版本需要改这些地方：

```text
lmdeploy/serve/openai/protocol.py
  新增 UpdateWeightsFromIpcRequest

lmdeploy/serve/openai/api_server.py
  新增 POST /update_weights_from_ipc
  限制先只支持 backend="pytorch"

lmdeploy/pytorch/engine/engine.py
  新增 async update_weights_from_ipc()
  复用 _run_weights_update() 串行化更新

lmdeploy/pytorch/engine/executor/base.py
lmdeploy/pytorch/engine/executor/base_worker.py
lmdeploy/pytorch/engine/executor/ray_executor.py
lmdeploy/pytorch/engine/mp_engine/base.py
lmdeploy/pytorch/engine/mp_engine/base_worker.py
  补齐 collective_rpc / worker 转发

lmdeploy/pytorch/engine/model_agent/agent.py
  新增 update_weights_from_ipc()
  调用 checkpoint-engine adapter
  复用现有 load_weights / finalize 逻辑

lmdeploy/pytorch/engine/model_agent/checkpoint_engine_adapter.py
  新增 CE from_ipc adapter
  负责 ZMQ socket、CUDA IPC handle、bucket metadata、ack/error
```

完成这些以后，lmdeploy 才能真正作为 checkpoint-engine broadcast 的 receiver 使用。