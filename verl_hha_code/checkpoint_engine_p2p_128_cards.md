# checkpoint-engine p2p 模式：128 卡局部实例重启场景解读

这篇文档只分析 checkpoint-engine 的 `p2p` 模式。

它会尽量沿用 broadcast 文档的写法，但要先说明一个核心区别：

```text
broadcast:
  一次同步更新所有 CE ranks / 所有推理实例。
  owner rank 通过 distributed broadcast 把 bucket 发给全体 ranks。

p2p:
  只更新指定的目标 ranks。
  目标 ranks 从已有 CE ranks 的 P2PStore 远程读取权重 bucket。
  读取到目标 ranks 后，再在目标 ranks 内部 broadcast。
```

所以 p2p 的价值不在于替代常规共卡 IPC，也不在于替代全量同步的 broadcast。

官方 README 里的原始定位是：

```text
new inference instances are dynamically added
```

也就是动态新增推理实例。

这篇文档先分析你现在更关心的一种子场景：

```text
CE ranks 没挂，仍然持有权重 cache。
某一组 SGLang worker 挂了，然后原地重启。
只需要把权重重新灌给这组重启后的 SGLang worker。
```

这种情况下，p2p 的意义是：

```text
只更新目标实例对应的 CE ranks / SGLang workers。
不触发全 128 卡 broadcast。
不影响其他 15 个健康推理实例。
```

CE rank 也挂掉的情况放到最后单独讨论。

## 1. 先把 p2p 核心过程说清楚

先用最短的话描述你这个场景下 p2p 到底在做什么。

前提是：

```text
CE ranks 是长活的。
CE ranks 已经 register_checkpoint()。
CE ranks 的 P2PStore 已经把 registered memory 暴露出来。
正常每次权重同步时，除了 trainer -> SGLang from_tensor，也同时更新 CE cache。
```

这样，CE cache 永远保存最新已发布权重：

```text
trainer 最新权重 version N
  -> SGLang 正常 from_tensor 更新到 version N
  -> CE pinned memory cache 也更新到 version N
```

然后 rollout 中某个 SGLang 实例挂了，例如 instance0：

```text
SGLang worker0..7 挂了并重启
CE rank0..127 没挂
CE cache 仍然是最新 version N
```

这时 p2p 的恢复过程是：

```text
1. 只更新 instance0 对应的 target CE rank0..7

2. target CE rank0..7 根据已有 global_metas 知道：
     每个 bucket 的 source owner CE rank 是谁，因为 ce 没有挂所以有些 bucket 的 source 就是自己
     source owner CE rank 的 P2PStore 地址是什么
     source memory ptr / offset / size 是什么

3. 对每个 bucket，checkpoint-engine 先选一个 target receiver rank
     例如 bucket_k:
       source owner = CE rank37
       target receiver = CE rank3

4. target receiver rank 通过 P2PStore / Mooncake / RDMA
   从 source owner rank 的 registered memory 远程读 bucket，mem 是 cpu 对象

5. bucket 先到 target receiver rank 的 GPU buffer， rdma 内部可能会做 cpu -> gpu 的 copy

6. target receiver rank 在 target ranks 内部做小范围 dist.broadcast
   例如 rank3 -> rank0..7

7. target CE rank0..7 都拿到这个 bucket 后，
   再通过原来的 ZMQ + CUDA IPC 让 SGLang worker0..7 访问本卡 CE GPU buffer

8. SGLang worker0..7 根据 metadata 切 tensor view，调用 model.load_weights()
```

所以核心链路是：

```text
source CE registered memory
  --P2PStore / Mooncake / RDMA read-->
target receiver CE GPU buffer
  --target ranks 内部 dist.broadcast-->
target CE rank0..7 GPU buffers
  --CUDA IPC-->
restarted SGLang worker0..7
  --model.load_weights()-->
模型参数
```

这里最关键的两个设计点是：

```text
P2P 负责跨 source/target 拉一次 bucket。
target 内 broadcast 负责把 bucket 扩散给目标实例内所有 ranks。
```

它不是让 source rank 直接 broadcast 给 target ranks。这样做的目的，是让 source ranks 尽量只是
被动暴露 registered memory，不主动进入恢复实例的 collective，减少对健康实例的影响。

也不是让 target rank0..7 都各自 RDMA 读同一个 bucket。那会把同一个 bucket 读 8 次，放大
source 网卡和 memory read 压力。

因此，p2p 是一个折中：

```text
跨 source/target:
  P2P 只拉一次

target 实例内部:
  小范围 broadcast 一次

推理引擎侧:
  仍然走 from_ipc / ZMQ / CUDA IPC / load_weights
```

理解了这个过程，后面 `ranks`、`receiver_rank`、`owner_rank`、`P2PStore`、`global_metas`
这些代码结构就比较自然了。

p2p 的最核心逻辑是：假设有 target 0~~~7 需要更新权重，因为权重分布在 source rank0 ~ rank 127 上，为了避免全局集合通信，target0~~~7需要想办法获取到全量权重，其实就是要得到 global meta 中记录的所有 bucket 实际上数据。以 bucket10 为例，假设其实际数据在 source rank17 上，那么就需要通过 rdma 方式将这个 source bucket 拉倒某个 target rank 上，然后这个 target rank 再进行内部广播，从而 target 所有 rank 都得到了这个 bucket10 实际数据，然后后续就是走 cuda ipc+zmq 了。

为啥不直接在source rank17+target0~7 上组成集合通信，这样就少了这个 rdma 过程，原因就是为了避免建立这个通信组，因此可能这个卡在做其他事情，强行参与通信可能有问题。而 rdma 是单独的 mem buffer，可以理解为独立进程，一定不会干扰正在运行的 source 进程的。

理解 p2p 逻辑的前提是要彻底理解 boardcast 模式的逻辑。

## 2. 先看 128 卡全局总览

先固定一个例子：

```text
总 GPU 数：128
SGLang 实例数：16
每个 SGLang 实例：8 卡
checkpoint-engine ranks：128 个，全部还活着
CE 侧状态：已经 register_checkpoint，并保留一份完整 HF 权重 cache
故障对象：SGLang instance0 的 worker0..7
重启对象：SGLang instance0 的 worker0..7
目标更新 ranks：CE rank0..7
```

主流程是：

```text
128 个 CE ranks:
  早就 register_checkpoint()
  早就把 CPU pinned memory 注册到 P2PStore
  早就 gather_metas()
  还活着，能被远程读

重启后的 SGLang instance0:
  worker0..7 重新启动
  等待 CE rank0..7 重新灌权重

CE rank0..7:
  只更新 ranks=0..7
  从 128 个 CE ranks 的 P2PStore 远程读权重
  在 rank0..7 内部 broadcast
  通过 ZMQ/CUDA IPC 交给重启后的 SGLang worker0..7
```

一句话总结：

```text
p2p 是“只让目标实例对应的 CE ranks 从已有 CE cache 拉权重并灌给该实例”，
不是“所有实例重新从 disk 加载”，也不是“全 128 卡重新 broadcast”。
```

## 3. 用户使用层面的差别其实很小

从用户调用 `ParameterServer` 的角度看，broadcast 和 p2p 前期逻辑基本一样。

共同部分是：

```text
ParameterServer 启动
register_checkpoint()
init_process_group()
gather_metas()
SGLang 启动并支持 /update_weights_from_ipc
CE bind ZMQ
HTTP POST /update_weights_from_ipc
SGLang worker connect ZMQ
CE 发 CUDA IPC handle
CE 发 bucket metadata
SGLang model.load_weights()
ack / post_hook
```

核心区别就是 `ps.update()` 时是否传 `ranks`。

不传 `ranks`：

```python
ps.update(checkpoint_name, req_func)
```

语义是 broadcast：

```text
更新所有 CE ranks / 所有推理实例。
owner rank 把 bucket broadcast 给所有 CE ranks。
```

传 `ranks`：

```python
ps.update(checkpoint_name, req_func, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
```

语义是 p2p：

```text
只更新 ranks 指定的目标 CE ranks。
目标 receiver rank 从 owner rank 的 P2PStore 远程读 bucket。
然后在目标 ranks 内部 broadcast。
```

所以可以记成：

```text
API 使用上:
  update(..., ranks=None)  -> broadcast
  update(..., ranks=[...]) -> p2p

系统语义上:
  broadcast -> 全量同步
  p2p      -> 局部恢复 / 局部新增

内部实现上:
  broadcast -> 全局 distributed broadcast
  p2p      -> P2P read + target ranks 内部 broadcast
```

不过 p2p 有额外前提：

```text
P2PStore 初始化成功
安装 checkpoint-engine[p2p] / Mooncake
owner ranks 的 registered memory 仍然活着
global_metas 仍然有效
```

## 4. 128 卡 p2p 端到端流程

这一节先写完整流程。后面章节再拆开解释。

### 4.1 CE/source 侧先持有一份可复用权重 cache

p2p 的前提是：已经有一批 source CE ranks 活着，并且它们的权重没有被释放。

例如，最初 128 卡集群启动时，先用 broadcast 或普通注册流程把 HF 权重加载进 CE：

```text
source CE rank0   注册一部分 HF tensor
source CE rank1   注册一部分 HF tensor
...
source CE rank127 注册一部分 HF tensor
```

每个 source CE rank 调：

```python
ps.register_checkpoint(checkpoint_name, ...)
```

注册后，每个 rank 持有自己负责的 CPU pinned memory：

```text
source CE rank0:
  CPU pinned memory: [tensor A, tensor B]

source CE rank1:
  CPU pinned memory: [tensor C, tensor D]

...

source CE rank127:
  CPU pinned memory: [tensor X, tensor Y]
```

如果 `P2PStore` 初始化成功，`register_checkpoint()` 后还会把这些 memory buffer 注册到
Mooncake TransferEngine：

```text
source CE rank0 P2PStore:
  register memory_pool_checkpoint_0 -> ptr0, size0

source CE rank1 P2PStore:
  register memory_pool_checkpoint_0 -> ptr1, size1

...
```

这样后续新实例可以通过 P2PStore 远程读取这些 buffer。

### 4.2 CE/source 侧生成并保留 global_metas

source CE ranks 执行：

```python
ps.gather_metas(checkpoint_name)
```

它会 all-gather 出全局 metadata。

这份 metadata 不只包含 tensor 的 name / shape / dtype / offset，还包含 P2P 需要的地址信息：

```text
owner rank
memory buffer ptr
memory buffer size
p2p_store_addr
rdma_device
host_ip
device_uuid
```

在当前主场景里，CE 进程一直活着，所以这份 metadata 保留在
`ps._current_global_parameter_metas` 里即可。

也就是说，这里不需要：

```python
metas = ps.get_metas()
ps.load_metas(metas)
```

也不需要：

```text
--save-metas-file
--load-metas-file
```

主场景应该是：

```text
CE 进程长活
register_checkpoint() 已经做过
gather_metas() 已经做过
global_metas 仍在 CE 进程内存里
后续 SGLang worker 重启时，直接 ps.update(..., ranks=target_ranks)
```

保存 metas 只适用于“新 CE 进程加入 / CE 也挂了 / controller 需要跨进程恢复”的变体，本文
最后单独讨论。

### 4.3 某个 SGLang 实例挂掉，但 CE ranks 还活着

假设 16 个 SGLang 实例中，instance0 挂了：

```text
SGLang instance0:
  worker0..7 挂了
```

但这里先明确假设：

```text
CE rank0..127 都还活着。
CE rank0..7 也还活着。
CE rank0..7 的 CPU pinned memory / P2PStore / global_metas 仍然有效。
```

也就是说，挂掉的是推理 worker，不是 checkpoint-engine。

现在原地重启 SGLang instance0：

```text
restarted SGLang instance0:
  worker0..7 重新启动
  重新等待权重灌入
```

这时候不需要拉起新的 CE rank0..7。

仍然使用原来还活着的：

```text
CE rank0..7
```

它们作为本次恢复的目标 ranks，重新创建 ZMQ endpoint，重新通过
`POST /update_weights_from_ipc` 通知重启后的 SGLang worker 来连接。

SGLang worker 重启后，它之前的 ZMQ 连接、CUDA IPC handle、IPC buffer view 都没了。虽然 CE rank0..7 还活着，但新 worker 不会自动知道。所以恢复时需要重新触发一次 POST, 这个 POST 发给重启后的 SGLang 实例入口，body 里带当前这次 CE 重新 bind 出来的 zmq_handles：

  {

    "zmq_handles": {

      "GPU-uuid-worker0": "ipc://@checkpoint-engine-GPU-uuid0-new.sock",

      ...

      "GPU-uuid-worker7": "ipc://@checkpoint-engine-GPU-uuid7-new.sock"

    }

  }

然后 SGLang 内部 fanout.

### 4.4 CE ranks 复用已有 global_metas

因为 CE 进程没有挂，所以最理想情况下：

```text
register_checkpoint() 的 CPU pinned memory 还在
P2PStore registered memory 还在
gather_metas() 生成的 global_metas 还在
```

这时恢复 SGLang worker 不需要重新从 HF disk 读完整模型，也不需要重新
`register_checkpoint()`。

CE 进程内部已经保留着 `current_global_parameter_metas`，直接复用即可。

这一步的含义是：

```text
CE rank0..7 知道：
  哪个 tensor 在哪个 source owner rank 上
  owner rank 的 P2PStore 地址是什么
  owner rank 的 memory buffer ptr 是多少
  后续应该从哪里远程读
```

这里的 source owner ranks 不是另一批新进程，而是当前仍然存活的 128 个 CE ranks。

所以同一批 CE 进程原地恢复 SGLang 时，关键动作就是后面调用：

```python
ps.update(checkpoint_name, req_func, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
```

### 4.5 调 ps.update(..., ranks=[0..7]) 进入 p2p 模式

p2p 的开关不是 `update_method` 字符串本身，而是 `ParameterServer.update()` 是否传了
`ranks`。

源码逻辑是：

```python
ps.update(checkpoint_name, req_func, ranks=list(range(inference_parallel_size)))
```

如果 `ranks` 是空或 `None`：

```text
走全量 broadcast。
```

如果 `ranks` 非空：

```text
走 p2p_update。
只更新 ranks 里指定的目标 ranks。
```

在这个例子里：

```text
ranks = [0, 1, 2, 3, 4, 5, 6, 7]
```

表示这次只灌权重给目标 8 个 CE ranks / 重启后的 8 个 SGLang workers。

### 4.6 target CE ranks 生成 p2p bucket 分配计划

p2p 仍然需要 bucket。

但和 broadcast 不同，p2p 生成的是：

```text
(receiver_rank, owner_rank, bucket)
```

其中：

```text
owner_rank:
  source CE rank，真正持有该 bucket 对应 CPU pinned memory 的 rank

receiver_rank:
  target CE rank，负责通过 P2P 从 owner_rank 远程读取这个 bucket 的 rank

bucket:
  这次要搬运的一组 tensor metadata / memory ranges
```

例子：

```text
bucket0: receiver=target rank0, owner=source rank0,   items=[A, B]
bucket1: receiver=target rank1, owner=source rank37,  items=[C, D]
bucket2: receiver=target rank2, owner=source rank80,  items=[E]
bucket3: receiver=target rank3, owner=source rank127, items=[X, Y]
...
```

这个 receiver 分配不是随便做的。

`_assign_receiver_ranks()` 会参考：

```text
local_topo:
  target CE ranks 的 RDMA device 分布

remote_topo:
  source owner ranks 的 RDMA device 分布
```

目标是尽量让多个 receiver 并行读不同 RDMA device 上的 source bucket，减少某一张网卡成为瓶颈。

### 4.7 p2p bucket 计划确定后是怎么执行的

生成 `(receiver_rank, owner_rank, bucket)` 之后，不是所有 bucket 完全串行，也不是所有 bucket
同时 broadcast。

源码里的执行方式更像：

```text
receiver 维度并行准备
target group 内按统一顺序 broadcast / load
```

先把 bucket 按 receiver rank 分组：

```text
target rank0: [bucket_a, bucket_d, bucket_g, ...]
target rank1: [bucket_b, bucket_e, bucket_h, ...]
target rank2: [bucket_c, bucket_f, bucket_i, ...]
...
```

执行时外层按轮次推进：

```text
for i in range(max_len):
  每个 receiver 准备自己的第 i 个 bucket
```

例如 target ranks 是 `rank0..7`，第 0 轮可能是：

```text
target rank0 从 source rank0   P2P read bucket0
target rank1 从 source rank37  P2P read bucket1
target rank2 从 source rank80  P2P read bucket2
target rank3 从 source rank127 P2P read bucket3
...
```

这些 P2P read 可以在 receiver 维度上并行准备，目的是尽量打满不同 receiver / source RDMA
device。

但后面的 broadcast 不能随便并行。

因为同一个 `ranks_group=[0..7]` 里的 collective 顺序必须完全一致。不能出现：

```text
rank0 正在 broadcast bucket0
rank1 同时 broadcast bucket1
rank2 同时 broadcast bucket2
```

这种情况下，各 rank 进入 collective 的顺序会错乱。

所以更准确的执行节奏是：

```text
第 i 轮准备阶段:
  多个 receiver 各自 P2P read / 准备自己的第 i 个 bucket

第 i 轮发送阶段:
  按所有 target ranks 一致的顺序：
    receiver0 broadcast 它的 bucket
    receiver1 broadcast 它的 bucket
    receiver2 broadcast 它的 bucket
    ...
```

因此它不是：

```text
全局 bucket0 完成后才开始准备 bucket1
```

也不是：

```text
所有 bucket 同时 broadcast
```

而是：

```text
receiver 维度并行准备 + target group 内按序 broadcast/load 的流水。
```

`h2d_buffer` 和 GPU double buffer 也是为这个流水服务的：

```text
h2d_buffer:
  用来提前准备 / 接收当前 receiver 的 bucket

double buffer:
  buffer[0:bucket_size]
  buffer[bucket_size:2*bucket_size]
  当前 bucket 被 worker 读时，另一个 half 可以服务下一轮准备
```

### 4.8 target CE ranks 仍然要重新创建 ZMQ 和 CUDA IPC buffer

到这里，SGLang 侧的消费链路和 broadcast 是一样的。

target CE rank0..7 会分别 bind ZMQ endpoint：

```text
target CE rank0 bind ipc://@checkpoint-engine-GPU-uuid0-0.sock
target CE rank1 bind ipc://@checkpoint-engine-GPU-uuid1-0.sock
...
target CE rank7 bind ipc://@checkpoint-engine-GPU-uuid7-0.sock
```

然后实例 leader 发一次：

```text
POST /update_weights_from_ipc
```

POST body 携带 restarted instance0 内 8 个 worker 的 `zmq_handles`。

SGLang worker0..7 收到 fanout 后，各自 connect 自己对应的 CE rank socket。

接着每个 target CE rank 分配 GPU double buffer：

```text
buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device="cuda")
```

并通过 ZMQ 把这个 buffer 的 CUDA IPC handle 发给本机 SGLang worker。

关键点：

```text
p2p 只改变 CE rank 之间怎么把 bucket 搬到目标 CE rank。
CE -> SGLang worker 仍然是同一套 ZMQ + CUDA IPC + metadata + load_weights。
```

### 4.9 p2p 远程读取 bucket 到 receiver rank

假设当前 bucket 是：

```text
bucket_k:
  receiver_rank = target rank3
  owner_rank    = source rank37
  items         = [layer10.mlp.up_proj.weight, layer10.mlp.down_proj.weight]
```

流程是：

```text
target CE rank3:
  根据 global_metas 找到 source rank37 的 p2p_store_addr
  根据 metadata 找到 source rank37 上对应 memory buffer 的 remote ptr / offset
  调 P2PStore.batch_transfer_sync_read(...)
  把 source rank37 CPU pinned memory 中的 bucket bytes 读到 target rank3 的 GPU buffer
```

对应源码关键逻辑：

```text
_copy_to_buffer(..., owner_rank=source_rank37)
  -> _get_addr_ptrs(owner_rank)
  -> batch_transfer_sync_read(target_addr, buf_ptrs, remote_ptrs, lens)
```

这里的 `target_addr` 是 source CE rank 的 P2PStore 地址。

`remote_ptrs` 是 source CE rank 中 registered memory buffer 的地址。

`buf_ptrs` 是 target CE receiver rank 的本地目标 buffer 地址。

### 4.10 receiver rank 在目标 ranks 内 broadcast

p2p 远程读完后，bucket 只在 receiver rank 的 GPU buffer 里。

但 SGLang instance0 有 8 个 worker。为了让 8 个 worker 都看到完整 HF 权重流，target CE
rank0..7 之间还要做一次小范围 broadcast：

```text
target CE rank3 作为 src
target CE rank0..7 调 dist.broadcast(buffer_b, src=target rank3, group=ranks_group)
```

broadcast 完成后：

```text
target CE rank0 的 GPU buffer 里有 bucket_k bytes
target CE rank1 的 GPU buffer 里有 bucket_k bytes
...
target CE rank7 的 GPU buffer 里有 bucket_k bytes
```

注意，这里不是 128 卡全量 broadcast。

这里只在目标 ranks 内 broadcast：

```text
group = ranks_group = [0..7]
```

这就是 p2p 和 broadcast 的关键差异：

```text
broadcast 模式:
  owner rank -> 128 个 CE ranks 全量 broadcast

p2p 模式:
  source owner rank -> 目标 receiver rank P2P read
  目标 receiver rank -> 目标 ranks 内部 broadcast
```

### 4.11 ZMQ 发送 bucket metadata，worker load_weights()

目标 ranks 内 broadcast 完成后，target CE rank0..7 都拥有当前 bucket bytes。

然后和 broadcast 模式一样：

```text
target CE rank i --ZMQ--> SGLang worker i:
  send(bucket metadata)

SGLang worker i:
  从 CUDA IPC buffer 切 tensor view
  调 model.load_weights()
  回 ack / error
```

TP / EP / MoE 的逻辑仍然在 SGLang `model.load_weights()` 里：

```text
TP rank 只保留自己的 shard
EP rank 只保留自己的 expert
不需要的 tensor 临时看到后跳过
```

### 4.12 重复所有 bucket，最后收尾

每个 bucket 都重复：

```text
选择 receiver rank
receiver rank 从 source owner rank P2P read
目标 ranks 内 broadcast
ZMQ 发 metadata
SGLang worker load_weights()
worker ack / error
```

所有 bucket 完成后：

```text
CE 发 None:
  worker 释放 IPC buffer

CE 再发 None:
  worker 执行 post_hook / flush cache
```

最终 restarted instance0 的 8 个 worker ready，可以重新加入 rollout。

## 5. 一张详细总图

```text
阶段 A：CE cache 先存在，CE ranks 没挂

  CE rank0..127:
    register_checkpoint()
    CPU pinned memory 保存各自负责的权重切片
    P2PStore.register_named_tensors(memory_pool_...)
    gather_metas()
    进程保持存活，global_metas 仍然有效


阶段 B：某个 SGLang 实例挂掉，然后原地重启

  old SGLang instance0 挂了
  restarted SGLang instance0 启动:
    worker0..7

  CE rank0..7 没挂:
    继续作为本次恢复的 target ranks


阶段 C：target CE ranks 复用已有 metadata

  CE rank0..127:
    已经有 global_metas

  target CE rank0..7 现在知道：
    tensor 在哪个 source owner rank
    source P2PStore 地址
    source memory buffer ptr
    source RDMA device


阶段 D：调用 ps.update(..., ranks=[0..7]) 进入 p2p

  p2p_update = True
  need_update = rank in [0..7]
  ranks_group = dist.new_group([0..7])


阶段 E：生成 p2p bucket plan

  global_metas
    |
    v
  _gen_h2d_buckets(..., ranks=[0..7])
    |
    v
  [(receiver_rank, owner_rank, bucket), ...]

  例子：
    bucket0: receiver=target rank0, owner=source rank0
    bucket1: receiver=target rank1, owner=source rank37
    bucket2: receiver=target rank2, owner=source rank80
    bucket3: receiver=target rank3, owner=source rank127


阶段 F：target CE 创建 ZMQ endpoint，触发 SGLang from_ipc

  target CE rank0 bind ipc://@checkpoint-engine-GPU-uuid0-0.sock
  target CE rank1 bind ipc://@checkpoint-engine-GPU-uuid1-0.sock
  ...
  target CE rank7 bind ipc://@checkpoint-engine-GPU-uuid7-0.sock

  target CE rank0 -> POST /update_weights_from_ipc -> restarted SGLang instance0

  POST body:
    zmq_handles = {
      GPU-uuid-worker0: ipc://@checkpoint-engine-GPU-uuid0-0.sock,
      ...
      GPU-uuid-worker7: ipc://@checkpoint-engine-GPU-uuid7-0.sock
    }

  SGLang instance0 fanout:
    worker0 connect target CE rank0 ZMQ
    ...
    worker7 connect target CE rank7 ZMQ


阶段 G：target CE 发送 CUDA IPC handle

  每个 target CE rank i:
    buffer_i = torch.empty(bucket_size * 2, dtype=torch.uint8, device="cuda")
    handle_i = reduce_tensor(buffer_i)
    send handle_i over ZMQ

  每个 SGLang worker i:
    recv handle_i
    rebuild IPC buffer_i


阶段 H：每个 bucket 的数据流

  假设：
    receiver = target rank3
    owner    = source rank37

  1. target rank3 P2P read:

     source rank37 CPU pinned memory
       --Mooncake / P2PStore.batch_transfer_sync_read-->
     target rank3 GPU buffer

  2. target rank3 在目标 ranks 内 broadcast:

     target rank3 GPU buffer
       --dist.broadcast(src=target rank3, group=[0..7])-->
     target rank0..7 GPU buffer

  3. target CE rank0..7 通过 ZMQ 发 metadata:

     target CE rank i --ZMQ--> SGLang worker i

  4. SGLang worker0..7 从 CUDA IPC buffer 切 tensor:

     raw = ipc_buffer[offset : offset + size]
     tensor = raw.view(dtype).view(shape)

  5. SGLang worker0..7 调:

     model.load_weights(named_tensors)

  6. SGLang worker0..7 回:

     ack / error


阶段 I：全部 bucket 完成

  CE --ZMQ--> worker:
    send(None)
    worker 释放 IPC buffer

  CE --ZMQ--> worker:
    send(None)
    worker post_hook / flush cache

  restarted SGLang instance0 ready
```

## 6. p2p 和 broadcast 到底差在哪

先看 broadcast：

```text
bucket owner rank:
  从自己的 CPU pinned memory 拷到 GPU buffer
  dist.broadcast 给所有 128 个 CE ranks

所有 CE ranks:
  都参与 broadcast
  都临时收到每个 bucket
  都通过 CUDA IPC 把 bucket 暴露给本机 worker
```

再看 p2p：

```text
source owner rank:
  不参与目标 ranks 的 dist.broadcast
  只需要 P2PStore 活着，registered memory 可被远程读

target receiver rank:
  通过 Mooncake P2P 从 source owner rank 读 bucket
  在目标 ranks 内做小范围 broadcast

target ranks:
  只有指定 ranks 接收并加载权重
```

所以 p2p 的数据路径是：

```text
source owner CPU pinned memory
  -> Mooncake P2P read
  -> target receiver GPU buffer
  -> target ranks 内 dist.broadcast
  -> target CE GPU buffers
  -> CUDA IPC
  -> target SGLang workers
```

而 broadcast 的数据路径是：

```text
owner CPU pinned memory
  -> owner GPU buffer
  -> 128 ranks 全量 dist.broadcast
  -> 128 CE GPU buffers
  -> CUDA IPC
  -> 128 SGLang workers
```

## 7. p2p 为什么仍然需要 dist.broadcast

这点很容易误解。

p2p 不是说每个目标 rank 都独立从 source 读完整权重。

源码里 p2p bucket 的单位是：

```text
(receiver_rank, owner_rank, bucket)
```

也就是某个 bucket 先被分配给一个目标 receiver rank。这个 receiver rank 通过 Mooncake 从
source owner rank 读到 bucket。

但推理实例内部有多个 ranks。

例如 restarted instance0 有 8 卡：

```text
target CE rank0..7
```

如果 bucket 只在 target rank3 上，worker0..2、worker4..7 就看不到这个 bucket。为了让每个
worker 都看到完整 HF 权重流，还需要：

```text
target rank3 -> target rank0..7 做一次目标组内 broadcast
```

所以 p2p 里的 broadcast 是小范围的：

```text
只在 ranks=[0..7] 内 broadcast
```

不是 128 卡全量 broadcast。

## 8. 为什么不让 source 直接 broadcast 给 target

一个自然的问题是：

```text
既然最后 target ranks 内还要 broadcast，
为什么不直接让 source owner rank broadcast 给 target rank0..7？
```

原因是 p2p 的目标不是“完全不用 broadcast”，而是：

```text
避免让 source ranks 主动参与目标实例的恢复 collective。
```

如果让 source rank37 直接 broadcast 给 target rank0..7，需要：

```text
source rank37 加入本次恢复用的 torch.distributed group
source rank37 主动进入 dist.broadcast()
target rank0..7 也进入同一个 collective
```

这会带来几个问题。

第一，source CE rank 必须主动参与恢复流程。

p2p 希望 source 侧只是：

```text
暴露 registered memory
被 target 远程读
```

而不是让 source CE rank 进入 Python update loop 和恢复实例的 collective。

第二，每个 bucket 的 owner 可能不同。

例如：

```text
bucket0: owner=source rank0
bucket1: owner=source rank37
bucket2: owner=source rank80
bucket3: owner=source rank127
```

如果直接 source broadcast，就意味着很多不同 source owner ranks 都要频繁参与目标恢复流程。

第三，source ranks 可能还服务着健康实例。

如果恢复 instance0 时让大量 source ranks 参与额外 collective，容易影响其他 15 个健康实例。

第四，source 和 target 不一定适合放进同一个恢复用 process group。

p2p 只需要：

```text
source P2PStore 地址
source memory ptr / offset / size
target 本地 buffer 地址
```

就可以让 target receiver rank 远程读。

而 source 直接 broadcast 需要更强的 distributed process group / collective 语义。

所以 p2p 采用的是折中设计：

```text
source owner rank:
  被动提供 registered memory
  不主动进入恢复 collective

target receiver rank:
  主动用 P2PStore / Mooncake / RDMA 从 source 读一次 bucket

target ranks:
  只在目标实例内部做小范围 broadcast
```

也就是：

```text
source rank37 CPU pinned memory
  --P2P/RDMA read，一次-->
target rank3 GPU buffer
  --target ranks 内部 broadcast-->
target rank0..7 GPU buffer
```

那为什么不让 target rank0..7 都各自 RDMA 读 source？

因为那会把同一个 bucket 读 8 次：

```text
source rank37 -> target rank0
source rank37 -> target rank1
...
source rank37 -> target rank7
```

这会放大 source 侧网卡和 memory read 压力。

当前设计是：

```text
跨 source/target:
  P2P 只拉一次，减少对 source 的影响

target 实例内部:
  小范围 broadcast，把 bucket 扩散给本实例所有 ranks
```

一句话总结：

```text
P2P 是为了让 source 侧尽量被动、少参与；
target 内 broadcast 是为了让目标实例内所有 rank 都拿到 bucket。
```

## 9. p2p 为什么要 gather/load metas

p2p 比 broadcast 更依赖 metadata。

因为目标 ranks 即使自己也持有一部分本地权重，也不一定持有完整模型。它们需要知道：

```text
哪些 owner rank 持有权重
每个 owner rank 的 P2PStore 地址是什么
每个 owner rank 的 RDMA device 是什么
每个 tensor 在 owner memory buffer 的 ptr / offset / size 是什么
```

这些都来自 `global_metas`。

source 侧：

```text
register_checkpoint()
gather_metas()
```

target 侧，在本文主场景里：

```text
直接复用 CE 进程内已有 global_metas
update(..., ranks=target_ranks)
```

如果不是同一批长活 CE 进程，或者需要显式恢复 metadata，才需要：

```text
get_metas()
load_metas(source_metas)
```

其中 `load_metas()` 的作用是把 source 侧 topology 和 P2PStore 地址装进 target CE ranks。

如果没有这份 source metas，target ranks 不知道该去哪里远程读权重。

## 10. p2p 下 ZMQ / CUDA IPC 和 broadcast 模式一样吗

对 SGLang 来说，几乎一样。

不管 CE rank 之间的数据来源是：

```text
全量 dist.broadcast
```

还是：

```text
Mooncake P2P read + 目标组内 broadcast
```

SGLang worker 看到的接口仍然是：

```text
POST /update_weights_from_ipc
ZMQ endpoint
CUDA IPC handle
bucket metadata
model.load_weights()
ack / error
post_hook
```

所以：

```text
p2p 改的是 CE 内部如何把 bucket 搬到目标 CE ranks。
SGLang from_ipc 消费链路不需要因为 p2p 改成另一套协议。
```

这也是为什么 lmdeploy 如果要接 checkpoint-engine，仍然只需要实现同一个
`/update_weights_from_ipc` receiver 协议。CE 后面用 broadcast 还是 p2p，对推理引擎来说
可以是透明的。

## 11. p2p 的 ranks 参数怎么理解

`ParameterServer.update()` 的 `ranks` 参数表示：

```text
这次要更新哪些目标 CE ranks。
```

如果是恢复一个 8 卡实例：

```python
ps.update(checkpoint_name, req_func, ranks=list(range(8)))
```

如果是一次恢复两个 8 卡实例，也可能是：

```python
ps.update(checkpoint_name, req_func, ranks=list(range(16)))
```

在 checkpoint-engine 示例里，join 流程通常写成：

```python
ps.update(
    checkpoint_name,
    req_func,
    ranks=list(range(inference_parallel_size)),
)
```

含义是：

```text
本次只恢复一个推理实例。
这个实例有 inference_parallel_size 张卡。
```

如果你是 128 卡、16 个实例、每实例 8 卡，并且只恢复 instance0，那么 target ranks 就是
CE rank0..7。

如果你一次恢复两个实例，需要确保：

```text
target ranks 覆盖两个实例的所有 CE ranks
req_func 能对两个实例分别发 POST
每个 POST 的 zmq_handles 只包含对应实例内 worker
```

## 12. CE 没挂时，p2p 对 SGLang 重启有什么价值

假设 rollout 过程中 instance0 的 SGLang worker0..7 挂了，但 CE rank0..127 都没挂。

如果从 HF disk 恢复：

```text
restarted instance0 启动
重新从 disk 读取 HF checkpoint
重新 register_checkpoint()
重新灌权重
```

耗时可能主要卡在 disk load / register pin memory。

如果走 p2p：

```text
128 个 CE ranks 已经有一份 registered CPU pinned memory
CE rank0..7 作为 target ranks
CE rank0..7 从全局 CE P2PStore 拉完整权重流
CE rank0..7 通过 ZMQ/CUDA IPC 灌给 restarted SGLang worker0..7
```

这样可以避免重启后的 SGLang 实例重新从 disk 读完整模型，也避免触发 128 卡全量 broadcast。

和 broadcast 对比：

```text
broadcast:
  所有 128 个 CE ranks 都参与全量更新
  所有 16 个 SGLang 实例都可能被触发 update

p2p:
  只更新 CE rank0..7
  只触发 restarted instance0
  其他 15 个健康实例不需要重新 load_weights
```

但前提是：

```text
CE 进程必须还活着
registered memory 不能被 unregister
P2PStore / Mooncake 可用
global_metas 不能过期
```

如果你只在本地保存一份 HF 文件，然后所有 CE 进程都退出了，p2p 没法凭空从 metadata 恢复
权重。那还是要走 disk load。

## 13. p2p 的内存开销怎么理解

source 侧需要保留：

```text
一份完整模型的 CPU pinned memory，分布在 source CE ranks 上
P2PStore registered memory 信息
```

target 侧恢复时需要临时分配：

```text
GPU double buffer: bucket_size * 2
可选 h2d_buffer: bucket_size
```

源码里 `_detect_bucket_size()` 会根据空闲显存判断是否启用 `h2d_buffer`：

```text
显存足够:
  使用 h2d_buffer
  可以更好地做流水和并行

显存不足:
  disable_h2d_buffer=True
  少分一块 buffer
  但流水能力下降
```

p2p 下还会把接收 buffer 注册到 P2PStore：

```text
p2p_ipc_buffer_name = "__ipc_buffer__"
P2PStore.register_named_tensors({p2p_ipc_buffer_name: buffer 或 h2d_buffer})
```

可以简单记：

```text
source 侧长期多留一份分布式 CPU pinned 权重 cache。
target 侧恢复期间额外占用 bucket 级 GPU buffer。
```

## 14. p2p 和你关心的共卡场景

如果是正常训练后同步权重，且训练进程和推理进程共卡：

```text
trainer GPU tensor -> inference worker
```

这种路径仍然优先直接 IPC / `update_weights_from_tensor`。

p2p 不适合作为每轮常规共卡权重同步的默认方案，因为它仍然需要：

```text
source CE cache
P2PStore
bucket 计划
ZMQ / CUDA IPC
目标组内 broadcast
model.load_weights()
```

p2p 更适合做恢复通道：

```text
SGLang 实例挂了并重启
CE cache 仍然活着
只更新这个实例对应的 target ranks
从 CE cache 拉权重比重新 disk load 快
```

所以对于你的目标，可以把三条路分清：

```text
正常每轮共卡同步:
  直接 IPC tensor / update_weights_from_tensor

全量 128 卡从 HF disk 同步:
  checkpoint-engine broadcast

局部 SGLang 实例挂掉后快速恢复，且 CE 没挂:
  checkpoint-engine p2p
```
