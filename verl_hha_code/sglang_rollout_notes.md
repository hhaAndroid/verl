# SGLang Rollout 关系梳理

这份笔记记录一次关于 `verl/workers/rollout/sglang_rollout` 的阅读结论。重点是普通 SGLang rollout，不展开 PD prefill/decode 分离。后续这块代码可能会改，尤其是 generate 是否继续走 Ray actor 直调，还是统一到 HTTP endpoint。

## 1. 先看顶层对象

普通 SGLang rollout 可以先只记住这几个对象：

```text
PPOTrainer
│
├── LLMServerManager
│   │
│   └── SGLangReplica
│       │
│       └── SGLangHttpServer
│           │
│           └── SGLang engine / tokenizer_manager
│
└── AgentLoopManager
    │
    └── AgentLoopWorker
        │
        └── LLMServerClient
            │
            └── 调用 SGLangHttpServer.generate()
```

职责简化：

```text
LLMServerManager
  看到 rollout.name == "sglang"，选择 SGLangReplica，并管理 server handle/address。

SGLangReplica
  负责创建 SGLangHttpServer Ray actor。它不生成 token。

SGLangHttpServer
  负责启动 SGLang engine，并处理 generate / sleep / abort / resume 等请求。

AgentLoopWorker
  跑具体 AgentLoop，不直接认识 SGLangHttpServer，只持有 llm_client。

LLMServerClient
  从 load balancer 中选 server，然后发 generate 请求。
```



## 2. 两条访问路径

这块容易乱，是因为同一个 SGLang server 有两种访问方式。

```text
                         SGLangHttpServer Ray actor
                                  │
                  ┌───────────────┴────────────────┐
                  │                                │
                  v                                v
           Ray actor 方法入口                  HTTP endpoint 入口
           generate.remote(...)                POST /generate 等
                  │                                │
                  v                                v
     SGLangHttpServer.generate()             SGLang HTTP app endpoint
                  │                                │
                  └───────────────┬────────────────┘
                                  v
                    SGLang engine / tokenizer_manager
```

更具体地说：

```text
路径 A：主生成链路，Ray actor 直调

AgentLoopWorker
└── LLMServerClient.generate()
    └── server_handle.generate.remote(...)
        └── SGLangHttpServer.generate()
            └── tokenizer_manager.generate_request(...)
```

```text
路径 B：HTTP adapter 链路，主要用于权重/显存/cache 控制

ActorRollout worker / CheckpointEngineWorker
└── ServerAdapter
    └── AsyncHttpServerAdapter
        └── POST/GET http://host:port/...
            ├── /update_weights_from_tensor
            ├── /release_memory_occupation
            ├── /resume_memory_occupation
            ├── /flush_cache
            ├── /load_lora_adapter_from_tensors
            └── /unload_lora_adapter
```

结论：

```text
主 generate 默认走路径 A。
权重同步、显存释放/恢复、cache flush 等控制逻辑主要走路径 B。
```

所以当前代码是混合模式：

```text
generate:
  Ray actor Python 方法 -> 显式调用 tokenizer_manager

control/update:
  HTTP adapter -> SGLang HTTP endpoint -> SGLang 内部自己处理 tokenizer_manager/engine
```



## 3. SGLangHttpServer 和 HttpServerAdapter 的关系

它们不是父子关系，也不是继承关系，而是 server 和 client 的关系。

```text
SGLangHttpServer
  被访问的服务端。负责启动 SGLang engine 和 uvicorn HTTP app。

HttpServerAdapter / AsyncHttpServerAdapter
  访问 SGLangHttpServer 的 HTTP client 封装。
```

关系图：

```text
SGLangReplica
└── creates
    └── SGLangHttpServer Ray actor
        ├── Ray actor method:
        │   ├── generate()
        │   ├── sleep()
        │   ├── abort_all_requests()
        │   └── resume_generation()
        │
        └── HTTP endpoints:
            ├── /generate
            ├── /flush_cache
            ├── /release_memory_occupation
            ├── /resume_memory_occupation
            └── /update_weights_from_tensor

ServerAdapter
└── creates
    └── AsyncHttpServerAdapter
        └── calls SGLangHttpServer HTTP endpoints
```

关键区别：

```text
LLMServerClient -> SGLangHttpServer:
  走 Ray actor handle。

HttpServerAdapter -> SGLangHttpServer:
  走 HTTP URL。
```



## 4. tokenizer_manager 是否需要 verl 显式调用

这取决于走哪条路径。

```text
路径 A：Ray actor 直调

verl 显式调用 tokenizer_manager：

SGLangHttpServer.generate()
└── tokenizer_manager.generate_request(...)
```

```text
路径 B：HTTP adapter

verl 不显式调用 tokenizer_manager。
verl 只是调用 SGLang HTTP endpoint，把 SGLang server 当黑盒。

AsyncHttpServerAdapter
└── POST /generate 或 /release_memory_occupation 等
    └── SGLang HTTP server 内部自己处理 tokenizer_manager
```

当前普通 rollout 的主生成链路是路径 A，所以 `async_sglang_server.py` 里会直接碰：

```text
tokenizer_manager.generate_request
tokenizer_manager.release_memory_occupation
tokenizer_manager.resume_memory_occupation
tokenizer_manager.pause_generation
tokenizer_manager.continue_generation
```

如果未来完全改成黑盒 HTTP 模式，这些调用理论上可以从 verl 侧消失，统一放到 SGLang server endpoint 内部。

## 5. 第二种 HTTP adapter 路径在哪里用

`HttpServerAdapter / AsyncHttpServerAdapter` 不是主 generate 路径，但确实在用。

入口在 worker 初始化：

```text
ActorRolloutRefWorker / TrainingWorker
└── get_rollout_class("sglang", "async")
    └── verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter
```

`ServerAdapter` 初始化后会在需要时连接已经启动好的 `SGLangHttpServer`：

```text
ServerAdapter._init_server_adapter()
│
├── ray.get_actor("sglang_server_{replica_rank}_{node_rank}")
│
├── server_actor.get_server_address.remote()
│
└── AsyncHttpServerAdapter(
      host=host,
      port=server_port,
      launch_server=False,
    )
```

之后这些操作会走 HTTP adapter：

```text
ServerAdapter.resume(tags=["weights"])
└── AsyncHttpServerAdapter.resume_memory_occupation()
    └── POST /resume_memory_occupation

ServerAdapter.update_weights(...)
└── AsyncHttpServerAdapter.update_weights_from_tensor()
    └── POST /update_weights_from_tensor

ServerAdapter.update_weights(...)
└── AsyncHttpServerAdapter.flush_cache()
    └── GET /flush_cache

ServerAdapter.release()
└── AsyncHttpServerAdapter.release_memory_occupation()
    └── POST /release_memory_occupation
```

权重同步链路可以简化成：

```text
CheckpointEngineManager.update_weights()
│
└── actor_wg.update_weights(...)
    │
    └── worker.rollout.update_weights(...)
        │
        └── ServerAdapter.update_weights(...)
            │
            ├── get_named_tensor_buckets(weights)
            │
            ├── sgl_update_weights(engine=AsyncHttpServerAdapter, ...)
            │   └── AsyncHttpServerAdapter.update_weights_from_tensor(...)
            │       └── HTTP /update_weights_from_tensor
            │
            ├── AsyncHttpServerAdapter.flush_cache()
            │
            └── server_actor.set_global_steps.remote(global_steps)
```



## 6. 为什么会让人觉得乱

因为 generate 和控制逻辑没有走同一种边界。

```text
生成请求:
  AgentLoopWorker
  -> LLMServerClient
  -> Ray actor handle
  -> SGLangHttpServer.generate()
  -> tokenizer_manager.generate_request()

权重 / 显存 / cache 控制:
  ActorRollout worker
  -> ServerAdapter
  -> AsyncHttpServerAdapter
  -> HTTP endpoint
  -> SGLang server 内部处理
```

这导致同一个 SGLang server 既被当作 Ray actor 调，也被当作 HTTP service 调。

更干净的架构大概有两种方向：

```text
方向 A：全部走 Ray actor

generate / update_weights / release_memory / resume_memory / flush_cache
全部变成 SGLangHttpServer 的 Ray actor 方法。
```

```text
方向 B：全部走 HTTP endpoint

generate / update_weights / release_memory / resume_memory / flush_cache
全部通过 AsyncHttpServerAdapter 调 SGLang HTTP endpoint。
```

当前实现是：

```text
generate 用方向 A。
control/update 用方向 B。
```

这就是阅读时最容易混乱的根源。

## 7. 文档里有没有解释为什么这么设计

目前没有找到一篇文档明确解释：

```text
为什么 generate 走 Ray actor 直调，
而 update/memory/cache 走 HTTP adapter。
```

但有几处文档能拼出部分设计动机。

### `docs/start/agentic_rl.rst`

这里解释了为什么 generate 使用基于 Ray actor 的 token-in-token-out API，而不是标准 chat completion API：

```text
训练阶段必须严格使用 LLM inference 返回的 tokens。
文本和 token 的转换可能不可逆。
例如 "<think>" 文本重新 tokenize 后可能不是模型真实生成的 token。
所以 server 需要提供 token-based API。
```

并且文档提到：

```text
SGLang AsyncServer uses the async_generate interface of the SGLang engine,
which is located on the first GPU of each TP group.
Therefore AsyncServer needs to remotely call async_generate through ray actor.
```

这能解释 generate 主链路为什么偏向 Ray actor token API。

### `docs/advance/agent_loop.rst`

这里描述了 rollout 阶段：

```text
AgentLoopWorker 调 LLMServerClient.generate
LLMServerClient 选择 server
AsyncLLMServer 收到请求后生成 response
```

也说明 vLLM 和 SGLang 都实现了 AsyncLLMServer。

### `docs/extend_guide.rst`

这里把两个抽象拆开：

```text
RolloutReplica:
  define how to launch your own inference server.

ServerAdapter:
  define how to update weights with your own inference server.
```

这能解释为什么代码天然分成两条线：

```text
RolloutReplica / AsyncServer 负责 server launch 和 generate。
ServerAdapter 负责 weight update。
```

但它没有解释为什么 SGLang generate 直调 tokenizer_manager，而 weight update 走 HTTP adapter。

### 代码 TODO

`async_sglang_server.py` 里有一个非常关键的 TODO：

```text
TODO: switch to `/generate` http endpoint once multi-modal support ready.
```

这个 TODO 暗示当前 generate 没走 HTTP endpoint，至少部分原因是当时 HTTP `/generate` 对多模态等能力还不够满足 verl 需要。

## 8. 当前理解结论

可以把现状记成：

```text
主生成链路:
  为了 token 精确性、AgentLoop、多轮训练和后端差异封装，
  当前走 Ray actor token API，并显式调用 tokenizer_manager。

权重/显存/cache 控制链路:
  为了复用 SGLang 已有 HTTP control endpoints，
  当前走 ServerAdapter + AsyncHttpServerAdapter。

文档:
  解释了 server mode、token-based generate、Rollo用 server_handle.generate.remote(...)
2. SGLangHttpServer.generate 是否还显式调用 tokenizer_manager.generate_request(...)
3. AsyncHttpServerAdapter.generate 是否变成主生成路径
4. ServerAdapter 是否仍只负责 update/memory/cache
```

utReplica/ServerAdapter 分工；
  没有明确解释 Ray generate + HTTP control 混合设计。

代码迹象:
  TODO 表示未来可能把 generate 也切到 HTTP /generate endpoint。

后续如果这块代码重构，重点观察：

```text
1. LLMServerClient.generate 是否还调
```

