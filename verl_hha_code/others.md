# 其他代码细节记录

## `task.add_done_callback(self.background_tasks.discard)` 是否必要

位置：

```text
verl/trainer/ppo/v1/agent_loop_tq.py
```

相关代码：

```python
task = asyncio.create_task(
    self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
)
self.background_tasks.add(task)
task.add_done_callback(self.background_tasks.discard)
```

这里是 fire-and-forget 的后台任务模式。`generate_sequences()` 只负责把每条 sample 对应的 `_run_prompt()` 创建成后台 task，然后立即返回，不会 `await` 这些 task。

`self.background_tasks.add(task)` 的作用是给后台 task 留一个强引用，避免 task 没有明确 owner，也方便 worker 维护这些正在运行的任务。

`task.add_done_callback(self.background_tasks.discard)` 的作用是 task 完成后自动从 set 里移除。否则 Ray actor 长时间运行时，完成的 task 会一直留在 `self.background_tasks` 里，连带保留 prompt、sampling params、异常 traceback 等对象，造成内存持续增长。

所以这行 callback 对短期功能正确性不一定明显，但对长期训练很重要：运行时保活 task，结束后释放引用。

对比 `_run_prompt()` 内部的 task：

```python
tasks = []
for i in range(n):
    task = asyncio.create_task(...)
    tasks.append(task)
await asyncio.gather(*tasks)
```

这些 task 被局部 `tasks` 列表持有，并且马上由 `asyncio.gather()` 管理生命周期，因此不需要再放进 `background_tasks`。

## `_dump_generations` 为什么用 `ThreadPoolExecutor(max_workers=1)`

位置：

```text
verl/trainer/ppo/v1/trainer_base.py
```

相关代码：

```python
self._dump_executor = ThreadPoolExecutor(max_workers=1)
```

以及：

```python
future = self._dump_executor.submit(
    self._write_generations,
    inputs,
    outputs,
    gts,
    scores,
    reward_extra_infos_dict,
    dump_path,
    global_steps,
)
```

这里虽然用了 `ThreadPoolExecutor`，但 `max_workers=1` 说明它不是为了多个线程并发写同一类 dump，而是为了把 JSONL 写盘从训练主流程里异步挪出去。

主要收益：

- 训练主线程提交写文件任务后可以继续做后续逻辑，不必同步等待 JSONL 写完。
- 单 worker 保证 dump 写入串行执行，避免多个 dump 同时写盘导致 I/O 抖动或输出顺序难以判断。
- `_dump_generations()` 会检查已经完成的 future，并通过 `f.result()` 及时暴露写文件异常。
- `_shutdown_dump_executor()` 会在训练结束时等待所有 pending dump 完成，避免退出时丢文件。

限制也比较明确：

- 如果每步 dump 很大，写入速度慢于训练推进速度，pending future 会积压并持有 inputs、outputs 等对象，增加内存占用。
- `json.dumps` 有 CPU 序列化开销，单独放到线程里不一定能完全避开 GIL；实际收益主要来自和磁盘 I/O、Ray/GPU 计算等待时间重叠。
- 如果 dump 后马上 shutdown，例如最后一步或 `val_only`，异步收益基本不存在，因为最后仍然要等写完。

结论：这里的单 worker executor 是有意义的。它的目标是异步化和串行化写盘，而不是提高写盘并行度。除非确认 dump 是瓶颈且存储系统能承受并发写，否则盲目增大 `max_workers` 不一定更好。
