from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
import ray
import torch
import warnings
from verl.single_controller.base.decorator import Dispatch, Execute, register
import os
warnings.filterwarnings("ignore")

ray.init()

# [4] 表示要启动 4 个 worker
# max_colocate_count 表示每个 worker 最多占用的 cpu 核数
resource_pool = RayResourcePool([4], use_gpu=False, max_colocate_count=2)

@ray.remote
class GPUAccumulator(Worker):
    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        self.value = torch.zeros(size=(1,), device="cpu") + self.rank

    @register(Dispatch.ONE_TO_ALL)
    def add(self, x):
        self.value += x
        print(f"rank {self.rank}, value: {self.value}")
        return self.value


class_with_args = RayClassWithInitArgs(cls=GPUAccumulator)
worker_group = RayWorkerGroup(resource_pool, class_with_args)
# print(worker_group.execute_all_sync("add", x=[1, 1, 1, 1]))
print(worker_group.add(x=10))
