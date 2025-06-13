from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup, create_colocated_worker_cls
import ray
import torch
import warnings
from verl.single_controller.base.decorator import Dispatch, Execute, register
import os
warnings.filterwarnings("ignore")

ray.init()

# [4] 表示要启动 4 个 worker
# max_colocate_count 表示每个 worker 最多占用的 cpu 核数
resource_pool = RayResourcePool([4], use_gpu=False, max_colocate_count=1)

@ray.remote
class GPUAccumulator(Worker):
    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        self.value = torch.zeros(size=(1,), device="cpu") + self.rank

    @register(Dispatch.ONE_TO_ALL)
    def add(self, x):
        self.value += x
        print(f"[GPUAccumulator] rank {self.rank}, value: {self.value}")
        return self.value


@ray.remote
class GPU2Accumulator(Worker):
    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        self.value = torch.zeros(size=(1,), device="cpu") + self.rank

    @register(Dispatch.ONE_TO_ALL)
    def add(self, x):
        self.value -= x
        print(f"[GPU2Accumulator] rank {self.rank}, value: {self.value}")
        return self.value

# 假设一共 4 张卡, class_with_args 在 4 张卡上均匀切分，class2_with_args 也在 4 张卡上均匀切分
class_with_args = RayClassWithInitArgs(cls=GPUAccumulator)
class2_with_args = RayClassWithInitArgs(cls=GPU2Accumulator)

# 通过如下操作，不仅colocated 而且实际上两个 class 在同一个进程
cls_dict = {'actor': class_with_args, 'critic': class2_with_args}
ray_cls_with_init = create_colocated_worker_cls(cls_dict)

wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
# 对外假装变成 2 个独立的 worker group,方便用户调用
spawn_wg = wg_dict.spawn(prefix_set=cls_dict.keys())
colocated_actor_wg = spawn_wg['actor']
colocated_critic_wg = spawn_wg['critic']

print(colocated_actor_wg.add(x=10))
print(colocated_critic_wg.add(x=10))
