set -x

cd /cpfs01/shared/llm_razor/huanghaian/code/verl

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export TRITON_CACHE_DIR="/tmp/triton"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONUNBUFFERED=1

if [ "$RANK" -eq 0 ]; then
    /cpfs01/shared/llm_razor/huanghaian/miniconda3/envs/verl/bin/ray start --head --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT
    sleep 30
else
    /cpfs01/shared/llm_razor/huanghaian/miniconda3/envs/verl/bin/ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT"  --block
fi
sleep 30

# -m debugpy --connect 5681
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" /cpfs01/shared/llm_razor/huanghaian/miniconda3/envs/verl/bin/ray job submit \
  --working-dir . -- /cpfs01/shared/llm_razor/huanghaian/miniconda3/envs/verl/bin/python /cpfs01/shared/llm_razor/huanghaian/code/verl/verl/trainer/main_ppo.py  \
    data.train_files=/cpfs01/shared/llm_razor/huanghaian/code/verl/data/gsm8k/train.parquet \
    data.val_files=/cpfs01/shared/llm_razor/huanghaian/code/verl/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=/cpfs01/shared/llm_razor/huanghaian/new_model/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=/cpfs01/shared/llm_razor/huanghaian/new_model/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log
