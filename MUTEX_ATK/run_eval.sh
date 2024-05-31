# MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 python mutex/eval.py \
#         benchmark_name=LIBERO_100 \
#         folder=dataset-path \
#         eval_spec_modalities=gl \
#         experiment_dir=experiments/mutex \
#         model_name=mutex_weights.pth

# inst, gl, img
# LIBERO_10
MUJOCO_EGL_DEVICE_ID=5 CUDA_VISIBLE_DEVICES=5 python mutex/eval.py \
        benchmark_name=LIBERO_10 \
        folder=./LIBERO/libero/datasets \
        eval_spec_modalities=gl \
        experiment_dir=mutex_pretrained \
        base_task_id=1 \
        model_name=mutex_weights.pth