MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 python mutex/attack.py \
        benchmark_name=LIBERO_10 \
        folder=./LIBERO/libero/datasets \
        eval_spec_modalities=img \
        experiment_dir=mutex_pretrained \
        model_name=mutex_weights.pth \
        base_task_id=0 \
        target_task_id=1 \
        attack_method=gl2img \