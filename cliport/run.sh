export CLIPORT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=0
task="stack-block-pyramid-seq-unseen-colors" # "packing-seen-google-objects-seq" # 

python cliport/demos.py n=10 \
                    task=${task} \
                    mode=test 

python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=${task} \
                       agent=cliport \
                       mode=test \
                       n_demos=3 \
                       train_demos=1000 \
                       save_results=True \
                       exp_folder=cliport_quickstart \
                       checkpoint_type=val_missing \
                       update_results=True \
                       disp=False \
                       record.save_video=True \