export CLIPORT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=3
task="multi-attr-put-block-in-bowl-unseen-colors" #"put-block-in-bowl-adversarial"  # "stack-block-pyramid-seq-unseen-colors" # "packing-seen-google-objects-seq" # 

#python cliport/demos.py n=10 \
#                    task="put-block-in-bowl-unseen-colors" \
#                    mode=test 

# python cliport/demos.py n=10 \
#      	             task="put-block-in-bowl-seen-colors" \
# 		            mode=test

python cliport/eval.py model_task=multi-language-conditioned \
                       eval_task=${task} \
                       agent=cliport \
                       mode=test \
                       n_demos=3 \
                       train_demos=1000 \
                       save_results=True \
                       exp_folder=cliport_quickstart \
                       checkpoint_type=val \
                       update_results=True \
                       disp=False \
                       record.save_video=True \
