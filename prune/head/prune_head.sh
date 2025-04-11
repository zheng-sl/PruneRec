export CUDA_VISIBLE_DEVICES="7"

category="Video_Games"
short_name="Games"
model_path=""
embedding_json_path=""

for alpha in 0.7
do 
    head_json_path=""
    save_path=""
    log_dir=""
    echo model_path: ${model_path}
    echo head_json_path: ${head_json_path}
    echo save_path: ${save_path}
    echo log_dir: ${log_dir}
    python ./sasrec_init_prune.py \
        --model_path ${model_path} \
        --embedding_json_path ${embedding_json_path} \
        --head_json_path ${head_json_path} \
        --save_model_path ${save_path} \
        --alpha ${alpha} \
        --log_dir ${log_dir} \
        --num_heads_to_prune 7
done










