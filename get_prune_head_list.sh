export CUDA_VISIBLE_DEVICES="7"
category="Video_Games"
short_name="Games"
val_data_path=""
model_path=""
tokenizer_path=""
save_path=""
for alpha_1 in 0.7
do
    python ./prune/head/get_ships_per_layer.py \
        --val_data_path ${val_data_path} \
        --model_path ${model_path} \
        --tokenizer_path ${tokenizer_path} \
        --save_path ${save_path} \
        --alpha ${alpha_1} \
        --category ${category} \
        --nums_head2prune 7
done
