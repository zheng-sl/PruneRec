export CUDA_VISIBLE_DEVICES="7" 

category="Video_Games"
short_name="Games"
model_path="/"
tokenizer_path=""
top_nums=
prune_mlp="true"
prune_attn="false"
data_path=""  

top_indices_save_path_mlp=""
top_indices_save_path_q=""
top_indices_save_path_k=""
top_indices_save_path_v=""

python weight_compare.py \
        --model_path ${model_path} \
        --data_path ${data_path} \
        --category ${category} \
        --tokenizer_path ${tokenizer_path} \
        --top_nums ${top_nums} \
        --top_indices_save_path_mlp ${top_indices_save_path_mlp} \
        --top_indices_save_path_q ${top_indices_save_path_q} \
        --top_indices_save_path_k ${top_indices_save_path_k} \
        --top_indices_save_path_v ${top_indices_save_path_v} \
        --prune_mlp "${prune_mlp}" \
        # --prune_attn "${prune_attn}" \

echo "weight_compare done"

attn_q_proj_path=${top_indices_save_path_q}
attn_k_proj_path=${top_indices_save_path_k}
attn_v_proj_path=${top_indices_save_path_v}
mlp_up_proj_path=${top_indices_save_path_mlp}
save_model_path=""
model_intermediate_size=${top_nums}

python mlp_prune.py \
    --model_path ${model_path} \
    --top_nums ${top_nums} \
    --attn_q_proj_path ${attn_q_proj_path} \
    --attn_k_proj_path ${attn_k_proj_path} \
    --attn_v_proj_path ${attn_v_proj_path} \
    --mlp_up_proj_path ${mlp_up_proj_path} \
    --save_model_path ${save_model_path} \
    --model_intermediate_size ${model_intermediate_size} \
    --prune_mlp_flag "${prune_mlp}" \



    