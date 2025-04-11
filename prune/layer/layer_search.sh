export CUDA_VISIBLE_DEVICES="2" 
dim_change=True
short_name="Games"
model_name=""
tokenizer_path=""

nsamples=32
dataset=alpaca
block_num=20
block_type=mix
save_model=""
ppl_seq_len=256
log_dir=${save_model}

echo dim_change: ${dim_change}
python layer_search.py \
        --model-path ${model_name}\
        --tokenizer_path ${tokenizer_path} \
        --save-model-path ${save_model} \
        --block-type ${block_type} \
        --cal-nsamples ${nsamples} \
        --del-block-num ${block_num} \
        --cal-dataset ${dataset} \
        --ppl-search-path ppls \
        --ppl-eval-batch-size 2 \
        --device cuda \
        --ppl-eval-seqlen ${ppl_seq_len} \
        --log_dir ${log_dir} \
        --dim_change ${dim_change} \
        
