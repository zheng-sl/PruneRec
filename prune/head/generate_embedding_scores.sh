export CUDA_VISIBLE_DEVICES="7" 

category="Video_Games"
short_name="Games"
model_path=""
data_path="" 

save_dir=""

echo model_path: ${model_path}
echo data_path: ${data_path}
echo category: ${category}
echo save_dir: ${save_dir}


python generate_embedding_scores.py --model_path ${model_path} --data_path ${data_path} --category ${category} --save_dir_path ${save_dir}