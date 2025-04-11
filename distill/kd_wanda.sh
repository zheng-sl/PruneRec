category="Video_Games"
short_name="Games"
dim_change=True
teacher_model="teacher_model_path"
train_data="train_data_path"
val_data="val_data_path"
type="kd"
batch_size=24                                                                                                               
skew_alpha=0.1
alpha=0.2

temperature=1.0



wandb_project="llm-distill"   
name="wanda_name"  
notes="notes"  

echo teacher_model: ${teacher_model}
echo train_data: ${train_data}
echo val_data: ${val_data}
echo type: ${type}
echo batch_size: ${batch_size}
echo skew_alpha: ${skew_alpha}
echo temperature: ${temperature}
echo dim_change: ${dim_change}

export WANDB_PROJECT=${}
export WANDB_API_KEY=

echo "job started at $(date)"


student_model="student_model_path"
output_dir="output_dir_path"
wandb_name="wandb_name"  

echo student_model: ${student_model}
echo output_dir: ${output_dir}
echo WANDB_NAME: ${wandb_name}

python distill.py --teacher_model_path ${teacher_model} \
    --student_model_path ${student_model} \
    --output_dir ${output_dir} \
    --type ${type} \
    --train_data ${train_data} \
    --val_data ${val_data} \
    --batch_size ${batch_size} \
    --skew_alpha ${skew_alpha} \
    --WANDB_PROJECT ${wandb_project} \
    --WANDB_NAME ${wandb_name} \
    --NOTES "${notes}" \
    --alpha ${alpha} \
    --dim_change ${dim_change}

echo "job for alpha1=${alpha1} finished at $(date)"


echo "all jobs finished at $(date)"

test_file=$(ls -f test_data_path)
info_file=$(ls -f info_data_path)
ckpt_file=$(ls -d ${output_dir})
tokenizer_file=$(ls -d ${teacher_model})
dim_change=True

echo ${test_file}
echo ${info_file}
echo ${ckpt_file}
echo "job started at $(date)"
python evaluate.py --base_model ${ckpt_file} \
    --test_data_path ${test_file} \
    --info_file ${info_file} \
    --result_json_data ${ckpt_file}/result.json \
    --category ${category} \
    --num_beams 20 \
    --tokenizer_path ${tokenizer_file} \
    --dim_change ${dim_change}
echo "job finished at $(date)"
python calc.py \
    --path ${ckpt_file}/result.json \
    --item_path ${info_file} \
    --log_dir ${ckpt_file}
echo ${ckpt_file}
