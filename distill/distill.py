import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq, Qwen2ForCausalLM
import torch.nn.functional as F
from losses import forward_kl, reverse_kl, symmetric_kl, js_distance, tv_distance
from losses import skewed_forward_kl, skewed_reverse_kl

import wandb
import json
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple
import pickle
import sys
from custom_model.custommodel import CustomQwen2ForCausalLM
import logging





class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

class VideoDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len, category):
        self.data = pd.read_csv(train_file)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.category = category
        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this."
        ]
        self.get_inputs()

    def __len__(self):
        return len(self.data)
    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response: 
{data_point["output"]}
"""
    
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response: 
{data_point["output"]}"""


    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ", \"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with several examples of the task, please combine the example and the final input to complete the  final example.

### Instruction:
{self.instructs[0]}
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        attention_mask = [1] * len(tokens)
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            # print(inputs[-1])
            
        self.inputs = inputs
    def __getitem__(self, idx):
        return self.inputs[idx]


def get_teacher_model(model_name):
    return Qwen2ForCausalLM.from_pretrained(model_name)

def get_student_model(model_name, dim_change):
    if dim_change:
        print("No")
        return CustomQwen2ForCausalLM.from_pretrained(model_name)
    else:
        print("yes")
        return Qwen2ForCausalLM.from_pretrained(model_name)
        


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=0.5, temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        student_logits = outputs.logits
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_outputs = self.teacher_model(**inputs, use_cache=False)
            teacher_logits = teacher_outputs.logits

        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)), 
            inputs["labels"].view(-1),
            ignore_index=-100
        )

        student_logits_distil = student_logits / self.temperature
        teacher_logits_distil = teacher_logits / self.temperature

        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(student_logits_distil, teacher_logits_distil, inputs, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(student_logits_distil, teacher_logits_distil, inputs, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(student_logits_distil, teacher_logits_distil, inputs)
        elif "tvd" in args.type:
            distil_loss = tv_distance(student_logits_distil, teacher_logits_distil, inputs)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(student_logits_distil, teacher_logits_distil, inputs)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(student_logits_distil, teacher_logits_distil, inputs)
        else:
            raise NotImplementedError
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * distil_loss
        wandb.log({
            "total_loss": total_loss.item(), 
            "ce_loss": ce_loss.item(), 
            "distil_loss": distil_loss.item()
        })
        return (total_loss, outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        eval_loss = eval_result.get("eval_loss", None)
        if eval_loss is not None:
            wandb.log({"eval_loss": eval_loss})
        return eval_result


def freeze_mlp_params(model, sparsity=0.5, start_layer=4, end_layer=22):
   
    for name, param in model.named_parameters():
        if "mlp" in name and "weight" in name:
            layer_number = int(name.split('.')[2])  

           
            if start_layer <= layer_number <= end_layer:
                row = int(param.size(0) * sparsity)
                col = int(param.size(1) * sparsity)

                param.data[row:, col:].requires_grad = False
        
        elif "lm_head" in name and "weight" in name:
            col = int(param.size(1) * sparsity)

            param.data[:, col:].requires_grad = False

def setup_optimizer(trainer):
    """
    设置优化器，只更新 requires_grad=True 的参数
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in trainer.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": trainer.args.weight_decay,
        },
        {
            "params": [p for n, p in trainer.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=trainer.args.learning_rate)
    trainer.optimizer = optimizer


def main(args):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = args.alpha
  
    teacher_model = get_teacher_model(args.teacher_model_path).to(device)
    student_model = get_student_model(args.student_model_path, args.dim_change).to(device)
    distill_temperature = args.temperature
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, trust_remote_code=True)
   
    with open(args.train_data, 'rb') as train_file:
        train_dataset = pickle.load(train_file)
    with open(args.val_data, 'rb') as val_file:
        val_dataset = pickle.load(val_file)

    print("Datasets loaded successfully!")

    wandb.init(project=args.WANDB_PROJECT, name=args.WANDB_NAME, notes=args.NOTES, config=args)
    
    gradient_accumulation_steps = 8

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        warmup_steps=20,
        learning_rate=3e-4,
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,
        logging_first_step=True,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        save_strategy="epoch",  
        save_total_limit=1, 
        load_best_model_at_end=True,  
        metric_for_best_model="loss",  
        greater_is_better=False, 
        
    )


    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  
        teacher_model=teacher_model,
        temperature=distill_temperature,
        alpha=alpha,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],  

    )
    
    trainer.train()
    
    student_model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    parser.add_argument('--teacher_model_path', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='teacher model name')
    parser.add_argument('--student_model_path', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='student model name')
    parser.add_argument('--type', type=str, default="kd", help='distillation type')
    parser.add_argument('--train_data', type=str, default="yahma/alpaca-cleaned", help='train data path')
    parser.add_argument('--val_data', type=str, default="yahma/alpaca-cleaned", help='val data path')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')
    parser.add_argument('--skew_alpha', type=float, default=0.1, help='skew alpha')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--WANDB_PROJECT', type=str, default="llm-distill", help='wandb project name')
    parser.add_argument('--WANDB_NAME', type=str, default="lora-alpaca", help='wandb run name')
    parser.add_argument('--NOTES', type=str, default="lora-alpaca", help='wandb run notes')
    parser.add_argument('--alpha', type=float, default=0.5, help='distill alpha')       
    parser.add_argument('--temperature', type=float, default=1, help='distill temperature')       
    parser.add_argument('--dim_change', type=bool, default=False, help='')       


    
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)