"Getting attribution scores of the example."

import os
import sys
import logging
import argparse
import random
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm, trange

from transformers import AutoTokenizer, Qwen2ForCausalLM

import numpy as np
import torch
import pdb
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def scaled_input(emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
    # shape of emb: (num_head, seq_len, seq_len)
    if baseline is None:
        baseline = torch.zeros_like(emb)   

    num_points = batch_size * num_batch
    scale = 1.0 / num_points
    if start_i is None:
        step = (emb - baseline) * scale
        res = torch.cat([torch.add(baseline.unsqueeze(0), step*i) for i in range(num_points)], dim=0)
        return res, step[0]
    else:
        step = (emb - baseline) * scale
        start_emb = torch.add(baseline, step*start_i)
        end_emb = torch.add(baseline, step*end_i)
        step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
        res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new*i) for i in range(num_points)], dim=0)
        return res, step_new[0]


def get_video_dataset(nsamples, seed, seqlen, tokenizer, data_path, category):

    with open(data_path, 'r') as f:
        valdata = json.load(f)
    
    print("load completed")
    

    random.seed(seed)
    print(f"nsamples: {nsamples}")
    print(f"len(valdata): {len(valdata)}")
    print("*******----------------------------*******")
    print(valdata[0])
    print("*******----------------------------*******")


    dataloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(valdata) - 1)
            example = valdata[i]
            

            input_text = example.get('input', '')
            output_text = example.get('output', '').strip()  
            

            combined_text = f"{input_text} The above contents are ${category} that users have liked before and are very important for predicting the ${category} that user will like. Based on the previous content, predict the next ${category} that the user will like: {output_text}"
            
     
            trainenc = tokenizer(combined_text, return_tensors='pt')
            

            if trainenc.input_ids.shape[1] > seqlen:
                break
        

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        

        attn_mask = trainenc.attention_mask[:, i:j]
        

        dataloader.append((inp, attn_mask))
    

    valenc = None
    
    return dataloader, valenc


def get_emb_imp_with_emb_grad(model, dataloader, device, save_dir):

    input_ids_list = []
    attention_mask_list = []

    for inp, attn_mask in dataloader:
        input_ids_list.append(inp)
        attention_mask_list.append(attn_mask)


    input_ids_tensor = torch.cat(input_ids_list, dim=0) 
    attention_mask_tensor = torch.cat(attention_mask_list, dim=0)  

 
    input_ids = input_ids_tensor.to(device)
    attention_mask = attention_mask_tensor.to(device)
    input_len = int(attention_mask.sum().item())

    output = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)
    embeddings = output.hidden_states[-1]

    print(f"embeddings.shape:",{embeddings.shape})
    logits = output.logits
    pred_label = int(torch.argmax(logits[:, -1, :]))
    grad = torch.autograd.grad(logits[-1, -1, pred_label], embeddings, retain_graph=True)[0]

    grad_emb_product = grad * embeddings  
    abs_grad_emb_product = torch.abs(grad_emb_product)  


    emb_vals = torch.norm(abs_grad_emb_product, p=2, dim=(0, 1))
 
    num_to_select = emb_vals.numel() // 2 
    top_indices = torch.topk(emb_vals, num_to_select).indices.tolist()  
    



    sorted_indices = sorted(top_indices)
  
    output_data = sorted_indices
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_file = os.path.join(save_dir, "top_indices_emb*grad_l2norm_max.json")
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Top 50% indices saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",
                        default="",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_path",
                        default="",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--category",
                        default="Video Games",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--save_dir_path",
                        default="./result/emb*grad",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    args = parser.parse_args()
    args.zero_baseline = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    model = Qwen2ForCausalLM.from_pretrained(args.model_path)
    model.to(device)

    ############################################################################################################
    ### Load the evaluation data 

    category_dict = {"Office_Products": "office products", "Digital_Music":"digital music", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies":"movie"}
    category = category_dict[args.category]
    dataloader, _ = get_video_dataset(1, 42, 128, tokenizer, args.data_path, category=category)
    

    get_emb_imp_with_emb_grad(model, dataloader, device, save_dir=args.save_dir_path)

    ############################################################################################################

if __name__ == "__main__":
    main()
    