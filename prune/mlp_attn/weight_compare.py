from typing import Optional, Tuple
import argparse
import json
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer
from transformers import set_seed; set_seed(42)
import utils
import random
import os
from custom_model.custommodel import CustomQwen2ForCausalLM
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer
from collections import Counter
import argparse
# os.environ['CUDA_VISIBLE_DEVICES']='6'


def get_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_path",type=str, default="", help="your model directory")
    parse.add_argument("--tokenizer_path",type=str, default="", help="your tokenizer directory")
    parse.add_argument("--category",type=str, default="", help="your category")
    parse.add_argument("--data_path",type=str, default="", help="your data directory")
    parse.add_argument("--top_nums",type=int, default=448, help="your top")
    parse.add_argument("--top_indices_save_path_mlp", type=str, default="", help="your top indice save path")
    parse.add_argument("--top_indices_save_path_q", type=str, default="", help="your top indice save path")
    parse.add_argument("--top_indices_save_path_k", type=str, default="", help="your top indice save path")
    parse.add_argument("--top_indices_save_path_v", type=str, default="", help="your top indice save path")
    parse.add_argument("--prune_mlp", type=bool, default=False, help="your top indice save path")
    parse.add_argument("--prune_attn", type=bool, default=False, help="your top indice save path")

    args = parse.parse_args()
    return args

def get_video_dataset(nsamples, seed, seqlen, tokenizer, data_path, category="video"):

    with open(data_path, 'r') as f:
        valdata = json.load(f)
    

    

    random.seed(seed)
    print(f"nsamples: {nsamples}")
    print(f"len(valdata): {len(valdata)}")
    print("*******----------------------------*******")
    print(valdata[0])
    print("*******----------------------------*******")


    dataloader = []
    selected = set()
    for _ in range(nsamples):
        i = random.randint(0, len(valdata) - 1)
        while i in selected:
            i = random.randint(0, len(valdata) - 1)
        selected.add(i)
        example = valdata[i]
        

        input_text = example.get('input', '')
        output_text = example.get('output', '').strip()  
        

        combined_text = f"{input_text} The above contents are ${category} that users have liked before and are very important for predicting the ${category} that user will like. Based on the previous content, predict the next ${category} that the user will like: {output_text}"
        before_output = f"{input_text} The above contents are ${category} that users have liked before and are very important for predicting the ${category} that user will like. Based on the previous content, predict the next ${category} that the user will like: "

  
        trainenc = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=False)
        input_ids = tokenizer(before_output, return_tensors='pt', truncation=True, padding=False)
        output_ids = tokenizer(output_text, return_tensors='pt', truncation=True, padding=False)

        real_seqlen = trainenc.input_ids.shape[1]
        input_len = input_ids.input_ids.shape[1]
        output_len = output_ids.input_ids.shape[1]


        if real_seqlen < seqlen:

            padding_length = seqlen - real_seqlen
            inp = trainenc.input_ids[:, :real_seqlen]
            inp = torch.cat([inp, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
        else:
    
            inp = trainenc.input_ids[:, :seqlen]
            real_seqlen = seqlen

        
        dataloader.append((inp, real_seqlen, input_len, output_len))
    

    return dataloader

def save_to_json(data, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    import numpy as np
    # Convert numpy arrays to lists
    if isinstance(data, np.ndarray):
        data_list = data.tolist()  
    elif isinstance(data, list):
        data_list = data
    with open(filename, 'w') as f:
        json.dump(data_list, f)

def get_attn_top_indices(rec_model, tokenizer, args, device):
        
    rec_layer_outputs = {}
    q_proj_save_indices = []
    k_proj_save_indices = []
    v_proj_save_indices = []


    def rec_make_q_proj_hook(layer_idx):
        def rec_q_proj_hook(module, module_input, module_output):
            print(f"[Q_Proj Hook] Called for layer {layer_idx}")


            half_dim = module_output.shape[-1] // 2
            print(f"half_dim of q_proj: {half_dim}")
            top_result = module_output.topk(half_dim, dim=-1)
            token_top_values = top_result.values
            token_top_indices = top_result.indices

            flattened_indices = token_top_indices.view(-1).cpu().detach().numpy()
            counter = Counter(flattened_indices)
            most_common = counter.most_common(half_dim)
            save_indices = [int(item[0]) for item in most_common]
            print(f"[Q_Proj Hook] layer {layer_idx}, save_indices: {save_indices}")

            rec_layer_outputs[layer_idx] = save_indices
            q_proj_save_indices.append(save_indices)
            print(f"[Q_Proj Hook] len of all_save_indices: {len(q_proj_save_indices)}")
        return rec_q_proj_hook

    def rec_make_k_proj_hook(layer_idx):
        def rec_k_proj_hook(module, module_input, module_output):
            print(f"[K_Proj Hook] Called for layer {layer_idx}")
            half_dim = module_output.shape[-1] // 2
            print(f"half_dim of k_proj: {half_dim}")
            top_result = module_output.topk(half_dim, dim=-1)
            token_top_values = top_result.values
            token_top_indices = top_result.indices

            flattened_indices = token_top_indices.view(-1).cpu().detach().numpy()
            counter = Counter(flattened_indices)
            most_common = counter.most_common(half_dim)
            save_indices = [int(item[0]) for item in most_common]
            print(f"[K_Proj Hook] layer {layer_idx}, save_indices: {save_indices}")

            rec_layer_outputs[layer_idx] = save_indices
            k_proj_save_indices.append(save_indices)
            print(f"[K_Proj Hook] len of all_save_indices: {len(k_proj_save_indices)}")
        return rec_k_proj_hook

    def rec_make_v_proj_hook(layer_idx):
        def rec_v_proj_hook(module, module_input, module_output):
            print(f"[V_Proj Hook] Called for layer {layer_idx}")
            half_dim = module_output.shape[-1] // 2
            print(f"half_dim of v_proj: {half_dim}")

            top_result = module_output.topk(half_dim, dim=-1)
            token_top_values = top_result.values
            token_top_indices = top_result.indices

            flattened_indices = token_top_indices.view(-1).cpu().detach().numpy()
            counter = Counter(flattened_indices)
            most_common = counter.most_common(half_dim)
            save_indices = [int(item[0]) for item in most_common]
            print(f"[V_Proj Hook] layer {layer_idx}, save_indices: {save_indices}")

            rec_layer_outputs[layer_idx] = save_indices
            v_proj_save_indices.append(save_indices)
            print(f"[V_Proj Hook] len of all_save_indices: {len(v_proj_save_indices)}")
        return rec_v_proj_hook


    q_layer_idx = 0
    k_layer_idx = 0
    v_layer_idx = 0
    # o_layer_idx = 0

    for name, module in rec_model.model.named_modules():
        if "q_proj" in name:
            module.register_forward_hook(rec_make_q_proj_hook(q_layer_idx))
            q_layer_idx += 1
        if "k_proj" in name:
            module.register_forward_hook(rec_make_k_proj_hook(k_layer_idx))
            k_layer_idx += 1
        if "v_proj" in name:
            module.register_forward_hook(rec_make_v_proj_hook(v_layer_idx))
            v_layer_idx += 1


    category_dict = {"Office_Products": "office products","Digital_Music":"digital music", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies":"movie", "Movie_Lens": "Movie_Lens"}

    category = category_dict[args.category]
    eval_dataloader = get_video_dataset(1, 42, 100, tokenizer, args.data_path, category=category)

    for i, (inp, real_seqlen, input_len, output_len) in enumerate(eval_dataloader):
        print(f"inp: {inp}")
        print(f"real_seqlen: {real_seqlen}")
        print(f"input_len: {input_len}")
        print(f"output_len: {output_len}")
        print("*******----------------------------*******")
        with torch.no_grad():
            rec_outputs = rec_model(input_ids=inp.to(device))
        

        save_to_json(q_proj_save_indices, args.top_indices_save_path_q)
        save_to_json(k_proj_save_indices, args.top_indices_save_path_k)
        save_to_json(v_proj_save_indices, args.top_indices_save_path_v)



def main():
    
    args = get_parser()
    print(f"args.prune_mlp: {args.prune_mlp}")
    print(f"args.prune_attn: {args.prune_attn}")
    rec_model = CustomQwen2ForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    rec_model = rec_model.to(device)
    save_top_nums = args.top_nums
    rec_layer_outputs = {}
    all_save_indices = []
    if args.prune_mlp:
        def rec_make_up_proj_hook(layer_idx):
            def rec_up_proj_hook(module, module_input, module_output):
                print(f"Hook called for layer {layer_idx}")
      
                top_result = module_output.topk(save_top_nums, dim=-1)
                token_10_values = top_result.values
                token_10_indices = top_result.indices  

               
                flattened_indices = token_10_indices.view(-1).cpu().detach().numpy()
                counter = Counter(flattened_indices)

                most_common = counter.most_common(save_top_nums)
                save_indices = [int(item[0]) for item in most_common]

                top_10_sum = token_10_values.sum(dim=-1).cpu().detach().numpy()
                total_sum = module_output.sum(dim=-1).cpu().detach().numpy()
                percentage = top_10_sum / total_sum
                bottom_10_sum = module_output.topk(save_top_nums, dim=-1, largest=False)[0].sum(dim=-1).cpu().detach().numpy()

       
                rec_layer_outputs[layer_idx] = save_indices
           
                all_save_indices.append(save_indices)
                print(f"len of all_save_indices: {len(all_save_indices)}")

            return rec_up_proj_hook
        layer_idx = 0
        for name, module in rec_model.model.named_modules():
            if "up_proj" in name:
                module.register_forward_hook(rec_make_up_proj_hook(layer_idx))
                layer_idx += 1
        category_dict = {"Office_Products": "office products","Digital_Music":"digital music", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies":"movie", "Movie_Lens": "Movie_Lens"}
        category = category_dict[args.category]
        eval_dataloader = get_video_dataset(1, 42, 100, tokenizer, args.data_path, category=category)

        for i, (inp, real_seqlen, input_len, output_len) in enumerate(eval_dataloader):
            print(f"inp: {inp}")
            print(f"real_seqlen: {real_seqlen}")
            print(f"input_len: {input_len}")
            print(f"output_len: {output_len}")
            print("*******----------------------------*******")
            with torch.no_grad():
                
                rec_outputs = rec_model(input_ids=inp.to(device))
            
            save_to_json(all_save_indices, args.top_indices_save_path_mlp)
    if args.prune_attn:
        get_attn_top_indices(rec_model, tokenizer, args, device)




if __name__ == "__main__" :
    main()
