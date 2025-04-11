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
import sys
sys.path.append("..")
from custom_model.custommodel import CustomQwen2ForCausalLM
import utils
import random
import os

def get_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_path",type=str, default="", help="your model directory")
    parse.add_argument("--data_path",type=str, default="", help="your head json directory")
    parse.add_argument("--top_nums",type=int, default=448, help="your top nums")
    parse.add_argument("--attn_q_proj_path",type=str, default="", help="your attn q proj json directory")
    parse.add_argument("--attn_k_proj_path",type=str, default="", help="your attn k proj json directory")
    parse.add_argument("--attn_v_proj_path",type=str, default="", help="your attn v proj json directory")
    parse.add_argument("--mlp_up_proj_path",type=str, default="", help="your mlp up proj json directory")
    parse.add_argument("--save_model_path",type=str, default="", help="your save model directory")
    parse.add_argument("--model_intermediate_size",type=int, default=448, help="your model intermediate size")
    parse.add_argument("--prune_mlp_flag", type=bool, default=False, help="your top indice save path")
    parse.add_argument("--prune_attn_flag", type=bool, default=False, help="your top indice save path")

    args = parse.parse_args()
    return args

# def get_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_parameters_sum(model):
    return sum(p.numel() for n,p in model.named_parameters() if n.find("emb")==-1)

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def find_attn_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        if name1 == 'self_attn':
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        
    return res


def find_mlp_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        if name1 == 'mlp':
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        # res.update(find_layers(
        #     child, layers=layers, name=name + '.' + name1 if name != '' else name1
        # ))
    return res

def prune_attn(model, q_mask, k_mask, v_mask):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        # print(layer)
        subset = find_attn_layers(layer)
        # print(subset)
        for name in subset:
            if name == 'self_attn.q_proj':
                weight = subset[name].weight
                mask = q_mask[i].to(weight.device)
                mask_flat = mask.flatten()
                # print(weight.size())
                filtered_weight = weight[mask_flat, :]
                new_layer = nn.Linear(filtered_weight.size(1), filtered_weight.size(0), bias=False)
                new_layer.weight.data = filtered_weight
                # new_layer.bias.data = subset[name].bias
                setattr(layer.self_attn, name.split('.')[-1], new_layer)
            elif name == 'self_attn.k_proj':
                weight = subset[name].weight
                mask = k_mask[i].to(weight.device)
                mask_flat = mask.flatten()
                #print(weight.size())
                filtered_weight = weight[mask_flat, :]
                new_layer = nn.Linear(filtered_weight.size(1), filtered_weight.size(0), bias=False)
                new_layer.weight.data = filtered_weight
                # new_layer.bias.data = subset[name].bias
                setattr(layer.self_attn, name.split('.')[-1], new_layer)
            elif name == 'self_attn.v_proj':
                weight = subset[name].weight
                mask = v_mask[i].to(weight.device)
                mask_flat = mask.flatten()
                # print(weight.size())
                filtered_weight = weight[mask_flat, :]
                new_layer = nn.Linear(filtered_weight.size(1), filtered_weight.size(0), bias=False)
                new_layer.weight.data = filtered_weight
                # new_layer.bias.data = subset[name].bias
                setattr(layer.self_attn, name.split('.')[-1], new_layer)
            elif name == 'self_attn.o_proj':
                weight = subset[name].weight
                mask = q_mask[i].to(weight.device)
                mask_flat = mask.flatten()
                filtered_weight = weight[:, mask_flat]  
                new_layer = nn.Linear(filtered_weight.size(1), filtered_weight.size(0), bias=False)
                new_layer.weight.data = filtered_weight
                # new_layer.bias.data = subset[name].bias
                setattr(layer.self_attn, name.split('.')[-1], new_layer)

def prune_mlp(model, mlp_mask):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        # print(layer)
        subset = find_layers(layer)
        # print(subset)
        for name in subset:
            if name == 'mlp.gate_proj' or name == 'mlp.up_proj':
                weight = subset[name].weight
                mask = mlp_mask[i].to(weight.device)
                mask_flat = mask.flatten()
                # print(weight.size())
                filtered_weight = weight[mask_flat, :]
                new_layer = nn.Linear(filtered_weight.size(1), filtered_weight.size(0), bias=False)
                new_layer.weight.data = filtered_weight
                # new_layer.bias.data = subset[name].bias
                setattr(layer.mlp, name.split('.')[-1], new_layer)
            elif name == 'mlp.down_proj':
                weight = subset[name].weight
                mask = mlp_mask[i].to(weight.device)
                mask_flat = mask.flatten()
                filtered_weight = weight[:, mask_flat]  
                new_layer = nn.Linear(filtered_weight.size(1), filtered_weight.size(0), bias=False)
                new_layer.weight.data = filtered_weight
                # new_layer.bias.data = subset[name].bias
                setattr(layer.mlp, name.split('.')[-1], new_layer)


def main():
    args = get_parser()
    print(f"args.prune_mlp: {args.prune_mlp_flag}")
    print(f"args.prune_attn: {args.prune_attn_flag}")
    if args.prune_attn_flag:
        
        model_path = args.model_path
        model=CustomQwen2ForCausalLM.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model =model.to(device)
        print(model)
        print(model.device)


        para_nums_before = get_parameters_sum(model)
        print(f"Original Parameters: {para_nums_before}")

        top = args.top_nums
        save_model_path = args.save_model_path
        mlp_up_proj_path = args.mlp_up_proj_path
        model_intermediate_size = top

        
        with open (args.attn_q_proj_path, 'r') as f:
            q_prune = json.load(f)  
        # print(len(mlp_prune))
        q_mask = (torch.zeros(24, 1, 896) == 1) # initialize as all False
        for i in range(len(q_prune)):
            for j in q_prune[i]:
                q_mask[i][:, j] = True
        print(q_mask.shape)
        print(len(q_prune))

        with open (args.attn_k_proj_path, 'r') as f:
            k_prune = json.load(f)  
        k_mask = (torch.zeros(24, 1, 128) == 1) # initialize as all False
        for i in range(len(k_prune)):
            for j in k_prune[i]:
                k_mask[i][:, j] = True
        print(k_mask.shape)
        print(len(k_prune))

        with open (args.attn_v_proj_path, 'r') as f:
            v_prune = json.load(f)  
        # print(len(mlp_prune))
        v_mask = (torch.zeros(24, 1, 128) == 1) # initialize as all False
        for i in range(len(v_prune)):
            for j in v_prune[i]:
                v_mask[i][:, j] = True
        print(v_mask.shape)
        print(len(v_prune))

        prune_attn(model, q_mask, k_mask, v_mask)



        with open (args.mlp_up_proj_path, 'r') as f:
            mlp_prune = json.load(f)  
        # print(len(mlp_prune))
        mlp_mask = (torch.zeros(24, 1, 4864) == 1) 
        for i in range(len(mlp_prune)):
            for j in mlp_prune[i]:
                mlp_mask[i][:, j] = True
        print(mlp_mask.shape)
        print(len(mlp_prune))

        prune_mlp(model, mlp_mask)

        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.contiguous()
        
        model.config.num_attention_heads = 7
        model.config.num_key_value_heads = 1
        model.config.hidden_size = model.config.hidden_size 
        model.config.intermediate_size = model_intermediate_size 
    else:
        args = get_parser()
        model_path = args.model_path
        model=CustomQwen2ForCausalLM.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model =model.to(device)
        print(model)
        print(model.device)


        para_nums_before = get_parameters_sum(model)
        print(f"Original Parameters: {para_nums_before}")

        top = args.top_nums
        save_model_path = args.save_model_path
        mlp_up_proj_path = args.mlp_up_proj_path
        model_intermediate_size = top

        

        with open (args.mlp_up_proj_path, 'r') as f:
            mlp_prune = json.load(f)  
        mlp_mask = (torch.zeros(24, 1, 4864) == 1) 
            for j in mlp_prune[i]:
                mlp_mask[i][:, j] = True
        print(mlp_mask.shape)
        print(len(mlp_prune))

        prune_mlp(model, mlp_mask)
        print(model)
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.contiguous()
        model.config.intermediate_size = model_intermediate_size 
    
if __name__ == "__main__":
    main()
