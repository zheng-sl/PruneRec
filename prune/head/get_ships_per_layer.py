
from tqdm import tqdm
from get_model import get_model, get_tokenizer
import warnings
warnings.filterwarnings("ignore", module="accelerate.utils.other")

from format import get_time_str, set_seed
import logging
import os
import time
import torch
import gc
import json
from accelerate import Accelerator
import accelerate
import copy
import pandas as pd
from pd_diff import kl_divergence, kl_divergence_per_token
from ships_utils import sort_ships_dict
import gc
import datetime
import datasets
from torch.utils.data import DataLoader, Dataset
import utils
import random
from collections import OrderedDict
from sortedcontainers import SortedSet
import os
import argparse
import numpy as np
from collections import defaultdict


def parse_args():

    parser = argparse.ArgumentParser(description="SHIPS")
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="./data/Video_Games.json",
        help="The path of the val data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="The path of the model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="path to save the json file",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help=""
    )
    parser.add_argument(
        "--category",
        type=str,
        default=""
    )
    parser.add_argument(
        "--nums_head2prune",
        type=int,
        default=7
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="The path of the tokenizer",
    )
    return parser.parse_args()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
    selected_data = set()
    for _ in range(nsamples):
        i = random.randint(0, len(valdata) - 1)
        while i in selected_data:
            i = random.randint(0, len(valdata) - 1)
        selected_data.add(i)
        example = valdata[i]
        
       
        input_text = example.get('input', '')
        output_text = example.get('output', '').strip()  
   
        combined_text = f"{input_text} The above contents are {category} that users have liked before and are very important for predicting the {category} that user will like. Based on the previous content, predict the next {category} that the user will like: {output_text}"

        before_output = f"{input_text} The above contents are {category} that users have liked before and are very important for predicting the {category} that user will like. Based on the previous content, predict the next {category} that the user will like: "
        

        trainenc = tokenizer(combined_text, return_tensors='pt')
        before_output_ids = tokenizer(before_output, return_tensors='pt')
        output_ids = tokenizer(output_text, return_tensors='pt')

        before_output_len = before_output_ids.input_ids.shape[1]
        output_len = output_ids.input_ids.shape[1]
        output_start = before_output_len
        output_end = before_output_len + output_len

        input_len = trainenc.input_ids.shape[1]
        if input_len < seqlen:
        
            padding_len = seqlen - input_len
       
            trainenc.input_ids = torch.cat([trainenc.input_ids, 
                                            torch.full((1, padding_len), tokenizer.pad_token_id, dtype=torch.long)], dim=1)
            trainenc.attention_mask = torch.cat([trainenc.attention_mask, 
                                                    torch.zeros((1, padding_len), dtype=torch.long)], dim=1)
        
        
        
        inp = trainenc.input_ids
        attn_mask = trainenc.attention_mask
        
        dataloader.append({
            "input_ids": inp,
            "attention_mask": attn_mask,
            "output_st": output_start,
            "output_end": output_end
        })

    return dataloader

def evaluate_ppl(
    model: torch.nn.Module, pad_token_id: int | None, testloader: DataLoader[dict[str, torch.Tensor]], silence=True
) -> float:
    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    nlls = []
    
    if not slice:
        logging.info("Calculating PPL")

    for batch in testloader:
        print(batch)


class SHIPS:
    def __init__(self, data_path, model_name,tokenizer_path, mask_cfg=None, device="cuda:0"):
        self.tokenizer, _ = get_tokenizer(tokenizer_path)
        self.model, self.accelerator = (
            get_model(model_name,get_custom=True, add_size=False))
        self.model.to(device)
        self.mask_cfg = mask_cfg
        self.data_path = data_path
        self.layers = self.model.config.num_hidden_layers
        self.heads = self.model.config.num_attention_heads

    @staticmethod
    def one_forward_pass(data, model, tokenizer, mask_cfg=None):
        # inputs = tokenizer(input_text, return_tensors='pt')
        input_ids = data['input_ids'].to(model.device)
        attn_mask = data['attention_mask'].to(model.device)
        with torch.no_grad():
            if mask_cfg is not None:
                head_mask = mask_cfg['head_mask']
                mask_type = mask_cfg['mask_type']
                scale_factor = mask_cfg['scale_factor']
            else:
                head_mask, mask_type, scale_factor = None, None, None
            output = model(input_ids, attention_mask=attn_mask,
                           head_mask=head_mask, mask_type=mask_type, scale_factor=scale_factor)
        return output.logits

    @staticmethod
    def _get_pd(logits):
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _get_last_logits(logits):
        return logits[:, -1, :]

    @staticmethod
    def _get_update_mask_cfg(can_seq, mask_cfg):

        temp_mask_cfg = copy.deepcopy(mask_cfg)
        
        if 'head_mask' not in temp_mask_cfg:
            temp_mask_cfg['head_mask'] = {}

        for layer, head in can_seq:
            temp_mask_cfg['head_mask'][(int(layer), int(head))] = mask_cfg['mask_qkv']
        
        sorted_head_mask = OrderedDict(
            sorted(temp_mask_cfg['head_mask'].items(), key=lambda x: (x[0][0], x[0][1]))
        )
        temp_mask_cfg['head_mask'] = sorted_head_mask
        return temp_mask_cfg


    def _get_base_pd(self, data, model, tokenizer):
        base_logits = self.one_forward_pass(data, model, tokenizer)
        base_pd = self._get_pd(self._get_last_logits(base_logits))
        return base_pd

    
    def ships_generate(self, input_text, top_ships, mask_num=2, top_k=5, max_new_tokens=32):
        test_mask_cfg = self.mask_cfg
        for idx, key in enumerate(top_ships.keys()):
            if idx == mask_num:
                break
            layer, head = key.split(sep='-')[0], key.split(sep='-')[1]
            test_mask_cfg = self._get_update_mask_cfg(layer, head, test_mask_cfg)
        generated_text = input_text
        cur_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        cur_ids = cur_ids.to(self.model.device)
        for _ in range(max_new_tokens):
            logits = self.one_forward_pass(generated_text, self.model, self.tokenizer, test_mask_cfg)
            softmax_logits = torch.softmax(logits[0, -1, :], dim=-1)
            topk_probs, topk_indices = torch.topk(softmax_logits, top_k)
            topk_token_ids = topk_indices.squeeze().tolist()

            chosen_token = torch.multinomial(topk_probs, num_samples=1).to(self.model.device)
            chosen_token = topk_indices[chosen_token]
            cur_ids = torch.cat([cur_ids, torch.reshape(chosen_token, (-1, 1))], dim=1)
            generated_text = self.tokenizer.decode(cur_ids.squeeze(), skip_special_tokens=True)
            if chosen_token.item() == self.tokenizer.eos_token_id:
                break
        return generated_text

    def ships_test(self, ships_res_path, mask_num=2, top_k=5, max_new_tokens=32, use_tem=False):
        ships_data = []
        with open(ships_res_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                ships_data.append(data)
        ships_res = copy.deepcopy(ships_data)
        generation_key = f"generation_mask-{mask_num}_top_k-{top_k}"
        if use_tem:
            generation_key += "-use_tem"
        with open(ships_res_path, "w+") as f:
            for idx, one_jsonl in tqdm(enumerate(ships_data)):
                for key, value in one_jsonl.items():
                    if "generation" not in key:
                        top_k_ships = value
                        if generation_key not in one_jsonl:
                            if use_tem:
                                input_text = f"## Query: {key}\n## Answer:"
                            else:
                                input_text = key
                            ships_generation = self.ships_generate(input_text, top_k_ships, mask_num, top_k, max_new_tokens)
                            ships_res[idx][generation_key] = ships_generation[len(input_text):]
                    else:
                        pass
            for idx, one_jsonl in enumerate(ships_res):
                f.write(json.dumps(one_jsonl) + '\n')
        with open(ships_res_path+"bat", "w+") as f:
            for idx, one_jsonl in enumerate(ships_res):
                f.write(json.dumps(one_jsonl) + '\n')
    

    def _get_base_pd_per_token(self, data, model, tokenizer):
        base_logits = self.one_forward_pass(data, model, tokenizer)
        base_pd = self._get_pd(base_logits)
        return base_pd

    def get_kl_loss_per_token(self, data, now_mask_cfg):
        base_pd = self._get_base_pd_per_token(data, self.model, self.tokenizer)
        logits = self.one_forward_pass(data, self.model, self.tokenizer, now_mask_cfg)
        now_pd = self._get_pd(logits)
        ships_score = kl_divergence_per_token(base_pd, now_pd)
        return ships_score
    def get_kl_loss_last_token(self, data, now_mask_cfg):
        base_pd = self._get_base_pd(data, self.model, self.tokenizer)
        logits = self.one_forward_pass(data, self.model, self.tokenizer, now_mask_cfg)
        now_pd = self._get_pd(self._get_last_logits(logits))
        ships_score = kl_divergence(base_pd, now_pd)
        return ships_score
    def get_kl_loss_only_output(self, data, now_mask_cfg):
        output_start = data['output_st']
        output_end = data['output_end']-1
        base_pd = self._get_base_pd_only_output(data, self.model, self.tokenizer, output_start, output_end) 
        logits = self.one_forward_pass(data, self.model, self.tokenizer, now_mask_cfg)
        output_logits = logits[:, output_start:output_end, :]
        now_pd = self._get_pd(output_logits)
        ships_score = kl_divergence_per_token(base_pd, now_pd)
        return ships_score
    def _get_base_pd_only_output(self, data, model, tokenizer, output_start, output_end):
        base_logits = self.one_forward_pass(data, model, tokenizer)
        base_output_logits = base_logits[:, output_start:output_end, :]
        base_pd = self._get_pd(base_output_logits)
        return base_pd

    def main(self, res_path, layers=None, heads=None, generate_flag=False, use_tem=False, beta = 2, category="games", nums_head_prune=7):
      
        test_loader = get_video_dataset(10, 42, 256, self.tokenizer, self.data_path, category = category)

        if layers is None:
            start_layer, end_layer = 0, self.layers
        else:
            start_layer, end_layer = layers[0], layers[1]
        if heads is None:
            start_head, end_head = 0, self.heads
        else:
            start_head, end_head = heads[0], heads[1]


        alpha = args.alpha
        print(f"alpha: {alpha}")
        candidate_layer_head = SortedSet(key=lambda x: (x[0], x[1]))
        prelayer_head_importance = []
        total_head_importance = []
        for layer in range(start_layer, end_layer, 1):
            current_layer_head_importance = []
            for head in range(start_head, end_head, 1):
                
                key = (layer, head)
                current_layer_head = frozenset(candidate_layer_head) | {(layer, head)}
                now_mask_cfg = self._get_update_mask_cfg(list(current_layer_head), self.mask_cfg)
                all_kl_loss = []
                
                for data in test_loader:
                    ships_score = self.get_kl_loss_last_token(data, now_mask_cfg)
                    
                    all_kl_loss.append(ships_score)
                head_loss = 1 - (sum(all_kl_loss) / len(all_kl_loss))
                current_layer_head_importance.append(head_loss)
            current_layer_head_importance = np.array([loss.cpu().numpy() for loss in current_layer_head_importance])

            min_kl_loss = np.min(current_layer_head_importance)
            max_kl_loss = np.max(current_layer_head_importance)
            head_importance_scores = (current_layer_head_importance - min_kl_loss) / (max_kl_loss - min_kl_loss)
            current_layer_head_importance = head_importance_scores 

            
            if len(prelayer_head_importance) > 0:
                prelayer_head_importance = np.array(prelayer_head_importance)
                current_layer_head_importance = prelayer_head_importance * alpha + (1 - alpha) * current_layer_head_importance
            
            prelayer_head_importance = current_layer_head_importance
            
          
            total_head_importance.append(prelayer_head_importance.tolist())
            
            num_heads_to_select = nums_head_prune
            top_heads_indices = np.argsort(current_layer_head_importance)[-num_heads_to_select:]

            for head_idx in top_heads_indices:
                candidate_layer_head.add((layer, head_idx))
        
      
        layer_to_heads = defaultdict(list)
        for layer, head in candidate_layer_head:
            layer_to_heads[layer].append(head)

        candidate_layer_head_list = []
        for layer in sorted(layer_to_heads.keys()):  
            candidate_layer_head_list.append(layer_to_heads[layer])

        candidate_layer_head_list = [[int(head) if isinstance(head, np.int64) else head for head in layer] for layer in candidate_layer_head_list]
        
        save_dir = res_path
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"alpha_{alpha:.1f}.json"), "w") as f:
            json.dump(candidate_layer_head_list, f)
        with open(os.path.join(save_dir, f"total_head_imp_alpha_{alpha:.1f}.json"), "w") as f:
            json.dump(total_head_importance, f) 

if __name__ == "__main__":

    args = parse_args()
    test_accelerator = Accelerator()
    mask_config = {
        "mask_qkv": ['q'],
        "scale_factor": 0.0001,
        "mask_type": "scale_mask",
    }

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    test = SHIPS(args.val_data_path,
                 args.model_path,
                 tokenizer_path=args.tokenizer_path,
                 mask_cfg=mask_config,
                 device=device)

    category_dict = {"Office_Products": "office products", "Digital_Music":"digital music", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies":"movie"}
    category = category_dict[args.category]
    test.main(args.save_path, generate_flag=False, beta = 2, category=category, nums_head_prune=args.nums_head2prune)
