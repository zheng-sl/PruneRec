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
# from custom_model.custommodel_1 import CustomQwen2ForCausalLM
from custom_model.custommodel import CustomQwen2ForCausalLM

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="bf16",
        help="Data type for computation ('bf16', 'fp32', 'fp64').",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to load the tokenizer",
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        default=None,
        help="Path to save the pruned model and tokenizer",
    )
    parser.add_argument(
        "--ppl-search-path",
        type=str,
        help="Path to save the perplexity search results.",
        default="ppls",
    )
    parser.add_argument(
        "--del-block-num",
        type=int,
        help="Number of blocks to delete.",
        default=0,
    )
    parser.add_argument(
        "--block-type",
        type=str,
        help="Block type for searching ('mha', 'mlp', 'mix').",
        choices=["mha", "mlp", "mix"],
        default="mix",
    )
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset for calibration.",
        choices=["wikitext2", "alpaca"],
        default="alpaca",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples for calibration.",
        default=128,
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save the log file.")
    parser.add_argument("--dim_change", type=bool, default=False, help="Whether to change the hidden size of the model.")
    return parser.parse_args()


class MaskedLlamaDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = None
        self.mlp = None
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.mask_block = ""

    def setting_layer(self, layer):
        if "mha" not in self.mask_block:
            self.input_layernorm = layer.input_layernorm
            self.self_attn = layer.self_attn
        else:
            self.input_layernorm = None
            self.self_attn = None
        if "mlp" not in self.mask_block:
            self.post_attention_layernorm = layer.post_attention_layernorm
            self.mlp = layer.mlp
        else:
            self.post_attention_layernorm = None
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        if "mha" not in self.mask_block:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual.to(hidden_states.device) + hidden_states
        else:
            self_attn_weights = None
            present_key_value = None

        if "mlp" not in self.mask_block:
        # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual.to(hidden_states.device) + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def get_model_params(model):
    return sum(int(p.nelement()) for p in model.parameters())
def get_model_params_wo_embedding(model):
    return sum(p.numel() for n,p in model.named_parameters() if n.find("emb")==-1)


@torch.no_grad
def block_search_by_ppl(args, model, test_loader=None, model_size=None, tokenizer=None):
    # Initialize best results dictionary
    best_results = {}
   

    # Split blocks into MHA and MLP lists
    mha_block_ids = list(range(model.config.num_hidden_layers)) if args.block_type != "mlp" else [] # You can use BI to reduce the search space if needed
    mlp_block_ids = list(range(model.config.num_hidden_layers)) if args.block_type != "mha" else []

    logging.info(f"mha_block_ids: {mha_block_ids}")
    logging.info(f"mlp_block_ids: {mlp_block_ids}")

    # iterate search process
    current_sequence = set()
    current_ppl = float('inf')
    # import pdb;
    # pdb.set_trace()
    pbar = tqdm(range(1, args.del_block_num+1), desc=f"searching block del order based on {args.cal_dataset} ppl")
    for del_num in pbar:
        best_candidate = None
        best_candidate_ppl = float('inf')

        candidate_layers = [layer_id for layer_id in range(len(model.model.layers)) if layer_id not in current_sequence]

        for layer_id in candidate_layers:
            candidate_sequence = frozenset(current_sequence) | {layer_id}
            del_layer_dict = apply_block_masks(model, candidate_sequence)
            candidate_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
            revert_block_masks(model, del_layer_dict)

            if candidate_ppl < best_candidate_ppl:
                best_candidate_ppl = candidate_ppl
                best_candidate = candidate_sequence

        if best_candidate is not None:
            current_sequence = best_candidate
            current_ppl = best_candidate_ppl

        del_order_list = list(current_sequence)
        best_results[str(del_num)] = sorted(del_order_list, reverse=False)

        print(f"best_ppl: {current_ppl}")
        # print(f"best_seq ({del_num}): {sorted(del_order_list, key=lambda x: x[1], reverse=False)}")
    file_dir = args.ppl_search_path
    os.makedirs(args.ppl_search_path, exist_ok=True)
    file_name = f"{args.ppl_search_path}/{args.model_path.split('/')[-1]}_{args.block_type}_{args.cal_dataset}_ns_{args.cal_nsamples}_del_order_list.json"

    with open(file_name, "w") as f:
        json.dump(best_results, f)
    # import pdb;
    # pdb.set_trace()
    logging.info(f"del_order_list path: {file_name}")
    print(f"prune_list:{best_results[str(args.del_block_num)]}")
    remove_redundant_layers(args, model, best_results)
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    logging.info(f"model saved to {args.save_model_path}")
    logging.info(f"pruned_model: {model}")
    pruned_model = get_model_params_wo_embedding(model)
    logging.info(f"pruned model size: {pruned_model/1e9:.3f}B")
    

@torch.no_grad
def remove_redundant_layers(args, model, best_results):
    prune_list = best_results[str(args.del_block_num)]
    logging.info(f"chosen del_block_list_before: {prune_list}")
    # prune_list = [11,12,13,14,15,16,17,18,19,20,21,22]
    logging.info(f"chosen del_block_list_after: {prune_list}")
    layers = model.model.layers
    sorted_list = sorted(prune_list, reverse=True) 
    prune_layer_num = len(sorted_list)
    first_del_layer = min(sorted_list)
    
    # remove layers from back to front
    for prune_layer in sorted_list:
        del(layers[prune_layer])

    sorted_list = sorted(prune_list) 

    for j in range(0, prune_layer_num):
        start = sorted_list[j] - j
        for i in range(start, model.config.num_hidden_layers - prune_layer_num):
            model.model.layers[i].self_attn.layer_idx = model.model.layers[i].self_attn.layer_idx - 1
    
    # Update model config
    model.config.num_hidden_layers -= prune_layer_num
    logging.info(f"Updated model config: num_hidden_layers={model.config.num_hidden_layers}")
    

def apply_block_masks(model, seq):
    del_layer_dict = {}
    for layer_id in seq:
        chosen_layer = model.model.layers[layer_id]
        if isinstance(chosen_layer, MaskedLlamaDecoderLayer):
            # chosen_layer.mask_block += 'mha'
            # chosen_layer.mask_block += 'mlp'
            chosen_layer.setting_layer(del_layer_dict[str(layer_id)])
        else:
            new_layer = MaskedLlamaDecoderLayer()
            new_layer.mask_block += 'mha'
            new_layer.mask_block += 'mlp'
            new_layer.setting_layer(chosen_layer)
            del_layer_dict[str(layer_id)] = chosen_layer
            model.model.layers[layer_id] = new_layer
    return del_layer_dict


def revert_block_masks(model, del_layer_dict):
    for k, v in del_layer_dict.items():
        layer_id = int(k)
        model.model.layers[layer_id] = v

def get_video_dataset(nsamples, seed, seqlen, tokenizer):
    with open('', 'r') as f:
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
            output_text = example.get('output', '').strip()  # 去除多余的换行符和空白
            
            combined_text = f"{input_text} The above contents are games that users have liked before and are very important for predicting the games that user will like. Based on the previous content, predict the next game that the user will like: {output_text}"
            
            
            trainenc = tokenizer(combined_text, return_tensors='pt')
            
            
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attn_mask = trainenc.attention_mask[:, i:j]
        
        dataloader.append({
            "input_ids": inp,
            "attention_mask": attn_mask
        })

    return dataloader


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_file = f"{args.log_dir}/layer_search.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ])
    
    logging.info(args)
    logging.info(f"PyTorch device: {args.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    if args.compute_dtype == "bf16":
        compute_dtype = torch.bfloat16
    elif args.compute_dtype == "fp32":
        compute_dtype = torch.float32
    elif args.compute_dtype == "fp64":
        compute_dtype = torch.float64
    else:
        raise NotImplementedError("Unsupported compute type.")
    if not args.dim_change:
        print("False")
        model = Qwen2ForCausalLM.from_pretrained(args.model_path, torch_dtype=compute_dtype, trust_remote_code=True, device_map="auto", use_cache=False)
    else:
        print("True")
        model = CustomQwen2ForCausalLM.from_pretrained(args.model_path, torch_dtype=compute_dtype, trust_remote_code=True, device_map="auto", use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    model_size = get_model_params_wo_embedding(model)

    logging.info(f"original model size: {model_size}")
    print(f"original model size: {model_size}")

    dataset = utils.get_dataset(args.cal_dataset)
    test_dataset = dataset["test"]  
    sampled_test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), args.cal_nsamples))
    test_loader = utils.prepare_test_dataloader(
        dataset=sampled_test_dataset, 
        tokenizer=tokenizer, 
        seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size
    )

    
    block_search_by_ppl(args, model, test_loader, model_size, tokenizer)
    logging.info("Block search finished.")
    model_size = get_model_params_wo_embedding(model)

    logging.info(f"after pruned model size: {model_size}")
    print(f"after pruned model size: {model_size}")


if __name__ == "__main__":
    main()
