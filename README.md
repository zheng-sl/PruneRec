# PruneRec
Our code is based on BIGRec (https://github.com/SAI990323/Grounding4Rec) and BlockPruner(https://github.com/MrGGLS/BlockPruner)
# Quick Start
## Setup 
### Installation
``` pip install -r requirements.txt```

## Step-by-step Instructions
Our code consists of the following steps:
- Prune heads: determine a list of heads to prune based on their importance.
- Get a sparse matrix for the embedding layer to align it with the MHA layer, prune heads and the embedding layer, and then recover via distillation.
- Obtain a sparse matrix for the MLP layer based on the importance of the intermediate dimensions， prune the MLP layer, and recover through distillation.
- Determine layers to prune based on the PPL metric, prune layers, and recover using distillation.
  we finetune LLM and evaluate the pruned model using BigRec's evaluation code.

1. get pruning head list
    ```bash get_prune_head_list.sh```
3. get embedding score
   ```bash ./prune/head/generate_embedding_scores.sh```
5. pruning head and embedding layer
    ``` bash ./prune/head/prune_head.sh ```
6. pruning mlp layer
   ``` bash ./prune/mlp_attn/mlp_attn_run.sh ```
7. pruning layer
   ``` bash ./prune/layer/layer_seqrch.sh ```
8. recovery
   ``` bash ./distill/kd.sh ```


