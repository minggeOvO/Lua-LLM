# Lua-LLM: Learning Unstructured-Sparsity Allocation for Large Language Models



We propose Lua-LLM (**L**earning **u**nstructured-sparsity **a**llocation in LLMs), a learning-based global pruning framework that explores the optimal unstructured sparsity allocation. Unlike existing pruning methods, which primarily focus on allocating per-layer sparsity, Lua-LLM achieves flexible allocation for both layer-wise and intra-layer sparsity. Furthermore, Lua-LLM leverages a soft Top-K operator to approximate the importance-based mask selection mechanism, enabling efficient binary mask learning. Experimental results on LLaMA and OPT families demonstrate significant performance improvements over existing methods.

## Requirements

1. The environment is based on torch 2.0.0 and transformers 4.44.0. For detailed package requirement information, see `environment.yml`.
2. Download the huggingface models and datasets into directory `./model` and `./datasets`, respectively. 

## Run Lua-LLM

### 1. Initialization: Preparing weight importance metric

```
cd $Lua_REPO/wanda
python main.py --model llama2-7b --local_dir ../model/Llama-2-7b --prune_method score --sparsity_type unstructured
```

### 2. Learning unstructured sparsity allocation
Prune LLaMA2-7B model at 60% sparsity level.
```
torchrun --nproc_per_node=2 lua_llama.py \
    --hf_model ../model/Llama-2-7b \
    --use_fsdp True \
    --p 0.4 \
    --lam 16.0 \
    --batch_size 1 \
    --total_n_step 500 \
    --hn_lr 1e-4 \
    --use_bf16 True \
    --out_dir ../output/llama2-7b \
    --weight_proxy "../wanda/llama2-7b-metric.pt"

python load_eval_llama.py \
    --hf_model ../model/Llama-2-7b \
    --p 0.4 \
    --out_dir ../output/llama2-7b \
    --weight_proxy "../wanda/llama2-7b-metric.pt"
```


## Acknowledgement

This repository is build upon the [Wanda](https://github.com/locuslab/wanda) and [DISP-LLM](https://github.com/ZhengaoLi/DISP-LLM-Dimension-Independent-Structural-Pruning) repositories.
