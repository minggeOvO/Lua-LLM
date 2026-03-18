import os
import time
import datetime
from functools import partial
from typing import Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer,AutoModelForCausalLM

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from datasets import IterableDataset

import warnings
warnings.filterwarnings("ignore")

# Custom modules and tools
from utils import DistributedEnv
from data import dataloader_creator, load_hf_dataset_wikitext
from models import LlamaTokenizer, PruneLlamaForCausalLM, PruneLlamaDecoderLayer
from pruning import topk_hypernetwork, collect_info_reg_llama, help_functions_hn

def main(
    exp_name: str = 'displlm',
    out_dir: str = None,
    hf_model: str = '../wanda/AI-ModelScope/Llama-2-7b-hf',
    learning_rate: float = None,
    total_n_step: int = 100000,
    start_iter: int = 0,
    batch_size: int = 1,
    use_fsdp: bool = True,
    num_workers: int = 2,
    rand_seed: int = None,
    non_hf_tokenizer_path: str = None,
    compile_flag: bool = True,
    p: float = 0.48,
    lam: float = 16.0,
    hn_block_size: int = 2048,
    hn_lr: float = 1e-3,
    min_hn_lr: float = 1e-3,
    use_sch: bool = False,
    weight_proxy: str = None,
):

    device=torch.device("cuda:0")

    # Load the tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer = hf_tokenizer
    ignored_token = tokenizer.bos_token_id

    # # Load the prunable LLaMA model and collect pruning information
    # model = PruneLlamaForCausalLM.from_pretrained(hf_model,torch_dtype=torch.float16)
    # model.config.use_cache = False
    # model.to(device)

    pruned_model = AutoModelForCausalLM.from_pretrained(
    hf_model, 
    torch_dtype=torch.float16)
    pruned_model.seqlen = 2048
    pruned_model.to(device) 
    

    # Load the hypernetwork
    # Use collect_info_reg_llama to compute pruning regularization
    # param_reg = collect_info_reg_llama(model, p=p, lam=lam)

    row_num = []
    col_num = []

    for layer in pruned_model.model.layers:
        dim = layer.self_attn.q_proj.weight.shape
        row_num.append(dim[0])
        col_num.append(dim[1])

        dim = layer.self_attn.k_proj.weight.shape
        row_num.append(dim[0])
        col_num.append(dim[1])

        dim = layer.self_attn.v_proj.weight.shape
        row_num.append(dim[0])
        col_num.append(dim[1])

        dim = layer.self_attn.o_proj.weight.shape
        row_num.append(dim[0])
        col_num.append(dim[1])

        dim = layer.mlp.gate_proj.weight.shape
        row_num.append(dim[0])
        col_num.append(dim[1])

        dim = layer.mlp.up_proj.weight.shape
        row_num.append(dim[0])
        col_num.append(dim[1])

        dim = layer.mlp.down_proj.weight.shape
        row_num.append(dim[0])
        col_num.append(dim[1])

    # Create hypernetwork for pruning
    c_dict = torch.load(weight_proxy)
    hn = topk_hypernetwork(row_num_structures=row_num, mask_structures=col_num, importance_dict=c_dict, p=p)
    hn.load_state_dict(torch.load(os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}.pt")))
    #hn_helper = help_functions_hn(param_reg.structures) 

    torch.cuda.empty_cache()

    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with autocast(device_type='cuda', dtype=data_type):
        with torch.inference_mode():
            # Generate pruning vectors using the hypernetwork
            vectors = hn.hard_output()
            #hn_helper.set_gate_vectors(model, vectors) 

    ind = 0
    
    for layer in pruned_model.model.layers:
        layer.self_attn.q_proj.weight.data *= vectors[ind].to(device)
        ind+=1

        layer.self_attn.k_proj.weight.data *= vectors[ind].to(device)
        ind+=1

        layer.self_attn.v_proj.weight.data *= vectors[ind].to(device)
        ind+=1

        layer.self_attn.o_proj.weight.data *= vectors[ind].to(device)
        ind+=1

        layer.mlp.gate_proj.weight.data *= vectors[ind].to(device)
        ind+=1

        layer.mlp.up_proj.weight.data *= vectors[ind].to(device)
        ind+=1

        layer.mlp.down_proj.weight.data *= vectors[ind].to(device)
        ind+=1

    torch.cuda.empty_cache()

    from lib.eval import eval_ppl
    ppl_test = eval_ppl(pruned_model, tokenizer)
    print(f"wikitext perplexity {ppl_test}")

    if p==0.5 or p==0.4 or p==0.3 or p==0.2 or p==0.1:
        # torch.save(pruned_model.state_dict(), os.path.join(out_dir, f"sparse_model-ckpt-{p:.2f}.pt"))
        pruned_model.save_pretrained(os.path.join(out_dir, f"sparse_model-ckpt-{p:.2f}"))
        tokenizer.save_pretrained(os.path.join(out_dir, f"sparse_model-ckpt-{p:.2f}"))


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    from jsonargparse import CLI
    CLI(main)
