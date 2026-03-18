DEBUG = False

if DEBUG:
    from torchviz import make_dot

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
from transformers import AutoTokenizer

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
from models import LlamaTokenizer, PruneLlamaForCausalLM, PruneLlamaDecoderLayer, MaskedLinear, MaskedLinearGQA


def round_to_block_size(current_rank, block_size=32):
    """Round `current_rank` down to the nearest multiple of `block_size`."""
    round_rank = max(block_size, (current_rank // block_size) * block_size)
    return round_rank


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
    use_bf16: bool = False,
    weight_proxy: str = None,
    mask_init_value: float = 0.5,
):

    # Initialize the distributed environment
    env = DistributedEnv()
    print(env)

    dist.init_process_group(
        backend="nccl",
        rank=env.global_rank,
        world_size=env.world_size,
        timeout=datetime.timedelta(seconds=3600 * 5),
    )

    # Use bf16 if supported, otherwise fallback to fp16
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Prepare output directory, random seed, and learning rate
    if out_dir is None:
        user_name = 'l'
        dateTimeObj = datetime.datetime.now()
        out_dir = os.path.join('output/', user_name, exp_name)

    if rand_seed is None:
        rand_seed = start_iter

    # Automatically calculate learning rate if not provided
    if learning_rate is None:
        llama_learning_rate_per_sample = 0.0003 / (4 * 1024 * 1024)
        learning_rate = min(
            llama_learning_rate_per_sample * batch_size * 4096 * env.world_size,
            0.0003
        )

    if env.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Set the current GPU
    device_id = env.local_rank
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Load the tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer = hf_tokenizer
    if non_hf_tokenizer_path:
        env.print_master('Using non_hf_tokenizer ...')
        tokenizer = LlamaTokenizer(non_hf_tokenizer_path, output_type='list')
    ignored_token = tokenizer.bos_token_id

    # Load the prunable LLaMA model and collect pruning information
    model = PruneLlamaForCausalLM.from_pretrained(hf_model,torch_dtype=torch.float16)
    model.config.use_cache = False
    env.print_master(model.config)
    print(model)
    
    # Load dataset
    tic = time.time()
    result_dataset = load_hf_dataset_wikitext('train', env.world_size * num_workers)

    train_dataloader_hn = dataloader_creator(
        dataset=result_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        block_size=hn_block_size,
        num_workers=num_workers,
        cycling=False,
        rank=env.global_rank,
        world_size=env.world_size,
        ignored_token=ignored_token,
    )
    toc = time.time() - tic
    env.print(f"Initializing training dataset - done. Time: {toc:.2f}s")

    model.load_and_apply_importance_scores(weight_proxy)

    for module in model.modules():
        if isinstance(module, (MaskedLinear, MaskedLinearGQA)):
            # Reset mask logits with the specified initial value
            torch.nn.init.constant_(module.mask_logits, mask_init_value)

    model.to(device_id)

    # Wrap with FSDP if enabled
    if use_fsdp:
        my_auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={PruneLlamaDecoderLayer}
        )
        if use_bf16:
            model = model.to(data_type).to(device_id)
            model = FSDP(
                model, 
                auto_wrap_policy=my_auto_wrap_policy,
                use_orig_params=True
                )
        else:
            model = FSDP(
                model,
                auto_wrap_policy=my_auto_wrap_policy,
                use_orig_params=True,
                mixed_precision=MixedPrecision(
                    param_dtype=data_type,
                    reduce_dtype=data_type,
                    buffer_dtype=data_type
                ),
            )
    else:
        if use_bf16:
            model = model.to(data_type).to(device_id)
        model = DDP(model)

    # Enable torch.compile
    if compile_flag:
        model = torch.compile(model)

    # Train hypernetwork
    train_hn(
        env,
        model,
        train_hn_data=train_dataloader_hn,
        ignored_token=ignored_token,
        max_iter=total_n_step,
        out_dir=out_dir,
        p=p,
        lam=lam,
        hn_block_size=hn_block_size,
        hn_lr=hn_lr,
        min_hn_lr=min_hn_lr,
        use_sch=use_sch,
        use_fsdp=use_fsdp,
    )
    
def train_hn(
    env: DistributedEnv,
    model: torch.nn.Module,
    train_hn_data: IterableDataset,
    start_iter=0,
    ignored_token=-1,
    log_interval=50,
    max_iter=100000,
    out_dir=None,
    p=None,
    lam=16.0,
    hn_block_size=2048,
    hn_lr=1e-3,
    min_hn_lr=1e-3,
    use_sch=False,
    use_fsdp=False,
):
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_id = env.local_rank
    iter_num = start_iter

    # Select the appropriate GradScaler (ShardedGradScaler for FSDP)
    if use_fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
    else:
        scaler = GradScaler()

    # Enable learning rate scheduling if configured
    mask_params = []
    for name, param in model.named_parameters():
        if 'mask_logits' in name:
            mask_params.append(param)
            param.requires_grad = True  # Ensure mask parameters are trainable
        else:
            param.requires_grad = False  # Freeze all regular parameters

    print("mask params",mask_params)
    optimizer = torch.optim.AdamW(mask_params, lr=hn_lr, weight_decay=0.05)
    if use_sch:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iter,
            eta_min=min_hn_lr,
            last_epoch=iter_num - 1
        )

    tic = time.time()
    torch.cuda.empty_cache()

    sum_ori_params = model.get_full_parameter_count()

    while True:
        for batch in train_hn_data:
            if iter_num >= max_iter:
                break

            with torch.no_grad():
                input_ids = batch['input_ids'].to(device_id)
                targets = batch['labels'].to(device_id)
                input_ids = input_ids[:, :hn_block_size]
                targets = targets[:, :hn_block_size]

            with autocast(device_type='cuda', dtype=data_type):
                # Forward pass
                output = model(input_ids)
                logits = output.logits if hasattr(output, 'logits') else output
                ce_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=ignored_token
                )
                sum_params = 0
                masked_modules = model.get_masked_linear_modules()
                for _, m in masked_modules:
                    sum_params += m.get_parameter_count()

                param_ratio = sum_params / sum_ori_params
                if param_ratio > p:
                    clamped_p_ratio = torch.clamp(param_ratio, min=p)
                    loss = torch.log(clamped_p_ratio / p)
                else:
                    clamped_p_ratio = torch.clamp(param_ratio, max=p)
                    loss = torch.log(p / clamped_p_ratio)

                reg_loss = lam * loss
                
                total_loss = ce_loss + reg_loss

            # Check for NaN loss values
            if torch.isnan(total_loss):
                env.print_master("!!! nan loss detected !!!")
                total_loss.fill_(0)

            print("ce loss: %.4f" % ce_loss, "reg_loss: %.4f" % reg_loss)

            # Backward pass and optimizer step
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if use_sch:
                scheduler.step()

            if iter_num % log_interval == 0:
                elapsed = time.time() - tic
                env.print_master(
                    f"Iter {iter_num}/{max_iter}, "
                    f"loss={total_loss.item():.4f}, "
                    f"reg={reg_loss.item():.4f}, "
                    f"time={elapsed*1000:.2f}ms"
                )
                tic = time.time()

            iter_num += 1
            if iter_num >= max_iter:
                break
        if iter_num >= max_iter:
            break

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    from jsonargparse import CLI
    CLI(main)
