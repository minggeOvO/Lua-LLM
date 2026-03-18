for p in 0.1; do
    torchrun --master_port 12323 --nproc_per_node=2 train_hypernetwork_llama.py \
        --hf_model ../model/llama-7b \
        --use_fsdp True \
        --p "$p" \
        --lam 16.0 \
        --batch_size 1 \
        --total_n_step 500 \
        --hn_lr 5e-3 \
        --min_hn_lr 1e-4 \
        --use_sch False \
        --use_bf16 True \
        --out_dir ../output_500/llama-7b \
        --exp_name "PruneLlama" \
        --weight_proxy "../llama-7b.pt"
done

for p in 0.1; do
    torchrun --master_port 12323 --nproc_per_node=2 train_hypernetwork_llama.py \
        --hf_model ../model/Llama-2-7b \
        --use_fsdp True \
        --p "$p" \
        --lam 16.0 \
        --batch_size 1 \
        --total_n_step 500 \
        --hn_lr 5e-3 \
        --min_hn_lr 1e-4 \
        --use_sch False \
        --use_bf16 True \
        --out_dir ../output_500/llama2-7b \
        --exp_name "PruneLlama" \
        --weight_proxy "../llama2-7b.pt"
done