#!/usr/bin/env python
# coding: utf-8

# ## Fine-tune Gemma 3 12B for Math (NuminaMath) - FIXED & SPEED OPTIMIZED
# Optimized for NERSC Perlmutter (8x A100 40GB)

import os
import sys
import warnings
import random
import numpy as np
import pandas as pd
import re
import time  # 引入时间库用于测速
from tqdm import tqdm
import subprocess 
import glob 

import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTTrainer 

# --- 1. SLURM & DDP Environment Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def setup_slurm_env():
    """自动配置 SLURM 和 DDP 环境变量"""
    if "SLURM_PROCID" in os.environ:
        try:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            
            node_list = os.environ["SLURM_NODELIST"]
            hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
            master_node = hostnames.split()[0].decode("utf-8")
            
            os.environ["MASTER_ADDR"] = master_node
            os.environ["MASTER_PORT"] = "29500" 
            
            if os.environ["RANK"] == "0":
                print(f"[DDP Setup] Master: {master_node}, World Size: {os.environ['WORLD_SIZE']}")
        except Exception as e:
            print(f"Warning: SLURM setup failed: {e}")

setup_slurm_env()

def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

SEED = 3407
set_deterministic(SEED)

# --- 2. Initialize Process Group ---
global_rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
device_string = f"cuda:{local_rank}"

if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

# --- 3. Load Model ---
GEMMA_PATH = "./gemma-3-12b-it-local" 

if local_rank == 0:
    print(f"[Rank {global_rank}] Loading model from {GEMMA_PATH}...")

# 显存紧张时，这里会有些慢，请耐心等待
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    dtype="auto", 
    device_map=device_string, 
    attn_implementation="sdpa" # 使用 PyTorch 原生加速
)
model.config.use_cache = False 
model.gradient_checkpointing_enable()
model.enable_input_require_grads() 

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
tokenizer.padding_side = "right" 
tokenizer.pad_token = tokenizer.eos_token

# --- 4. Load Dataset ---
dataset_path = "./processed_gemma_math_new"

if local_rank == 0:
    print(f"Loading dataset from: {dataset_path}")

try:
    dataset = load_from_disk(dataset_path)
    train_data = dataset["train"]      
    val_data = dataset["validation"]   
    test_data = dataset["test"]        
except Exception as e:
    if local_rank == 0:
        print(f"Error loading data: {e}")
    sys.exit(1)

# --- 5. Parallel Inference Utilities (Speed Optimized) ---
def extract_boxed_answer(text):
    if not text: return None
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None

def normalize_answer(ans):
    if not ans: return ""
    return str(ans).strip().replace(" ", "").lower()

def run_parallel_evaluation(model, tokenizer, eval_dataset, phase_name, num_samples=None, batch_size=8):
    """
    [极速版] 修复 EOS 问题 + 实时测速
    """
    if num_samples is None: num_samples = len(eval_dataset)
    
    # 1. 临时设置：开启缓存以加速推理
    old_use_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    
    # ⚠️ 关键设置：Batch 推理必须 Left Padding
    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left" 
    
    # 2. 数据切片
    subset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    sharded_subset = subset.shard(num_shards=world_size, index=global_rank)
    
    if local_rank == 0:
        print(f"\n>>> [{phase_name}] Starting Eval ({world_size} GPUs, Batch={batch_size}) <<<")

    results = []
    
    # 🚀 关键修复 1: 明确指定停止符 (EOS + End_of_Turn)
    # 必须包含 <end_of_turn>，否则 Gemma 会一直生成直到 2048 长度
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]
    
    # 3. 推理循环
    with torch.inference_mode(): 
        dataset_len = len(sharded_subset)
        iterator = range(0, dataset_len, batch_size)
        
        if local_rank == 0:
            iterator = tqdm(iterator, desc=f"GPU {global_rank} Inferencing")

        for i in iterator:
            batch_indices = range(i, min(i + batch_size, dataset_len))
            batch_data = sharded_subset.select(batch_indices)
            problems = batch_data['problem']
            true_solutions = batch_data['solution']
            
            # 构造 Prompts (Gemma Chat Template)
            formatted_prompts = []
            for p in problems:
                messages = [{"role": "user", "content": p}]
                # add_generation_prompt=True 引导模型开始回答
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                formatted_prompts.append(text)
            
            inputs = tokenizer(
                formatted_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            ).to(model.device)
            
            # ⏱️ 测速开始
            start_time = time.time()
            
            # 🚀 关键修复 2: 显式 eos_token_id
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048, # 即使设为 2048，只要 EOS 生效，也会在讲完后立即停止
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators 
            )
            
            # ⏱️ 测速结束
            end_time = time.time()
            duration = end_time - start_time
            
            # 计算实际生成的 Token 数 (用于 Debug)
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            
            # 解码
            decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # 🚀 Debug 信息：只在 Rank 0 打印一次，确认是否生效
            if i == 0 and local_rank == 0:
                gen_len = torch.sum(generated_tokens[0] != tokenizer.pad_token_id).item()
                tokens_per_sec = (gen_len * len(batch_indices)) / duration
                print(f"\n🔍 [Speed Check] Batch Time: {duration:.2f}s | Speed: {tokens_per_sec:.1f} tok/s | Sample Length: {gen_len}")
                if gen_len >= 2048:
                    print("⚠️ 警告: 样本达到了最大长度，EOS 可能未生效！")
            
            for j, model_output in enumerate(decoded_batch):
                true_sol = true_solutions[j]
                pred_boxed = extract_boxed_answer(model_output)
                true_boxed = extract_boxed_answer(true_sol)
                
                is_correct = False
                if pred_boxed and true_boxed:
                    if normalize_answer(pred_boxed) == normalize_answer(true_boxed):
                        is_correct = True
                
                results.append({
                    "true_boxed": true_boxed,
                    "pred_boxed": pred_boxed, 
                    "is_correct": is_correct
                })

    # 4. 保存与合并结果
    partial_filename = f"{phase_name}_rank{global_rank}.csv"
    pd.DataFrame(results).to_csv(partial_filename, index=False)
    dist.barrier()
    
    if global_rank == 0:
        print(f"Merging results for {phase_name}...")
        all_files = glob.glob(f"{phase_name}_rank*.csv")
        if all_files:
            combined_df = pd.concat([pd.read_csv(f) for f in all_files])
            final_filename = f"{phase_name}_final.csv"
            combined_df.to_csv(final_filename, index=False)
            acc = combined_df['is_correct'].mean()
            print(f"\n✅ [{phase_name}] Accuracy: {acc:.2%} ({combined_df['is_correct'].sum()}/{len(combined_df)})")
            print(f"Results saved to: {final_filename}")
            for f in all_files: os.remove(f)
            
    # 恢复配置
    model.config.use_cache = old_use_cache
    tokenizer.padding_side = old_pad_side
    model.train()

# --- 6. Training Config (Optimized for 40GB VRAM) ---
peft_config = LoraConfig(
    lora_alpha=128, lora_dropout=0.001, r=64, bias="none", 
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    use_dora=True
)

training_arguments = TrainingArguments(
    output_dir="logs_math_12b_16k",
    seed=SEED,
    num_train_epochs=1,
    # 🔴 显存优化: 40GB 卡建议设为 2，设 4 可能在反向传播时 OOM
    per_device_train_batch_size=2, 
    # 保持总 Batch = 128 (2 * 8卡 * 8累积 = 128)
    gradient_accumulation_steps=8, 
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    optim="adamw_torch_fused",
    learning_rate=2e-4, 
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True, 
    logging_steps=10,
    eval_strategy="steps", eval_steps=30,
    save_strategy="steps", save_steps=30,
    report_to="tensorboard"
)
training_arguments.max_seq_length = 2048

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data, 
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments
)

# ==========================================
# WORKFLOW EXECUTION
# ==========================================

# 1. Baseline Test
if dist.is_initialized(): dist.barrier()
# 🚀 Batch=8 是 A100 40G 的安全甜蜜点。
# 预期耗时: 3-5 分钟 (500 samples)
run_parallel_evaluation(model, tokenizer, test_data, "baseline_test", num_samples=500, batch_size=8)

# 2. Training
if dist.is_initialized(): dist.barrier()
if local_rank == 0: print("\n[Phase 2] Fine-tuning...")
trainer.train()

if local_rank == 0:
    print("Saving model...")
    trainer.model.save_pretrained("trained-model-math-12b")
    tokenizer.save_pretrained("trained-model-math-12b")

# 3. Final Validation
if dist.is_initialized(): dist.barrier()
torch.cuda.empty_cache()
run_parallel_evaluation(model, tokenizer, val_data, "final_validation", num_samples=500, batch_size=8)

# 4. Final Test
if dist.is_initialized(): dist.barrier()
torch.cuda.empty_cache()
run_parallel_evaluation(model, tokenizer, test_data, "final_test", num_samples=500, batch_size=8)

if local_rank == 0:
    print(f"\n[Rank {global_rank}] All Jobs Finished Successfully.")
