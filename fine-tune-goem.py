#!/usr/bin/env python
# coding: utf-8

# ## Fine-tune Gemma 3 12B - DDP Speedup Version (Complete)
# Strategy: Manual DDP Init + SLURM Bridge + SFTTrainer Reuse
# Includes: Loss Plotting, Multi-label Metrics, Confusion Heatmap

import os
import sys
import warnings
import random
import time
import json
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # Added for heatmap

import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# --- 1. CRITICAL SLURM & DDP SETUP ---
if "SLURM_PROCID" in os.environ:
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    
    try:
        if "MASTER_ADDR" not in os.environ:
            cmd = "scontrol show hostnames " + os.environ["SLURM_JOB_NODELIST"]
            stdout = subprocess.check_output(cmd.split())
            master_node = stdout.decode().splitlines()[0]
            os.environ["MASTER_ADDR"] = master_node
            os.environ["MASTER_PORT"] = "29500"
    except Exception as e:
        pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

SEED = 42
set_deterministic(SEED)

global_rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
device_string = f"cuda:{local_rank}"

torch.cuda.set_device(local_rank)
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
    if global_rank == 0:
        print(f" DDP Initialized. World Size: {world_size}")
        print(f" Master Node: {os.environ.get('MASTER_ADDR')}")

# --- 2. Load Model ---
# *** Model Configuration ***
# Use the official Hugging Face ID so others can reproduce the work.
# If running on NERSC/Offline, you can change this to your local path.
GEMMA_PATH = "google/gemma-3-12b-it"

if global_rank == 0:
    print(f"Loading model from {GEMMA_PATH}...")

model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    dtype="auto", 
    device_map=device_string, 
    attn_implementation="sdpa",
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads() 

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
model.config.use_cache = False 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
EOS_TOKEN = tokenizer.eos_token

# --- 3. Prepare Dataset ---

train_file = "train_cleaned.jsonl"
val_file = "dev_cleaned.jsonl"
test_file = "test_cleaned.jsonl"

def load_data(filepath):
    if not os.path.exists(filepath):
        if global_rank == 0: print(f"⚠️ {filepath} missing")
        return pd.DataFrame()
    return pd.read_json(filepath, lines=True)

df_train = load_data(train_file)
df_eval = load_data(val_file)
df_test = load_data(test_file)

if global_rank == 0:
    print(f"Data: Train={len(df_train)}, Test={len(df_test)}")

# Prompts
def create_training_prompt(data_point):
    return f"""{data_point["instruction"]}\n\n[{data_point["input"]}] = {data_point["output"]}""".strip() + EOS_TOKEN

def create_inference_prompt(data_point):
    return f"""{data_point["instruction"]}\n\n[{data_point["input"]}] = """.strip()

df_train["text"] = df_train.apply(create_training_prompt, axis=1)
df_eval["text"] = df_eval.apply(create_training_prompt, axis=1)

# List 格式用于手动推理
test_prompts_list = df_test.apply(create_inference_prompt, axis=1).tolist()
test_labels_list = df_test['output'].tolist()

train_data = Dataset.from_pandas(df_train)
eval_data = Dataset.from_pandas(df_eval)

# --- 4. DDP Accelerated Inference Function ---
def distributed_predict(all_prompts, model, tokenizer):
    total_samples = len(all_prompts)
    if total_samples == 0: return []
    
    my_indices = list(range(global_rank, total_samples, world_size))
    my_prompts = [all_prompts[i] for i in my_indices]
    
    my_results = []
    batch_size = 64 
    
    iterator = tqdm(range(0, len(my_prompts), batch_size), desc="DDP Inference") if global_rank == 0 else range(0, len(my_prompts), batch_size)

    for i in iterator:
        batch_prompts = my_prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device_string)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for text in decoded:
            try:
                ans = text.split("=")[-1].lower().strip()
            except:
                ans = "none"
            my_results.append(ans)
            
    my_data_pairs = list(zip(my_indices, my_results))
    all_data_pairs = [None for _ in range(world_size)]
    
    dist.all_gather_object(all_data_pairs, my_data_pairs)
    
    if global_rank == 0:
        flat_pairs = [item for sublist in all_data_pairs if sublist for item in sublist]
        flat_pairs.sort(key=lambda x: x[0]) 
        return [x[1] for x in flat_pairs]
    else:
        return []

def evaluate_metrics(y_true, y_pred, phase="Final"):
    y_true = [str(y).lower().strip() for y in y_true]
    y_pred = [str(y).lower().strip() for y in y_pred]
    acc = accuracy_score(y_true, y_pred)
    print(f'\n=== {phase} Results ===')
    print(f'Exact Match Accuracy: {acc:.4f}')

# --- 5. Baseline Evaluation (DDP) ---
dist.barrier()
if global_rank == 0: print("Starting Fast Baseline Evaluation...")

model.gradient_checkpointing_disable() 
model.config.use_cache = True

start_time = time.time()
y_pred_base = distributed_predict(test_prompts_list, model, tokenizer)

if global_rank == 0:
    print(f"Baseline finished in {time.time() - start_time:.2f}s")
    if len(test_labels_list) > 0:
        evaluate_metrics(test_labels_list, y_pred_base, phase="Baseline")

model.config.use_cache = False
model.gradient_checkpointing_enable()
dist.barrier()

# --- 6. Training Configuration ---

peft_config = LoraConfig(
    lora_alpha=128, lora_dropout=0.05, r=64, bias="none", 
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    use_dora=True
)

training_arguments = SFTConfig(
    output_dir="logs_goemotions_fast",
    seed=SEED,
    num_train_epochs=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    ddp_find_unused_parameters=True,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4,
    
    dataloader_num_workers=0,
    
    optim="adamw_torch_fused",
    learning_rate=4e-4, 
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    bf16=True, 
    report_to="tensorboard",
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=20,
    per_device_eval_batch_size=4,
)
training_arguments.max_seq_length = 256

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
)

# --- 7. Training ---
if global_rank == 0: print("Starting Training...")
trainer.train()

# --- 8. Final Evaluation & Plotting (DDP) ---
dist.barrier()
if global_rank == 0:
    print("--- Training Finished. Saving Model... ---")
    trainer.model.save_pretrained("trained-model-goemotions-final")
    tokenizer.save_pretrained("trained-model-goemotions-final")

    print("Generating Loss Plot...")
    log_history = trainer.state.log_history
    
    train_steps = [x["step"] for x in log_history if "loss" in x]
    train_losses = [x["loss"] for x in log_history if "loss" in x]
    
    eval_steps = [x["step"] for x in log_history if "eval_loss" in x]
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    
    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, label="Training Loss", color="blue", alpha=0.6)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="orange", linewidth=2)
        
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("loss_plot_goemotions.png")
    print("Saved loss chart to 'loss_plot_goemotions.png'")

    print("Starting Final Evaluation...")

model.gradient_checkpointing_disable() 
model.config.use_cache = True

start_time = time.time()
y_pred_final = distributed_predict(test_prompts_list, model, tokenizer)

if global_rank == 0:
    print(f"Final Eval finished in {time.time() - start_time:.2f}s")
    if len(test_labels_list) > 0:
        evaluate_metrics(test_labels_list, y_pred_final, phase="Final")
    
    csv_filename = "goemotions_results.csv"
    results_df = pd.DataFrame({'input': df_test['input'], 'true': test_labels_list, 'pred': y_pred_final})
    results_df.to_csv(csv_filename, index=False)
    print(f"Saved results csv to {csv_filename}")

    print("Generating Detailed Classification Report & Heatmap...")
    
    def parse_labels(text):
        if pd.isna(text) or str(text).lower() == "none":
            return []
        return [t.strip().lower() for t in str(text).split(',') if t.strip()]

    y_true = results_df['true'].apply(parse_labels)
    y_pred = results_df['pred'].apply(parse_labels)

    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    labels = mlb.classes_

    print(f"\n📊 Total Classes Detected: {len(labels)}")
    
    print("="*60)
    print("           DETAILED CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true_bin, y_pred_bin, target_names=labels, zero_division=0)
    print(report)

    confusion_matrix = np.zeros((len(labels), len(labels)))

    for true_labels, pred_labels in zip(y_true, y_pred):
        for t in true_labels:
            if t in labels:
                t_idx = np.where(labels == t)[0][0]
                for p in pred_labels:
                    if p in labels:
                        p_idx = np.where(labels == p)[0][0]
                        confusion_matrix[t_idx, p_idx] += 1

    plt.figure(figsize=(20, 16))
    sns.heatmap(confusion_matrix, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("GoEmotions Multi-label Confusion Heatmap")
    plt.tight_layout()
    plt.savefig("goemotions_confusion_heatmap.png")
    print("Saved heatmap to 'goemotions_confusion_heatmap.png'")
    # 👆👆👆 评估结束 👆👆👆
