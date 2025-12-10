#!/usr/bin/env python
# coding: utf-8

# ## Fine-tune Gemma 3 12B-it for Sentiment Analysis (4-Class)
# Dataset: Twitter Entity Sentiment Analysis (Positive, Negative, Neutral, Irrelevant)
# Optimized for NERSC Perlmutter (8x A100 40GB)

import os
import sys
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Environment Setup ---
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

# --- 2. Load Model ---
# Get Ranks
global_rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_string = f"cuda:{local_rank}"

# *** 12B 模型路径 ***
GEMMA_PATH = "./gemma-3-12b-it-local" 

if local_rank == 0:
    print(f"[Rank {global_rank}] Loading model from {GEMMA_PATH} onto {device_string}...")
    print("Forcing 'sdpa' attention implementation.")

attn_implementation = "sdpa"

# 12B 模型加载 (数据并行：每个GPU加载完整模型)
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    dtype="auto", 
    device_map=device_string, 
    attn_implementation=attn_implementation
)

# 显存优化
model.gradient_checkpointing_enable()
model.enable_input_require_grads() # 修复 DDP 兼容性

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
model.config.use_cache = False 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

EOS_TOKEN = tokenizer.eos_token

if local_rank == 0:
    print(f"Model loaded on {device_string}. DType: {model.dtype}")

# --- 3. Prepare Dataset (New Format: 4 Classes) ---

train_file = "twitter_training.csv"
val_file = "twitter_validation.csv"

if global_rank == 0:
    print(f"Loading datasets: {train_file} and {val_file}...")

def load_twitter_data(filepath):
    # 读取无标题的 CSV，手动指定列名
    df = pd.read_csv(filepath, header=None, names=["id", "entity", "sentiment", "text"])
    initial_len = len(df)
    df = df.dropna(subset=["text"])
    df = df.reset_index(drop=True)
    df['sentiment'] = df['sentiment'].str.lower()
    if global_rank == 0:
        print(f"Loaded {filepath}: {len(df)} rows (Dropped {initial_len - len(df)} bad rows)")
    return df

df_train = load_twitter_data(train_file)
df_eval = load_twitter_data(val_file)

# 使用验证集作为测试集
X_test = df_eval.copy()

# Prompt Formatting
def create_training_prompt(data_point):
    return f"""generate_prompt
            Analyze the sentiment of the tweet enclosed in square brackets.
            Determine if it is positive, negative, neutral, or irrelevant.
            Return the answer as one of the labels: "positive", "negative", "neutral", "irrelevant".

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip() + EOS_TOKEN

def create_test_prompt(data_point):
    return f"""
            Analyze the sentiment of the tweet enclosed in square brackets.
            Determine if it is positive, negative, neutral, or irrelevant.
            Return the answer as one of the labels: "positive", "negative", "neutral", "irrelevant".

            [{data_point["text"]}] = 

            """.strip()

df_train["text"] = df_train.apply(create_training_prompt, axis=1)
df_eval["text"] = df_eval.apply(create_training_prompt, axis=1)
X_test_prompts = X_test.apply(create_test_prompt, axis=1)

train_data = Dataset.from_pandas(df_train)
eval_data = Dataset.from_pandas(df_eval)

if global_rank == 0:
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation/Test samples: {len(eval_data)}")


# --- 4. Metrics & Predict Functions ---

def evaluate_metrics(y_true, y_pred, phase="Final"):
    labels = ['positive', 'negative', 'neutral', 'irrelevant']
    y_true = [str(y).lower().strip() for y in y_true]
    y_pred = [str(y).lower().strip() for y in y_pred]
    
    acc = accuracy_score(y_true, y_pred)
    print(f'\n--- {phase} Evaluation Results ---')
    print(f'Overall Accuracy: {acc:.3f}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, labels=labels))
    print('\nConfusion Matrix (Labels: Pos, Neg, Neu, Irr):')
    print(confusion_matrix(y_true, y_pred, labels=labels))

def predict(prompts, model, tokenizer):
    y_pred = []
    batch_size = 8 
    
    # 只有调用此函数的主进程会显示进度条
    if global_rank == 0:
        iterator = tqdm(range(0, len(prompts), batch_size), desc="Inference")
    else:
        iterator = range(0, len(prompts), batch_size)

    for i in iterator:
        batch_prompts = prompts[i:i+batch_size].tolist()
        current_device = model.device
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(current_device)
        
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for output in decoded_outputs:
            try:
                answer = output.split("=")[-1].lower().strip()
            except:
                answer = "none"
            
            if "positive" in answer: y_pred.append("positive")
            elif "negative" in answer: y_pred.append("negative")
            elif "neutral" in answer: y_pred.append("neutral")
            elif "irrelevant" in answer: y_pred.append("irrelevant")
            else: y_pred.append("none")
                
    return y_pred


# --- 5. Training Configuration ---

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_dora=True
)

# 1. 初始化 SFTConfig (手动设置 max_seq_length)
training_arguments = SFTConfig(
    output_dir="logs_twitter_12b",
    seed=SEED,
    num_train_epochs=1,
    
    # 12B + DDP 关键配置
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    ddp_find_unused_parameters=True,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    optim="adamw_torch_fused",
    learning_rate=2e-4, 
    lr_scheduler_type="cosine",  # Cosine Decay 是目前最稳的策略
    warmup_ratio=0.03,           # 3% 的步数用于预热，防止刚开始梯度爆炸      
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    report_to="tensorboard",
    save_strategy="no", 
    
    # *** 新增：开启训练中评估 (Valid Loss Curve) ***
    eval_strategy="steps",    # 按步数评估
    eval_steps=50,            # 每 50 步评估一次 (画点)
    logging_steps=50,         # 日志频率与评估频率一致
    per_device_eval_batch_size=2, # 必须限制评估时的 batch size，防止 OOM
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


# --- 6. Baseline Evaluation (Pre-training) ---
# 使用 Barrier 确保所有 GPU 同步

if torch.distributed.is_initialized():
    torch.distributed.barrier() 

if global_rank == 0:
    print("\n--- Starting Baseline Evaluation (Pre-training) on Rank 0 ---")
    # 临时优化推理设置
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    
    # 跑全量验证集 (1000条)
    print(f"Evaluating on the FULL validation set ({len(X_test)} samples)...")
    
    y_pred_base = predict(X_test_prompts, model, tokenizer)
    y_true_base = X_test['sentiment'].tolist()
    
    evaluate_metrics(y_true_base, y_pred_base, phase="Baseline")
    
    # 恢复训练设置
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("--- Baseline Evaluation Finished ---\n")

if torch.distributed.is_initialized():
    torch.distributed.barrier() 


# --- 7. Training Execution ---

if global_rank == 0:
    print("--- Starting Training on Twitter Dataset (12B Model) ---")

trainer.train()

if global_rank == 0:
    print("--- Training Finished. Saving Model... ---")
    trainer.model.save_pretrained("trained-model-twitter-12b")
    
    # --- 升级版绘图代码：画出 Training 和 Validation Loss ---
    log_history = trainer.state.log_history
    
    # 提取数据
    train_steps = [x["step"] for x in log_history if "loss" in x]
    train_losses = [x["loss"] for x in log_history if "loss" in x]
    
    eval_steps = [x["step"] for x in log_history if "eval_loss" in x]
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    
    if train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_losses, label="Training Loss", color="blue")
        
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label="Validation Loss", color="orange", linewidth=2)
        
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss (12B Model)")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_plot_twitter_12b.png")
        print("Saved loss plot with validation curve.")


# --- 8. Final Evaluation ---
if torch.distributed.is_initialized():
    torch.distributed.barrier()

model.gradient_checkpointing_disable() 
model.config.use_cache = True

if global_rank == 0:
    print("\n--- Starting Final Evaluation on Full Validation Set ---")
    y_pred = predict(X_test_prompts, model, tokenizer)
    y_true = X_test['sentiment'].tolist()
    evaluate_metrics(y_true, y_pred, phase="Final")
    
    results_df = pd.DataFrame({'text': X_test['text'], 'true_label': y_true, 'pred_label': y_pred})
    results_df.to_csv("twitter_predictions_12b.csv", index=False)
    print("Saved predictions to twitter_predictions_12b.csv")
