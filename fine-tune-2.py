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
from sklearn.model_selection import train_test_split 

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
global_rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_string = f"cuda:{local_rank}"
GEMMA_PATH = "./gemma-3-12b-it-local" 

if local_rank == 0:
    print(f"[Rank {global_rank}] Loading model from {GEMMA_PATH} onto {device_string}...")

attn_implementation = "sdpa"
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    dtype="auto", 
    device_map=device_string, 
    attn_implementation=attn_implementation
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads() 
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
model.config.use_cache = False 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
EOS_TOKEN = tokenizer.eos_token

if local_rank == 0:
    print(f"Model loaded on {device_string}. DType: {model.dtype}")

# --- 3. Prepare Dataset (Strict Deduplication & 9:1 Split) ---

train_file = "twitter_training.csv"
val_file = "twitter_validation.csv" 

def load_twitter_data(filepath):
    df = pd.read_csv(filepath, header=None, names=["id", "entity", "sentiment", "text"])
    initial_len = len(df)
    df = df.dropna(subset=["text"])
    # 基础去重：文件内部去重
    df = df.drop_duplicates(subset=["text"], keep="first")
    df = df.reset_index(drop=True)
    df['sentiment'] = df['sentiment'].str.lower()
    if global_rank == 0:
        print(f"Loaded {filepath}: {len(df)} rows (Dropped {initial_len - len(df)} rows)")
    return df

# 1. 加载数据
df_full_train = load_twitter_data(train_file) # ~70k
df_test_blind = load_twitter_data(val_file)   # ~1k (盲测集)

# (--- 新增: 这里的逻辑比您要求的更严格 ---)
# 防止【盲测集】泄露到【训练集】中
# 在切分前，先把盲测集里出现过的文本从总训练数据里剔除
if global_rank == 0:
    print("--- Strict Decontamination Step 1: Removing Blind Test data from Training Set ---")
    blind_texts = set(df_test_blind["text"].tolist())
    initial_size = len(df_full_train)
    df_full_train = df_full_train[~df_full_train["text"].isin(blind_texts)]
    print(f"Removed {initial_size - len(df_full_train)} rows that overlapped with Blind Test Set.")

# 2. 切分数据 (9:1)
if global_rank == 0:
    print("Splitting into Train (90%) and Validation (10%)...")

df_train, df_eval = train_test_split(
    df_full_train,
    test_size=0.1, # (--- 修改点: 9:1 比例 ---)
    random_state=SEED,
    shuffle=True,
    stratify=df_full_train['sentiment']
)

# (--- 新增: 您要求的进阶检查 ---)
# 3. 确保验证集数据没有在训练集里出现过 (防止切分后的任何残留重叠)
if global_rank == 0:
    print("--- Strict Decontamination Step 2: Ensuring No Validation Leakage ---")
    val_texts = set(df_eval["text"].tolist())
    initial_train_size = len(df_train)
    df_train = df_train[~df_train["text"].isin(val_texts)]
    print(f"Cleaned Train set size: {len(df_train)} (Removed {initial_train_size - len(df_train)} overlaps)")

# 4. 设置最终测试集
X_test = df_test_blind.copy()

if global_rank == 0:
    print(f"--- Final Data Statistics ---")
    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_eval)}")
    print(f"Test set size: {len(X_test)}")

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

# --- 4. Metrics & Predict Functions ---
def evaluate_metrics(y_true, y_pred, phase="Final"):
    labels = ['positive', 'negative', 'neutral', 'irrelevant']
    y_true = [str(y).lower().strip() for y in y_true]
    y_pred = [str(y).lower().strip() for y in y_pred]
    acc = accuracy_score(y_true, y_pred)
    print(f'\n--- {phase} Evaluation Results ---')
    print(f'Overall Accuracy: {acc:.3f}')
    print(classification_report(y_true, y_pred, labels=labels))
    print(confusion_matrix(y_true, y_pred, labels=labels))

def predict(prompts, model, tokenizer):
    y_pred = []
    batch_size = 64 
    
    # 只有调用此函数的主进程会显示进度条
    if global_rank == 0:
        iterator = tqdm(range(0, len(prompts), batch_size), desc="Inference")
    else:
        iterator = range(0, len(prompts), batch_size)

    # --- 关键修改：解包模型，防止 DDP 死锁 ---
    # 如果 model 是 DDP 包装过的，它会有 .module 属性
    # 我们直接取 .module 来做单卡推理，不需要其他 GPU 配合
    inference_model = model.module if hasattr(model, "module") else model

    for i in iterator:
        batch_prompts = prompts[i:i+batch_size].tolist()
        current_device = model.device
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(current_device)
        
        # 使用 inference_model 而不是 model
        outputs = inference_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
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
    lora_alpha=128, lora_dropout=0.05, r=64, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_dora=True
)

training_arguments = SFTConfig(
    output_dir="logs_twitter_12b",
    seed=SEED,
    num_train_epochs=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    ddp_find_unused_parameters=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    optim="adamw_torch_fused",
    learning_rate=2e-4, 
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=False, bf16=True,
    max_grad_norm=1.0,
    report_to="tensorboard",
    save_strategy="no", 
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    per_device_eval_batch_size=2,
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

# --- 6. Baseline Evaluation ---
if torch.distributed.is_initialized(): torch.distributed.barrier() 

if global_rank == 0:
    print("\n--- Starting Baseline Evaluation (Blind Test) ---")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    y_pred_base = predict(X_test_prompts, model, tokenizer)
    y_true_base = X_test['sentiment'].tolist()
    evaluate_metrics(y_true_base, y_pred_base, phase="Baseline")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

if torch.distributed.is_initialized(): torch.distributed.barrier() 

# --- 7. Training ---
if global_rank == 0: print("--- Starting Training ---")
trainer.train()

if global_rank == 0:
    print("--- Training Finished. Saving Model... ---")
    trainer.model.save_pretrained("trained-model-twitter-12b")
    
    # Plotting
    log_history = trainer.state.log_history
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

# --- 8. Final Evaluation ---
if torch.distributed.is_initialized(): torch.distributed.barrier()
model.gradient_checkpointing_disable() 
model.config.use_cache = True

if global_rank == 0:
    print("\n--- Starting Final Evaluation on Blind Test Set ---")
    y_pred = predict(X_test_prompts, model, tokenizer)
    y_true = X_test['sentiment'].tolist()
    evaluate_metrics(y_true, y_pred, phase="Final")
    
    results_df = pd.DataFrame({'text': X_test['text'], 'true_label': y_true, 'pred_label': y_pred})
    results_df.to_csv("twitter_predictions_12b.csv", index=False)
