# Gemma-12B GoEmotions Fine-tuning

This repository contains code for fine-tuning Google's Gemma-3-12B-IT model on the GoEmotions dataset. The project implements parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) with DoRA and supports distributed training on SLURM clusters via PyTorch Distributed Data Parallel (DDP).

## Overview

The GoEmotions dataset is a large-scale corpus of Reddit comments annotated with emotion labels. This project fine-tunes Gemma-12B to classify these emotions across three granularities:

* **Original:** 27 emotion labels + Neutral.
* **Ekman:** Mapped to 6 basic emotions (Anger, Disgust, Fear, Joy, Sadness, Surprise).
* **Sentiment:** Mapped to Positive, Negative, or Ambiguous.

## Features

* **Distributed Training:** DDP support for multi-GPU training on SLURM clusters.
* **Efficient Fine-tuning:** LoRA + DoRA implementation on linear projection layers.
* **Comprehensive Analysis:** Automated generation of confusion matrices, loss curves, and multi-label metrics (Precision, Recall, F1).
* **Comparative Experimentation:** Includes scripts and logs for different learning rate strategies.

## Requirements

### Hardware
- Multi-GPU setup recommended (tested with 8+ GPUs)
- CUDA-capable GPUs with sufficient VRAM for 12B parameter model
- SLURM cluster environment (for distributed training)

### Software
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- Transformers library
- PEFT (Parameter-Efficient Fine-Tuning)
- TRL (Transformer Reinforcement Learning)
- Datasets
- scikit-learn
- pandas
- matplotlib
- seaborn
- tqdm

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Troylyw/Gemma-12B-GoEmotions-Finetune.git
cd Gemma-12B-GoEmotions-Finetune
```

2. Install required dependencies:
```bash
pip install torch transformers peft trl datasets scikit-learn pandas matplotlib seaborn tqdm
```

3. Ensure you have access to the Gemma-3-12B-IT model from Hugging Face (requires authentication):
```bash
huggingface-cli login
```

## Dataset Preparation

The project includes a dedicated pipeline for handling the GoEmotions dataset located in the `original dataset and prepossessing/` directory.

* **Raw Data:** The original `train.tsv`, `dev.tsv`, and `test.tsv` files are stored here.
* **Preprocessing:** Use the `dataset_clean.py` script in this folder to convert the raw TSV data into the JSONL format required by the model.
* **Mappings:** Files like `emotions.txt` and `ekman_mapping.json` are provided here to handle label conversions.

**To prepare the data, run:**

```bash
python "original dataset and prepossessing/dataset_clean.py"
```
**Thg dataset should be in JSONL format with the following structure:**
```json
{"instruction": "Analyze the text and identify the emotions.", "input": "Text to analyze", "output": "emotion_label"}
```

The project expects three files:
- `train_cleaned.jsonl`: Training data
- `dev_cleaned.jsonl`: Validation/development data
- `test_cleaned.jsonl`: Test data

## Usage

### SLURM Cluster (Recommended)

1. Modify `run_job.slurm` to match your cluster configuration:
   - Update account (`-A m4431`)
   - Adjust number of nodes and GPUs
   - Update paths and Python environment

2. Submit the job:
```bash
sbatch run_job.slurm
```

### Local Multi-GPU Training

For local multi-GPU training, you can use:

```bash
torchrun --nproc_per_node=<num_gpus> fine-tune-goem.py
```

### Single GPU Training

Modify the script to set `WORLD_SIZE=1` and `RANK=0` if running on a single GPU (note: memory constraints may apply for a 12B model).

## Scripts

### Main Training (Recommended)
`fine-tune-goem.py` is the optimized training script (Learning Rate: 4e-4). It handles the full fine-tuning lifecycle and generates the final model artifacts.

### Preliminary Experiments
`fine-tune-goem-original_1e-5.py` is provided for experimental comparison using a lower learning rate (1e-5).

* The results from these early runs (including basic loss curves `loss_plot_goemotions.png` and initial reports) are stored in the `original_fine_tune/` folder.
* You can use these artifacts to analyze the impact of hyperparameters on model convergence compared to the final version.

### Configuration Files

- **`ekman_mapping.json`**: Maps original emotions to Ekman's six basic emotions (anger, disgust, fear, joy, sadness, surprise)
- **`sentiment_mapping.json`**: Maps emotions to sentiment categories (positive, negative, ambiguous)
- **`run_job.slurm`**: SLURM batch job configuration

### Utility Scripts

- **`extract_words.py`**: Utility for extracting emotion-related words from the dataset

## Model Configuration

### LoRA Configuration
- **LoRA Rank (r)**: 64
- **LoRA Alpha**: 128
- **LoRA Dropout**: 0.05
- **Target Modules**: All linear projection layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **DoRA**: Enabled for improved performance

### Training Hyperparameters
- **Learning Rate**: 4e-4 (configurable via different scripts)
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps
- **Effective Batch Size**: 16 per device
- **Epochs**: 1
- **Optimizer**: AdamW (fused)
- **LR Scheduler**: Cosine with 3% warmup
- **Weight Decay**: 0.01
- **Precision**: bfloat16
- **Max Sequence Length**: 256 tokens
- **Gradient Checkpointing**: Enabled

## Output Files

After training, the following files are generated:

### Model Checkpoints
- `trained-model-goemotions-final/`: Final fine-tuned model and tokenizer
- `logs_goemotions_fast/`: Training checkpoints and logs

### Evaluation Results
- `goemotions_results.csv`: Test set predictions with true and predicted labels
- `goemotions_original_report.csv`: Classification report for original emotion labels
- `goemotions_ekman_report.csv`: Classification report for Ekman emotions
- `goemotions_sentiment_report.csv`: Classification report for sentiment classification

### Visualizations
- `loss_plot_goemotions.png`: Training and validation loss curves
- `classified_report.png`: gpemotions accuracy and detailed classification result
- `goemotions_ekman_heatmap.png`: Heatmap for Ekman emotion categories
- `goemotions_sentiment_heatmap.png`: Heatmap for sentiment classification
- `baseline.png`: zero shot accuracy

## Evaluation Metrics

The model evaluation includes:
- **Exact Match Accuracy**: Exact string match between predicted and true labels
- **Multi-label Classification Metrics**:
  - Precision, Recall, F1-score per class
  - Macro and micro averages
- **Confusion Matrices**: Visual representation of classification performance

## DDP Training Details

The implementation uses PyTorch Distributed Data Parallel (DDP) with:
- Automatic SLURM environment detection
- Manual DDP initialization for better control
- Distributed inference for faster evaluation
- Barrier synchronization for coordinated training steps

## Notes

- The model uses gradient checkpointing to reduce memory usage
- Inference cache is disabled during training and enabled during evaluation
- The tokenizer uses left padding for batch processing
- Results are saved only on rank 0 to avoid duplicates
