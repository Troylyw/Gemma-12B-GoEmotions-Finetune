# Gemma-12B GoEmotions Fine-tuning

This repository contains code for fine-tuning Google's Gemma-3-12B-IT model on the GoEmotions dataset for emotion classification tasks. The project implements efficient fine-tuning using LoRA (Low-Rank Adaptation) with DoRA (Weight-Decomposed Low-Rank Adaptation) and supports distributed training via PyTorch Distributed Data Parallel (DDP).

## Overview

The GoEmotions dataset is a large-scale corpus of Reddit comments annotated with emotion labels. This project fine-tunes the Gemma-12B model to perform emotion classification, supporting both original emotion labels and aggregated categories (Ekman's six basic emotions and sentiment classification).

## Features

- **Distributed Training**: Supports DDP for multi-GPU training on SLURM clusters
- **Efficient Fine-tuning**: Uses LoRA with DoRA for parameter-efficient training
- **Multiple Learning Rates**: Includes variants with different learning rates (5e-4, 5e-5)
- **Comprehensive Evaluation**:
  - Baseline and post-training evaluation
  - Multi-label classification metrics
  - Confusion matrix heatmaps
  - Training loss visualization
- **Emotion Mappings**: Supports original labels, Ekman emotions, and sentiment classification
- **Fast Inference**: Optimized distributed inference for evaluation

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

## Dataset Format

The dataset should be in JSONL format with the following structure:

```json
{
  "instruction": "Analyze the text and identify the emotions.",
  "input": "Text to analyze",
  "output": "emotion_label"
}
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

### Main Training Scripts

- **`fine-tune-goem.py`**: Main training script with learning rate 4e-4 (default)
- **`fine-tune-goem_5e-4.py`**: Variant with learning rate 5e-4
- **`fine-tune-goem-5e-5.py`**: Variant with learning rate 5e-5

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
- `goemotions_confusion_heatmap.png`: Confusion matrix heatmap for emotion classification
- `goemotions_original_heatmap.png`: Heatmap for original emotion labels
- `goemotions_ekman_heatmap.png`: Heatmap for Ekman emotion categories
- `goemotions_sentiment_heatmap.png`: Heatmap for sentiment classification

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

## Troubleshooting

### Out of Memory Errors

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `max_seq_length`

### DDP Issues

- Ensure SLURM environment variables are properly set
- Check NCCL backend initialization
- Verify network connectivity between nodes

### Model Loading Issues

- Ensure you have Hugging Face authentication configured
- Check internet connectivity or configure offline model paths
- Verify sufficient disk space for model downloads

## License

This project uses the Gemma model from Google, which is subject to Google's Gemma Terms of Use. Please review the license terms before use.

## Citation

If you use this code, please cite:

- The GoEmotions dataset
- The Gemma model
- PEFT library for LoRA implementation

## Acknowledgments

- Google for the Gemma model
- Hugging Face for transformers and PEFT libraries
- The GoEmotions dataset creators
