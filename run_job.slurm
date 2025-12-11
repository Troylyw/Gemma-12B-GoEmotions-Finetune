#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J gemma-12b
#SBATCH -t 04:00:00
#SBATCH -A m4431
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32

module load python
source /pscratch/sd/t/troylyw/nanogpt-env/bin/activate
cd /pscratch/sd/t/troylyw/gemma


export OMP_NUM_THREADS=8
export PYTHONFAULTHANDLER=1
export TORCHAO_CPP_EXT_SKIP=1

export TORCHAO_CPP_EXT_SKIP=1 


echo "Job started on $(hostname) at $(date)"

srun -u python fine-tune-goem.py

echo "Job finished at $(date)"
