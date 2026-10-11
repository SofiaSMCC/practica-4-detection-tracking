#!/bin/bash
# Uso: sbatch scripts/slurm_run.sh <script.py> [args...]
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o results/slurm_%j.log
ROOT=/home/marcelino.maita/courses/deep_learning/practica-4
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$ROOT/.venv/bin/python" -I "$@"
