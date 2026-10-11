#!/bin/bash
# Uso: sbatch -J nombre scripts/slurm_cpu.sh <script.py> [args...]     (nodo de CPU; el script debe recibir --device cpu)
#SBATCH -p standard
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH -o results/slurm_%j.log
ROOT=/home/marcelino.maita/courses/deep_learning/practica-4
cd "$ROOT"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
exec "$ROOT/.venv/bin/python" -I "$@"
