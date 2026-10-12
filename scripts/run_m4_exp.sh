#!/bin/bash
# FASE 1 del metodo 4 en Khipu (nodo de CPU): solo usa detecciones cacheadas (results/_cache/{yolo26,rtdetr,sahi,gmc})
# y el GT; no hace falta GPU. Un solo job encadenado:
#   prefuse -> fusion (E1, 12 variantes) -> post (E2/E3, grilla de post-proceso con ablacion) para YR y YRS_slot
# Salidas en results/m4_exp/. Uso: bash scripts/run_m4_exp.sh      (opciones extra de sbatch en OPTS)
cd "$(dirname "$0")/.."
S="scripts/slurm_cpu.sh"; P="scripts/m4_experiments.py"; J="--jobs 16"
j=$(sbatch --parsable $OPTS -J m4exp -p standard -c 16 --mem=24G -t 12:00:00 -o "results/slurm_%j.log" \
    --wrap "bash $S $P prefuse $J && bash $S $P fusion $J && bash $S $P post --variants YR YRS_slot $J")
echo "job: $j  (squeue -u \$USER ; tail -f results/slurm_$j.log)"
echo "al terminar:  python -I $P final --from results/m4_exp/post_final_<variante>.json --name method4_sahi_fusion_bytetrack"
echo "              python -I scripts/evaluate.py --names method4_sahi_fusion_bytetrack method5_fusion_bytetrack_cv"
