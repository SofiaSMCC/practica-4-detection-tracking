#!/bin/bash
# FASE 2 del metodo 4 en Khipu (GPU): corrida END-TO-END sobre las imagenes con la configuracion elegida por validacion
# cruzada (results/m4_exp/post_final_<variante>.json). Mide FPS reales y deja las predicciones MOT de las 20 secuencias.
#   job 1: las 5 primeras secuencias (las mismas en que se midieron los FPS de los metodos 1-3 -> FPS comparables)
#   job 2: las otras 15 (para tener las 20 secuencias evaluadas con la corrida en GPU)
#   SAHI320=1 agrega un job 3 (opcional): cache de SAHI con slices de 320 px para probarlo en la fusion
# Uso: bash scripts/run_m4_gpu.sh [variante]        (por defecto YRS_slot; opciones extra de sbatch en OPTS)
cd "$(dirname "$0")/.."
V="${1:-YRS_slot}"; S="scripts/slurm_run.sh"; NAME="method4_final_${V}_gpu"
CFG="results/m4_exp/post_final_$V.json"; [ -f "$CFG" ] || { echo "falta $CFG"; exit 1; }
FIRST5="M0203 M0205 M0208 M0209 M0403"
REST15="M0601 M0602 M0606 M0701 M0801 M0802 M1001 M1004 M1007 M1009 M1101 M1301 M1302 M1303 M1401"
j1=$(sbatch --parsable $OPTS -J m4fps5 $S scripts/method4_final.py --from "$CFG" --name "$NAME" --seqs $FIRST5)
j2=$(sbatch --parsable $OPTS -J m4fps15 $S scripts/method4_final.py --from "$CFG" --name "$NAME" --seqs $REST15)
echo "jobs: primeras 5 = $j1 | otras 15 = $j2   (logs: results/slurm_<job>.log)"
if [ "$SAHI320" = "1" ]; then
  j3=$(sbatch --parsable $OPTS -J sahi320 $S scripts/cache_detections.py --detector sahi --slice 320 --tag sahi320 --device cuda:0)
  echo "cache SAHI 320: $j3"
fi
echo "al terminar (squeue -u \$USER vacio):"
echo "  FPS comparables:  .venv/bin/python -I scripts/evaluate.py --names $NAME --seqs $FIRST5"
echo "  20 secuencias:    .venv/bin/python -I scripts/evaluate.py --names $NAME"
