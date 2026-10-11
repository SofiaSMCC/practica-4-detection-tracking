#!/bin/bash
# Lanza los 4 metodos (+ variante con NMS del metodo 2) como jobs separados en paralelo y luego evalua todo.
# Uso: bash scripts/run_all.sh first5 | rest15 | [--seqs M0208 M1004]      (sin args = las 20 secuencias de test)
# Opciones de sbatch extra por variable OPTS, p.ej.: OPTS="-p data-science --gres=shard:rtxa6000:8 -t 00:20:00 -c 4 --mem=12G"
cd "$(dirname "$0")/.."
# Grupos: first5 = las 5 primeras secuencias de test ; rest15 = las otras 15 ; otro = se pasa tal cual (--seqs ...)
FIRST5="M0203 M0205 M0208 M0209 M0403"
REST15="M0601 M0602 M0606 M0701 M0801 M0802 M1001 M1004 M1007 M1009 M1101 M1301 M1302 M1303 M1401"
case "$1" in first5) EXTRA="--seqs $FIRST5";; rest15) EXTRA="--seqs $REST15";; *) EXTRA="$@";; esac
S="scripts/slurm_run.sh"
j1=$(sbatch --parsable $OPTS -J m1 $S scripts/method1_yolo11_sort.py $EXTRA)
j2=$(sbatch --parsable $OPTS -J m2 $S scripts/method2_yolo26_bytetrack.py $EXTRA)
j2n=$(sbatch --parsable $OPTS -J m2nms $S scripts/method2_yolo26_bytetrack.py --with_nms $EXTRA)
j3=$(sbatch --parsable $OPTS -J m3 $S scripts/method3_rtdetr_botsort.py $EXTRA)
j4=$(sbatch --parsable $OPTS -J m4 $S scripts/method4_sahi.py $EXTRA)
echo "jobs: m1=$j1 m2=$j2 m2nms=$j2n m3=$j3 m4=$j4"
# El cluster limita a 5 jobs por usuario, asi que la evaluacion se corre a mano cuando terminen (squeue -u $USER):
echo "al terminar:  python -I scripts/evaluate.py --names method1_yolo11_sort method2_yolo26_bytetrack method2_yolo26_nms_bytetrack method3_rtdetrm_botsort method4_sahi_yolo26_bytetrack"
