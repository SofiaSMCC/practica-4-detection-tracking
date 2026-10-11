# Histórico: primeros 5 jobs (primeras 5 secuencias de test)

Copia de seguridad de los resultados de los jobs 55477–55481 (m1, m2, m2nms, m3, m4).
Secuencias: M0203, M0205, M0208, M0209, M0403 (4 008 frames). Configuración por defecto de cada método.
Es una COPIA: los jobs de las otras 15 secuencias (55560–55564) siguen escribiendo en results/method*/.

| Método | MOTA (prom.) | IDF1 | IDS | FP | FN | FPS |
|---|---|---|---|---|---|---|
| 1 YOLOv11 + SORT | 0,607 | 0,735 | 205 | 6 038 | 13 663 | 39,0 |
| 2 YOLO26 + ByteTrack (sin NMS) | 0,541 | 0,776 | 39 | 7 755 | 9 588 | 39,0 |
| 2 con NMS | 0,542 | 0,778 | 85 | 8 593 | 9 140 | 32,2 |
| 3 RT-DETRv4-M + BoT-SORT | -0,058 | 0,658 | 36 | 46 568 | 573 | 17,3 |
| 4 SAHI + YOLO26 + ByteTrack | 0,373 | 0,711 | 71 | 11 494 | 9 317 | 9,1 |

Contenido: <método>/<seq>.txt (predicciones MOT), timing.json, metrics.csv; slurm_*.log; comparison_mean.csv y comparison_overall.csv.
