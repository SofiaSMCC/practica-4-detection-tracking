# Experimentos del método 4 (SAHI + heurísticas)

Todo corrió en CPU sobre las detecciones cacheadas (`results/_cache/{yolo26,rtdetr,sahi,gmc}`) y el GT de las 20 secuencias de test.
Código: `scripts/m4_lib.py` (fusión, ByteTrack, post-proceso, evaluación) y `scripts/m4_experiments.py` (etapas y validación cruzada).

**Protocolo.** Mismos 4 pliegues que `tune_cv.py` (secuencia i → pliegue i % 4). En cada pliegue se elige la configuración con 15 secuencias y se mide en las 5 no vistas. `cv_mota` es el promedio de los 4 pliegues (estimación honesta) y `mota_20seq` es la configuración final elegida con las 20 secuencias. La evaluación es la misma de `evaluate.py`: motmetrics con IoU ≥ 0,5, regiones ignoradas y MOTA promediado sobre secuencias.
Verificación: con la configuración del método 5 de Sofía, este harness reproduce MOTA 0,549 / IDF1 0,754 / IDS 125.

## Método 4 final

```
frame → YOLO26 (imagen completa) ─┐  mismo "voto": SAHI solo entra donde YOLO26 no vio nada
      → SAHI 640 + YOLO26 ────────┤
      → RT-DETRv4 (VisDrone) ─────┴→ fusión ponderada de cajas (IoU 0,55) → ByteTrack (alto 0,65 / bajo 0,2)
                                     → filtro de escala relativa (0,6) → largo mínimo 20 frames → MOT
```

Configuración: `post_final_YRS_slot.json`. Salidas: `results/method4_sahi_fusion_bytetrack/`.

| | MOTA (prom. 20 sec.) | IDF1 | IDS | FP | FN |
|---|---|---|---|---|---|
| Método 4 anterior (SAHI + ByteTrack ajustado) | 0,473 | 0,691 | 162 | 57 705 | 118 068 |
| Método 5 anterior (fusión YOLO26 + RT-DETR) | 0,549 | 0,754 | 125 | 65 158 | 90 953 |
| **Método 4 final** | **0,563** | **0,756** | **88** | 62 874 | 91 073 |

Mejora a la fusión anterior en 15 de 20 secuencias (`per_seq_m4_vs_anteriores.csv`). La mayor ganancia es M1101, de noche (−0,42 → −0,19).

## Ablación (validación cruzada, secuencias completas): `ablation_m4.csv`

| Paso | Configuración | CV MOTA | CV IDF1 | IDS |
|---|---|---|---|---|
| 1 | SAHI 640 + YOLO26 (método 4 anterior) | 0,465 | 0,651 | 82 |
| 2 | YOLO26 solo (método 2) | 0,485 | 0,686 | 238 |
| 3 | YOLO26 + RT-DETR fusionados (método 5 anterior) | 0,549 | 0,754 | 125 |
| 4 | + SAHI como respaldo de YOLO26 | 0,544 | 0,758 | 135 |
| 5 | + filtro de escala relativa | 0,557 | 0,753 | 129 |
| 6 | + largo mínimo 20 = **método 4 final** | **0,559** | **0,754** | **87** |
| control | paso 6 sin SAHI | 0,565 | 0,751 | 109 |
| control | paso 6 + interpolación + confianza + estático | 0,557 | 0,752 | 86 |

Recall de detección por tamaño (`size_recall_m4.csv`; terciles de 18,3 y 28,4 px): con SAHI, 0,494 / 0,783 / 0,906; sin SAHI, 0,470 / 0,762 / 0,894. Es decir, +2,4 puntos en pequeños a cambio de +8 % de FP.

## Hallazgos

1. **SAHI como tercer votante perjudica** (YRS: CV 0,485). YOLO26 y SAHI usan los mismos pesos, así que su acuerdo no es consenso real e infla la confianza de sus errores comunes. Usar **SAHI como respaldo** (un solo voto compartido) lo corrige: sube el recall de pequeños y mejora 12 de 20 secuencias, pero suma FP en escenas nocturnas u oblicuas (M1101, M1007, M1001).
2. **Los FP persistentes no son vehículos estacionados.** Con el movimiento de cámara compensado (GMC), solo el 3 % de los frames de los FP de M1101 se mueve menos de 10 px. Lo que sí los distingue es el tamaño: lado mediano de 15 px frente a 38 px de los vehículos reales. De ahí sale el **filtro de escala relativa**: en video de dron la altura es casi constante, así que se borran los tracks con lado < 0,6 × el lado típico de la escena, estimado con los propios tracks y sin usar GT. Es el único componente con ganancia robusta en validación cruzada.
3. Interpolación de huecos, filtro por confianza media y filtro estático: ganancia ≈ 0. ByteTrack ya recupera los huecos cortos con su segunda ronda de baja confianza.
4. **El ajuste por descenso de coordenadas sobreajusta** (`tune_YR_all_descent.json`: 0,54 a 0,62 en ajuste, pero 0,534 en validación). Por eso se fijó ByteTrack con la configuración de E1, que fue estable en los 4 pliegues, y se ajustó solo una grilla pequeña de post-proceso.
5. El control sin SAHI (0,565) y el método final (0,559) están dentro del ruido entre pliegues. SAHI se mantiene porque sube el recall de vehículos pequeños y el MOTA global ponderado por frames (0,548 frente a 0,544).

## Reproducir

```bash
python -I scripts/m4_experiments.py prefuse                       # fusiones -> results/_cache/fx_<variante>/
python -I scripts/m4_experiments.py fusion                        # E1 -> fusion_cv.json
python -I scripts/m4_experiments.py post --variants YR YRS_slot S Y   # E2/E3 -> post_cv_*.json, post_final_*.json
python -I scripts/m4_experiments.py final --from results/m4_exp/post_final_YRS_slot.json --name method4_sahi_fusion_bytetrack
python -I scripts/evaluate.py --names method4_sahi_fusion_bytetrack
```

En Khipu: `bash scripts/run_m4_exp.sh` (CPU, todo lo anterior) y `bash scripts/run_m4_gpu.sh YRS_slot` (FPS end-to-end en GPU).
