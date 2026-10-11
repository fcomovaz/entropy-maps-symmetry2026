# Resultados medidos

Medianas entre los casos de cada tamaño; cada caso ya es mediana de batches. No son intervalos de confianza ni un workload representativo del paper. Unidades ns/patch. Consultar CSV para distribución y dispersión.

| N | E1 | E2 | E3 | E4 | H1+E1 | H1+E4 | Mejor pipeline (mediana) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | 16684 | 12853 | 7995 | 8245 | 29590 | 22308 | H1+E3: 22251 |
| 1024 | 11238 | 8293 | 8135 | 8383 | 67034 | 58111 | H1+E3: 56747 |
| 4096 | 38798 | 27811 | 8409 | 8408 | 225636 | 197747 | H3_Counter+E3: 153719 |
| 16384 | 48322 | 32287 | 8483 | 8486 | 851670 | 755490 | H3_Counter+E3: 502725 |

## Histograma aislado

| Tamaño | H1 | H2 array | H5 unroll4 | H3 Counter | E5 tracking |
|---:|---:|---:|---:|---:|---:|
| 16 | 12381 | 23568 | 17354 | 43911 | 21570 |
| 32 | 50924 | 97665 | 69181 | 63591 | 72840 |
| 64 | 197412 | 367115 | 288223 | 143658 | 274578 |
| 128 | 764740 | 1377850 | 1095255 | 485420 | 1050665 |

## E5 y fracción de entropía

Cada ratio compara pipelines medidos: >1 favorece tracking. Fracciones estimadas con componentes aislados; overhead/caché aparecen como residual en CSV.

| Tamaño | Caso | H1+E4 / E5+E4 | H1+E1 / E5+E1 | Fracción H1+E1 entropía | Fracción H1+E4 entropía |
|---:|---|---:|---:|---:|---:|
| 16 | single_level | 1.229 | 0.960 | 27.6% | 40.6% |
| 16 | few_levels | 2.543 | 1.494 | 24.7% | 35.0% |
| 16 | uniform_256 | 0.462 | 0.716 | 84.6% | 41.7% |
| 16 | random | 0.773 | 0.812 | 72.3% | 39.3% |
| 16 | real | 0.743 | 0.850 | 58.2% | 40.3% |
| 32 | single_level | 0.855 | 0.697 | 7.7% | 13.4% |
| 32 | few_levels | 0.807 | 0.818 | 3.9% | 12.1% |
| 32 | uniform_256 | 0.232 | 0.931 | 55.0% | 33.1% |
| 32 | random | 0.433 | 1.173 | 31.3% | 5.2% |
| 32 | real | 0.704 | 0.645 | 20.0% | 14.1% |
| 64 | single_level | 0.896 | 0.570 | 1.2% | 2.1% |
| 64 | few_levels | 0.484 | 0.765 | 2.5% | 3.9% |
| 64 | uniform_256 | 0.679 | 0.703 | 28.1% | 11.2% |
| 64 | random | 0.716 | 0.749 | 17.0% | 2.9% |
| 64 | real | 0.707 | 0.747 | 18.4% | 4.5% |
| 128 | single_level | 0.712 | 0.782 | 0.4% | 0.8% |
| 128 | few_levels | 0.783 | 0.610 | 1.6% | 2.8% |
| 128 | uniform_256 | 0.683 | 0.718 | 6.3% | 0.8% |
| 128 | random | 0.420 | 0.411 | 5.7% | 1.0% |
| 128 | real | 0.706 | 0.893 | 6.1% | 1.2% |

## LUT: construcción y amortización

Amortización frente a E1 = build_ns / (mediana E1 − mediana LUT). Es una estimación dependiente de estos casos, sin carga desde ROM.

| N | LUT | Entradas | C f64 bytes | C f32/u32 bytes | C u16 bytes | Python bytes | Build ns | Patches para amortizar |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | E3 | 257 | 2056 | 1028 | 514 | 8280 | 30375 | 4 |
| 256 | E4 | 257 | 2056 | 1028 | 514 | 8280 | 54305 | 7 |
| 1024 | E3 | 1025 | 8200 | 4100 | 2050 | 32856 | 200975 | 65 |
| 1024 | E4 | 1025 | 8200 | 4100 | 2050 | 32856 | 369495 | 130 |
| 4096 | E3 | 4097 | 32776 | 16388 | 8194 | 131160 | 730950 | 25 |
| 4096 | E4 | 4097 | 32776 | 16388 | 8194 | 131160 | 1122615 | 37 |
| 16384 | E3 | 16385 | 131080 | 65540 | 32770 | 524376 | 2042480 | 52 |
| 16384 | E4 | 16385 | 131080 | 65540 | 32770 | 524376 | 4697595 | 118 |

## Error y ranking

Grupo all, pares estrictos con tolerancia 1e-12. Posiciones y top-k usan orden descendente sin tolerancia y desempate por índice, por eso pueden variar incluso con 0 inversiones estrictas.

| Tamaño | Método | MAE | Máx abs | Máx relativo (H>1e-12) | Inversiones | Empates nuevos | Empates rotos | Posiciones distintas | Top-1 | Top-10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | E2 | 4.7e-16 | 2.66e-15 | 3.82e-15 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| 16 | E3 | 4.7e-16 | 2.66e-15 | 3.82e-15 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| 16 | E4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| 16 | Q16 | 2.4e-05 | 0.000118 | 3.71e-05 | 2 | 3 | 1 | 10 | 1.0 | 1.0 |
| 16 | Q30 | 1.52e-09 | 7.71e-09 | 1.99e-09 | 0 | 0 | 1 | 0 | 1.0 | 1.0 |
| 32 | E2 | 3.04e-15 | 1.6e-14 | 1.14e-13 | 0 | 0 | 0 | 4 | 1.0 | 1.0 |
| 32 | E3 | 3.04e-15 | 1.6e-14 | 1.14e-13 | 0 | 0 | 0 | 4 | 1.0 | 1.0 |
| 32 | E4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| 32 | Q16 | 5.76e-05 | 0.000243 | 0.000392 | 0 | 2 | 0 | 2 | 1.0 | 1.0 |
| 32 | Q30 | 3.93e-09 | 1.49e-08 | 4.12e-08 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| 64 | E2 | 3.72e-15 | 1.42e-14 | 5.57e-14 | 0 | 0 | 0 | 2 | 1.0 | 1.0 |
| 64 | E3 | 3.72e-15 | 1.42e-14 | 5.57e-14 | 0 | 0 | 0 | 2 | 1.0 | 1.0 |
| 64 | E4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| 64 | Q16 | 0.000127 | 0.000698 | 0.000256 | 34 | 23 | 0 | 56 | 1.0 | 1.0 |
| 64 | Q30 | 5.83e-09 | 2.07e-08 | 3.01e-08 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| 128 | E2 | 1.17e-15 | 7.99e-15 | 3.06e-13 | 0 | 0 | 0 | 2 | 1.0 | 1.0 |
| 128 | E3 | 1.17e-15 | 7.99e-15 | 3.06e-13 | 0 | 0 | 0 | 2 | 1.0 | 1.0 |
| 128 | E4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |
| 128 | Q16 | 3.13e-05 | 0.00129 | 0.00397 | 282 | 477 | 0 | 85 | 1.0 | 1.0 |
| 128 | Q30 | 6.53e-09 | 2.14e-08 | 2.02e-07 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |

## Top-k: casos reales y casi empatados

| Tamaño | Grupo | Método | Inversiones | Mínimo agreement entre k evaluados |
|---:|---|---|---:|---:|
| 16 | real | E3 | 0 | 1.0000 |
| 16 | real | E4 | 0 | 1.0000 |
| 16 | real | Q16 | 2 | 1.0000 |
| 16 | real | Q30 | 0 | 1.0000 |
| 16 | near_tie | E3 | 0 | 1.0000 |
| 16 | near_tie | E4 | 0 | 1.0000 |
| 16 | near_tie | Q16 | 0 | 1.0000 |
| 16 | near_tie | Q30 | 0 | 1.0000 |
| 32 | real | E3 | 0 | 1.0000 |
| 32 | real | E4 | 0 | 1.0000 |
| 32 | real | Q16 | 0 | 1.0000 |
| 32 | real | Q30 | 0 | 1.0000 |
| 32 | near_tie | E3 | 0 | 0.9000 |
| 32 | near_tie | E4 | 0 | 1.0000 |
| 32 | near_tie | Q16 | 0 | 1.0000 |
| 32 | near_tie | Q30 | 0 | 1.0000 |
| 64 | real | E3 | 0 | 1.0000 |
| 64 | real | E4 | 0 | 1.0000 |
| 64 | real | Q16 | 0 | 1.0000 |
| 64 | real | Q30 | 0 | 1.0000 |
| 64 | near_tie | E3 | 0 | 1.0000 |
| 64 | near_tie | E4 | 0 | 1.0000 |
| 64 | near_tie | Q16 | 34 | 1.0000 |
| 64 | near_tie | Q30 | 0 | 1.0000 |
| 128 | real | E3 | 0 | 1.0000 |
| 128 | real | E4 | 0 | 1.0000 |
| 128 | real | Q16 | 0 | 1.0000 |
| 128 | real | Q30 | 0 | 1.0000 |
| 128 | near_tie | E3 | 0 | 1.0000 |
| 128 | near_tie | E4 | 0 | 1.0000 |
| 128 | near_tie | Q16 | 282 | 0.8750 |
| 128 | near_tie | Q30 | 0 | 1.0000 |

## Repetición completa

Run1 se conserva sin sobrescribir. Cambió la resolución de la ruta del experimento 01, sin cambiar kernels ni bytes de entrada. La primera ejecución coincidió con actividad externa en el workspace. Las diferencias muestran variabilidad del host, no aceleraciones del algoritmo.

| Tamaño | Pipeline | Run1 ns | Run2 ns | Cambio |
|---:|---|---:|---:|---:|
| 16 | H1+E1 | 32126 | 29590 | -7.9% |
| 16 | H1+E4 | 24531 | 22308 | -9.1% |
| 16 | H3_Counter+E4 | 55671 | 55225 | -0.8% |
| 16 | H1_sparse+E4 | 29736 | 31142 | +4.7% |
| 32 | H1+E1 | 88080 | 67034 | -23.9% |
| 32 | H1+E4 | 58914 | 58111 | -1.4% |
| 32 | H3_Counter+E4 | 74472 | 71883 | -3.5% |
| 32 | H1_sparse+E4 | 88697 | 84259 | -5.0% |
| 64 | H1+E1 | 240766 | 225636 | -6.3% |
| 64 | H1+E4 | 228474 | 197747 | -13.4% |
| 64 | H3_Counter+E4 | 174708 | 154855 | -11.4% |
| 64 | H1_sparse+E4 | 319175 | 282142 | -11.6% |
| 128 | H1+E1 | 824825 | 851670 | +3.3% |
| 128 | H1+E4 | 856795 | 755490 | -11.8% |
| 128 | H3_Counter+E4 | 497675 | 513465 | +3.2% |
| 128 | H1_sparse+E4 | 1161025 | 1104625 | -4.9% |
