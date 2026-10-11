# Resultados medidos

Medianas entre los casos de cada tamaño; cada caso ya es mediana de batches. No son intervalos de confianza ni un workload representativo del paper. Unidades ns/patch. Consultar CSV para distribución y dispersión.

| N | E1 | E2 | E3 | E4 | H1+E1 | H1+E4 | Mejor pipeline (mediana) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | 18433 | 14408 | 8530 | 8454 | 32126 | 24531 | H1+E3: 22970 |
| 1024 | 10040 | 8822 | 8815 | 8488 | 88080 | 58914 | H1+E4: 58914 |
| 4096 | 41310 | 27823 | 8575 | 8419 | 240766 | 228474 | H3_Counter+E4: 174708 |
| 16384 | 50372 | 30646 | 8593 | 8327 | 824825 | 856795 | H3_Counter+E3: 496910 |

## Histograma aislado

| Tamaño | H1 | H2 array | H5 unroll4 | H3 Counter | E5 tracking |
|---:|---:|---:|---:|---:|---:|
| 16 | 12606 | 21351 | 15884 | 43216 | 21403 |
| 32 | 47603 | 82109 | 67926 | 62244 | 71559 |
| 64 | 204567 | 342737 | 271837 | 150055 | 287144 |
| 128 | 834285 | 1298655 | 1295215 | 490465 | 1124665 |

## E5 y fracción de entropía

Cada ratio compara pipelines medidos: >1 favorece tracking. Fracciones estimadas con componentes aislados; overhead/caché aparecen como residual en CSV.

| Tamaño | Caso | H1+E4 / E5+E4 | H1+E1 / E5+E1 | Fracción H1+E1 entropía | Fracción H1+E4 entropía |
|---:|---|---:|---:|---:|---:|
| 16 | single_level | 1.163 | 0.670 | 20.1% | 31.3% |
| 16 | few_levels | 1.205 | 0.995 | 29.7% | 46.0% |
| 16 | uniform_256 | 0.439 | 0.699 | 80.0% | 38.8% |
| 16 | random | 0.520 | 0.744 | 76.3% | 41.9% |
| 16 | real | 0.922 | 0.767 | 59.7% | 40.1% |
| 32 | single_level | 0.966 | 0.774 | 7.4% | 12.5% |
| 32 | few_levels | 0.838 | 0.668 | 8.8% | 9.9% |
| 32 | uniform_256 | 0.607 | 0.796 | 41.4% | 10.3% |
| 32 | random | 0.607 | 1.102 | 52.6% | 15.6% |
| 32 | real | 0.723 | 0.773 | 29.6% | 15.1% |
| 64 | single_level | 0.733 | 0.744 | 4.4% | 7.8% |
| 64 | few_levels | 0.611 | 0.754 | 2.3% | 3.6% |
| 64 | uniform_256 | 0.692 | 0.718 | 21.9% | 4.3% |
| 64 | random | 0.948 | 0.553 | 11.8% | 2.1% |
| 64 | real | 0.712 | 0.719 | 18.2% | 4.2% |
| 128 | single_level | 0.769 | 0.760 | 0.3% | 0.6% |
| 128 | few_levels | 0.746 | 0.771 | 0.4% | 0.6% |
| 128 | uniform_256 | 0.712 | 0.699 | 9.4% | 1.3% |
| 128 | random | 0.685 | 0.681 | 4.6% | 0.7% |
| 128 | real | 0.670 | 0.697 | 5.6% | 1.1% |

## LUT: construcción y amortización

Amortización frente a E1 = build_ns / (mediana E1 − mediana LUT). Es una estimación dependiente de estos casos, sin carga desde ROM.

| N | LUT | Entradas | C f64 bytes | C f32/u32 bytes | C u16 bytes | Python bytes | Build ns | Patches para amortizar |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | E3 | 257 | 2056 | 1028 | 514 | 8280 | 31310 | 4 |
| 256 | E4 | 257 | 2056 | 1028 | 514 | 8280 | 54925 | 6 |
| 1024 | E3 | 1025 | 8200 | 4100 | 2050 | 32856 | 127835 | 105 |
| 1024 | E4 | 1025 | 8200 | 4100 | 2050 | 32856 | 221980 | 144 |
| 4096 | E3 | 4097 | 32776 | 16388 | 8194 | 131160 | 782250 | 24 |
| 4096 | E4 | 4097 | 32776 | 16388 | 8194 | 131160 | 1440985 | 44 |
| 16384 | E3 | 16385 | 131080 | 65540 | 32770 | 524376 | 3074700 | 74 |
| 16384 | E4 | 16385 | 131080 | 65540 | 32770 | 524376 | 5285560 | 126 |

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
