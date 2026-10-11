| Method | Median ns/pixel across all cases | Median speedup vs H1 | Role |
|---|---:|---:|---|
| H1 direct-address list | 53.699 | 1.000x | embedded candidate |
| H1-local direct-address list | 57.688 | 0.980x | embedded candidate |
| H2 array.array | 92.270 | 0.608x | embedded candidate |
| H3 collections.Counter | 45.141 | 1.481x | mapping/hash |
| H4 defaultdict | 97.987 | 0.590x | mapping/hash |
| H5 unroll x4 | 74.630 | 0.748x | embedded candidate |
| H5 unroll x8 | 71.924 | 0.765x | embedded candidate |
| H6 Pillow histogram [library/C implementation baseline] | 1.971 | 29.408x | native library |
