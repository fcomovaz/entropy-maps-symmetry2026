| Method | Median ns/pixel across all cases | Median speedup vs H1 | Role |
|---|---:|---:|---|
| H1 direct-address list | 56.709 | 1.000x | embedded candidate |
| H1-local direct-address list | 58.495 | 0.996x | embedded candidate |
| H2 array.array | 94.070 | 0.625x | embedded candidate |
| H3 collections.Counter | 44.021 | 1.384x | mapping/hash |
| H4 defaultdict | 96.874 | 0.513x | mapping/hash |
| H5 unroll x4 | 73.748 | 0.762x | embedded candidate |
| H5 unroll x8 | 74.810 | 0.780x | embedded candidate |
| H6 Pillow histogram [library/C implementation baseline] | 1.657 | 31.428x | native library |
