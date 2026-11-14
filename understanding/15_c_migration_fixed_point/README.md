# Lab 15: Fixed-Point Arithmetic, Shannon Entropy LUTs, BN Folding, and CNN Requantization

## Purpose

Following the **Understanding-First Rule**, this laboratory validates the numerical foundations required before migrating `x94` feature extraction, `lr_dot` / `mlp_16_dot`, and the 7 lightweight CNN architectures into portable C99 for desktop verification and future ESP32-CAM deployment.

## Experiments

| Script | Topic | Key Verification |
|---|---|---|
| `01_q_formats.py` | `Qm.n` representation, `q15_mul_round`, saturation vs wrap-around | Tests `-32768 * -32768 -> 32767` saturation and Q0.15 error bounds (`< 5e-5`) |
| `02_entropy_fixed_point.py` | `nlogn_full`, `term_per_level`, and `log2_mantissa` (`clz` + 257-entry LUT) | Compares max/mean error vs `float64` oracle and LUT flash size (~64 KB vs ~44 KB vs ~0.5 KB) |
| `03_bn_folding.py` | Analytical `Conv2d + BatchNorm2d` folding into `(W_fold, b_fold)` | Verifies `max |Conv(fold) - BN(Conv)| < 1e-6` in `eval()` mode |
| `04_requantization.py` | Power-of-two `int16` + `int64` accumulator + `shift_out` across fan-in `Cin*k^2` | Shows why `int32` overflows past 2 MACs in Q15 and validates `int64` headroom |

## Design Recommendation for Stage 4 / `configs/c_imp/*.toml`

- **Feature storage**: `Q0.15` (`feature_frac_bits = 15`), saturating `1.0` to `32767` (max error `3.05e-5`).
- **Rounding**: `half_up` (`+ (1 << (shift - 1))`), matching the issue's `q15_mul_round` kernel and single-cycle add-and-shift hardware instructions.
- **Shannon LUT mode**: `"log2_mantissa"` (`log2_mantissa_bits = 8`, 257 entries, ~514 B in flash) is recommended as the MCU default because it keeps max error below `2e-3` while reducing flash usage by >100× compared to `nlogn_full`; all three modes remain supported cleanly for desktop trade-off checks.
- **Accumulators**: Use `int64_t` in portable C99 reference kernels so arbitrary `Cin * k^2` fan-in never overflows `int32_t`.

## Manual Execution (PowerShell from `code/`)

```powershell
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/01_q_formats.py
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/02_entropy_fixed_point.py
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/03_bn_folding.py
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/04_requantization.py
.\.venv\Scripts\python.exe -m unittest discover understanding/15_c_migration_fixed_point/tests -v
```
