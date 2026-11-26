# Hardware Test Bench Results (`results/hw_test/`)

Compiler-measured footprints and physical microcontroller execution results for the portable C99 pipeline (`PC → MCU → PC`), produced by [experiments/hw_test.py](../../experiments/hw_test.py) following [hw_test/README.md](../../hw_test/README.md).

> [!NOTE]
> Heavy per-run directories under `results/hw_test/*` (`<board>/<run_id>/`, `build_cache.json`) and local `fit_report.{csv,md}` files are git-ignored (and can be cleaned at any time with `python -m experiments.hw_test clean`); this tracked `README.md` preserves the audited Stage 0, Stage 5, and Stage 9 measurements.

---

## Stage 0 — Compiler Output Mapping and AVR Size Spike (`measured`)

Measured with `arduino-cli 1.5.2-rc.1` (`arduino:avr 1.8.8`, `esp8266:esp8266 3.1.2`, `esp32:esp32 3.3.11`) using `--format json` on a trivial empty sketch (`setup`/`loop`) and an AVR streaming spike linking `CIMP_DescStream` + `lr_dot` + `PROGMEM`.

### Section Mapping in `arduino-cli compile --format json` (`measured`)

| Core / Board (`FQBN`) | JSON / Output Field Parsed | Flash Used / Max (Trivial Sketch) | Static RAM Used / Max (Trivial Sketch) | Core Notes |
|---|---|---|---|---|
| `arduino:avr:nano:cpu=atmega328old` | `builder_result.executable_sections_size` (`text`, `data`) | `444 / 30,720 B` (`measured`) | `9 / 2,048 B` (`measured`) | `text` = Flash, `data` = static SRAM (globals + statics). |
| `esp8266:esp8266:nodemcuv2` | `compiler_err` (`sizes.py` regex; `executable_sections_size` absent) | `231,620 / 1,048,576 B` (`IROM`, `measured`) | `28,008 / 80,192 B` (`DATA=1496, RODATA=920, BSS=25592`, `measured`); `IRAM=59,143 / 65,536 B` | Unannotated `const` (`RODATA`) stays in DRAM; requires `CIMP_ROM` (`PROGMEM`) + `pgm_read_*`. |
| `esp32:esp32:esp32cam:PartitionScheme=huge_app` | `builder_result.executable_sections_size` (`text`, `data`) | `267,520 / 3,145,728 B` (`measured`) | `22,172 / 327,680 B` (`measured`) | No `PSRAM=enabled` menu option in `boards.txt` (PSRAM is enabled automatically by the board definition). |

### AVR ATmega328P Size Spike (`arduino:avr:nano:cpu=atmega328old`, `measured`)

Spike sketch with streaming state (`band_hist[8*16]` + `coarse_hist[21*16]` + `fine_ent[64]`, `1,060` B), `x94` vector (`188` B), RX buffer (`148` B), `Serial`, LUTs in `PROGMEM`, and `lr_dot` classifier (`int32_t` weights in `PROGMEM`, `int64_t` accumulator):

- **Flash (`text`):** `3,864 / 30,720 B` (`12.6 %`, `measured`)
- **Static SRAM (`data`):** `1,432 / 2,048 B` (`69.9 %`, `measured`) → `616 B` free for stack.
- **Finding:** Streaming descriptor + `lr_dot` fits comfortably in ATmega328P; `mlp_16_dot` (~`6.2` KiB weights in `PROGMEM`) adds `0` B of static SRAM for weights.

---

## Stage 5 — Cross-Board `arduino-cli` Compilation (`measured`)

All **25 configured builds** across the 4 boards compiled with `compile_ok = True` and `fits_budget = True` (cached in `results/hw_test/<board>/build_cache.json`). All 7 CNNs are marked `does_not_fit` on AVR and ESP8266 because their activation arenas (`163,840 B` to `2,162,688 B` `[predicted]`) exceed usable SRAM (`1,792 B` on AVR, `45,056 B` on ESP8266 without PSRAM), and `torchvision_mobilenet` is restricted to `q15` (Decision D3).

| Board (`core`) | Model | Mode | Flash Used / Max (`measured`) | Static RAM Used / Max (`measured`) | Status |
|---|---|---|---:|---:|---|
| `nano` (`arduino:avr 1.8.8`) | `lr_dot` | `q15` | `12,780 / 30,720 B` | `1,636 / 2,048 B` | `fits_budget` (`412 B` stack headroom) |
| `nano` (`arduino:avr 1.8.8`) | `mlp_16_dot` | `q15` | `19,038 / 30,720 B` | `1,636 / 2,048 B` | `fits_budget` (`412 B` stack headroom) |
| `uno` (`arduino:avr 1.8.8`) | `lr_dot` | `q15` | `12,780 / 32,256 B` | `1,636 / 2,048 B` | `fits_budget` (`412 B` stack headroom) |
| `uno` (`arduino:avr 1.8.8`) | `mlp_16_dot` | `q15` | `19,038 / 32,256 B` | `1,636 / 2,048 B` | `fits_budget` (`412 B` stack headroom) |
| `nodemcu` (`esp8266 3.1.2`) | `lr_dot` | `q15` | `243,572 / 1,048,576 B` | `49,204 / 80,192 B` | `fits_budget` |
| `nodemcu` (`esp8266 3.1.2`) | `lr_dot` | `f32` | `243,556 / 1,048,576 B` | `49,396 / 80,192 B` | `fits_budget` |
| `nodemcu` (`esp8266 3.1.2`) | `mlp_16_dot` | `q15` | `249,796 / 1,048,576 B` | `49,200 / 80,192 B` | `fits_budget` |
| `nodemcu` (`esp8266 3.1.2`) | `mlp_16_dot` | `f32` | `249,540 / 1,048,576 B` | `49,392 / 80,192 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `lr_dot` | `q15` / `f32` | `289,344` / `289,636 / 3,145,728 B` | `43,116` / `43,300 / 327,680 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `mlp_16_dot` | `q15` / `f32` | `295,512` / `295,608 / 3,145,728 B` | `43,116` / `43,300 / 327,680 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `ultra_tiny` | `q15` / `f32` | `292,616` / `293,676 / 3,145,728 B` | `40,204` / `40,204 / 327,680 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `ds_cnn` | `q15` / `f32` | `317,260` / `339,556 / 3,145,728 B` | `40,204` / `40,204 / 327,680 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `tiny_cnn` | `q15` / `f32` | `336,764` / `382,244 / 3,145,728 B` | `40,204` / `40,204 / 327,680 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `mobilenet_like` | `q15` / `f32` | `436,548` / `568,520 / 3,145,728 B` | `40,204` / `40,204 / 327,680 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `tiny_resnet` | `q15` / `f32` | `445,408` / `596,368 / 3,145,728 B` | `40,204` / `40,204 / 327,680 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `small_cnn` | `q15` / `f32` | `485,592` / `678,272 / 3,145,728 B` | `40,204` / `40,204 / 327,680 B` | `fits_budget` |
| `esp32cam` (`esp32 3.3.11`) | `torchvision_mobilenet` | `q15` | `1,928,852 / 3,145,728 B` | `40,204 / 327,680 B` | `fits_budget` |

---

## Stage 9 — Physical ESP32-CAM Execution (`COM5`, UART `460800`, `2026-11-26`, `measured`)

Hardware tested: **ESP32-CAM (`ESP32-D0WDQ6`, 4 MiB Flash, `4,194,304` B PSRAM)** mounted on an **ESP32-CAM-MB (CH340)** carrier on `COM5` (`upload_baud = 115200` for `esptool v5.3.1` stub stability; runtime UART protocol at `460800` with `serial_reset_on_open = false` and `0` retries). Executed **12 physical runs × 8 images** (`hw_test/img2send/`, `20000..20003.png` and `20042, 20056, 20068, 20084.png`), achieving **96/96 images with `100.0 %` parity**:

- **All `q15` models (9/9):** `0` bit-for-bit raw logit difference vs. desktop Python/C (`max_logit_abs_diff = 0.0`) and `8/8` `x94_crc_ok` on tabular models.
- **All tested `f32` models (3/3):** maximum logit difference $\le 5.66\times 10^{-6}$ (well below `f32_logit_abs_tol = 1e-4`) and `8/8` `x94_crc_ok` on tabular models.
- **Contiguous internal DRAM finding on `ultra_tiny/q15`:** Although the `q15` arena is `163,840 B` ($\le 200,000$ B budget) and `free_heap_min = 213,208 B`, the `auto` policy fell back to **PSRAM (`arena_location = psram`)** because the largest single contiguous free block in ESP32 internal DRAM is under `160 KiB`. Even in PSRAM, `ultra_tiny/q15` finished in `170.008 ms` ($\le 500$ ms) and remains **`fits_viable`**.

| Model | Mode | Arena `[predicted]` (`location` `[measured]`) | `free_heap_min` (`measured`) | `stack_hwm` (`measured`) | `free_psram_min` (`measured`) | `T_sel` ms (`measured`) | `T_model` ms (`measured`) | `T_total` ms (`measured`) | Parity (`measured`) | ROC-AUC (`offline`) | Verdict (`max_latency = 500 ms`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `lr_dot` | `q15` | `0 B` (`none`) | `210,296 B` | `6,908 B` | `4,192,124 B` | `3.178` | `0.012` | **`3.191`** | `8/8` (`diff=0`) | `0.8063` | `fits_viable` |
| `mlp_16_dot` | `q15` | `0 B` (`none`) | `210,296 B` | `6,908 B` | `4,192,124 B` | `3.179` | `0.187` | **`3.366`** | `8/8` (`diff=0`) | `0.8335` | `fits_viable` |
| `lr_dot` | `f32` | `0 B` (`none`) | `210,112 B` | `6,892 B` | `4,192,124 B` | `5.663` | `0.085` | **`5.748`** | `8/8` (`5.66e-6`) | `0.8061` | `fits_viable` |
| `mlp_16_dot` | `f32` | `0 B` (`none`) | `210,112 B` | `6,892 B` | `4,192,124 B` | `5.668` | `1.405` | **`7.073`** | `8/8` (`2.75e-6`) | `0.8334` | `fits_viable` |
| `ultra_tiny` | `q15` | `163,840 B` (`psram`) | `213,208 B` | `4,944 B` | `4,024,184 B` | `0.000` | `170.008` | **`170.008`** | `8/8` (`diff=0`) | `0.8815` | `fits_viable` |
| `ultra_tiny` | `f32` | `327,680 B` (`psram`) | `213,208 B` | `4,928 B` | `3,856,248 B` | `0.000` | `677.834` | `677.834` | `8/8` (`7.11e-7`) | `0.8815` | `fits_exceeds_latency` |
| `ds_cnn` | `q15` | `1,081,344 B` (`psram`) | `213,208 B` | `4,944 B` | `3,078,008 B` | `0.000` | `1991.234` | `1991.234` | `8/8` (`diff=0`) | `0.8778` | `fits_exceeds_latency` |
| `torchvision_mobilenet` | `q15` | `327,680 B` (`psram`) | `213,208 B` | `4,944 B` | `3,856,248 B` | `0.000` | `3031.870` | `3031.870` | `8/8` (`diff=0`) | `0.7411` | `fits_exceeds_latency` |
| `mobilenet_like` | `q15` | `524,288 B` (`psram`) | `213,208 B` | `4,944 B` | `3,651,448 B` | `0.000` | `3421.959` | `3421.959` | `8/8` (`diff=0`) | `0.8623` | `fits_exceeds_latency` |
| `tiny_cnn` | `q15` | `1,081,344 B` (`psram`) | `213,208 B` | `4,944 B` | `3,078,008 B` | `0.000` | `6195.723` | `6195.723` | `8/8` (`diff=0`) | `0.8713` | `fits_exceeds_latency` |
| `small_cnn` | `q15` | `1,081,344 B` (`psram`) | `213,208 B` | `4,944 B` | `3,078,008 B` | `0.000` | `8629.045` | `8629.045` | `8/8` (`diff=0`) | `0.8925` | `fits_exceeds_latency` |
| `tiny_resnet` | `q15` | `425,984 B` (`psram`) | `213,208 B` | `4,944 B` | `3,757,944 B` | `0.000` | `8664.952` | `8664.952` | `8/8` (`diff=0`) | `0.9213` | `fits_exceeds_latency` |

Full 53-row (`board × model × mode`) fit tables are generated via `python -m experiments.hw_test report --config configs/hw_test/boards.toml` at `results/hw_test/fit_report.csv` and `results/hw_test/fit_report.md`.

Five publication-grade MDPI 300-DPI figures summarizing all 4 MCUs together plus one dedicated multi-panel figure per microcontroller are generated under `results/hw_test/figures/` via `.\.venv\Scripts\python.exe -m experiments.visualize_hw_test --config configs/visualization/hw_test.toml --overwrite`:
- `results/hw_test/figures/fig1_cross_board_all_mcus.png` — All 4 MCUs combined (feasibility matrix, arena/ceiling ratios, and live ROC-AUC vs. latency Pareto across 16 runs).
- `results/hw_test/figures/fig2_board_esp32cam.png` — `ESP32-CAM` (`COM5`: Flash/PSRAM usage, 12-run latency breakdown vs. `500 ms`, 96-sample parity).
- `results/hw_test/figures/fig3_board_nano.png` — `Arduino Nano` (`COM4`: streaming 2 KB SRAM vs. impossible CNN arenas, per-image timing, 16/16 Q15 parity).
- `results/hw_test/figures/fig4_board_uno.png` — `Arduino Uno` (`COM6`: Optiboot footprint, per-image Nano-vs-Uno `<0.04%` clock reproducibility, 16/16 Q15 parity).
- `results/hw_test/figures/fig5_board_nodemcu.png` — `NodeMCU / ESP-12E` (`0 B` PSRAM: `4/4` tabular compile fit + `COM5` `ch341ser.sys` note vs. `14/14` mathematical CNN impossibility).

---

## Live 8-bit AVR: Arduino Nano (`COM4`) & Arduino Uno (`COM6`) (`ATmega328P`, `2026-11-26`)

Hardware tested: **Arduino Nano (`ATmega328P`, 8-bit @ 16 MHz, `30,720 B` Flash, `2,048 B` SRAM, `0 B` PSRAM; Optiboot `cpu=atmega328` over FTDI FT232R `COM4`)** and **Arduino Uno (`ATmega328P`, 8-bit @ 16 MHz, `32,256 B` Flash, `2,048 B` SRAM, `0 B` PSRAM; Optiboot over ATmega16U2 `COM6`)** (`uart_baud = 115200`, `serial_reset_on_open = true`, row-by-row `CIMP_DescStream` with `band_rows = 1`, `max_payload = 132 B`). Executed **4 physical runs × 8 images** (`hw_test/img2send/`), achieving **32/32 images with `100.0 %` bit-exact parity (`diff = 0`)**:

| Board | Model | Mode | Flash / Max (`measured`) | Static RAM / Max (`measured`) | `free_ram_boot` (`measured`) | `stack_hwm` (`measured`) | `T_sel` ms (`measured`) | `T_model` ms (`measured`) | `T_total` ms (`measured`) | Parity (`measured`) | Verdict (`max_latency = 500 ms`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `nano` | `lr_dot` | `q15` | `12,780 / 30,720 B` (`41.6%`) | `1,636 / 2,048 B` (`79.9%`) | `222 B` | `146 B` | `76.838` | `1.576` | **`78.418`** | `8/8` (`diff=0`) | `fits_viable` |
| `nano` | `mlp_16_dot` | `q15` | `19,038 / 30,720 B` (`62.0%`) | `1,636 / 2,048 B` (`79.9%`) | `230 B` | `154 B` | `77.020` | `24.830` | **`101.844`** | `8/8` (`diff=0`) | `fits_viable` |
| `uno` | `lr_dot` | `q15` | `12,780 / 32,256 B` (`39.6%`) | `1,636 / 2,048 B` (`79.9%`) | `222 B` | `146 B` | `76.816` | `1.574` | **`78.396`** | `8/8` (`diff=0`) | `fits_viable` |
| `uno` | `mlp_16_dot` | `q15` | `19,038 / 32,256 B` (`59.0%`) | `1,636 / 2,048 B` (`79.9%`) | `230 B` | `154 B` | `77.060` | `24.830` | **`101.882`** | `8/8` (`diff=0`) | `fits_viable` |

All 7 CNNs (`q15`) are physically impossible on `nano` and `uno` (`does_not_fit`): predicted activation arenas (`163,840–1,081,344 B`) exceed `1,792 B` usable SRAM by `91.4×–603.4×`. *(Engineering note: official Arduino Uno ATmega16U2 `usbser.sys` requires `boot_wait_s = 2.5` and keeping a constant `Serial` per-read timeout so mid-stream `SetCommTimeouts` calls do not drop USB-CDC bytes.)*

---

## NodeMCU / ESP-12E (`nodemcu`) Feasibility & Live `COM5` Attempt (`2026-11-26`)

- **Compiled & Feasible Tabular Models (`4/4` `compile_ok = True`):**
  - `lr_dot/q15`: `243,572 / 1,048,576 B` IROM (`23.2%`), `49,204 / 80,192 B` DRAM (`61.4%`), `384 B` weights, `0 B` arena.
  - `lr_dot/f32`: `243,556 / 1,048,576 B` IROM (`23.2%`), `49,396 / 80,192 B` DRAM (`61.6%`), `384 B` weights, `0 B` arena.
  - `mlp_16_dot/q15`: `249,796 / 1,048,576 B` IROM (`23.8%`), `49,200 / 80,192 B` DRAM (`61.4%`), `6,216 B` weights, `0 B` arena.
  - `mlp_16_dot/f32`: `249,540 / 1,048,576 B` IROM (`23.8%`), `49,392 / 80,192 B` DRAM (`61.6%`), `6,216 B` weights, `0 B` arena.
- **Why All 7 CNNs Are Physically Impossible on ESP8266 (`14/14` `does_not_fit`):**
  - ESP8266 (`nodemcuv2`) has **`0 B` external PSRAM** and only `45,056 B` usable heap (out of `80,192 B` physical DRAM after SDK/WiFi/stack reserve). Even the smallest CNN (`ultra_tiny/q15`) requires a `163,840 B` activation arena (`3.64×` usable DRAM and `2.04×` total physical DRAM), while `ds_cnn`, `tiny_cnn`, and `small_cnn` require `1,081,344 B` (`q15`) to `2,162,688 B` (`f32`) (`24.0×–48.0×` usable DRAM). Furthermore, `torchvision_mobilenet` weights (`1,633,714 B`) exceed the entire `1,048,576 B` (`1 MiB`) IROM app partition (`1.56×`).
- **Live `COM5` UART Attempt (`2026-11-26`):** `nodemcu_lr_dot_q15_uart` compiled and flashed via `arduino-cli`, but host serial configuration (`pyserial` at `115200`/`230400`/`460800` and `.NET System.IO.Ports.SerialPort` at `115200`) failed with Windows `ERROR_GEN_FAILURE` (code 31) on `USB-SERIAL CH340` (`VID_1A86:PID_7523`) under WCH driver `ch341ser.sys 3.9.2024.9` (which rejects `SetCommState()` on non-genuine CH340 bridges). Live `nodemcu` latency therefore remains `not_measured` on this host unless run via an FTDI/CP2102 USB-TTL adapter, an older `ch341ser.sys` driver, or `WiFi/TCP` (`secrets.toml`).

---

## Items Not Verified on Physical Hardware

- **Board `nodemcu` (ESP8266):** All 4 tabular configs (`lr_dot`, `mlp_16_dot` in `q15` and `f32`) are compile-verified (`compile_ok = True`), and `nodemcu_lr_dot_q15_uart` flashed, but live UART on `COM5` was blocked at host `SetCommState()` by WCH `ch341ser.sys 3.9.2024.9` (all 7 CNNs on AVR and ESP8266 are mathematically `does_not_fit` due to SRAM/Flash/0-PSRAM limits).
- **`WiFi/TCP` transport (`nodemcu` and `esp32cam`):** Verified with `TcpTransport` loopback and mock socket server, **not verified on physical hardware** (requires local WiFi credentials in `configs/hw_test/secrets.toml`).
- **Remaining 5 `f32` PSRAM CNNs on `esp32cam` (`ds_cnn`, `tiny_cnn`, `mobilenet_like`, `tiny_resnet`, `small_cnn`):** Compiled in Stage 5 (`fits_budget = True`, marked `not_measured`), **not executed on physical hardware**.
