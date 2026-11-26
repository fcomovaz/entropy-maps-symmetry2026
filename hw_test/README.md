# Microcontroller Hardware Test Bench (`hw_test/`)

Reproducible **PC → MCU → PC** test bench verifying bit-exact numerical parity, compiler-measured memory footprint (Flash, static SRAM, heap minimum, stack high-water mark, and PSRAM), and on-device execution latency for the multiscale $\mathbf{x}_{94}$ descriptor, folded tabular classifiers (`lr_dot`, `mlp_16_dot`), and the 7 lightweight CNNs in `c_imp/`.

---

## 1. Architecture and Timing Principle

```text
PC (Python: src/hw_test + experiments/hw_test.py)
  │  1. HELLO  ─────────────────────────────▶  MCU (hw_avr / hw_esp8266 / hw_esp32)
  │  ◀───────────────────────────── INFO (65 B: board, model, mode, streaming, boot free RAM)
  │  2. IMG_BEGIN (id, 16384 B, CRC32, reps, warmup) + IMG_CHUNK (1..8 rows/chunk)
  │  ◀───────────────────────────── CHUNK_ACK (idempotent retries on lost ACK)
  │  ◀───────────────────────────── ACK_RX (full 16,384 B CRC32 verification)
  │  3. Inference timed EXCLUSIVELY on the MCU (never includes T_transfer_host)
  │  ◀───────────────────────────── RESULT (101 B: logits, label, x94_crc32, stats[5][3], RAM/PSRAM)
  │  4. [Optional --dump-descriptor] GET_X94 ──▶ RESULT_X94 (188 B Q15 / 376 B F32)
```

- **One firmware per model (Decision D4):** Each build stages only the runtime files and model headers required by the selected model into `hw_test/build/<board>_<model>_<mode>_<transport>/`, generating a single-model `cimp_models_registry.h` and `hw_build_config.h` in staging without modifying `src/c_export/` or `c_imp/runtime/cimp_platform.c`.
- **Arithmetic modes (Decision D3):** `q15` is the default mode across all boards; `f32` is enabled only on ESP8266/ESP32 and excluded for `torchvision_mobilenet` (`CIMP_ENABLE_F32=0` strips `<math.h>` and all `*_F32` weight/LUT arrays).
- **Images (Decision D7):** Strictly `128×128` 8-bit grayscale PNGs (`mode="L"`, `16,384` raw bytes) in `hw_test/img2send/`, with no automatic resizing or mode conversion (any invalid file aborts early listing all invalid files). Parity is verified against `results/c_imp/<target>/vectors/manifest.csv` and `c_imp/generated/<target>/golden_*.csv`; images outside the manifest report `parity = n/a`.

---

## 2. Binary Wire Protocol (UART and WiFi/TCP)

All frames are packed **little-endian** with no padding bytes:

```text
magic (u16 = 0x5345) | version (u8 = 1) | type (u8) | seq (u16) | length (u16) | payload[length] | crc16 (u16)
```

- **CRC16-CCITT-FALSE** (`poly = 0x1021`, `init = 0xFFFF`): covers `version` through the end of `payload` (table-free bitwise implementation in `hw_crc.h`, `0 B` RAM).
- **CRC32 IEEE 802.3** (`0xEDB88320`): verifies the complete `16,384` B image and the `x94` feature vector produced by the MCU.

| Message Type | Code | Direction | Payload Size | Fields |
|---|---:|---|---:|---|
| `HELLO` | `0x01` | PC → MCU | 0 B | Resets protocol state and requests board/model metadata |
| `INFO` | `0x02` | MCU → PC | 65 B | `board_id[16]`, `model[24]`, `mode`, `protocol`, `width`, `height`, `streaming`, `max_payload`, `free_ram_boot`, `psram_size`, `git_hash[8]` |
| `IMG_BEGIN` | `0x10` | PC → MCU | 16 B | `image_id`, `total`, `crc32`, `repeats`, `warmup` |
| `IMG_CHUNK` | `0x11` | PC → MCU | `4 + N` B | `offset` (`u32`) + raw band bytes (`128` B on AVR, `1024` B on ESP) |
| `CHUNK_ACK` | `0x12` | MCU → PC | 5 B | `offset`, `status` (`0` OK, `1` duplicate ignored, `<0` error) |
| `ACK_RX` | `0x13` | MCU → PC | 5 B | `image_id`, `crc_ok` (`1` valid, `0` mismatch) |
| `RESULT` | `0x20` | MCU → PC | 101 B | `image_id`, `status`, `label` (`i8`), `n_logits`, `logit_raw[2]`, `frac_bits`, `x94_crc32`, `timer_resolution_us`, `n_iters`, `stats[5][3]` (`first, min, median, mean, max` × `selector, model, total` in µs), `free_heap_min`, `stack_hwm`, `free_psram_min`, `arena_bytes`, `arena_location` |
| `RESULT_X94` | `0x21` | MCU → PC | 188 / 376 B | `94 × int16_t` (`q15`) or `94 × float` (`f32`) |
| `GET_X94` | `0x22` | PC → MCU | 0 B | Requests the last computed `x94` vector (tabular models only) |
| `ERROR` | `0x7F` | MCU → PC | `3 + L` B | `code` (`u16`), `len` (`<= 48` B), UTF-8 message |

Fields unavailable on a given platform (e.g., `free_heap_min` on AVR or `stack_hwm` on ESP8266) are encoded as `0xFFFFFFFF`.

---

## 3. Directory Layout

```text
code/
├── configs/hw_test/
│   ├── boards.toml              # Stage 0 measured budgets, FQBNs, per-model overrides, fit thresholds
│   ├── secrets.example.toml     # WiFi STA template (copy to secrets.toml for --transport wifi)
│   └── secrets.toml             # Local WiFi credentials (git-ignored)
├── hw_test/
│   ├── README.md                # This guide
│   ├── img2send/                # 128×128 grayscale ('L') PNGs to send to the MCU
│   ├── firmware/
│   │   ├── common/              # hw_crc.h, hw_protocol.{h,c}, hw_board.h, hw_runner.{h,c}
│   │   ├── hw_avr/hw_avr.ino    # 1-row streaming sketch (Nano / UNO)
│   │   ├── hw_esp8266/          # NodeMCU ESP-12E (UART or WiFi TCP server)
│   │   └── hw_esp32/            # ESP32-CAM (UART or WiFi TCP + heap_caps_malloc SRAM/PSRAM arena)
│   └── build/                   # Staged sketches and arduino-cli build artifacts (git-ignored)
├── src/hw_test/                 # config.py, protocol.py, transport.py, images.py, firmware.py, session.py, report.py, fit_report.py
├── experiments/hw_test.py       # Unified CLI (`python -m experiments.hw_test`)
├── tests/test_hw_*.py           # Unit & integration tests (FakeTransport + mocked arduino-cli)
└── results/hw_test/             # build_cache.json, <board>/<run_id>/ runs, fit_report.{csv,md}
```

---

## 4. Wiring and Boot Modes per Board

### Arduino Nano (`nano`) and UNO (`uno`)
- **Connection:** Direct USB cable (`UART` at `115200` baud).
- **Serial behavior:** `serial_reset_on_open = true`, `boot_wait_s = 2.0`. Opening the port resets the ATmega328P via DTR and waits for bootloader completion before sending `HELLO`.
- **CH340 Nano clones:** Default to `arduino:avr:nano:cpu=atmega328old`. If `arduino-cli upload` fails with `not in sync`, switch `fqbn` to `cpu=atmega328` in `configs/hw_test/boards.toml`.

### NodeMCU ESP-12E (`nodemcu`)
- **Connection:** Direct USB cable (`UART` at `460800` baud, or WiFi STA TCP port `5000`).
- **Serial behavior:** `serial_reset_on_open = false` (clears `DTR=False, RTS=False` prior to `open()` so the USB-UART bridge does not trigger auto-reset/flash circuitry during `send`).

### ESP32-CAM (`esp32cam`)
- **FQBN:** `esp32:esp32:esp32cam:PartitionScheme=huge_app` (`3,145,728` B app partition). **Do not append `PSRAM=enabled`:** in `esp32:esp32 3.3.11`, PSRAM is enabled automatically by the `esp32cam` board definition and `arduino-cli` rejects `PSRAM=enabled`.
- **With a generic USB-TTL adapter:** Wire `5V → 5V`, `GND → GND`, `U0T → RX`, `U0R → TX`. To flash, jumper **`GPIO0 → GND`** and press **RST**; after upload completes, remove the `GPIO0` jumper and press **RST** again.
- **With an ESP32-CAM-MB carrier board (CH340, verified on `COM3`/`COM5`):**
  - RTS auto-reset enters download mode automatically without manual `GPIO0` jumpering (pass `--no-prompt` to `bench` to skip the interactive pause).
  - **Upload speed vs. runtime UART speed:** `esptool v5.3.1` over CH340 fails at `460800`/`230400` after stub handoff; `boards.toml` sets **`upload_baud = 115200`** (`--upload-property upload.speed=115200`), while runtime binary protocol communication runs at **`uart_baud = 460800`** with `0` retries.
  - **No reset on serial open:** `serial_reset_on_open = false` is required on Windows so `pyserial` disables `DTR`/`RTS` before `open()` and does not hold the board in reset/bootloader mode.
  - **UART0 quiet rule:** In `UART` mode no text is printed to `Serial` because `UART0` carries binary protocol frames. In `WiFi` mode, the firmware prints a single `HW_TEST_WIFI_IP=<ip> PORT=<port>` line to `Serial` upon WiFi connection.

---

## 5. WiFi Credentials (`secrets.toml`)

Required only when `--transport wifi` is selected on `nodemcu` or `esp32cam`:

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code
Copy-Item configs/hw_test/secrets.example.toml configs/hw_test/secrets.toml
# Edit configs/hw_test/secrets.toml with your local WiFi SSID and password
```

`configs/hw_test/secrets.toml` is git-ignored and is never loaded for `uart` runs.

---

## 6. CLI Commands (`experiments.hw_test`)

Run all commands in PowerShell from `code/` using `.venv`:

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code

# 1. List configured boards from TOML and detected serial ports
.\.venv\Scripts\python.exe -m experiments.hw_test boards --config configs/hw_test/boards.toml

# 2. Validate images in hw_test/img2send/ and check manifest.csv parity matching
.\.venv\Scripts\python.exe -m experiments.hw_test images --config configs/hw_test/boards.toml --all

# 3. Dry-run checks (validate config, images, and toolchain without a connected board)
.\.venv\Scripts\python.exe -m experiments.hw_test build --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --dry-run
.\.venv\Scripts\python.exe -m experiments.hw_test send --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --transport uart --all --dry-run

# 4. Compile single-model firmware (updates results/hw_test/<board>/build_cache.json)
.\.venv\Scripts\python.exe -m experiments.hw_test build --config configs/hw_test/boards.toml --board esp32cam --model ultra_tiny --mode q15

# 5. Compile and flash a board over serial (e.g. ESP32-CAM on COM5 or Nano on COM4)
.\.venv\Scripts\python.exe -m experiments.hw_test flash --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --port COM5

# 6. Send images over UART (optional --dump-descriptor fetches all 94 x94 features on tabular models)
.\.venv\Scripts\python.exe -m experiments.hw_test send --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --transport uart --port COM5 --all --dump-descriptor

# 7. Send a single image over UART
.\.venv\Scripts\python.exe -m experiments.hw_test send --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --transport uart --port COM5 --image 20042.png

# 8. Send images over WiFi/TCP (after flashing with --transport wifi and reading the device IP)
.\.venv\Scripts\python.exe -m experiments.hw_test flash --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --transport wifi --port COM5
.\.venv\Scripts\python.exe -m experiments.hw_test send --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --transport wifi --host 192.168.1.50 --all

# 9. Unattended compile sweep across all models for a board without flashing
.\.venv\Scripts\python.exe -m experiments.hw_test bench --config configs/hw_test/boards.toml --board nano --compile-only

# 10. Compile all + flash and measure a subset of models on ESP32-CAM (--no-prompt skips GPIO0 pause)
.\.venv\Scripts\python.exe -m experiments.hw_test bench --config configs/hw_test/boards.toml --board esp32cam --transport uart --port COM5 --models lr_dot mlp_16_dot ultra_tiny --modes q15 f32 --all --no-prompt

# 11. Generate/refresh consolidated results/hw_test/fit_report.{csv,md}
.\.venv\Scripts\python.exe -m experiments.hw_test report --config configs/hw_test/boards.toml

# 12. Clean generated staging builds and/or run reports (always preserves results/hw_test/README.md and hw_test/img2send/)
.\.venv\Scripts\python.exe -m experiments.hw_test clean --config configs/hw_test/boards.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.hw_test clean --config configs/hw_test/boards.toml --results-only --keep-build-cache
.\.venv\Scripts\python.exe -m experiments.hw_test clean --config configs/hw_test/boards.toml --staging-only
.\.venv\Scripts\python.exe -m experiments.hw_test clean --config configs/hw_test/boards.toml

# 13. Run automated hw_test unit and integration tests
.\.venv\Scripts\python.exe -m pytest tests/test_hw_config.py tests/test_hw_protocol.py tests/test_hw_session.py tests/test_hw_cli.py tests/test_hw_fit.py -v
```

### Manual PowerShell Cleanup Snippet

Equivalent direct PowerShell cleanup (removes `hw_test/build/` and all generated files under `results/hw_test/` while keeping `results/hw_test/README.md` intact):

```powershell
if (Test-Path "hw_test/build") { Remove-Item -Recurse -Force "hw_test/build" }
if (Test-Path "results/hw_test") {
  Get-ChildItem "results/hw_test" -Exclude "README.md", ".gitkeep" | Remove-Item -Recurse -Force
}
```

---

## 7. AVR Timing Caveat (`nano` / `uno`)

The ATmega328P has only `2,048` B of physical SRAM and **cannot buffer a full `128×128` image (`16,384` B)**:

- Uses **row-by-row streaming** (`streaming = true`, `band_rows = 1`, `sizeof(CIMP_DescStream) = 1,060` B), updating histogram accumulators on each 128-byte chunk.
- Executes as a **compute-only single pass with no warm-up** (`warmup_iters = 0`, `timed_iters = 1`, UART wait time excluded).
- Because the 16 MHz AVR `micros()` timer has **4 µs resolution** and each of the 128 rows is timed individually, selector time carries a **quantization error bound of ±4 µs/row = ±512 µs** over 128 rows (`avr_quantization_bound_us = 512` in `run_meta.json` and `summary.json`).
- Host transfer time (`T_transfer_host_ms`) on AVR includes per-row descriptor compute prior to the final `ACK_RX` and is flagged **`includes_compute = true`**.

---

## 8. Adding a New Board in `configs/hw_test/boards.toml`

1. Add a `[hw_test.boards.<id>]` table specifying `fqbn`, `firmware` (`hw_avr`, `hw_esp8266`, or `hw_esp32`), `family` (`avr | esp8266 | esp32`), `transports`, `uart_baud`, `serial_reset_on_open`, `boot_wait_s`, `physical_sram_bytes`, `usable_sram_budget_bytes`, `flash_app_bytes`, `psram_bytes`, `cpu_hz`, `streaming`, `band_rows` (divisor of `128`), `max_payload_bytes` (`>= 4 + band_rows * 128`), `numeric_modes`, `models`, and `arena_location` (`none | sram | psram | auto`).
2. Optionally add `upload_baud` and per-model overrides under `[hw_test.boards.<id>.model_overrides.<model>]`.
3. Validate configuration and compilation via `experiments.hw_test build --dry-run` and `experiments.hw_test build`.
