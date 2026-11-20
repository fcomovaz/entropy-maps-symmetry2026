# Task 12 — Microcontroller test bench (PC → MCU → PC)

> Revision 2 (after EGM review, see `tasks/task_12_review.md`). Do not execute anything from this file until the author approves it.
> Part I explains the experiment (reusable for the paper's Methods section). Part II is the build runbook for the implementing agent.
> Read first: `AGENTS.md`, `c_imp/README.md`, `tasks/task_11.md`, `src/c_export/` (all modules).

## Objective

The C99 pipeline in `c_imp/` is verified on the desktop (`experiments.verify_c_pipeline`, bit-exact `q15` parity with Python). This task runs it on real microcontrollers and answers three questions:

1. Does it give the **same outputs** on the device as on the desktop?
2. How long does it take **on the device** (descriptor time and model time, never the time to send the image)?
3. Which models **fit** in each board's memory, and which fit but are not worth running?

Boards: Arduino Nano (ATmega328P; the UNO uses the same chip), ESP-12E NodeMCU v3 (ESP8266), ESP32-CAM. The Nano is connected first.
Images are sent from the PC over UART (all boards) or WiFi/TCP (ESP8266 and ESP32 only). `arduino-cli` (with the `arduino:avr`, `esp8266:esp8266` and `esp32:esp32` cores already installed) builds and flashes.

---

# Part I — Experiment design (paper-ready)

## I.1 Terms used

| Term | Meaning |
|---|---|
| `x94` descriptor | 94 numbers per image: 85 Shannon entropies of 16-bin intensity histograms over a spatial pyramid of 1×1, 2×2, 4×4 and 8×8 regions (1+4+16+64 = 85), plus 9 asymmetry values (left-right, top-bottom, 180° rotation, each at scales 2, 4, 8) computed **from the 85 entropies**, not from pixels. |
| "selector" time | The project's name for descriptor extraction (`T_selector`). `T_total = T_selector + T_model` (AGENTS.md). For CNNs, which read the raw image, `T_selector = 0`. |
| `q15` | Signed 16-bit fixed point with 15 fractional bits (`int16_t`, value = integer / 32768). Logits are `int32_t` with the fractional bits reported by the pipeline. |
| arena | Pre-reserved scratch buffer for CNN layer activations; its size is `peak_arena_q15_bytes` in `results/c_imp/<target>/memory_report.json`. |
| Models | `lr_dot` (logistic regression on `x94`), `mlp_16_dot` (MLP, 16 hidden units, on `x94`), and CNNs on the raw 128×128 image: `ultra_tiny`, `ds_cnn`, `tiny_cnn`, `mobilenet_like`, `tiny_resnet`, `small_cnn`, `torchvision_mobilenet`. |
| Task | KolektorSDD2 surface-defect detection (binary defect/good), 128×128 8-bit grayscale (config: `configs/c_imp/kolektorsdd2.toml`). |

## I.2 Hardware limits (usable values go in the TOML; sources must be cited there)

Image = 128 × 128 × 1 B = 16 384 B. Physical memory is not usable memory: the SDK, WiFi stack and bootloader take a share.

| Board | CPU | Physical RAM | Usable for this app (to be measured) | Flash (physical → app partition) | Holds a whole image? |
|---|---|---|---|---|---|
| Nano / UNO | ATmega328P, 16 MHz, 8-bit, no FPU | 2 KiB | ≈ 1.6 KiB after core buffers | 32 KiB → ≈ 30 KiB | **No** |
| NodeMCU ESP-12E | ESP8266, 80/160 MHz, no FPU | ≈ 80 KiB DRAM | ≈ 40–45 KiB free heap with WiFi | 4 MiB → ≈ 1 MiB app | Yes (≈ 36–41 % of free heap) |
| ESP32-CAM | ESP32, 240 MHz, FPU | 520 KiB (≈ 300 KiB usable) + 4 MiB PSRAM | ≈ 300 KiB + PSRAM | 4 MiB → 3 MiB (`huge_app`) | Yes |

### Why the ESP8266 cannot run what the ESP32-CAM runs

Flash is not the problem (the ESP8266 has 4 MiB, the weights are small). **Activation memory is.** `ultra_tiny`, the smallest CNN, needs a 163 840 B `q15` arena (the `f32` arena is 327 680 B). The ESP8266 has about 40–45 KiB free and no PSRAM. The other six CNNs need 327 680 B – 1 081 344 B. Only `ultra_tiny` could fit the ESP32's internal RAM; the rest need its PSRAM. Making a CNN fit on the ESP8266 would need layer fusion or tiling, new kernels that break the bit-exact `q15` parity with the desktop; that is out of scope. **The ESP8266 therefore runs the descriptor + `lr_dot` + `mlp_16_dot` only**, and the report marks the CNNs `does_not_fit` with the arena size as the reason.

## I.3 Timing method

1. The PC sends the image. The device checks its CRC32 and replies `ACK_RX`. The device calls `flush()` on its transmit buffer, **then** starts its own timer. Transfer time is measured on the PC in a separate column (`T_transfer_host`) and is never added to any device time.
2. The device repeats `[selector, model]` for `warmup_iters` untimed and `timed_iters` timed iterations on the same stored image. For each timed iteration it records `t_selector`, `t_model` and `t_total = t_selector + t_model` (the same iteration; statistics are taken over the per-iteration totals, so median of totals ≠ sum of medians). The first timed iteration (cold caches) is reported separately from min/median/mean/max of the rest.
3. Timers count in native ticks (`micros()` ≥ 1 µs; 4 µs on AVR 16 MHz) as `uint32_t`; conversion to nanoseconds happens once, on the PC. The timer resolution is written to `run_meta.json`.
4. WiFi runs have background interrupts that UART runs do not. Both transports are measured for the same model; the difference is reported, not hidden.

### AVR exception (cannot store the image)

On the Nano the image arrives one 128-byte row at a time and each row updates the histograms immediately. The timer runs only around descriptor-update calls (UART waiting is excluded), plus the finish step (levels 4, 2, 1 and the 9 asymmetries), plus the model. So AVR timing is **compute-only, single run, no warm-up**, and the paper must say so. Each row call is timed separately with a 4 µs timer, so the selector time carries up to ±4 µs per row (±512 µs over 128 rows) quantization error; this bound is reported next to the result. `T_transfer_host` on AVR includes selector compute (the device ACKs after the last row) and is flagged `includes_compute = true`.

## I.4 Streaming descriptor (AVR)

State kept in SRAM: one band of 8 fine cells of 16 bins (8×16×2 B = 256 B), coarse accumulators for 4×4 (512 B), 2×2 (128 B), 1×1 (32 B), and the final 85 `q15` entropies (170 B) + 9 asymmetries (18 B): ≈ 1.1 KiB. The row buffer is one row (128 B). Row buffer + state + Arduino serial buffers + stack must be ≤ 2 KiB: **verify with a compile-size spike (Stage 0) before changing runtime code.**

When 16 image rows (one 8×8-grid band) are complete, the 8 level-8 entropies of that band are computed, the band is added to the coarse accumulators, and the band is cleared. `finish` computes levels 4, 2, 1 with the existing single-cell entropy routine (made non-`static`) and the 9 asymmetries with the existing `cimp_asymmetry_q15` on the 85 entropies. Required invariant: **bit-identical** `x94` `q15` output to `cimp_descriptor_x94_q15` for every test image.

Duplicate chunks (a lost ACK and a retry) must not be counted twice: the device keeps `expected_offset` and answers an already-seen offset with the ACK only.

## I.5 Verdict taxonomy (Stage 8)

| Verdict | Rule (evaluated in this order) |
|---|---|
| `does_not_fit` | Compile fails, or flash / usable SRAM / PSRAM budget exceeded (reason column says which). |
| `not_measured` | Fits, but no device run exists yet. |
| `fits_exceeds_latency` | Median `t_total` > `max_latency_ms`. |
| `fits_dominated` | Another fitting model on the same board is faster **and** has higher offline AUC. `lr_dot` is the baseline and is never judged against itself. |
| `fits_viable` | None of the above. |

Every report value is tagged `predicted` (from `memory_report.json`) or `measured` (from compile output or device). Offline AUC/F1 are read from the existing `results/` files (computed on the desktop, in `f32` unless the file says otherwise) and are labeled `offline`.

## I.6 Decision D4 — one firmware per model, or one firmware with several models?

**Adopted: one firmware per model for measurement; no multi-model firmware initially.**

| | One firmware per model | One firmware, runtime model switch |
|---|---|---|
| Flash/RAM numbers | Exact per model, straight from the compiler | Mixed: you measure the sum, not one model |
| Fit decisions | Clean (a model either compiles or not) | A big model can hide or break a small one |
| Fits on AVR / ESP8266 | Required (2 KiB / ~40 KiB RAM) | Impossible for CNNs; only useful for `lr_dot` + `mlp_16_dot` |
| Cost | Build + flash per model. Builds are unattended; the ESP32-CAM needs a manual GPIO0→GND + reset on every upload | One flash |

Mitigations: `bench` builds **all** model/board combinations unattended (compile-only gives the fit table); only the ones you choose with `--models` are flashed and run. If ESP32-CAM re-flashing becomes too slow, add a multi-model ESP32 profile later; `torchvision_mobilenet` (1.6 MB weights) stays separate. Decision for the author: confirm or request the multi-model ESP32 profile.

---

# Part II — Implementation runbook

Rules: all numbers and paths come from TOML (AGENTS.md). Frozen dataclasses, type hints, `logging`, `main(argv) -> int`. Work on branch `feat/hw-test`. Every stage ends with a **DoD** verified by execution. Whatever needs a physical board and was not run is reported as **not verified on hardware**.

## II.1 Decisions

| # | Decision |
|---|---|
| D1 | Timing is on the device only (I.3). |
| D2 | One binary protocol over UART and TCP. Per-chunk ACKs are **required on UART** and also sent on TCP (one parser). |
| D3 | Default arithmetic `q15`. `f32` is built only for ESP8266/ESP32 and only for models that are not `torchvision_mobilenet`; builds without `f32` must drop all `*_F32` arrays and `<math.h>` (compile guard `CIMP_ENABLE_F32`). |
| D4 | Per-model firmware (I.6). Model chosen at stage time by writing a single-model `cimp_models_registry.h` into the staging folder; **no change to `src/c_export`** and no change to `cimp_platform.c`. |
| D5 | WiFi: ESP is a TCP server (STA). Credentials in `configs/hw_test/secrets.toml` (git-ignored; template `secrets.example.toml`); required only when `wifi` is used. |
| D6 | New dependency: `pyserial` only. |
| D7 | Images: PNG 128×128 mode `L` in `hw_test/img2send/`, no resizing (wrong size = error). Golden outputs come from `results/c_imp/<target>/vectors/manifest.csv` + `c_imp/generated/<target>/golden_*.csv`. Images outside the manifest report `parity = n/a`. |
| D8 | UNO and Nano are two TOML entries sharing one firmware (different FQBN). Nano clones usually need `arduino:avr:nano:cpu=atmega328old`; fallback `cpu=atmega328` is a TOML edit. |

## II.2 Layout

```text
code/
├── configs/hw_test/{boards.toml, secrets.example.toml, secrets.toml(git-ignored)}
├── hw_test/
│   ├── README.md
│   ├── img2send/            # PNGs to send (+ .gitkeep, a few samples)
│   ├── firmware/{common/, hw_avr/, hw_esp8266/, hw_esp32/}   # common: protocol, CRC, board probes, state machine
│   └── build/               # staging, git-ignored
├── src/hw_test/{config,protocol,transport,images,firmware,session,fit_report,report}.py
├── experiments/hw_test.py   # CLI
├── c_imp/runtime/           # MODIFIED (see Stage 4)
├── c_imp/tests/test_streaming.c   # NEW (separate target)
├── src/c_export/build_c.py  # MODIFIED: new function building test_streaming.c (with generated_dir include + cimp_descriptor.c)
├── tests/test_hw_*.py       # NEW
├── results/hw_test/<board>/<run_id>/  # outputs (git-ignored except README.md)
├── requirements.txt, .gitignore, c_imp/README.md, results/README.md   # MODIFIED
```

## II.3 `configs/hw_test/boards.toml` — schema (values are proposals; cite the source of each in a comment)

Global tables: `[hw_test]` (`c_imp_config`, `secrets_config`, `seed`), `[hw_test.paths]` (`images_dir`, `firmware_dir`, `staging_dir`, `results_dir`, `runtime_dir`), `[hw_test.images]` (`allowed_extensions`, `width`, `height`, `mode`), `[hw_test.protocol]` (`magic`, `version`, `ack_timeout_s`, `result_timeout_s`, `max_retries`; the CRC type is fixed in code, not configurable), `[hw_test.timing]` (`warmup_iters`, `timed_iters`), `[hw_test.fit]` (`max_latency_ms`, metric source dirs), `[hw_test.arduino_cli]` (`executable`, `compile_extra_args`).

Per-board keys (override globals where they overlap):
`fqbn`, `firmware`, `family`, `transports`, `uart_baud` (ESP default 460800, 921600 optional), `boot_wait_s`, `tcp_port` (wifi boards), `physical_sram_bytes`, `usable_sram_budget_bytes` (physical minus SDK/WiFi/stack reserve), `flash_app_bytes`, `psram_bytes`, `cpu_hz`, `streaming`, `band_rows` (AVR = 1), `max_payload_bytes` (counts the whole `payload[length]` field including the 4-byte offset), `warmup_iters`, `timed_iters`, `result_timeout_s` (per-board and per-model override, e.g. PSRAM CNNs get fewer repeats and a longer timeout), `numeric_modes`, `models`, `arena_location` (`sram|psram|auto`; `auto` compares against `usable_sram_budget_bytes`).
Board overrides: `nano`/`uno` → `streaming = true`, `band_rows = 1`, `warmup_iters = 0`, `timed_iters = 1`, `numeric_modes = ["q15"]`, `models = ["lr_dot","mlp_16_dot"]`. `nodemcu` → `models = ["lr_dot","mlp_16_dot"]`. `esp32cam` → all 9 models, `PartitionScheme=huge_app`, `PSRAM=enabled`.
Ports are CLI flags (`--port`, `--host`), never in TOML.

## II.4 Wire protocol (shared; freeze exact layouts in `protocol.py` and `hw_protocol.h`, little-endian, packed)

Frame: `magic u16 | version u8 | type u8 | seq u16 | length u16 | payload | crc16 u16` (CRC16-CCITT-FALSE over `version…payload`; AVR uses the bitwise form, and its CRC32 is the table-free nibble or bitwise form, because a 1 KiB table does not fit).

| Type | Direction | Fixed payload layout |
|---|---|---|
| `HELLO` 0x01 | PC→dev | empty |
| `INFO` 0x02 | dev→PC | `board_id char[16]`, `model char[24]`, `mode u8`, `protocol u8`, `width u16`, `height u16`, `streaming u8`, `max_payload u16`, `free_ram_boot u32`, `psram_size u32`, `git_hash char[8]` (NUL-padded strings). Compile-time flash/RAM sizes are **not** sent (a firmware cannot know its own link size); the PC reads them from the compiler output. |
| `IMG_BEGIN` 0x10 | PC→dev | `image_id u32`, `total u32`, `crc32 u32`, `repeats u16`, `warmup u16` |
| `IMG_CHUNK` 0x11 | PC→dev | `offset u32`, bytes |
| `CHUNK_ACK` 0x12 | dev→PC | `offset u32`, `status i8` |
| `ACK_RX` 0x13 | dev→PC | `image_id u32`, `crc_ok u8` |
| `RESULT` 0x20 | dev→PC | `image_id u32`, `status i16`, `label i8`, `n_logits u8`, `logit_raw i32[2]`, `frac_bits u8`, `x94_crc32 u32` (0 for CNNs), `timer_resolution_us u8`, `n_iters u16`, then 5×(selector, model, total) `u32` µs for `first, min, median, mean, max`, `free_heap_min u32`, `stack_hwm u32`, `free_psram_min u32`, `arena_bytes u32`, `arena_location u8`. Fields a board cannot measure (stack on ESP8266, heap minimum on AVR) are `0xFFFFFFFF` = "not available". |
| `RESULT_X94` 0x21 | dev→PC | 94×`i16` (debug, `--dump-descriptor`) |
| `ERROR` 0x7F | dev→PC | `code u16`, `len u8`, text ≤ 48 B |

State machine rules: `HELLO` or `IMG_BEGIN` in any state resets the state; a TCP disconnect resets it. Streaming devices compute only after a chunk is fully received and `CHUNK_ACK` is sent, so the host waits for the ACK before sending the next row (this also protects the 64-byte AVR hardware UART buffer). No logging on the serial port after `HELLO` (UART0 is the protocol channel on ESP32-CAM).

## II.5 Stages

### Stage 0 — Spikes (no repo changes)
1. Run `arduino-cli compile --format json` for a trivial sketch on the three FQBNs; record the actual JSON section names for flash and RAM per core; write them into the `firmware.py` plan.
2. AVR size spike: compile a sketch that links the current `cimp_descriptor.c` (q15) + `lr_dot` + a 128 B row buffer and a 512 B stub to see flash/RAM. If it cannot fit, stop and report; do not alter the math.
- **DoD:** a short table in `results/hw_test/README.md` (spike numbers, labeled measured).

### Stage 1 — Config
`src/hw_test/config.py`: frozen dataclasses `HwTestConfig`, `BoardSpec`, `ProtocolSpec`, `TimingSpec`, `FitSpec`, `ImagesSpec`, `ArduinoCliSpec`, `Paths`, `WifiSecrets`; `load_hw_test_config(path, project_root)` and `load_wifi_secrets(path)` (secrets only loaded for wifi use). Reuse the `_require` pattern from `src/c_export/config.py`. Fail fast: missing keys (name the key), `transports` not in {uart, wifi}, wifi without `tcp_port`, models not in the c_imp config, image size ≠ descriptor size, `band_rows` not dividing 128, budgets ≤ 0. The loader does **not** check that firmware directories exist (that is `firmware.py`'s job). Add `pyserial` (pinned) to `requirements.txt` and install into `.venv`. No defaults in code.
- **DoD:** `pytest tests/test_hw_config.py` (valid + each invalid case).

### Stage 2 — Protocol and transports
`protocol.py` (encode/decode with the exact layouts above, resync on magic, CRC16, `zlib.crc32`), `transport.py` (`Transport` interface via `typing.Protocol`; `SerialTransport` waits `boot_wait_s` after open; `TcpTransport`; `FakeTransport`).
- **DoD:** `pytest tests/test_hw_protocol.py`: round trip every type, bad CRC rejected, split/merged reads, resync after garbage, duplicate offset handling.

### Stage 3 — Images and session (no hardware)
`images.py` (sorted discovery, `--image`/`--glob`/`--all`, fail-fast list of bad files, CRC32), `session.py` (`run_image`, `run_batch`, records `T_transfer_host` and `includes_compute`). Test double: a small fake device in `tests/` that answers with the golden rows from the manifest (no re-implementation of the algorithm).
- **DoD:** `pytest tests/test_hw_session.py` sends images from a temp `img2send/` and gets golden values back.

### Stage 4 — C runtime changes (desktop-verified)
All changes are opt-in so desktop output does not change.
1. `extern "C"` guards (`#ifdef __cplusplus`) in every `c_imp/runtime/*.h` (Arduino `.ino` files are C++).
2. `CIMP_ENABLE_F32` guard around `f32` kernels, `*_F32` weight arrays and `#include <math.h>` (default on for desktop; `c_writer.py` emits arrays under the guard).
3. `CIMP_ROM` / `CIMP_ROM_READ_{U8,I16,I32,I64,…}` in `cimp_fixed_point.h`: empty on desktop; `PROGMEM` + `pgm_read_*` on AVR **and ESP8266** (plain `const` on ESP8266 stays in DRAM). `I64` = two `pgm_read_dword`. Replace every dereference of a ROM table in `cimp_descriptor.c`, `infra_sklearn.c` (including the `int64_t b1_acc` bias) and `cimp_pipeline.c` with the macros. `c_writer.py` emits `CIMP_ROM` only when a flag is set; headers that include only `<stdint.h>` must include `cimp_fixed_point.h` when the flag is on. Note: `mlp_16_dot` weights are 32-bit (≈ 6.2 KiB `q15` path), not 3 KiB.
4. Streaming API in `cimp_descriptor.{h,c}`: `int cimp_desc_stream_init(CIMP_DescStream *s)`, `int cimp_desc_stream_push_rows(CIMP_DescStream *s, const uint8_t *rows, int32_t n_rows)`, `int cimp_desc_stream_finish_q15(CIMP_DescStream *s, q15_t out_x94[CIMP_N_X94])` (return `CIMP_OK` or negative). `cimp_eval_single_cell_q15` becomes non-`static`. `CIMP_DescStream` holds no `CIMP_DescriptorWorkspace` (that struct is 2 720 B).
5. The platform hook stays as is (weak `cimp_mcu_time_ns`); each `.ino` defines it as `extern "C"`. Do not edit `cimp_platform.c`.
6. New `c_imp/tests/test_streaming.c`: for the 4 embedded test images and `band_rows ∈ {1,2,4,8,16}`, streaming `x94` == `cimp_descriptor_x94_q15` bit for bit; print `sizeof(CIMP_DescStream)`. New builder function in `src/c_export/build_c.py` (include path = `generated_dir`, links `cimp_descriptor.c`); `test_runtime.c` stays standalone.
- **DoD:** `python -m experiments.verify_c_pipeline --config configs/c_imp/kolektorsdd2.toml` still passes; `test_streaming` passes; `sizeof(CIMP_DescStream)` is recorded.

### Stage 5 — Firmware and build staging
`hw_test/firmware/common/` (protocol, CRC16/CRC32, `hw_board.h` probes: AVR stack-pointer gap, ESP8266 `ESP.getFreeHeap()`, ESP32 `heap_caps_get_minimum_free_size` + `uxTaskGetStackHighWaterMark` + `ESP.getPsramSize()`; unavailable → `0xFFFFFFFF`) and one `.ino` per family. Model, mode, baud, payload size, repeats come from a generated `hw_build_config.h`. ESP32 arena: `heap_caps_malloc` once at boot (SRAM or PSRAM per `arena_location`), not a static array; ESP8266/AVR use static buffers. Inference stays allocation-free.
`src/hw_test/firmware.py`: `stage_sketch` (copies only the needed `common/`, runtime and generated files for the one model, writes `hw_build_config.h` and a single-model `cimp_models_registry.h` **in staging**; selects files per family so AVR does not compile `infra_pytorch.c`), `compile` (`arduino-cli compile --format json --build-path …`; parse sizes using the Stage 0 mapping per family), `upload`. Fail fast with the exact `arduino-cli core install …` message.
- **DoD:** `build` succeeds for `lr_dot`/`q15` on all three FQBNs and for every ESP32 CNN (or a logged `does_not_fit` with the compiler message).

### Stage 6 — CLI (`python -m experiments.hw_test <cmd> --config configs/hw_test/boards.toml`)
`boards` (list + `arduino-cli board list`; unknown ports shown as such), `images`, `build --board --model --mode`, `flash --board --model --mode --port`, `send --board --model --mode --transport {uart,wifi} (--port|--host) (--image|--glob|--all) [--repeats] [--dump-descriptor]`, `bench --board --transport (--port|--host) [--models …]` (compile all unattended; flash/run only the selected ones, prompting before each ESP32-CAM upload), `report`.
`build_parser()`, `main(argv=None) -> int`, `logging`; top-level catch of `OSError, ValueError, KeyError, RuntimeError, serial.SerialException, TimeoutError` → `SystemExit("error: …")`. `--dry-run` validates config/images/toolchain but does **not** require a connected board.
- **DoD:** `pytest tests/test_hw_cli.py` (injected `FakeTransport`, mocked `arduino-cli` runner).

### Stage 7 — Results and parity
`results/hw_test/<board>/<run_id>/`, `run_id = YYYYMMDDTHHMMSSZ_<model>_<mode>_<transport>` (no `:`; Windows-safe). Files: `run_meta.json` (config snapshot, git hash, FQBN, core and arduino-cli versions, timer resolution, port/host), `per_image.csv` (`image, label, logit0, logit1, margin, golden_*, parity_ok, x94_crc_ok, first/min/median/mean/max for selector/model/total, T_transfer_host_ms, includes_compute, free_heap_min, stack_hwm`; tabular models fill `logit0` only), `summary.json`. Parity: `q15` must be 100 % bit-exact on logits (CNNs: both logits); `f32` within `c_imp.verification` tolerances (`f32_logit_abs_tol` in `configs/c_imp/*.toml`). A mismatch is a defect to debug, not a result to tune.
- **DoD:** fake-device run writes all files; hardware parity rate recorded per board.

### Stage 8 — Fit report
`fit_report.py` → `results/hw_test/fit_report.{csv,md}` per (board, model, mode): predicted flash/arena (`memory_report.json`), measured flash/RAM (compiler), measured runtime memory, latency, offline AUC/F1, verdict (I.5), every column tagged `predicted|measured|offline`.
- **DoD:** unit tests for verdict order (including `lr_dot` baseline and `not_measured`) on synthetic inputs.

### Stage 9 — Hardware runs (needs the author)
Nano on COM4 first (`flash --board nano --model lr_dot --mode q15 --port COM4`; if upload says "not in sync", switch the FQBN `cpu=` in TOML), then `mlp_16_dot`; then NodeMCU (UART, then WiFi); then ESP32-CAM (GPIO0→GND + reset to flash). Finish with `report`.
- **DoD:** per-board `summary.json` + `fit_report.md`; unrun items listed as not verified.

### Stage 10 — Docs
`hw_test/README.md` (Spanish like the other READMEs: wiring, ESP32-CAM boot mode, secrets, every command, AVR timing caveat, adding a board), update `c_imp/README.md` (streaming API, `CIMP_ROM`, `CIMP_ENABLE_F32`, `extern "C"`), `results/README.md`, project `README.md` command list.

## II.6 Risks to verify during implementation
1. AVR fit (flash and the 2 KiB SRAM) — resolved by the Stage 0 spike, not by assumption.
2. AVR 64-byte UART buffer: protected by one-row chunks and ACK-before-next-row.
3. ESP8266 watchdog: `yield()` only between repeats, outside timed regions.
4. ESP32-CAM has no USB: needs a USB-TTL adapter; UART0 doubles as the log port. Stable baud may be below 921600.
5. `torchvision_mobilenet` on ESP32: 1.6 MB weights, PSRAM arena; likely `fits_exceeds_latency` or `does_not_fit`, but hypothesis only until measured. It also requires a long `result_timeout_s` and fewer repeats.
6. AVR timer: 4 µs resolution and per-row timing error (I.3).

## II.7 Files touched
New: everything under `configs/hw_test/`, `hw_test/`, `src/hw_test/`, `experiments/hw_test.py`, `c_imp/tests/test_streaming.c`, `tests/test_hw_{config,protocol,session,cli,fit}.py`, `results/hw_test/README.md`.
Modified: `c_imp/runtime/{cimp_descriptor.h,cimp_descriptor.c,cimp_fixed_point.h,infra_sklearn.c,cimp_pipeline.c}` and all runtime headers (`extern "C"`), `src/c_export/{c_writer.py,luts.py,linear_export.py,build_c.py}` (guarded/opt-in output only), `requirements.txt`, `.gitignore`, `c_imp/README.md`, `results/README.md`, `README.md`.
