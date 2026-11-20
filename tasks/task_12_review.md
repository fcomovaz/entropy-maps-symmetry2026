# EGM review of task_12.md (consolidated)

Reviewers: zoom-out, comprehension, design critic, style editor, implementation readiness. Stages were run in parallel, not sequentially; there was no separate intermediate zoom-out pass.

## Blockers found (all addressed in revision 2)
| # | Finding | Source | Resolution |
|---|---|---|---|
| 1 | `PROGMEM` plan ignored every read site, the 64-bit bias, ESP8266 DRAM `.rodata`, and mlp weights being 32-bit (≈6.2 KiB, not 3 KiB); `f32` arrays and `<math.h>` not guarded | zoom-out, design, readiness | Stage 4 items 2–3 |
| 2 | AVR 2 KiB budget omitted the row buffer and the 9 asymmetries; 512-byte chunks overflowed the payload limit | comprehension, design | `band_rows = 1`, I.4, Stage 0 spike |
| 3 | Streaming `finish` reused functions needing a 2 720 B workspace; signatures missing | readiness | Stage 4 item 4 |
| 4 | Retried chunks would double-count pixels on AVR | comprehension, design | `expected_offset` rule (I.4, II.4) |
| 5 | Wire layouts undefined; `INFO` carried sizes a firmware cannot know | readiness, design | Fixed layouts in II.4 |
| 6 | `extern "C"` missing for Arduino C++ sketches | readiness | Stage 4 item 1 |
| 7 | `result_timeout_s` 60 s too short for PSRAM CNNs; global warm-up impossible on AVR | design, comprehension | Per-board/model overrides (II.3) |

## Major fixes
- Timing: flush before starting timer; per-iteration `t_total`; cold iteration reported; native-tick timers; AVR per-row error bound; `T_transfer_host` flagged on AVR.
- `src/c_export` and `cimp_platform.c` are no longer edited for registry/timer (staging-only registry, `.ino` defines the hook); streaming test is a separate target.
- Config: `boot_wait_s`, usable vs physical SRAM, flash app partition, per-board `warmup_iters`; CRC and `allow_resize` removed from TOML.
- Verdict taxonomy with order; `lr_dot` exempt from self-comparison; CLI flags `--model/--mode` added; dry-run needs no board; exception list extended; `run_id` Windows-safe.
- Readability: objective statement, glossary, ESP8266-vs-ESP32 explanation, Part I (paper-ready) vs Part II (runbook), KiB/B units, no duplicated file lists.

## Not adopted
- Dropping the protocol or the fake device (zoom-out): kept, but the fake device only replays golden rows.
- Dropping `fit_report.py` (zoom-out): kept; it is small and encodes the verdicts you asked for.
- Single multi-model firmware: left as an author decision (I.6).
