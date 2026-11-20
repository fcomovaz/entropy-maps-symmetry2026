"""Unit and integration tests for `hw_test` session protocol and run reports.

Manual execution (PowerShell from `code/` directory):
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_hw_session.py -v
"""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hw_test.config import load_hw_test_config
from src.hw_test.images import GoldenLookup, prepare_images
from src.hw_test.protocol import (
    ARENA_LOC_NONE,
    ARENA_LOC_SRAM,
    CHUNK_STATUS_BAD_OFFSET,
    CHUNK_STATUS_DUPLICATE,
    CHUNK_STATUS_OK,
    CHUNK_STATUS_OVERFLOW,
    MSG_ACK_RX,
    MSG_CHUNK_ACK,
    MSG_ERROR,
    MSG_GET_X94,
    MSG_HELLO,
    MSG_IMG_BEGIN,
    MSG_IMG_CHUNK,
    MSG_INFO,
    MSG_RESULT,
    MSG_RESULT_X94,
    PROTOCOL_VERSION,
    UNAVAILABLE_U32,
    AckRxPayload,
    ChunkAckPayload,
    ErrorPayload,
    FrameDecoder,
    ImgBeginPayload,
    ImgChunkPayload,
    InfoPayload,
    ResultPayload,
    ResultX94Payload,
    TimingStats,
    TimingTriple,
    crc32_ieee,
    decode_payload,
    encode_ack_rx,
    encode_chunk_ack,
    encode_error,
    encode_frame,
    encode_info,
    encode_result,
    encode_result_x94,
    pack_f32_logit,
)
from src.hw_test.report import write_run_report
from src.hw_test.session import run_batch
from src.hw_test.transport import FakeTransport


class GoldenReplayDevice:
    """In-memory fake MCU answering protocol frames from manifest/golden CSVs."""

    def __init__(
        self,
        *,
        board_id: str,
        model: str,
        mode: str,
        streaming: bool,
        max_payload: int,
        is_cnn: bool,
        golden_lookup: GoldenLookup,
        drop_ack_for_offset: int | None = None,
        force_bad_crc: bool = False,
        override_q15_logit0: int | None = None,
    ) -> None:
        self.board_id = board_id
        self.model = model
        self.mode = mode
        self.streaming = streaming
        self.max_payload = max_payload
        self.is_cnn = is_cnn
        self.golden_lookup = golden_lookup
        self.drop_ack_for_offset = drop_ack_for_offset
        self._dropped_once = False
        self.force_bad_crc = force_bad_crc
        self.override_q15_logit0 = override_q15_logit0

        self.decoder = FrameDecoder()
        self.img_id = 0
        self.total = 0
        self.expected_crc32 = 0
        self.expected_offset = 0
        self.buf = bytearray()
        self.last_x94: ResultX94Payload | None = None

    def handle_bytes(self, tx_bytes: bytes) -> list[bytes]:
        self.decoder.feed(tx_bytes)
        out: list[bytes] = []
        for frame in self.decoder.decode_all():
            if frame.msg_type == MSG_HELLO:
                self.buf.clear()
                self.expected_offset = 0
                info = InfoPayload(
                    board_id=self.board_id,
                    model=self.model,
                    mode=self.mode,
                    protocol_version=PROTOCOL_VERSION,
                    width=128,
                    height=128,
                    streaming=self.streaming,
                    max_payload=self.max_payload,
                    free_ram_boot=616 if self.streaming else 200000,
                    psram_size=0 if self.streaming else 4194304,
                    git_hash="test123",
                )
                out.append(encode_frame(MSG_INFO, frame.seq, encode_info(info)))

            elif frame.msg_type == MSG_IMG_BEGIN:
                req = decode_payload(frame)
                assert isinstance(req, ImgBeginPayload)
                self.img_id = req.image_id
                self.total = req.total_bytes
                self.expected_crc32 = req.crc32
                self.expected_offset = 0
                self.buf.clear()
                self.last_x94 = None
                out.append(
                    encode_frame(
                        MSG_CHUNK_ACK,
                        frame.seq,
                        encode_chunk_ack(ChunkAckPayload(0xFFFFFFFF, CHUNK_STATUS_OK)),
                    )
                )

            elif frame.msg_type == MSG_IMG_CHUNK:
                chunk = decode_payload(frame)
                assert isinstance(chunk, ImgChunkPayload)
                if chunk.offset < self.expected_offset:
                    out.append(
                        encode_frame(
                            MSG_CHUNK_ACK,
                            frame.seq,
                            encode_chunk_ack(ChunkAckPayload(chunk.offset, CHUNK_STATUS_DUPLICATE)),
                        )
                    )
                    continue
                if chunk.offset > self.expected_offset:
                    out.append(
                        encode_frame(
                            MSG_CHUNK_ACK,
                            frame.seq,
                            encode_chunk_ack(ChunkAckPayload(chunk.offset, CHUNK_STATUS_BAD_OFFSET)),
                        )
                    )
                    continue
                if self.expected_offset + len(chunk.data) > self.total:
                    out.append(
                        encode_frame(
                            MSG_CHUNK_ACK,
                            frame.seq,
                            encode_chunk_ack(ChunkAckPayload(chunk.offset, CHUNK_STATUS_OVERFLOW)),
                        )
                    )
                    continue

                self.buf.extend(chunk.data)
                self.expected_offset += len(chunk.data)

                if (
                    self.drop_ack_for_offset is not None
                    and chunk.offset == self.drop_ack_for_offset
                    and not self._dropped_once
                ):
                    self._dropped_once = True
                    # Simulate lost ACK on the wire: do not append CHUNK_ACK
                    continue

                out.append(
                    encode_frame(
                        MSG_CHUNK_ACK,
                        frame.seq,
                        encode_chunk_ack(ChunkAckPayload(chunk.offset, CHUNK_STATUS_OK)),
                    )
                )

                if self.expected_offset == self.total:
                    actual_crc = crc32_ieee(self.buf)
                    crc_ok = (actual_crc == self.expected_crc32) and (not self.force_bad_crc)
                    out.append(
                        encode_frame(
                            MSG_ACK_RX,
                            frame.seq,
                            encode_ack_rx(AckRxPayload(self.img_id, crc_ok)),
                        )
                    )
                    if crc_ok:
                        out.append(self._build_result_frame(frame.seq))

            elif frame.msg_type == MSG_GET_X94:
                if self.last_x94 is None:
                    out.append(
                        encode_frame(
                            MSG_ERROR,
                            frame.seq,
                            encode_error(ErrorPayload(5, "x94 unavailable")),
                        )
                    )
                else:
                    out.append(
                        encode_frame(
                            MSG_RESULT_X94,
                            frame.seq,
                            encode_result_x94(self.last_x94),
                        )
                    )
        return out

    def _build_result_frame(self, seq: int) -> bytes:
        import hashlib

        sha = hashlib.sha256(self.buf).hexdigest()
        ref = self.golden_lookup.lookup(sha, self.model)

        x94_crc = 0
        if not self.is_cnn:
            if ref is not None and self.mode == "q15" and ref.expected_x94_q15 is not None:
                self.last_x94 = ResultX94Payload("q15", ref.expected_x94_q15)
            elif ref is not None and self.mode == "f32" and ref.expected_x94_f64 is not None:
                self.last_x94 = ResultX94Payload("f32", tuple(float(v) for v in ref.expected_x94_f64))
            else:
                self.last_x94 = ResultX94Payload(
                    self.mode,
                    tuple(0 for _ in range(94)) if self.mode == "q15" else tuple(0.0 for _ in range(94)),
                )
            x94_crc = crc32_ieee(encode_result_x94(self.last_x94))

        if ref is not None:
            if self.mode == "q15":
                label = ref.expected_q15_label
                l0 = (
                    self.override_q15_logit0
                    if self.override_q15_logit0 is not None
                    else ref.expected_q15_logits_raw[0]
                )
                l1 = ref.expected_q15_logits_raw[1] if self.is_cnn else 0
                frac_bits = 16
            else:
                label = ref.expected_f32_label
                l0 = pack_f32_logit(ref.expected_f32_logits[0])
                l1 = pack_f32_logit(ref.expected_f32_logits[1]) if self.is_cnn else 0
                frac_bits = 0
        else:
            label = 0
            l0 = 0
            l1 = 0
            frac_bits = 16 if self.mode == "q15" else 0

        sel_us = 0 if self.is_cnn else 4200
        mod_us = 12000 if self.is_cnn else 310
        tot_us = sel_us + mod_us
        triple = TimingTriple(sel_us, mod_us, tot_us)
        stats = TimingStats(triple, triple, triple, triple, triple)

        res = ResultPayload(
            image_id=self.img_id,
            status=0,
            label=label,
            n_logits=2 if self.is_cnn else 1,
            logit_raw0=l0,
            logit_raw1=l1,
            frac_bits=frac_bits,
            x94_crc32=x94_crc,
            timer_resolution_us=4 if self.streaming else 1,
            n_iters=1 if self.streaming else 10,
            stats=stats,
            free_heap_min=UNAVAILABLE_U32 if self.streaming else 140000,
            stack_hwm=612,
            free_psram_min=UNAVAILABLE_U32,
            arena_bytes=163840 if self.is_cnn else 0,
            arena_location=ARENA_LOC_SRAM if self.is_cnn else ARENA_LOC_NONE,
        )
        return encode_frame(MSG_RESULT, seq, encode_result(res))


class TestHwSession(unittest.TestCase):
    """Verify `run_batch`, retry/duplicate handling, parity checks, and Stage 7 reports."""

    def setUp(self) -> None:
        self.base_cfg = load_hw_test_config(Path("configs/hw_test/boards.toml"), PROJECT_ROOT)
        self.golden = GoldenLookup.from_config(self.base_cfg)

    def test_streaming_tabular_session_with_temp_img2send_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            img_dir = tmp_root / "img2send"
            results_dir = tmp_root / "results"
            img_dir.mkdir(parents=True)
            for name in ("20000.png", "20042.png", "20056.png"):
                shutil.copy2(self.base_cfg.paths.images_dir / name, img_dir / name)
            # Add one unknown PNG outside manifest -> parity="n/a"
            Image.fromarray(np.full((128, 128), 77, dtype=np.uint8), mode="L").save(
                img_dir / "99999_unknown.png"
            )

            cfg = replace(
                self.base_cfg,
                paths=replace(self.base_cfg.paths, images_dir=img_dir, results_dir=results_dir),
                protocol=replace(self.base_cfg.protocol, ack_timeout_s=0.05),
            )
            images = prepare_images(img_dir, cfg.images, select_all=True)
            self.assertEqual(len(images), 4)

            dev = GoldenReplayDevice(
                board_id="nano",
                model="lr_dot",
                mode="q15",
                streaming=True,
                max_payload=132,
                is_cnn=False,
                golden_lookup=self.golden,
                drop_ack_for_offset=256,  # exercises timeout retry + duplicate chunk status=1
            )
            transport = FakeTransport(handler=dev.handle_bytes, max_read_chunk=7)
            with transport:
                batch = run_batch(
                    transport,
                    cfg,
                    board="nano",
                    images=images,
                    model="lr_dot",
                    mode="q15",
                    dump_descriptor=True,
                    golden_lookup=self.golden,
                    transport_name="uart",
                    port_or_host="COM4",
                )

            statuses = {r.image.name: r.parity_status for r in batch.records}
            self.assertEqual(statuses["20000.png"], "pass")
            self.assertEqual(statuses["20042.png"], "pass")
            self.assertEqual(statuses["20056.png"], "pass")
            self.assertEqual(statuses["99999_unknown.png"], "n/a")
            self.assertTrue(all(r.includes_compute for r in batch.records))
            self.assertTrue(all(r.dumped_x94 is not None for r in batch.records))

            paths = write_run_report(cfg, batch, host_git_hash="deadbee")
            self.assertTrue(paths.run_meta_path.is_file())
            self.assertTrue(paths.per_image_csv_path.is_file())
            self.assertTrue(paths.summary_json_path.is_file())
            self.assertIsNotNone(paths.x94_dump_path)
            assert paths.x94_dump_path is not None
            self.assertTrue(paths.x94_dump_path.is_file())
            self.assertNotIn(":", paths.run_id)

            summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["parity"]["n_pass"], 3)
            self.assertEqual(summary["parity"]["n_fail"], 0)
            self.assertEqual(summary["parity"]["n_na"], 1)
            self.assertEqual(summary["parity"]["parity_rate"], 1.0)
            self.assertEqual(summary["avr_quantization_bound_us"], 512)

            with paths.per_image_csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            # Tabular models fill logit0 only and leave logit1 blank
            self.assertNotEqual(rows[0]["logit0"], "")
            self.assertEqual(rows[0]["logit1"], "")
            self.assertEqual(rows[0]["free_heap_min"], "")

    def test_cnn_q15_and_f32_parity_and_mismatch_failures(self) -> None:
        images = prepare_images(self.base_cfg.paths.images_dir, self.base_cfg.images, image="20042.png")

        # 1. CNN f32 pass
        dev_f32 = GoldenReplayDevice(
            board_id="esp32cam",
            model="ultra_tiny",
            mode="f32",
            streaming=False,
            max_payload=1028,
            is_cnn=True,
            golden_lookup=self.golden,
        )
        with FakeTransport(handler=dev_f32.handle_bytes) as tr:
            batch_f32 = run_batch(
                tr,
                self.base_cfg,
                board="esp32cam",
                images=images,
                model="ultra_tiny",
                mode="f32",
                golden_lookup=self.golden,
                transport_name="wifi",
                port_or_host="192.168.1.50",
            )
        self.assertEqual(batch_f32.records[0].parity_status, "pass")
        self.assertIsNotNone(batch_f32.records[0].margin_float)

        # 2. Q15 bit-exact mismatch detected
        dev_bad_q15 = GoldenReplayDevice(
            board_id="esp32cam",
            model="ultra_tiny",
            mode="q15",
            streaming=False,
            max_payload=1028,
            is_cnn=True,
            golden_lookup=self.golden,
            override_q15_logit0=-999999,
        )
        with FakeTransport(handler=dev_bad_q15.handle_bytes) as tr:
            batch_bad = run_batch(
                tr,
                self.base_cfg,
                board="esp32cam",
                images=images,
                model="ultra_tiny",
                mode="q15",
                golden_lookup=self.golden,
            )
        self.assertEqual(batch_bad.records[0].parity_status, "fail")
        self.assertFalse(batch_bad.records[0].parity_ok)

        # 3. Firmware identity mismatch fails fast
        with FakeTransport(handler=dev_f32.handle_bytes) as tr:
            with self.assertRaises(RuntimeError):
                run_batch(
                    tr,
                    self.base_cfg,
                    board="esp32cam",
                    images=images,
                    model="ds_cnn",
                    mode="f32",
                )

        # 4. Bad ACK_RX CRC fails fast
        dev_crc = GoldenReplayDevice(
            board_id="esp32cam",
            model="ultra_tiny",
            mode="q15",
            streaming=False,
            max_payload=1028,
            is_cnn=True,
            golden_lookup=self.golden,
            force_bad_crc=True,
        )
        with FakeTransport(handler=dev_crc.handle_bytes) as tr:
            with self.assertRaises(RuntimeError):
                run_batch(
                    tr,
                    self.base_cfg,
                    board="esp32cam",
                    images=images,
                    model="ultra_tiny",
                    mode="q15",
                )


if __name__ == "__main__":
    unittest.main()
