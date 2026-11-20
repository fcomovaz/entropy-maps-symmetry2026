"""Unit tests for `hw_test` wire protocol, transports, and image validation/golden lookup.

Manual execution (PowerShell from `code/` directory):
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_hw_protocol.py -v
"""

from __future__ import annotations

import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hw_test.config import ImagesSpec, load_hw_test_config
from src.hw_test.images import GoldenLookup, discover_images, load_images, prepare_images
from src.hw_test.protocol import (
    ACK_RX_STRUCT,
    ARENA_LOC_PSRAM,
    ARENA_LOC_SRAM,
    CHUNK_ACK_STRUCT,
    CHUNK_STATUS_BAD_OFFSET,
    CHUNK_STATUS_DUPLICATE,
    CHUNK_STATUS_OK,
    ERROR_TEXT_MAX_BYTES,
    HEADER_STRUCT,
    IMG_BEGIN_STRUCT,
    INFO_STRUCT,
    MAGIC,
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
    RESULT_STRUCT,
    RESULT_X94_F32_STRUCT,
    RESULT_X94_Q15_STRUCT,
    UNAVAILABLE_U32,
    AckRxPayload,
    ChunkAckPayload,
    ErrorPayload,
    FrameDecoder,
    GetX94Payload,
    HelloPayload,
    ImgBeginPayload,
    ImgChunkPayload,
    InfoPayload,
    ProtocolError,
    ResultPayload,
    ResultX94Payload,
    TimingStats,
    TimingTriple,
    crc16_ccitt_false,
    crc32_ieee,
    decode_frame,
    decode_payload,
    encode_ack_rx,
    encode_chunk_ack,
    encode_error,
    encode_frame,
    encode_img_begin,
    encode_img_chunk,
    encode_info,
    encode_result,
    encode_result_x94,
    pack_f32_logit,
    unpack_f32_logit,
)
from src.hw_test.transport import FakeTransport, SerialTransport, TcpTransport, Transport


def _bitwise_crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc


class TestHwProtocol(unittest.TestCase):
    """Validate wire struct sizes, CRC routines, round trips, resync, and error handling."""

    def test_struct_sizes_and_crc_vectors(self) -> None:
        self.assertEqual(HEADER_STRUCT.size, 8)
        self.assertEqual(INFO_STRUCT.size, 65)
        self.assertEqual(IMG_BEGIN_STRUCT.size, 16)
        self.assertEqual(CHUNK_ACK_STRUCT.size, 5)
        self.assertEqual(ACK_RX_STRUCT.size, 5)
        self.assertEqual(RESULT_STRUCT.size, 101)
        self.assertEqual(RESULT_X94_Q15_STRUCT.size, 188)
        self.assertEqual(RESULT_X94_F32_STRUCT.size, 376)

        self.assertEqual(crc16_ccitt_false(b"123456789"), 0x29B1)
        sample = bytes(range(256))
        self.assertEqual(crc16_ccitt_false(sample), _bitwise_crc16_ccitt_false(sample))
        self.assertEqual(crc32_ieee(b"123456789"), 0xCBF43926)

    def test_round_trip_every_message_type(self) -> None:
        stats = TimingStats(
            first=TimingTriple(100, 200, 300),
            minimum=TimingTriple(90, 180, 270),
            median=TimingTriple(95, 190, 285),
            mean=TimingTriple(96, 192, 288),
            maximum=TimingTriple(110, 210, 320),
        )
        raw_x94_q15 = tuple(i - 40 for i in range(94))
        raw_x94_f32 = tuple(float(i) * 0.125 for i in range(94))
        x94_q15_crc = crc32_ieee(encode_result_x94(ResultX94Payload("q15", raw_x94_q15)))

        f32_l0 = pack_f32_logit(-1.25)
        f32_l1 = pack_f32_logit(2.75)

        cases = [
            (MSG_HELLO, 1, b"", HelloPayload()),
            (MSG_GET_X94, 2, b"", GetX94Payload()),
            (
                MSG_INFO,
                3,
                encode_info(
                    InfoPayload(
                        board_id="esp32cam",
                        model="torchvision_mobilenet",
                        mode="q15",
                        protocol_version=PROTOCOL_VERSION,
                        width=128,
                        height=128,
                        streaming=False,
                        max_payload=1028,
                        free_ram_boot=245760,
                        psram_size=4194304,
                        git_hash="abc1234",
                    )
                ),
                InfoPayload(
                    board_id="esp32cam",
                    model="torchvision_mobilenet",
                    mode="q15",
                    protocol_version=PROTOCOL_VERSION,
                    width=128,
                    height=128,
                    streaming=False,
                    max_payload=1028,
                    free_ram_boot=245760,
                    psram_size=4194304,
                    git_hash="abc1234",
                ),
            ),
            (
                MSG_IMG_BEGIN,
                4,
                encode_img_begin(
                    ImgBeginPayload(image_id=7, total_bytes=16384, crc32=0x12345678, repeats=10, warmup=3)
                ),
                ImgBeginPayload(image_id=7, total_bytes=16384, crc32=0x12345678, repeats=10, warmup=3),
            ),
            (
                MSG_IMG_CHUNK,
                5,
                encode_img_chunk(ImgChunkPayload(offset=128, data=bytes(range(128)))),
                ImgChunkPayload(offset=128, data=bytes(range(128))),
            ),
            (
                MSG_CHUNK_ACK,
                6,
                encode_chunk_ack(ChunkAckPayload(offset=0xFFFFFFFF, status=CHUNK_STATUS_OK)),
                ChunkAckPayload(offset=0xFFFFFFFF, status=CHUNK_STATUS_OK),
            ),
            (
                MSG_CHUNK_ACK,
                7,
                encode_chunk_ack(ChunkAckPayload(offset=256, status=CHUNK_STATUS_BAD_OFFSET)),
                ChunkAckPayload(offset=256, status=CHUNK_STATUS_BAD_OFFSET),
            ),
            (
                MSG_ACK_RX,
                8,
                encode_ack_rx(AckRxPayload(image_id=7, crc_ok=True)),
                AckRxPayload(image_id=7, crc_ok=True),
            ),
            (
                MSG_RESULT,
                9,
                encode_result(
                    ResultPayload(
                        image_id=7,
                        status=0,
                        label=1,
                        n_logits=1,
                        logit_raw0=296482,
                        logit_raw1=0,
                        frac_bits=16,
                        x94_crc32=x94_q15_crc,
                        timer_resolution_us=4,
                        n_iters=1,
                        stats=stats,
                        free_heap_min=UNAVAILABLE_U32,
                        stack_hwm=616,
                        free_psram_min=UNAVAILABLE_U32,
                        arena_bytes=0,
                        arena_location=0,
                    )
                ),
                ResultPayload(
                    image_id=7,
                    status=0,
                    label=1,
                    n_logits=1,
                    logit_raw0=296482,
                    logit_raw1=0,
                    frac_bits=16,
                    x94_crc32=x94_q15_crc,
                    timer_resolution_us=4,
                    n_iters=1,
                    stats=stats,
                    free_heap_min=UNAVAILABLE_U32,
                    stack_hwm=616,
                    free_psram_min=UNAVAILABLE_U32,
                    arena_bytes=0,
                    arena_location=0,
                ),
            ),
            (
                MSG_RESULT,
                10,
                encode_result(
                    ResultPayload(
                        image_id=8,
                        status=-2,
                        label=-1,
                        n_logits=2,
                        logit_raw0=f32_l0,
                        logit_raw1=f32_l1,
                        frac_bits=0,
                        x94_crc32=0,
                        timer_resolution_us=1,
                        n_iters=5,
                        stats=stats,
                        free_heap_min=140000,
                        stack_hwm=2048,
                        free_psram_min=3000000,
                        arena_bytes=163840,
                        arena_location=ARENA_LOC_PSRAM,
                    )
                ),
                ResultPayload(
                    image_id=8,
                    status=-2,
                    label=-1,
                    n_logits=2,
                    logit_raw0=f32_l0,
                    logit_raw1=f32_l1,
                    frac_bits=0,
                    x94_crc32=0,
                    timer_resolution_us=1,
                    n_iters=5,
                    stats=stats,
                    free_heap_min=140000,
                    stack_hwm=2048,
                    free_psram_min=3000000,
                    arena_bytes=163840,
                    arena_location=ARENA_LOC_PSRAM,
                ),
            ),
            (
                MSG_RESULT_X94,
                11,
                encode_result_x94(ResultX94Payload("q15", raw_x94_q15)),
                ResultX94Payload("q15", raw_x94_q15),
            ),
            (
                MSG_RESULT_X94,
                12,
                encode_result_x94(ResultX94Payload("f32", raw_x94_f32)),
                ResultX94Payload("f32", raw_x94_f32),
            ),
            (
                MSG_ERROR,
                13,
                encode_error(ErrorPayload(code=0x0042, text="bad offset")),
                ErrorPayload(code=0x0042, text="bad offset"),
            ),
        ]

        for msg_type, seq, raw_payload, expected_obj in cases:
            with self.subTest(msg_type=hex(msg_type), seq=seq):
                wire = encode_frame(msg_type, seq, raw_payload)
                frame = decode_frame(wire)
                self.assertEqual(frame.version, PROTOCOL_VERSION)
                self.assertEqual(frame.msg_type, msg_type)
                self.assertEqual(frame.seq, seq)
                decoded = decode_payload(frame)
                self.assertEqual(decoded, expected_obj)

        # Verify float32 and q15 logit conversions and arena location name
        res_f32 = cases[9][3]
        assert isinstance(res_f32, ResultPayload)
        self.assertEqual(res_f32.label, -1)
        self.assertEqual(res_f32.arena_location_name, "psram")
        self.assertEqual(res_f32.logits_as_float("f32"), (-1.25, 2.75))
        self.assertAlmostEqual(unpack_f32_logit(f32_l0), -1.25)

        res_q15 = cases[8][3]
        assert isinstance(res_q15, ResultPayload)
        self.assertAlmostEqual(res_q15.logits_as_float("q15")[0], 296482 / 65536.0)

        # Verify ERROR text truncation to <= 48 B
        long_err = encode_error(ErrorPayload(code=9, text="x" * 100))
        dec_err = decode_payload(decode_frame(encode_frame(MSG_ERROR, 99, long_err)))
        assert isinstance(dec_err, ErrorPayload)
        self.assertEqual(len(dec_err.text.encode("utf-8")), ERROR_TEXT_MAX_BYTES)

    def test_bad_crc_and_bad_header_rejected(self) -> None:
        raw = bytearray(encode_frame(MSG_CHUNK_ACK, 5, encode_chunk_ack(ChunkAckPayload(128, 0))))
        bad_crc = bytearray(raw)
        bad_crc[-1] ^= 0x01
        with self.assertRaises(ProtocolError):
            decode_frame(bytes(bad_crc))

        bad_payload = bytearray(raw)
        bad_payload[8] ^= 0x80
        with self.assertRaises(ProtocolError):
            decode_frame(bytes(bad_payload))

        bad_ver = encode_frame(MSG_HELLO, 1, b"", version=2)
        with self.assertRaises(ProtocolError):
            decode_frame(bad_ver)

        bad_magic = bytearray(raw)
        bad_magic[0] = 0x00
        with self.assertRaises(ProtocolError):
            decode_frame(bytes(bad_magic))

        with self.assertRaises(ProtocolError):
            decode_frame(bytes(raw[:-1]))

        dec = FrameDecoder()
        dec.feed(bytes(bad_crc))
        with self.assertRaises(ProtocolError):
            dec.next_frame()
        # Decoder remains usable after CRC error for a subsequent valid frame
        dec.feed(bytes(raw))
        recovered = dec.next_frame()
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.seq, 5)

    def test_split_merged_reads_and_garbage_resync(self) -> None:
        f1 = encode_frame(MSG_HELLO, 1)
        f2 = encode_frame(MSG_IMG_BEGIN, 2, encode_img_begin(ImgBeginPayload(0, 16384, 0xCAFEBABE, 1, 0)))
        f3 = encode_frame(MSG_ACK_RX, 3, encode_ack_rx(AckRxPayload(0, True)))

        # Include "ESP-ROM..." (starts with 'E''S' = little-endian MAGIC!) and repeated 0x45 before f1
        garbage = b"ESP-ROM:esp32\r\n\x00\xff\x45\x45"
        stream = garbage + f1 + f2 + b"\xaa\xbb" + f3

        dec = FrameDecoder()
        decoded_frames = []
        # Feed 1 byte at a time to exercise magic split across reads and partial header/payload
        for idx in range(len(stream)):
            dec.feed(stream[idx : idx + 1])
            decoded_frames.extend(dec.decode_all())

        self.assertEqual(len(decoded_frames), 3)
        self.assertEqual([f.seq for f in decoded_frames], [1, 2, 3])
        self.assertEqual(dec.discarded_bytes, len(garbage) + 2)

        # Merged reads in a single feed
        dec2 = FrameDecoder()
        dec2.feed(f1 + f2 + f3)
        merged = dec2.decode_all()
        self.assertEqual([f.msg_type for f in merged], [MSG_HELLO, MSG_IMG_BEGIN, MSG_ACK_RX])


class TestHwTransportAndImages(unittest.TestCase):
    """Validate `FakeTransport`, `TcpTransport`, `SerialTransport`, and `images.py`."""

    def test_fake_transport_fragmented_reads_and_duplicate_offset_handling(self) -> None:
        expected_offset = 0
        rx_buf = bytearray()
        decoder_dev = FrameDecoder()

        def fake_mcu_handler(tx_bytes: bytes) -> list[bytes]:
            nonlocal expected_offset
            decoder_dev.feed(tx_bytes)
            out_frames: list[bytes] = []
            for frame in decoder_dev.decode_all():
                if frame.msg_type == MSG_IMG_BEGIN:
                    expected_offset = 0
                    rx_buf.clear()
                    ack = encode_frame(
                        MSG_CHUNK_ACK,
                        frame.seq,
                        encode_chunk_ack(ChunkAckPayload(0xFFFFFFFF, CHUNK_STATUS_OK)),
                    )
                    out_frames.append(ack)
                elif frame.msg_type == MSG_IMG_CHUNK:
                    chunk = decode_payload(frame)
                    assert isinstance(chunk, ImgChunkPayload)
                    if chunk.offset < expected_offset:
                        status = CHUNK_STATUS_DUPLICATE
                    elif chunk.offset > expected_offset:
                        status = CHUNK_STATUS_BAD_OFFSET
                    else:
                        rx_buf.extend(chunk.data)
                        expected_offset += len(chunk.data)
                        status = CHUNK_STATUS_OK
                    out_frames.append(
                        encode_frame(
                            MSG_CHUNK_ACK,
                            frame.seq,
                            encode_chunk_ack(ChunkAckPayload(chunk.offset, status)),
                        )
                    )
            return out_frames

        transport = FakeTransport(handler=fake_mcu_handler, max_read_chunk=3)
        self.assertIsInstance(transport, Transport)

        with self.assertRaises(RuntimeError):
            transport.write(b"x")

        host_dec = FrameDecoder()
        with transport:
            # Instant timeout when nothing is queued
            with self.assertRaises(TimeoutError):
                transport.recv_frame(host_dec, timeout_s=1.0)

            transport.write(
                encode_frame(MSG_IMG_BEGIN, 1, encode_img_begin(ImgBeginPayload(0, 256, 0, 1, 0)))
            )
            begin_ack = decode_payload(transport.recv_frame(host_dec, timeout_s=1.0))
            self.assertEqual(begin_ack, ChunkAckPayload(0xFFFFFFFF, CHUNK_STATUS_OK))

            row0 = bytes([7] * 128)
            row1 = bytes([9] * 128)
            transport.write(encode_frame(MSG_IMG_CHUNK, 2, encode_img_chunk(ImgChunkPayload(0, row0))))
            self.assertEqual(
                decode_payload(transport.recv_frame(host_dec, timeout_s=1.0)),
                ChunkAckPayload(0, CHUNK_STATUS_OK),
            )

            # Retry duplicate offset 0 -> status=1 (duplicate), buffer not double-counted
            transport.write(encode_frame(MSG_IMG_CHUNK, 3, encode_img_chunk(ImgChunkPayload(0, row0))))
            self.assertEqual(
                decode_payload(transport.recv_frame(host_dec, timeout_s=1.0)),
                ChunkAckPayload(0, CHUNK_STATUS_DUPLICATE),
            )

            transport.write(encode_frame(MSG_IMG_CHUNK, 4, encode_img_chunk(ImgChunkPayload(128, row1))))
            self.assertEqual(
                decode_payload(transport.recv_frame(host_dec, timeout_s=1.0)),
                ChunkAckPayload(128, CHUNK_STATUS_OK),
            )

        self.assertEqual(bytes(rx_buf), row0 + row1)

    def test_tcp_transport_loopback(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        def server_worker() -> None:
            conn, _ = srv.accept()
            with conn:
                dec = FrameDecoder()
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    dec.feed(data)
                    for fr in dec.decode_all():
                        if fr.msg_type == MSG_HELLO:
                            info = InfoPayload(
                                board_id="nodemcu",
                                model="lr_dot",
                                mode="q15",
                                protocol_version=PROTOCOL_VERSION,
                                width=128,
                                height=128,
                                streaming=False,
                                max_payload=1028,
                                free_ram_boot=42000,
                                psram_size=0,
                                git_hash="deadbee",
                            )
                            conn.sendall(encode_frame(MSG_INFO, fr.seq, encode_info(info)))
                            return

        t = threading.Thread(target=server_worker, daemon=True)
        t.start()
        try:
            with TcpTransport("127.0.0.1", port, connect_timeout_s=2.0) as tcp:
                tcp.write(encode_frame(MSG_HELLO, 1))
                frame = tcp.recv_frame(FrameDecoder(), timeout_s=2.0)
                info = decode_payload(frame)
                assert isinstance(info, InfoPayload)
                self.assertEqual(info.board_id, "nodemcu")
                self.assertEqual(info.free_ram_boot, 42000)
        finally:
            t.join(timeout=2.0)
            srv.close()

    def test_serial_transport_boot_wait_and_mocked_read(self) -> None:
        rx_stream = bytearray(encode_frame(MSG_HELLO, 11))
        created: list[_FakePySerial] = []

        class _FakePySerial:
            def __init__(self) -> None:
                self.port = ""
                self.baudrate = 0
                self.timeout = 0.0
                self.write_timeout = 0.0
                self.dtr = True
                self.rts = True
                self.dtr_on_open: bool | None = None
                self.rts_on_open: bool | None = None
                self.is_open = False
                self.reset_in_called = 0
                self.reset_out_called = 0
                self.written = bytearray()
                created.append(self)

            def open(self) -> None:
                self.dtr_on_open = self.dtr
                self.rts_on_open = self.rts
                self.is_open = True

            @property
            def in_waiting(self) -> int:
                return len(rx_stream)

            def reset_input_buffer(self) -> None:
                self.reset_in_called += 1

            def reset_output_buffer(self) -> None:
                self.reset_out_called += 1

            def write(self, data: bytes) -> int:
                self.written.extend(data)
                return len(data)

            def flush(self) -> None:
                return None

            def read(self, size: int) -> bytes:
                out = bytes(rx_stream[:size])
                del rx_stream[:size]
                return out

            def close(self) -> None:
                self.is_open = False

        with mock.patch("src.hw_test.transport.serial.Serial", side_effect=_FakePySerial), mock.patch(
            "src.hw_test.transport.time.sleep"
        ) as sleep_mock:
            ser = SerialTransport("COM4", 115200, boot_wait_s=2.0, reset_on_open=True)
            with ser:
                sleep_mock.assert_called_once_with(2.0)
                self.assertTrue(created[-1].dtr_on_open)
                self.assertTrue(created[-1].rts_on_open)
                ser.write(b"ping")
                frame = ser.recv_frame(FrameDecoder(), timeout_s=1.0)
                self.assertEqual(frame.msg_type, MSG_HELLO)
                self.assertEqual(frame.seq, 11)

            # reset_on_open=False must deassert DTR and RTS BEFORE open()
            rx_stream.extend(encode_frame(MSG_HELLO, 12))
            ser_esp = SerialTransport("COM3", 460800, boot_wait_s=0.0, reset_on_open=False)
            with ser_esp:
                self.assertFalse(created[-1].dtr_on_open)
                self.assertFalse(created[-1].rts_on_open)
                frame2 = ser_esp.recv_frame(FrameDecoder(), timeout_s=1.0)
                self.assertEqual(frame2.seq, 12)

    def test_images_discovery_validation_and_golden_lookup(self) -> None:
        cfg = load_hw_test_config(Path("configs/hw_test/boards.toml"), PROJECT_ROOT)
        all_paths = discover_images(cfg.paths.images_dir, select_all=True)
        self.assertEqual([p.name for p in all_paths], sorted(p.name for p in all_paths))
        self.assertEqual(len(all_paths), 8)

        glob_paths = discover_images(cfg.paths.images_dir, glob_pattern="2000*.png")
        self.assertEqual([p.name for p in glob_paths], ["20000.png", "20001.png", "20002.png", "20003.png"])

        single = prepare_images(cfg.paths.images_dir, cfg.images, image="20042.png")
        self.assertEqual(len(single), 1)
        img = single[0]
        self.assertEqual(len(img.raw_bytes), 16384)
        self.assertEqual(img.crc32, crc32_ieee(img.raw_bytes))

        lookup = GoldenLookup.from_config(cfg)
        ref_lr = lookup.lookup(img, "lr_dot")
        self.assertIsNotNone(ref_lr)
        assert ref_lr is not None
        self.assertEqual(ref_lr.index, 0)
        self.assertEqual(ref_lr.expected_q15_logits_raw, (296482,))
        self.assertEqual(ref_lr.expected_q15_label, 1)
        assert ref_lr.expected_x94_q15 is not None
        self.assertEqual(len(ref_lr.expected_x94_q15), 94)

        ref_cnn = lookup.lookup(img, "ultra_tiny")
        self.assertIsNotNone(ref_cnn)
        assert ref_cnn is not None
        self.assertEqual(ref_cnn.expected_q15_logits_raw, (-372472, 284405))
        self.assertIsNone(ref_cnn.expected_x94_q15)

        # Unknown synthetic image returns None from GoldenLookup (parity = "n/a")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            valid_img = tmp_dir / "valid.png"
            arr = np.full((128, 128), 123, dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(valid_img)
            loaded_valid = load_images([valid_img], cfg.images)
            self.assertIsNone(lookup.lookup(loaded_valid[0], "lr_dot"))

            # Collect all invalid files in a single ValueError (no resizing, no conversion)
            bad_ext = tmp_dir / "a_bad_ext.jpg"
            Image.fromarray(arr, mode="L").save(bad_ext)
            bad_rgb = tmp_dir / "b_bad_rgb.png"
            Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB").save(bad_rgb)
            bad_size = tmp_dir / "c_bad_size.png"
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L").save(bad_size)
            bad_corrupt = tmp_dir / "d_corrupt.png"
            bad_corrupt.write_bytes(b"not a png")

            with self.assertRaises(ValueError) as ctx:
                prepare_images(tmp_dir, ImagesSpec((".png",), 128, 128, "L"), select_all=True)
            msg = str(ctx.exception)
            for fname in ("a_bad_ext.jpg", "b_bad_rgb.png", "c_bad_size.png", "d_corrupt.png"):
                self.assertIn(fname, msg)


if __name__ == "__main__":
    unittest.main()
