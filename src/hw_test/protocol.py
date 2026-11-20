"""Frozen little-endian binary wire protocol for PC <-> MCU `hw_test` communication.

Frame layout (packed, little-endian):
    magic (u16, 0x5345) | version (u8, 1) | type (u8) | seq (u16) | length (u16)
    | payload[length] | crc16 (u16)

CRC16-CCITT-FALSE (poly=0x1021, init=0xFFFF) covers `version .. payload` (bytes 2..8+length).
CRC32 uses standard IEEE 802.3 (`zlib.crc32(data) & 0xFFFFFFFF`).
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Sequence
import zlib

MAGIC: int = 0x5345
PROTOCOL_VERSION: int = 1

MSG_HELLO: int = 0x01
MSG_INFO: int = 0x02
MSG_IMG_BEGIN: int = 0x10
MSG_IMG_CHUNK: int = 0x11
MSG_CHUNK_ACK: int = 0x12
MSG_ACK_RX: int = 0x13
MSG_RESULT: int = 0x20
MSG_RESULT_X94: int = 0x21
MSG_GET_X94: int = 0x22
MSG_ERROR: int = 0x7F

UNAVAILABLE_U32: int = 0xFFFFFFFF

MODE_Q15: int = 0
MODE_F32: int = 1
MODE_TO_CODE: dict[str, int] = {"q15": MODE_Q15, "f32": MODE_F32}
CODE_TO_MODE: dict[int, str] = {MODE_Q15: "q15", MODE_F32: "f32"}

ARENA_LOC_NONE: int = 0
ARENA_LOC_SRAM: int = 1
ARENA_LOC_PSRAM: int = 2
ARENA_LOC_NAMES: dict[int, str] = {
    ARENA_LOC_NONE: "none",
    ARENA_LOC_SRAM: "sram",
    ARENA_LOC_PSRAM: "psram",
}

CHUNK_STATUS_OK: int = 0
CHUNK_STATUS_DUPLICATE: int = 1
CHUNK_STATUS_BAD_OFFSET: int = -1
CHUNK_STATUS_OVERFLOW: int = -2

ERROR_TEXT_MAX_BYTES: int = 48
N_X94: int = 94
MAX_DEFAULT_PAYLOAD: int = 4096

HEADER_STRUCT = struct.Struct("<HBBHH")          # 8 B
CRC16_STRUCT = struct.Struct("<H")               # 2 B
INFO_STRUCT = struct.Struct("<16s24sBBHHBHII8s") # 65 B
IMG_BEGIN_STRUCT = struct.Struct("<IIIHH")       # 16 B
IMG_CHUNK_HDR_STRUCT = struct.Struct("<I")       # 4 B + pixel bytes
CHUNK_ACK_STRUCT = struct.Struct("<Ib")          # 5 B
ACK_RX_STRUCT = struct.Struct("<IB")             # 5 B
# Byte 6 is signed label (i8), byte 7 is n_logits (u8) per task_12.md II.4.
RESULT_STRUCT = struct.Struct("<IhbBiiBIBH15IIIIIB")  # 101 B
RESULT_X94_Q15_STRUCT = struct.Struct("<94h")    # 188 B
RESULT_X94_F32_STRUCT = struct.Struct("<94f")    # 376 B
ERROR_HDR_STRUCT = struct.Struct("<HB")          # 3 B + <=48 B UTF-8 text

_F32_PACK = struct.Struct("<f")
_I32_PACK = struct.Struct("<i")


class ProtocolError(ValueError):
    """Raised when wire framing, checksums, versions, or payload layouts are invalid."""


def _build_crc16_table() -> tuple[int, ...]:
    table: list[int] = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC16_TABLE: tuple[int, ...] = _build_crc16_table()


def crc16_ccitt_false(data: bytes | bytearray | memoryview) -> int:
    """Compute CRC16-CCITT-FALSE (poly=0x1021, init=0xFFFF, xorout=0x0000)."""
    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) ^ _CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc


def crc32_ieee(data: bytes | bytearray | memoryview) -> int:
    """Compute standard unsigned IEEE 802.3 CRC32."""
    return zlib.crc32(data) & 0xFFFFFFFF


def pack_f32_logit(value: float) -> int:
    """Reinterpret IEEE 754 float32 bits as signed int32 for `RESULT.logit_raw`."""
    return int(_I32_PACK.unpack(_F32_PACK.pack(float(value)))[0])


def unpack_f32_logit(raw_i32: int) -> float:
    """Reinterpret signed int32 bits from `RESULT.logit_raw` as float32."""
    return float(_F32_PACK.unpack(_I32_PACK.pack(int(raw_i32)))[0])


def _pack_fixed_ascii(text: str, size: int, field_name: str) -> bytes:
    encoded = text.encode("ascii")
    if len(encoded) > size:
        raise ValueError(f"{field_name} exceeds {size} bytes: {text!r}")
    return encoded.ljust(size, b"\x00")


def _unpack_fixed_ascii(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


@dataclass(frozen=True, slots=True)
class Frame:
    """Decoded wire protocol frame."""

    version: int
    msg_type: int
    seq: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class HelloPayload:
    """Empty `MSG_HELLO` (0x01) payload."""


@dataclass(frozen=True, slots=True)
class GetX94Payload:
    """Empty `MSG_GET_X94` (0x22) payload."""


@dataclass(frozen=True, slots=True)
class InfoPayload:
    """Decoded `MSG_INFO` (0x02, 65 B) device capabilities and build metadata."""

    board_id: str
    model: str
    mode: str
    protocol_version: int
    width: int
    height: int
    streaming: bool
    max_payload: int
    free_ram_boot: int
    psram_size: int
    git_hash: str


@dataclass(frozen=True, slots=True)
class ImgBeginPayload:
    """Decoded `MSG_IMG_BEGIN` (0x10, 16 B) image transfer header."""

    image_id: int
    total_bytes: int
    crc32: int
    repeats: int
    warmup: int


@dataclass(frozen=True, slots=True)
class ImgChunkPayload:
    """Decoded `MSG_IMG_CHUNK` (0x11, 4 + N B) pixel slice."""

    offset: int
    data: bytes


@dataclass(frozen=True, slots=True)
class ChunkAckPayload:
    """Decoded `MSG_CHUNK_ACK` (0x12, 5 B) chunk receipt acknowledgment."""

    offset: int
    status: int


@dataclass(frozen=True, slots=True)
class AckRxPayload:
    """Decoded `MSG_ACK_RX` (0x13, 5 B) whole-image CRC verification status."""

    image_id: int
    crc_ok: bool


@dataclass(frozen=True, slots=True)
class TimingTriple:
    """Microseconds tuple `(selector_us, model_us, total_us)` for one summary row."""

    selector_us: int
    model_us: int
    total_us: int


@dataclass(frozen=True, slots=True)
class TimingStats:
    """Five-row timing summary `[first, min, median, mean, max] x [sel, model, total]`."""

    first: TimingTriple
    minimum: TimingTriple
    median: TimingTriple
    mean: TimingTriple
    maximum: TimingTriple

    def to_flat_15(self) -> tuple[int, ...]:
        """Flatten stats in wire order `[first, min, median, mean, max] x [sel, model, total]`."""
        out: list[int] = []
        for row in (self.first, self.minimum, self.median, self.mean, self.maximum):
            out.extend((row.selector_us, row.model_us, row.total_us))
        return tuple(out)

    @classmethod
    def from_flat_15(cls, values: Sequence[int]) -> TimingStats:
        """Construct `TimingStats` from 15 unsigned microsecond integers."""
        if len(values) != 15:
            raise ValueError(f"Expected 15 timing stats, got {len(values)}")
        rows = [
            TimingTriple(int(values[i]), int(values[i + 1]), int(values[i + 2]))
            for i in range(0, 15, 3)
        ]
        return cls(
            first=rows[0],
            minimum=rows[1],
            median=rows[2],
            mean=rows[3],
            maximum=rows[4],
        )


@dataclass(frozen=True, slots=True)
class ResultPayload:
    """Decoded `MSG_RESULT` (0x20, 101 B) inference output, timing, and memory telemetry."""

    image_id: int
    status: int
    label: int
    n_logits: int
    logit_raw0: int
    logit_raw1: int
    frac_bits: int
    x94_crc32: int
    timer_resolution_us: int
    n_iters: int
    stats: TimingStats
    free_heap_min: int
    stack_hwm: int
    free_psram_min: int
    arena_bytes: int
    arena_location: int

    @property
    def arena_location_name(self) -> str:
        """Human-readable arena location name (`none`, `sram`, `psram`, or `unknown`)."""
        return ARENA_LOC_NAMES.get(self.arena_location, f"unknown({self.arena_location})")

    def logits_as_float(self, mode: str) -> tuple[float, ...]:
        """Convert active raw logits into floating-point values for `q15` or `f32`."""
        raws = (self.logit_raw0, self.logit_raw1)[: self.n_logits]
        norm_mode = mode.strip().lower()
        if norm_mode == "f32":
            return tuple(unpack_f32_logit(r) for r in raws)
        if norm_mode == "q15":
            scale = float(1 << self.frac_bits) if self.frac_bits > 0 else 1.0
            return tuple(float(r) / scale for r in raws)
        raise ValueError(f"Unsupported numeric mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class ResultX94Payload:
    """Decoded `MSG_RESULT_X94` (0x21) descriptor vector (94 x i16 or 94 x f32)."""

    mode: str
    values: tuple[int, ...] | tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ErrorPayload:
    """Decoded `MSG_ERROR` (0x7F) device error code and short UTF-8 description."""

    code: int
    text: str


Payload = (
    HelloPayload
    | InfoPayload
    | ImgBeginPayload
    | ImgChunkPayload
    | ChunkAckPayload
    | AckRxPayload
    | ResultPayload
    | ResultX94Payload
    | GetX94Payload
    | ErrorPayload
)


def encode_frame(
    msg_type: int,
    seq: int,
    payload: bytes = b"",
    *,
    version: int = PROTOCOL_VERSION,
    magic: int = MAGIC,
) -> bytes:
    """Pack and checksum one binary wire frame."""
    if not (0 <= magic <= 0xFFFF):
        raise ValueError(f"magic out of uint16 range: {magic}")
    if not (0 <= version <= 0xFF):
        raise ValueError(f"version out of uint8 range: {version}")
    if not (0 <= msg_type <= 0xFF):
        raise ValueError(f"msg_type out of uint8 range: {msg_type}")
    if not (0 <= seq <= 0xFFFF):
        raise ValueError(f"seq out of uint16 range: {seq}")
    if len(payload) > 0xFFFF:
        raise ValueError(f"payload exceeds uint16 length limit: {len(payload)}")

    header = HEADER_STRUCT.pack(magic, version, msg_type, seq, len(payload))
    crc_region = header[2:] + payload
    crc16 = crc16_ccitt_false(crc_region)
    return header + payload + CRC16_STRUCT.pack(crc16)


def decode_frame(
    raw: bytes,
    *,
    expected_version: int = PROTOCOL_VERSION,
    expected_magic: int = MAGIC,
) -> Frame:
    """Decode and verify one exact wire frame buffer (rejects truncation, trailing bytes, bad CRC)."""
    if len(raw) < HEADER_STRUCT.size + CRC16_STRUCT.size:
        raise ProtocolError(f"Frame too short ({len(raw)} B)")

    magic, version, msg_type, seq, length = HEADER_STRUCT.unpack_from(raw, 0)
    if magic != expected_magic:
        raise ProtocolError(f"Invalid magic 0x{magic:04X}, expected 0x{expected_magic:04X}")

    expected_total = HEADER_STRUCT.size + length + CRC16_STRUCT.size
    if len(raw) != expected_total:
        raise ProtocolError(
            f"Frame length mismatch: header declares {expected_total} B, buffer has {len(raw)} B"
        )

    crc_expected = CRC16_STRUCT.unpack_from(raw, HEADER_STRUCT.size + length)[0]
    crc_actual = crc16_ccitt_false(memoryview(raw)[2 : HEADER_STRUCT.size + length])
    if crc_actual != crc_expected:
        raise ProtocolError(
            f"CRC16 mismatch: got 0x{crc_actual:04X}, expected 0x{crc_expected:04X}"
        )

    if version != expected_version:
        raise ProtocolError(
            f"Unsupported protocol version {version}, expected {expected_version}"
        )

    payload = bytes(raw[HEADER_STRUCT.size : HEADER_STRUCT.size + length])
    return Frame(version=version, msg_type=msg_type, seq=seq, payload=payload)


class FrameDecoder:
    """Incremental byte-stream decoder supporting split reads, merged frames, and magic resync."""

    def __init__(
        self,
        *,
        expected_version: int = PROTOCOL_VERSION,
        expected_magic: int = MAGIC,
        max_payload_bytes: int = MAX_DEFAULT_PAYLOAD,
    ) -> None:
        self.expected_version = expected_version
        self.expected_magic = expected_magic
        self.max_payload_bytes = max_payload_bytes
        self._magic_bytes = struct.pack("<H", expected_magic)
        self._buf = bytearray()
        self.discarded_bytes = 0

    def feed(self, data: bytes | bytearray | memoryview) -> None:
        """Append incoming raw transport bytes to the internal stream buffer."""
        if data:
            self._buf.extend(data)

    def clear(self) -> None:
        """Drop all buffered bytes without resetting counters."""
        self._buf.clear()

    def next_frame(self) -> Frame | None:
        """Extract the next valid frame, resyncing past garbage bytes before MAGIC.

        Returns None when more bytes are needed. Raises ProtocolError on CRC failure
        or unsupported protocol version with valid CRC.
        """
        while True:
            idx = self._buf.find(self._magic_bytes)
            if idx < 0:
                # Preserve a trailing first magic byte in case magic spans two reads.
                if self._buf and self._buf[-1] == self._magic_bytes[0]:
                    drop = len(self._buf) - 1
                    if drop > 0:
                        del self._buf[:drop]
                        self.discarded_bytes += drop
                elif self._buf:
                    self.discarded_bytes += len(self._buf)
                    self._buf.clear()
                return None

            if idx > 0:
                del self._buf[:idx]
                self.discarded_bytes += idx

            if len(self._buf) < HEADER_STRUCT.size:
                return None

            _, version, msg_type, seq, length = HEADER_STRUCT.unpack_from(self._buf, 0)
            if length > self.max_payload_bytes:
                # False magic sequence inside text/garbage (e.g. "ESP-ROM..."); slide 1 byte.
                del self._buf[:1]
                self.discarded_bytes += 1
                continue

            total_len = HEADER_STRUCT.size + length + CRC16_STRUCT.size
            if len(self._buf) < total_len:
                return None

            crc_expected = CRC16_STRUCT.unpack_from(self._buf, HEADER_STRUCT.size + length)[0]
            crc_actual = crc16_ccitt_false(memoryview(self._buf)[2 : HEADER_STRUCT.size + length])
            if crc_actual != crc_expected:
                if version != self.expected_version:
                    del self._buf[:1]
                    self.discarded_bytes += 1
                    continue
                del self._buf[:1]
                self.discarded_bytes += 1
                raise ProtocolError(
                    f"CRC16 mismatch on msg_type=0x{msg_type:02X} seq={seq}: "
                    f"got 0x{crc_actual:04X}, expected 0x{crc_expected:04X}"
                )

            if version != self.expected_version:
                del self._buf[:total_len]
                raise ProtocolError(
                    f"Unsupported protocol version {version}, expected {self.expected_version}"
                )

            payload = bytes(self._buf[HEADER_STRUCT.size : HEADER_STRUCT.size + length])
            del self._buf[:total_len]
            return Frame(version=version, msg_type=msg_type, seq=seq, payload=payload)

    def decode_all(self) -> list[Frame]:
        """Decode all complete frames currently buffered."""
        frames: list[Frame] = []
        while True:
            frame = self.next_frame()
            if frame is None:
                return frames
            frames.append(frame)


def encode_info(payload: InfoPayload) -> bytes:
    """Serialize `InfoPayload` into 65-byte `MSG_INFO` wire payload."""
    mode_key = payload.mode.strip().lower()
    if mode_key not in MODE_TO_CODE:
        raise ValueError(f"Unsupported mode {payload.mode!r}, expected 'q15' or 'f32'")
    return INFO_STRUCT.pack(
        _pack_fixed_ascii(payload.board_id, 16, "board_id"),
        _pack_fixed_ascii(payload.model, 24, "model"),
        MODE_TO_CODE[mode_key],
        payload.protocol_version,
        payload.width,
        payload.height,
        1 if payload.streaming else 0,
        payload.max_payload,
        payload.free_ram_boot,
        payload.psram_size,
        _pack_fixed_ascii(payload.git_hash, 8, "git_hash"),
    )


def decode_info(payload: bytes) -> InfoPayload:
    """Deserialize 65-byte `MSG_INFO` wire payload."""
    if len(payload) != INFO_STRUCT.size:
        raise ProtocolError(f"INFO payload must be {INFO_STRUCT.size} B, got {len(payload)} B")
    (
        raw_board,
        raw_model,
        mode_code,
        proto_ver,
        width,
        height,
        streaming_u8,
        max_payload,
        free_ram_boot,
        psram_size,
        raw_git,
    ) = INFO_STRUCT.unpack(payload)
    if mode_code not in CODE_TO_MODE:
        raise ProtocolError(f"Invalid mode code {mode_code} in INFO payload")
    return InfoPayload(
        board_id=_unpack_fixed_ascii(raw_board),
        model=_unpack_fixed_ascii(raw_model),
        mode=CODE_TO_MODE[mode_code],
        protocol_version=proto_ver,
        width=width,
        height=height,
        streaming=bool(streaming_u8),
        max_payload=max_payload,
        free_ram_boot=free_ram_boot,
        psram_size=psram_size,
        git_hash=_unpack_fixed_ascii(raw_git),
    )


def encode_img_begin(payload: ImgBeginPayload) -> bytes:
    """Serialize `ImgBeginPayload` into 16-byte `MSG_IMG_BEGIN` wire payload."""
    return IMG_BEGIN_STRUCT.pack(
        payload.image_id,
        payload.total_bytes,
        payload.crc32,
        payload.repeats,
        payload.warmup,
    )


def decode_img_begin(payload: bytes) -> ImgBeginPayload:
    """Deserialize 16-byte `MSG_IMG_BEGIN` wire payload."""
    if len(payload) != IMG_BEGIN_STRUCT.size:
        raise ProtocolError(
            f"IMG_BEGIN payload must be {IMG_BEGIN_STRUCT.size} B, got {len(payload)} B"
        )
    image_id, total_bytes, crc32, repeats, warmup = IMG_BEGIN_STRUCT.unpack(payload)
    return ImgBeginPayload(
        image_id=image_id,
        total_bytes=total_bytes,
        crc32=crc32,
        repeats=repeats,
        warmup=warmup,
    )


def encode_img_chunk(payload: ImgChunkPayload) -> bytes:
    """Serialize `ImgChunkPayload` into `4 + N`-byte `MSG_IMG_CHUNK` wire payload."""
    if not payload.data:
        raise ValueError("IMG_CHUNK data must not be empty")
    return IMG_CHUNK_HDR_STRUCT.pack(payload.offset) + payload.data


def decode_img_chunk(payload: bytes) -> ImgChunkPayload:
    """Deserialize `MSG_IMG_CHUNK` wire payload (`offset u32` + bytes)."""
    if len(payload) <= IMG_CHUNK_HDR_STRUCT.size:
        raise ProtocolError(
            f"IMG_CHUNK payload must be > {IMG_CHUNK_HDR_STRUCT.size} B, got {len(payload)} B"
        )
    (offset,) = IMG_CHUNK_HDR_STRUCT.unpack_from(payload, 0)
    return ImgChunkPayload(offset=offset, data=bytes(payload[IMG_CHUNK_HDR_STRUCT.size :]))


def encode_chunk_ack(payload: ChunkAckPayload) -> bytes:
    """Serialize `ChunkAckPayload` into 5-byte `MSG_CHUNK_ACK` wire payload."""
    return CHUNK_ACK_STRUCT.pack(payload.offset, payload.status)


def decode_chunk_ack(payload: bytes) -> ChunkAckPayload:
    """Deserialize 5-byte `MSG_CHUNK_ACK` wire payload."""
    if len(payload) != CHUNK_ACK_STRUCT.size:
        raise ProtocolError(
            f"CHUNK_ACK payload must be {CHUNK_ACK_STRUCT.size} B, got {len(payload)} B"
        )
    offset, status = CHUNK_ACK_STRUCT.unpack(payload)
    return ChunkAckPayload(offset=offset, status=status)


def encode_ack_rx(payload: AckRxPayload) -> bytes:
    """Serialize `AckRxPayload` into 5-byte `MSG_ACK_RX` wire payload."""
    return ACK_RX_STRUCT.pack(payload.image_id, 1 if payload.crc_ok else 0)


def decode_ack_rx(payload: bytes) -> AckRxPayload:
    """Deserialize 5-byte `MSG_ACK_RX` wire payload."""
    if len(payload) != ACK_RX_STRUCT.size:
        raise ProtocolError(f"ACK_RX payload must be {ACK_RX_STRUCT.size} B, got {len(payload)} B")
    image_id, crc_ok_u8 = ACK_RX_STRUCT.unpack(payload)
    return AckRxPayload(image_id=image_id, crc_ok=bool(crc_ok_u8))


def encode_result(payload: ResultPayload) -> bytes:
    """Serialize `ResultPayload` into 101-byte `MSG_RESULT` wire payload."""
    if not (0 <= payload.n_logits <= 2):
        raise ValueError(f"n_logits must be in [0, 2], got {payload.n_logits}")
    flat_stats = payload.stats.to_flat_15()
    return RESULT_STRUCT.pack(
        payload.image_id,
        payload.status,
        payload.label,
        payload.n_logits,
        payload.logit_raw0,
        payload.logit_raw1,
        payload.frac_bits,
        payload.x94_crc32,
        payload.timer_resolution_us,
        payload.n_iters,
        *flat_stats,
        payload.free_heap_min,
        payload.stack_hwm,
        payload.free_psram_min,
        payload.arena_bytes,
        payload.arena_location,
    )


def decode_result(payload: bytes) -> ResultPayload:
    """Deserialize 101-byte `MSG_RESULT` wire payload."""
    if len(payload) != RESULT_STRUCT.size:
        raise ProtocolError(f"RESULT payload must be {RESULT_STRUCT.size} B, got {len(payload)} B")
    unpacked = RESULT_STRUCT.unpack(payload)
    (
        image_id,
        status,
        label,
        n_logits,
        logit_raw0,
        logit_raw1,
        frac_bits,
        x94_crc32,
        timer_resolution_us,
        n_iters,
    ) = unpacked[:10]
    if n_logits > 2:
        raise ProtocolError(f"RESULT n_logits out of range [0, 2]: {n_logits}")
    stats = TimingStats.from_flat_15(unpacked[10:25])
    free_heap_min, stack_hwm, free_psram_min, arena_bytes, arena_location = unpacked[25:30]
    return ResultPayload(
        image_id=image_id,
        status=status,
        label=label,
        n_logits=n_logits,
        logit_raw0=logit_raw0,
        logit_raw1=logit_raw1,
        frac_bits=frac_bits,
        x94_crc32=x94_crc32,
        timer_resolution_us=timer_resolution_us,
        n_iters=n_iters,
        stats=stats,
        free_heap_min=free_heap_min,
        stack_hwm=stack_hwm,
        free_psram_min=free_psram_min,
        arena_bytes=arena_bytes,
        arena_location=arena_location,
    )


def encode_result_x94(payload: ResultX94Payload) -> bytes:
    """Serialize `ResultX94Payload` into 188 B (`q15`) or 376 B (`f32`) wire payload."""
    if len(payload.values) != N_X94:
        raise ValueError(f"RESULT_X94 expects {N_X94} elements, got {len(payload.values)}")
    mode = payload.mode.strip().lower()
    if mode == "q15":
        return RESULT_X94_Q15_STRUCT.pack(*(int(v) for v in payload.values))
    if mode == "f32":
        return RESULT_X94_F32_STRUCT.pack(*(float(v) for v in payload.values))
    raise ValueError(f"Unsupported RESULT_X94 mode: {payload.mode!r}")


def decode_result_x94(payload: bytes) -> ResultX94Payload:
    """Deserialize 188 B (`q15`) or 376 B (`f32`) `MSG_RESULT_X94` wire payload."""
    if len(payload) == RESULT_X94_Q15_STRUCT.size:
        return ResultX94Payload(mode="q15", values=RESULT_X94_Q15_STRUCT.unpack(payload))
    if len(payload) == RESULT_X94_F32_STRUCT.size:
        return ResultX94Payload(mode="f32", values=RESULT_X94_F32_STRUCT.unpack(payload))
    raise ProtocolError(
        f"RESULT_X94 payload must be {RESULT_X94_Q15_STRUCT.size} B (q15) or "
        f"{RESULT_X94_F32_STRUCT.size} B (f32), got {len(payload)} B"
    )


def encode_error(payload: ErrorPayload) -> bytes:
    """Serialize `ErrorPayload` into `<HB` + UTF-8 text (<= 48 bytes)."""
    text_bytes = payload.text.encode("utf-8")
    if len(text_bytes) > ERROR_TEXT_MAX_BYTES:
        text_bytes = text_bytes[:ERROR_TEXT_MAX_BYTES]
    return ERROR_HDR_STRUCT.pack(payload.code, len(text_bytes)) + text_bytes


def decode_error(payload: bytes) -> ErrorPayload:
    """Deserialize `MSG_ERROR` wire payload."""
    if len(payload) < ERROR_HDR_STRUCT.size:
        raise ProtocolError(f"ERROR payload too short: {len(payload)} B")
    code, text_len = ERROR_HDR_STRUCT.unpack_from(payload, 0)
    if text_len > ERROR_TEXT_MAX_BYTES:
        raise ProtocolError(f"ERROR text_len {text_len} exceeds {ERROR_TEXT_MAX_BYTES} B")
    if len(payload) != ERROR_HDR_STRUCT.size + text_len:
        raise ProtocolError(
            f"ERROR payload length mismatch: declared {ERROR_HDR_STRUCT.size + text_len} B, "
            f"got {len(payload)} B"
        )
    text = payload[ERROR_HDR_STRUCT.size :].decode("utf-8", errors="replace")
    return ErrorPayload(code=code, text=text)


def decode_payload(frame: Frame) -> Payload:
    """Decode `frame.payload` into its typed payload dataclass based on `frame.msg_type`."""
    mtype = frame.msg_type
    if mtype == MSG_HELLO:
        if frame.payload:
            raise ProtocolError(f"HELLO payload must be empty, got {len(frame.payload)} B")
        return HelloPayload()
    if mtype == MSG_GET_X94:
        if frame.payload:
            raise ProtocolError(f"GET_X94 payload must be empty, got {len(frame.payload)} B")
        return GetX94Payload()
    if mtype == MSG_INFO:
        return decode_info(frame.payload)
    if mtype == MSG_IMG_BEGIN:
        return decode_img_begin(frame.payload)
    if mtype == MSG_IMG_CHUNK:
        return decode_img_chunk(frame.payload)
    if mtype == MSG_CHUNK_ACK:
        return decode_chunk_ack(frame.payload)
    if mtype == MSG_ACK_RX:
        return decode_ack_rx(frame.payload)
    if mtype == MSG_RESULT:
        return decode_result(frame.payload)
    if mtype == MSG_RESULT_X94:
        return decode_result_x94(frame.payload)
    if mtype == MSG_ERROR:
        return decode_error(frame.payload)
    raise ProtocolError(f"Unknown message type 0x{mtype:02X}")
