"""Host-side PC <-> MCU communication session, retry logic, and golden parity checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Sequence

from configs.logging import logger
from src.hw_test.config import BoardSpec, HwTestConfig
from src.hw_test.images import GoldenLookup, GoldenReference, PreparedImage
from src.hw_test.protocol import (
    CHUNK_STATUS_DUPLICATE,
    CHUNK_STATUS_OK,
    MSG_GET_X94,
    MSG_HELLO,
    MSG_IMG_BEGIN,
    MSG_IMG_CHUNK,
    N_X94,
    AckRxPayload,
    ChunkAckPayload,
    ErrorPayload,
    FrameDecoder,
    ImgBeginPayload,
    ImgChunkPayload,
    InfoPayload,
    ProtocolError,
    ResultPayload,
    ResultX94Payload,
    crc32_ieee,
    decode_payload,
    encode_frame,
    encode_img_begin,
    encode_img_chunk,
    encode_result_x94,
)
from src.hw_test.transport import Transport


class _SeqCounter:
    def __init__(self, start: int = 1) -> None:
        self._val = start & 0xFFFF

    def next(self) -> int:
        cur = self._val
        self._val = (self._val + 1) & 0xFFFF
        return cur


@dataclass(frozen=True, slots=True)
class ImageRunRecord:
    """Complete per-image execution telemetry, outputs, and golden parity check."""

    image: PreparedImage
    model: str
    mode: str
    is_cnn: bool
    result: ResultPayload
    logits_float: tuple[float, ...]
    margin_float: float | None
    t_transfer_host_ms: float
    includes_compute: bool
    golden: GoldenReference | None
    parity_status: str          # "pass" | "fail" | "n/a"
    parity_ok: bool | None
    x94_crc_ok: bool | None
    logit_max_abs_diff: float | None
    dumped_x94: ResultX94Payload | None = None


@dataclass(frozen=True, slots=True)
class BatchRunResult:
    """Aggregated result of running a batch of images against one flashed MCU."""

    board_id: str
    model: str
    mode: str
    transport_name: str
    port_or_host: str
    started_at_utc: str
    finished_at_utc: str
    info: InfoPayload
    warmup_requested: int
    repeats_requested: int
    records: tuple[ImageRunRecord, ...]


def _make_decoder(cfg: HwTestConfig, max_payload_bytes: int = 4096) -> FrameDecoder:
    return FrameDecoder(
        expected_version=cfg.protocol.version,
        expected_magic=cfg.protocol.magic,
        max_payload_bytes=max(4096, max_payload_bytes),
    )


def _recv_typed(
    transport: Transport,
    decoder: FrameDecoder,
    timeout_s: float,
    expected_cls: type,
    context: str,
):
    frame = transport.recv_frame(decoder, timeout_s=timeout_s)
    payload = decode_payload(frame)
    if isinstance(payload, ErrorPayload):
        raise RuntimeError(f"MCU error during {context} (code={payload.code}): {payload.text}")
    if not isinstance(payload, expected_cls):
        raise ProtocolError(
            f"Unexpected message during {context}: got {type(payload).__name__}, "
            f"expected {expected_cls.__name__}"
        )
    return payload


def run_hello(
    transport: Transport,
    cfg: HwTestConfig,
    *,
    decoder: FrameDecoder | None = None,
    seq_counter: _SeqCounter | None = None,
) -> InfoPayload:
    """Send `MSG_HELLO` and return the MCU's `InfoPayload` capability banner."""
    dec = decoder if decoder is not None else _make_decoder(cfg)
    seqs = seq_counter if seq_counter is not None else _SeqCounter()
    transport.write(encode_frame(MSG_HELLO, seqs.next(), b"", version=cfg.protocol.version, magic=cfg.protocol.magic))
    info = _recv_typed(transport, dec, cfg.protocol.ack_timeout_s, InfoPayload, "HELLO")
    logger.info(
        "Connected MCU: board=%s model=%s mode=%s streaming=%s max_payload=%d free_ram_boot=%d psram=%d git=%s",
        info.board_id,
        info.model,
        info.mode,
        info.streaming,
        info.max_payload,
        info.free_ram_boot,
        info.psram_size,
        info.git_hash,
    )
    return info


def _evaluate_parity(
    *,
    cfg: HwTestConfig,
    mode: str,
    is_cnn: bool,
    result: ResultPayload,
    logits_float: tuple[float, ...],
    golden: GoldenReference | None,
    dumped_x94: ResultX94Payload | None,
) -> tuple[str, bool | None, bool | None, float | None]:
    if golden is None:
        return "n/a", None, None, None

    x94_crc_ok: bool | None = None
    if not is_cnn:
        if mode == "q15" and golden.expected_x94_q15 is not None:
            expected_crc = crc32_ieee(
                encode_result_x94(ResultX94Payload(mode="q15", values=golden.expected_x94_q15))
            )
            x94_crc_ok = result.x94_crc32 == expected_crc
        elif mode == "f32" and dumped_x94 is not None and golden.expected_x94_f64 is not None:
            raw_crc = crc32_ieee(encode_result_x94(dumped_x94))
            desc_diff = max(
                abs(float(a) - float(b))
                for a, b in zip(dumped_x94.values, golden.expected_x94_f64)
            )
            x94_crc_ok = (
                raw_crc == result.x94_crc32
                and desc_diff <= cfg.c_imp_cfg.verification.f32_descriptor_abs_tol
            )

    if mode == "q15":
        expected_raw = golden.expected_q15_logits_raw
        actual_raw = (result.logit_raw0,) if not is_cnn else (result.logit_raw0, result.logit_raw1)
        if len(actual_raw) != len(expected_raw):
            raise ValueError(
                f"Logit count mismatch for {golden.model}: device sent {len(actual_raw)}, golden has {len(expected_raw)}"
            )
        max_diff = float(max(abs(int(a) - int(b)) for a, b in zip(actual_raw, expected_raw)))
        logits_ok = actual_raw == expected_raw
        label_ok = result.label == golden.expected_q15_label
        crc_gate = x94_crc_ok is not False
        parity_ok = logits_ok and label_ok and crc_gate
        return ("pass" if parity_ok else "fail"), parity_ok, x94_crc_ok, max_diff

    if mode == "f32":
        expected_f32 = golden.expected_f32_logits
        if len(logits_float) != len(expected_f32):
            raise ValueError(
                f"Logit count mismatch for {golden.model}: device sent {len(logits_float)}, golden has {len(expected_f32)}"
            )
        max_diff = float(max(abs(float(a) - float(b)) for a, b in zip(logits_float, expected_f32)))
        tol = cfg.c_imp_cfg.verification.f32_logit_abs_tol
        label_ok = result.label == golden.expected_f32_label
        crc_gate = x94_crc_ok is not False
        parity_ok = (max_diff <= tol) and label_ok and crc_gate
        return ("pass" if parity_ok else "fail"), parity_ok, x94_crc_ok, max_diff

    raise ValueError(f"Unsupported mode: {mode!r}")


def run_image(
    transport: Transport,
    cfg: HwTestConfig,
    board: BoardSpec,
    item: PreparedImage,
    model: str,
    mode: str,
    info: InfoPayload,
    *,
    repeats: int | None = None,
    warmup: int | None = None,
    dump_descriptor: bool = False,
    golden: GoldenReference | None = None,
    decoder: FrameDecoder | None = None,
    seq_counter: _SeqCounter | None = None,
) -> ImageRunRecord:
    """Send one image, collect device timing/telemetry, and check golden parity."""
    norm_mode = mode.strip().lower()
    is_cnn = cfg.is_cnn_model(model)
    eff = board.effective_settings(model)

    use_repeats = eff.timed_iters if repeats is None else repeats
    use_warmup = eff.warmup_iters if warmup is None else warmup
    if use_repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {use_repeats}")
    if use_warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {use_warmup}")
    if board.streaming:
        use_repeats = 1
        use_warmup = 0

    max_chunk_data = min(board.max_payload_bytes, info.max_payload) - 4
    if max_chunk_data <= 0:
        raise ValueError(
            f"Negotiated max_payload ({board.max_payload_bytes}, {info.max_payload}) too small for chunks"
        )
    target_band_bytes = board.band_rows * cfg.images.width
    chunk_size = min(target_band_bytes, max_chunk_data)
    if board.streaming and (chunk_size % cfg.images.width) != 0:
        raise ValueError(
            f"Streaming chunk size ({chunk_size}) must be a multiple of image width ({cfg.images.width})"
        )

    dec = decoder if decoder is not None else _make_decoder(cfg, board.max_payload_bytes)
    seqs = seq_counter if seq_counter is not None else _SeqCounter()

    t_tx_start = time.perf_counter()

    # 1. IMG_BEGIN + ACK
    begin_payload = encode_img_begin(
        ImgBeginPayload(
            image_id=item.image_id,
            total_bytes=len(item.raw_bytes),
            crc32=item.crc32,
            repeats=use_repeats,
            warmup=use_warmup,
        )
    )
    transport.write(
        encode_frame(
            MSG_IMG_BEGIN,
            seqs.next(),
            begin_payload,
            version=cfg.protocol.version,
            magic=cfg.protocol.magic,
        )
    )
    begin_ack: ChunkAckPayload = _recv_typed(
        transport, dec, cfg.protocol.ack_timeout_s, ChunkAckPayload, "IMG_BEGIN"
    )
    if begin_ack.offset != 0xFFFFFFFF or begin_ack.status != CHUNK_STATUS_OK:
        raise RuntimeError(
            f"Unexpected IMG_BEGIN ACK: offset=0x{begin_ack.offset:08X} status={begin_ack.status}"
        )

    # 2. IMG_CHUNK loop with timeout retries
    total = len(item.raw_bytes)
    offset = 0
    while offset < total:
        slice_bytes = item.raw_bytes[offset : offset + chunk_size]
        chunk_frame = encode_frame(
            MSG_IMG_CHUNK,
            seqs.next(),
            encode_img_chunk(ImgChunkPayload(offset=offset, data=slice_bytes)),
            version=cfg.protocol.version,
            magic=cfg.protocol.magic,
        )
        ack: ChunkAckPayload | None = None
        for attempt in range(cfg.protocol.max_retries + 1):
            transport.write(chunk_frame)
            try:
                ack = _recv_typed(
                    transport,
                    dec,
                    cfg.protocol.ack_timeout_s,
                    ChunkAckPayload,
                    f"IMG_CHUNK@{offset}",
                )
                break
            except TimeoutError as exc:
                if attempt >= cfg.protocol.max_retries:
                    raise TimeoutError(
                        f"Timed out waiting for CHUNK_ACK at offset={offset} "
                        f"after {attempt + 1} attempts"
                    ) from exc
                logger.warning(
                    "Timeout waiting for CHUNK_ACK at offset=%d (attempt %d/%d); retrying",
                    offset,
                    attempt + 1,
                    cfg.protocol.max_retries + 1,
                )

        assert ack is not None
        if ack.offset != offset:
            raise ProtocolError(f"CHUNK_ACK offset mismatch: got {ack.offset}, expected {offset}")
        if ack.status not in (CHUNK_STATUS_OK, CHUNK_STATUS_DUPLICATE):
            raise RuntimeError(
                f"MCU rejected chunk at offset={offset} with status={ack.status}"
            )
        offset += len(slice_bytes)

    # 3. ACK_RX
    ack_rx: AckRxPayload = _recv_typed(
        transport, dec, cfg.protocol.ack_timeout_s, AckRxPayload, "ACK_RX"
    )
    t_transfer_host_ms = (time.perf_counter() - t_tx_start) * 1000.0
    if ack_rx.image_id != item.image_id:
        raise ProtocolError(
            f"ACK_RX image_id mismatch: got {ack_rx.image_id}, expected {item.image_id}"
        )
    if not ack_rx.crc_ok:
        raise RuntimeError(f"MCU reported image CRC32 failure for {item.name} (id={item.image_id})")

    # 4. RESULT
    result: ResultPayload = _recv_typed(
        transport, dec, eff.result_timeout_s, ResultPayload, "RESULT"
    )
    if result.image_id != item.image_id:
        raise ProtocolError(
            f"RESULT image_id mismatch: got {result.image_id}, expected {item.image_id}"
        )
    if result.status != 0:
        raise RuntimeError(
            f"MCU inference failed for {item.name} (id={item.image_id}): status={result.status}"
        )

    # 5. Optional GET_X94 debug descriptor fetch
    dumped_x94: ResultX94Payload | None = None
    if dump_descriptor and not is_cnn:
        transport.write(
            encode_frame(
                MSG_GET_X94,
                seqs.next(),
                b"",
                version=cfg.protocol.version,
                magic=cfg.protocol.magic,
            )
        )
        dumped_x94 = _recv_typed(
            transport, dec, cfg.protocol.ack_timeout_s, ResultX94Payload, "GET_X94"
        )
        if len(dumped_x94.values) != N_X94 or dumped_x94.mode != norm_mode:
            raise ProtocolError(
                f"Unexpected RESULT_X94 shape/mode: len={len(dumped_x94.values)}, mode={dumped_x94.mode}"
            )
        dumped_crc = crc32_ieee(encode_result_x94(dumped_x94))
        if dumped_crc != result.x94_crc32:
            raise ProtocolError(
                f"RESULT_X94 CRC32 mismatch vs RESULT.x94_crc32: "
                f"0x{dumped_crc:08X} != 0x{result.x94_crc32:08X}"
            )

    logits_float = result.logits_as_float(norm_mode)
    margin_float = (
        float(logits_float[1] - logits_float[0]) if len(logits_float) >= 2 else None
    )

    parity_status, parity_ok, x94_crc_ok, max_diff = _evaluate_parity(
        cfg=cfg,
        mode=norm_mode,
        is_cnn=is_cnn,
        result=result,
        logits_float=logits_float,
        golden=golden,
        dumped_x94=dumped_x94,
    )

    return ImageRunRecord(
        image=item,
        model=model,
        mode=norm_mode,
        is_cnn=is_cnn,
        result=result,
        logits_float=logits_float,
        margin_float=margin_float,
        t_transfer_host_ms=t_transfer_host_ms,
        includes_compute=board.streaming,
        golden=golden,
        parity_status=parity_status,
        parity_ok=parity_ok,
        x94_crc_ok=x94_crc_ok,
        logit_max_abs_diff=max_diff,
        dumped_x94=dumped_x94,
    )


def run_batch(
    transport: Transport,
    cfg: HwTestConfig,
    board: str,
    images: Sequence[PreparedImage],
    model: str,
    mode: str,
    *,
    repeats: int | None = None,
    warmup: int | None = None,
    dump_descriptor: bool = False,
    golden_lookup: GoldenLookup | None = None,
    transport_name: str = "uart",
    port_or_host: str = "",
) -> BatchRunResult:
    """Perform HELLO verification and execute inference across all `images`."""
    if not images:
        raise ValueError("No images provided to run_batch")
    board_spec = cfg.get_board(board)
    eff = board_spec.effective_settings(model)
    norm_mode = mode.strip().lower()
    if norm_mode not in eff.numeric_modes:
        raise ValueError(
            f"Mode '{norm_mode}' not allowed for ({board}, {model}). Allowed: {eff.numeric_modes}"
        )

    use_repeats = eff.timed_iters if repeats is None else repeats
    use_warmup = eff.warmup_iters if warmup is None else warmup
    if board_spec.streaming:
        use_repeats = 1
        use_warmup = 0

    decoder = _make_decoder(cfg, board_spec.max_payload_bytes)
    seqs = _SeqCounter()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    info = run_hello(transport, cfg, decoder=decoder, seq_counter=seqs)
    mismatches: list[str] = []
    if info.board_id != board_spec.board_id:
        mismatches.append(f"board_id={info.board_id!r} != {board_spec.board_id!r}")
    if info.model != model:
        mismatches.append(f"model={info.model!r} != {model!r}")
    if info.mode != norm_mode:
        mismatches.append(f"mode={info.mode!r} != {norm_mode!r}")
    if info.width != cfg.images.width or info.height != cfg.images.height:
        mismatches.append(
            f"dimensions={info.width}x{info.height} != {cfg.images.width}x{cfg.images.height}"
        )
    if info.streaming != board_spec.streaming:
        mismatches.append(f"streaming={info.streaming} != {board_spec.streaming}")
    if mismatches:
        raise RuntimeError(
            "Connected firmware does not match requested target: " + "; ".join(mismatches)
        )

    records: list[ImageRunRecord] = []
    for item in images:
        golden = golden_lookup.lookup(item, model) if golden_lookup is not None else None
        rec = run_image(
            transport,
            cfg,
            board_spec,
            item,
            model,
            norm_mode,
            info,
            repeats=use_repeats,
            warmup=use_warmup,
            dump_descriptor=dump_descriptor,
            golden=golden,
            decoder=decoder,
            seq_counter=seqs,
        )
        records.append(rec)

    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return BatchRunResult(
        board_id=board_spec.board_id,
        model=model,
        mode=norm_mode,
        transport_name=transport_name,
        port_or_host=port_or_host,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        info=info,
        warmup_requested=use_warmup,
        repeats_requested=use_repeats,
        records=tuple(records),
    )
