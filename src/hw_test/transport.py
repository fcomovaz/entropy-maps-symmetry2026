"""Transport abstractions and concrete implementations (UART, TCP, and in-memory fake)."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import socket
import threading
import time
from typing import Callable, Iterable, Protocol, runtime_checkable

import serial

from configs.logging import logger
from src.hw_test.protocol import Frame, FrameDecoder


@runtime_checkable
class Transport(Protocol):
    """Byte-stream transport contract shared by UART, WiFi/TCP, and test doubles."""

    def open(self) -> None:
        """Open the underlying connection and prepare it for framing."""

    def close(self) -> None:
        """Close the connection safely (idempotent)."""

    def write(self, data: bytes) -> None:
        """Transmit all bytes in `data` to the peer."""

    def read(self, n: int, timeout_s: float) -> bytes:
        """Read up to `n` available bytes, waiting at most `timeout_s` seconds."""

    def read_exactly(self, n: int, timeout_s: float) -> bytes:
        """Read exactly `n` bytes before `timeout_s` expires, or raise TimeoutError."""

    def recv_frame(self, decoder: FrameDecoder, timeout_s: float) -> Frame:
        """Read from transport until `decoder` yields one complete valid frame."""

    def flush_input(self) -> None:
        """Discard any pending unread bytes from the receive buffer."""


def _read_exactly_loop(
    read_fn: Callable[[int, float], bytes],
    n: int,
    timeout_s: float,
    label: str,
    *,
    break_on_empty: bool = False,
) -> bytes:
    """Shared bounded-deadline loop collecting exactly `n` bytes."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return b""
    buf = bytearray()
    deadline = time.monotonic() + max(0.0, timeout_s)
    while len(buf) < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 and buf:
            break
        chunk = read_fn(n - len(buf), max(0.0, remaining))
        if chunk:
            buf.extend(chunk)
            continue
        if break_on_empty or time.monotonic() >= deadline:
            break
    if len(buf) < n:
        raise TimeoutError(f"Timed out reading {n} bytes from {label}; received {len(buf)} B")
    return bytes(buf)


def _recv_frame_loop(
    read_fn: Callable[[int, float], bytes],
    decoder: FrameDecoder,
    timeout_s: float,
    label: str,
    *,
    read_chunk_size: int = 512,
    break_on_empty: bool = False,
) -> Frame:
    """Shared bounded-deadline loop extracting the next decoded frame."""
    existing = decoder.next_frame()
    if existing is not None:
        return existing

    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"Timed out waiting for frame from {label} ({timeout_s:.2f}s)")
        chunk = read_fn(read_chunk_size, remaining)
        if chunk:
            decoder.feed(chunk)
            frame = decoder.next_frame()
            if frame is not None:
                return frame
        elif break_on_empty or time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for frame from {label} ({timeout_s:.2f}s)")


class SerialTransport:
    """UART transport over `pyserial` with MCU reset boot wait and input buffer clearing."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        boot_wait_s: float = 2.0,
        *,
        reset_on_open: bool = True,
    ) -> None:
        if not port.strip():
            raise ValueError("Serial port must not be empty")
        if baudrate <= 0:
            raise ValueError(f"baudrate must be positive, got {baudrate}")
        if boot_wait_s < 0.0:
            raise ValueError(f"boot_wait_s must be >= 0, got {boot_wait_s}")
        self.port = port
        self.baudrate = baudrate
        self.boot_wait_s = boot_wait_s
        self.reset_on_open = reset_on_open
        self._ser: serial.Serial | None = None

    def __enter__(self) -> SerialTransport:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> serial.Serial:
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError(f"Serial transport {self.port} is not open")
        return self._ser

    def open(self) -> None:
        if self._ser is not None and self._ser.is_open:
            return
        logger.debug(
            "Opening serial port %s @ %d baud (boot_wait=%.2fs, reset_on_open=%s)",
            self.port,
            self.baudrate,
            self.boot_wait_s,
            self.reset_on_open,
        )
        ser = serial.Serial()
        ser.port = self.port
        ser.baudrate = self.baudrate
        # Keep a fixed short per-read timeout configured once before open(); mutating
        # ser.timeout on an open Windows usbser.sys CDC handle (e.g. Arduino Uno
        # ATmega16U2) triggers SetCommTimeouts mid-stream and drops incoming bytes.
        ser.timeout = 0.05
        ser.write_timeout = 5.0
        if not self.reset_on_open:
            ser.dtr = False
            ser.rts = False
        ser.open()
        try:
            if self.boot_wait_s > 0.0:
                time.sleep(self.boot_wait_s)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            ser.close()
            raise
        self._ser = ser

    def close(self) -> None:
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            finally:
                self._ser = None

    def write(self, data: bytes) -> None:
        ser = self._require_open()
        ser.write(data)
        ser.flush()

    def read(self, n: int, timeout_s: float) -> bytes:
        ser = self._require_open()
        if n <= 0:
            return b""
        waiting = ser.in_waiting
        if waiting > 0:
            return bytes(ser.read(min(n, waiting)))
        if timeout_s <= 0.0:
            return b""
        first = bytes(ser.read(1))
        if not first:
            return b""
        if n > 1:
            extra_waiting = ser.in_waiting
            if extra_waiting > 0:
                return first + bytes(ser.read(min(n - 1, extra_waiting)))
        return first

    def read_exactly(self, n: int, timeout_s: float) -> bytes:
        self._require_open()
        return _read_exactly_loop(self.read, n, timeout_s, f"serial:{self.port}")

    def recv_frame(self, decoder: FrameDecoder, timeout_s: float) -> Frame:
        self._require_open()
        return _recv_frame_loop(self.read, decoder, timeout_s, f"serial:{self.port}")

    def flush_input(self) -> None:
        ser = self._require_open()
        ser.reset_input_buffer()


class TcpTransport:
    """WiFi/TCP stream transport using `TCP_NODELAY` for low-latency ACK frames."""

    def __init__(self, host: str, port: int, connect_timeout_s: float = 5.0) -> None:
        if not host.strip():
            raise ValueError("TCP host must not be empty")
        if not (1 <= port <= 65535):
            raise ValueError(f"TCP port must be in [1, 65535], got {port}")
        if connect_timeout_s <= 0.0:
            raise ValueError(f"connect_timeout_s must be > 0, got {connect_timeout_s}")
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self._sock: socket.socket | None = None

    def __enter__(self) -> TcpTransport:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError(f"TCP transport {self.host}:{self.port} is not open")
        return self._sock

    def open(self) -> None:
        if self._sock is not None:
            return
        logger.debug("Connecting TCP %s:%d (timeout=%.2fs)", self.host, self.port, self.connect_timeout_s)
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def write(self, data: bytes) -> None:
        sock = self._require_open()
        sock.settimeout(self.connect_timeout_s)
        sock.sendall(data)

    def read(self, n: int, timeout_s: float) -> bytes:
        sock = self._require_open()
        if n <= 0:
            return b""
        sock.settimeout(max(0.0, timeout_s))
        try:
            chunk = sock.recv(n)
        except (TimeoutError, socket.timeout, BlockingIOError):
            return b""
        if not chunk:
            raise ConnectionError(f"TCP peer {self.host}:{self.port} closed connection")
        return chunk

    def read_exactly(self, n: int, timeout_s: float) -> bytes:
        self._require_open()
        return _read_exactly_loop(self.read, n, timeout_s, f"tcp:{self.host}:{self.port}")

    def recv_frame(self, decoder: FrameDecoder, timeout_s: float) -> Frame:
        self._require_open()
        return _recv_frame_loop(self.read, decoder, timeout_s, f"tcp:{self.host}:{self.port}")

    def flush_input(self) -> None:
        sock = self._require_open()
        prev_timeout = sock.gettimeout()
        try:
            sock.settimeout(0.0)
            while True:
                try:
                    data = sock.recv(4096)
                except BlockingIOError:
                    break
                if not data:
                    break
        finally:
            sock.settimeout(prev_timeout)


FakeHandler = Callable[[bytes], bytes | Iterable[bytes] | None]


class FakeTransport:
    """In-memory deterministic transport for unit testing protocol framing and sessions.

    Supports pre-queued RX chunks, callback handlers invoked on `write()`, byte-level
    fragmentation via `max_read_chunk`, and instant timeout when `instant_timeout=True`.
    """

    def __init__(
        self,
        *,
        handler: FakeHandler | None = None,
        max_read_chunk: int | None = None,
        instant_timeout: bool = True,
    ) -> None:
        if max_read_chunk is not None and max_read_chunk <= 0:
            raise ValueError(f"max_read_chunk must be positive, got {max_read_chunk}")
        self.handler = handler
        self.max_read_chunk = max_read_chunk
        self.instant_timeout = instant_timeout
        self._rx_chunks: deque[bytes] = deque()
        self._cond = threading.Condition()
        self._open = False
        self.tx_chunks: list[bytes] = []

    def __enter__(self) -> FakeTransport:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        """Whether the fake transport is currently open."""
        return self._open

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError("FakeTransport is not open")

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        with self._cond:
            self._open = False
            self._cond.notify_all()

    def push_rx(self, data: bytes) -> None:
        """Enqueue a raw byte chunk into the host-side receive buffer."""
        if not data:
            return
        with self._cond:
            self._rx_chunks.append(bytes(data))
            self._cond.notify_all()

    def push_rx_chunks(self, chunks: Iterable[bytes]) -> None:
        """Enqueue multiple discrete byte chunks to simulate fragmented or noisy reads."""
        with self._cond:
            for chunk in chunks:
                if chunk:
                    self._rx_chunks.append(bytes(chunk))
            self._cond.notify_all()

    def written_bytes(self) -> bytes:
        """Return concatenation of all transmitted payloads written so far."""
        return b"".join(self.tx_chunks)

    def write(self, data: bytes) -> None:
        self._require_open()
        payload = bytes(data)
        self.tx_chunks.append(payload)
        if self.handler is not None:
            response = self.handler(payload)
            if response is None:
                return
            if isinstance(response, (bytes, bytearray, memoryview)):
                self.push_rx(bytes(response))
            else:
                self.push_rx_chunks(bytes(c) for c in response)

    def read(self, n: int, timeout_s: float) -> bytes:
        self._require_open()
        if n <= 0:
            return b""
        limit = n if self.max_read_chunk is None else min(n, self.max_read_chunk)
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cond:
            while not self._rx_chunks:
                if self.instant_timeout or not self._open:
                    return b""
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return b""
                self._cond.wait(timeout=remaining)
            head = self._rx_chunks.popleft()
            if len(head) <= limit:
                return head
            self._rx_chunks.appendleft(head[limit:])
            return head[:limit]

    def read_exactly(self, n: int, timeout_s: float) -> bytes:
        self._require_open()
        return _read_exactly_loop(
            self.read,
            n,
            timeout_s,
            "fake",
            break_on_empty=self.instant_timeout,
        )

    def recv_frame(self, decoder: FrameDecoder, timeout_s: float) -> Frame:
        self._require_open()
        chunk_sz = self.max_read_chunk if self.max_read_chunk is not None else 512
        return _recv_frame_loop(
            self.read,
            decoder,
            timeout_s,
            "fake",
            read_chunk_size=chunk_sz,
            break_on_empty=self.instant_timeout,
        )

    def flush_input(self) -> None:
        self._require_open()
        with self._cond:
            self._rx_chunks.clear()
