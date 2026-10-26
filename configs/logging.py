"""One shared logger, configured once by main.py. Standard library only."""

import logging
import sys

# from logging.handlers import RotatingFileHandler  # Previous size-based setup.
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from time import gmtime
from zipfile import ZIP_DEFLATED, ZipFile

# Every module imports this exact same Logger object.
logger = logging.getLogger("research")


def zip_name(default_name):
    """Keep the timestamp intact so the handler can find and prune old ZIPs."""
    return default_name + ".zip"


def zip_rotator(source, destination):
    """Archive the closed log; remove it only after the ZIP is complete."""
    source_path = Path(source)
    temporary = Path(destination + ".tmp")
    # Publish a completed archive, rather than writing directly to its final name.
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(source_path, arcname=source_path.name)
    temporary.replace(destination)
    source_path.unlink()


def configure_logging(level=logging.INFO):
    """Send records to the console and one log file (rotation is optional)."""
    logger.setLevel(level)

    # This logger owns its handlers. Do not also forward records to root handlers.
    logger.propagate = False

    # Calling setup twice must not add duplicate handlers/messages.
    # Handlers default to NOTSET, so the logger's level controls both outputs.
    if logger.handlers:
        return

    # Resolve relative to this file, not the terminal's working directory.
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        # fmt="%(asctime)s.%(msecs)03dZ | %(levelname)-10s | %(name)s | "
        fmt="%(asctime)s.%(msecs)03dZ |"
        " %(levelname)-7s | "
        "%(filename)15s:%(lineno)-4d |"
        # " %(funcName)s |"
        " %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = gmtime  # UTC timestamps; Z means UTC.

    console = logging.StreamHandler(sys.stdout)

    # ---- Pick exactly ONE file handler below ----
    #
    # A `namer` renames the rolled file, and a `rotator` receives (source, dest)
    # when the handler rotates. Both are inherited from BaseRotatingHandler,
    # so they work for RotatingFileHandler AND TimedRotatingFileHandler.
    # Do NOT attach them to a plain FileHandler: it never rotates, so the
    # hooks are never called.

    # (1) No rotation: one plain-text file that grows forever.
    log_file = logging.FileHandler(log_dir / "research.log", encoding="utf-8")

    # (2) Size-based rotation: a new file every time the current one passes
    #     maxBytes. Rolled files are suffixed .1, .2, ...; backupCount keeps
    #     the N newest and prunes the rest. With the ZIP hooks below, the
    #     active research.log stays plain text and each .1/.2/... becomes .zip.
    # log_file = RotatingFileHandler(
    #     log_dir / "research.log",
    #     maxBytes=5_000,
    #     backupCount=10,
    #     encoding="utf-8",
    # )
    # log_file.namer = zip_name       # research.log.1 -> research.log.1.zip
    # log_file.rotator = zip_rotator  # write the .zip, then delete the .log.1

    # (3) Time-based rotation: rotate on the first log record *after* the
    #     interval has elapsed (no background timer). The new file is named
    #     with a timestamp instead of a numeric suffix. `utc=True` makes the
    #     timestamp in the backup filename UTC as well.
    # log_file = TimedRotatingFileHandler(
    #     log_dir / "research.log",
    #     when="S",  # S = seconds, M = minutes, H = hours, D = days.
    #     interval=2,
    #     backupCount=5,
    #     encoding="utf-8",
    #     utc=True,
    # )
    # log_file.namer = zip_name
    # log_file.rotator = zip_rotator

    # Attach both sinks. Handlers keep their default level (NOTSET), so the
    # logger's level — set at the top of this function — filters both outputs.
    for handler in (console, log_file):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
