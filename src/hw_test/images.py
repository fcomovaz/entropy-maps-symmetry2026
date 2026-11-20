"""Image discovery, strict format validation, and golden parity reference lookup."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

from configs.logging import logger
from src.hw_test.config import HwTestConfig, ImagesSpec
from src.hw_test.protocol import N_X94, crc32_ieee


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Validated grayscale image payload ready for MCU transmission."""

    image_id: int
    name: str
    path: Path
    width: int
    height: int
    mode: str
    raw_bytes: bytes
    crc32: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One row from `results/c_imp/<target>/vectors/manifest.csv`."""

    index: int
    file_name: str
    split: str
    label: int
    bin_relpath: str
    sha256: str


@dataclass(frozen=True, slots=True)
class GoldenReference:
    """Expected desktop golden outputs for one manifest image and one exported model."""

    index: int
    file_name: str
    true_label: int
    sha256: str
    model: str
    is_cnn: bool
    expected_q15_logits_raw: tuple[int, ...]
    expected_q15_label: int
    expected_f32_logits: tuple[float, ...]
    expected_f32_label: int
    expected_x94_q15: tuple[int, ...] | None = None
    expected_x94_f64: tuple[float, ...] | None = None


def discover_images(
    images_dir: Path,
    *,
    image: str | Path | None = None,
    glob_pattern: str | None = None,
    select_all: bool = False,
) -> tuple[Path, ...]:
    """Discover image files deterministically via `--image`, `--glob`, or `--all`."""
    modes_selected = sum(
        1 for flag in (image is not None, glob_pattern is not None, select_all) if flag
    )
    if modes_selected != 1:
        raise ValueError(
            "Specify exactly one image selector: `image`, `glob_pattern`, or `select_all=True`"
        )

    base_dir = Path(images_dir).resolve()
    if image is not None:
        candidate = Path(image)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        elif (base_dir / candidate).exists():
            resolved = (base_dir / candidate).resolve()
        else:
            resolved = candidate.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image!r} (checked {base_dir / candidate} and {resolved})"
            )
        return (resolved,)

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {base_dir}")

    if glob_pattern is not None:
        if not glob_pattern.strip():
            raise ValueError("glob_pattern must not be empty")
        matched = sorted(
            (
                p.resolve()
                for p in base_dir.glob(glob_pattern)
                if p.is_file() and not p.name.startswith(".")
            ),
            key=lambda p: p.as_posix(),
        )
        if not matched:
            raise ValueError(f"No images matched glob {glob_pattern!r} in {base_dir}")
        return tuple(matched)

    # select_all == True
    all_files = sorted(
        (
            p.resolve()
            for p in base_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ),
        key=lambda p: p.as_posix(),
    )
    if not all_files:
        raise ValueError(f"No images found in {base_dir}")
    return tuple(all_files)


def load_images(paths: Sequence[Path], spec: ImagesSpec) -> tuple[PreparedImage, ...]:
    """Load and strictly validate images without resizing or mode conversion.

    Inspects every file in `paths` and raises a single `ValueError` listing all
    invalid files and failure reasons when any file fails validation.
    """
    if not paths:
        raise ValueError("No image paths provided to load_images")

    allowed_exts = {ext.lower() for ext in spec.allowed_extensions}
    expected_bytes = spec.width * spec.height
    prepared: list[PreparedImage] = []
    errors: list[str] = []

    for img_path in paths:
        p = Path(img_path)
        reasons: list[str] = []
        if not p.is_file():
            errors.append(f"{p}: file does not exist")
            continue

        if p.suffix.lower() not in allowed_exts:
            reasons.append(
                f"invalid extension {p.suffix!r} (allowed: {sorted(allowed_exts)})"
            )

        raw_bytes: bytes | None = None
        actual_w = -1
        actual_h = -1
        actual_mode = ""
        try:
            with Image.open(p) as im:
                actual_mode = im.mode
                actual_w, actual_h = im.size
                if actual_mode != spec.mode:
                    reasons.append(f"mode={actual_mode!r} != expected {spec.mode!r}")
                if (actual_w, actual_h) != (spec.width, spec.height):
                    reasons.append(
                        f"size={actual_w}x{actual_h} != expected {spec.width}x{spec.height}"
                    )
                if not reasons:
                    im.load()
                    raw_bytes = im.tobytes()
                    if len(raw_bytes) != expected_bytes:
                        reasons.append(
                            f"byte length={len(raw_bytes)} != expected {expected_bytes}"
                        )
        except Exception as exc:
            reasons.append(f"failed to open image ({exc})")

        if reasons:
            errors.append(f"{p}: {'; '.join(reasons)}")
            continue

        assert raw_bytes is not None
        prepared.append(
            PreparedImage(
                image_id=len(prepared),
                name=p.name,
                path=p.resolve(),
                width=actual_w,
                height=actual_h,
                mode=actual_mode,
                raw_bytes=raw_bytes,
                crc32=crc32_ieee(raw_bytes),
                sha256=hashlib.sha256(raw_bytes).hexdigest(),
            )
        )

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(
            f"Image validation failed for {len(errors)} file(s) (no resizing/conversion allowed):\n  - {detail}"
        )

    logger.info(
        "Validated %d image(s) (%dx%d mode=%s)",
        len(prepared),
        spec.width,
        spec.height,
        spec.mode,
    )
    return tuple(prepared)


def prepare_images(
    images_dir: Path,
    spec: ImagesSpec,
    *,
    image: str | Path | None = None,
    glob_pattern: str | None = None,
    select_all: bool = False,
) -> tuple[PreparedImage, ...]:
    """Discover and validate images in one step."""
    paths = discover_images(
        images_dir,
        image=image,
        glob_pattern=glob_pattern,
        select_all=select_all,
    )
    return load_images(paths, spec)


@dataclass(frozen=True, slots=True)
class _ModelGoldenRow:
    index: int
    file_name: str
    true_label: int
    is_cnn: bool
    q15_logits_raw: tuple[int, ...]
    q15_label: int
    f32_logits: tuple[float, ...]
    f32_label: int


class GoldenLookup:
    """Lazy reader over exported vector manifests, golden x94 descriptors, and model logits."""

    def __init__(self, vectors_dir: Path, generated_dir: Path) -> None:
        self.vectors_dir = Path(vectors_dir)
        self.generated_dir = Path(generated_dir)
        self._by_sha256: dict[str, ManifestEntry] = {}
        self._by_filename: dict[str, ManifestEntry] = {}
        self._x94_q15: dict[int, tuple[int, ...]] | None = None
        self._x94_f64: dict[int, tuple[float, ...]] | None = None
        self._model_cache: dict[str, dict[int, _ModelGoldenRow] | None] = {}
        self._load_manifest()

    @classmethod
    def from_dirs(cls, vectors_dir: Path, generated_dir: Path) -> GoldenLookup:
        """Create a `GoldenLookup` from explicit `vectors/` and `generated/` directories."""
        return cls(vectors_dir=vectors_dir, generated_dir=generated_dir)

    @classmethod
    def from_config(cls, cfg: HwTestConfig) -> GoldenLookup:
        """Create a `GoldenLookup` anchored to the target configured in `HwTestConfig`."""
        return cls(
            vectors_dir=cfg.c_imp_cfg.results_dir / "vectors",
            generated_dir=cfg.c_imp_cfg.generated_dir,
        )

    def _load_manifest(self) -> None:
        manifest_path = self.vectors_dir / "manifest.csv"
        if not manifest_path.is_file():
            logger.warning(
                "Golden manifest not found at %s; parity will report 'n/a'",
                manifest_path,
            )
            return

        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"index", "file_name", "split", "label", "bin_relpath", "sha256"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Malformed manifest header in {manifest_path}: expected {sorted(required)}"
                )
            for row in reader:
                entry = ManifestEntry(
                    index=int(row["index"]),
                    file_name=row["file_name"].strip(),
                    split=row["split"].strip(),
                    label=int(row["label"]),
                    bin_relpath=row["bin_relpath"].strip(),
                    sha256=row["sha256"].strip().lower(),
                )
                self._by_sha256.setdefault(entry.sha256, entry)
                self._by_filename.setdefault(entry.file_name, entry)

    def _load_x94_q15(self) -> dict[int, tuple[int, ...]]:
        if self._x94_q15 is not None:
            return self._x94_q15
        path = self.vectors_dir / "golden_x94_q15.csv"
        table: dict[int, tuple[int, ...]] = {}
        if path.is_file():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                feat_cols = [c for c in (reader.fieldnames or []) if c not in ("index", "label")]
                if len(feat_cols) != N_X94:
                    raise ValueError(
                        f"Expected {N_X94} x94 columns in {path}, found {len(feat_cols)}"
                    )
                for row in reader:
                    table[int(row["index"])] = tuple(int(row[c]) for c in feat_cols)
        self._x94_q15 = table
        return table

    def _load_x94_f64(self) -> dict[int, tuple[float, ...]]:
        if self._x94_f64 is not None:
            return self._x94_f64
        path = self.vectors_dir / "golden_x94_f64.csv"
        table: dict[int, tuple[float, ...]] = {}
        if path.is_file():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                feat_cols = [c for c in (reader.fieldnames or []) if c not in ("index", "label")]
                if len(feat_cols) != N_X94:
                    raise ValueError(
                        f"Expected {N_X94} x94 columns in {path}, found {len(feat_cols)}"
                    )
                for row in reader:
                    table[int(row["index"])] = tuple(float(row[c]) for c in feat_cols)
        self._x94_f64 = table
        return table

    def _load_model_rows(self, model: str) -> Mapping[int, _ModelGoldenRow] | None:
        if model in self._model_cache:
            return self._model_cache[model]

        tabular_path = self.generated_dir / f"golden_{model}.csv"
        cnn_path = self.generated_dir / f"golden_cnn_{model}.csv"

        rows: dict[int, _ModelGoldenRow] = {}
        if tabular_path.is_file():
            with tabular_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row["index"])
                    rows[idx] = _ModelGoldenRow(
                        index=idx,
                        file_name=row["file_name"].strip(),
                        true_label=int(row["true_label"]),
                        is_cnn=False,
                        q15_logits_raw=(int(row["py_q15_logit_raw"]),),
                        q15_label=int(row["py_q15_label"]),
                        f32_logits=(float(row["py_f32_logit"]),),
                        f32_label=int(row["py_f32_label"]),
                    )
            self._model_cache[model] = rows
            return rows

        if cnn_path.is_file():
            with cnn_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row["index"])
                    rows[idx] = _ModelGoldenRow(
                        index=idx,
                        file_name=row["file_name"].strip(),
                        true_label=int(row["true_label"]),
                        is_cnn=True,
                        q15_logits_raw=(
                            int(row["py_q15_logit0_raw"]),
                            int(row["py_q15_logit1_raw"]),
                        ),
                        q15_label=int(row["py_q15_label"]),
                        f32_logits=(
                            float(row["py_f32_logit0"]),
                            float(row["py_f32_logit1"]),
                        ),
                        f32_label=int(row["py_f32_label"]),
                    )
            self._model_cache[model] = rows
            return rows

        logger.debug("No golden CSV found for model %r under %s", model, self.generated_dir)
        self._model_cache[model] = None
        return None

    def lookup_manifest(self, image: PreparedImage | str) -> ManifestEntry | None:
        """Look up a `ManifestEntry` by `PreparedImage.sha256` or explicit sha256/filename."""
        if isinstance(image, PreparedImage):
            return self._by_sha256.get(image.sha256.lower())
        key = image.strip()
        if key.lower() in self._by_sha256:
            return self._by_sha256[key.lower()]
        return self._by_filename.get(key)

    def get_x94_q15(self, index: int) -> tuple[int, ...] | None:
        """Return golden `q15` x94 descriptor tuple for manifest row `index`, if available."""
        return self._load_x94_q15().get(index)

    def get_x94_f64(self, index: int) -> tuple[float, ...] | None:
        """Return golden `f64` x94 descriptor tuple for manifest row `index`, if available."""
        return self._load_x94_f64().get(index)

    def lookup(self, image: PreparedImage | str, model: str) -> GoldenReference | None:
        """Return golden parity expectations for `(image, model)`, or None if not in manifest."""
        entry = self.lookup_manifest(image)
        if entry is None:
            return None
        model_rows = self._load_model_rows(model)
        if model_rows is None:
            return None
        mrow = model_rows.get(entry.index)
        if mrow is None:
            return None

        x94_q15 = None if mrow.is_cnn else self.get_x94_q15(entry.index)
        x94_f64 = None if mrow.is_cnn else self.get_x94_f64(entry.index)
        return GoldenReference(
            index=entry.index,
            file_name=entry.file_name,
            true_label=mrow.true_label,
            sha256=entry.sha256,
            model=model,
            is_cnn=mrow.is_cnn,
            expected_q15_logits_raw=mrow.q15_logits_raw,
            expected_q15_label=mrow.q15_label,
            expected_f32_logits=mrow.f32_logits,
            expected_f32_label=mrow.f32_label,
            expected_x94_q15=x94_q15,
            expected_x94_f64=x94_f64,
        )
