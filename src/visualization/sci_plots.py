"""
sci_plots.py — Módulo maestro de visualización científica (Pure Matplotlib + NumPy).

Diseñado para producir figuras sobrias, vectoriales y listas para publicación en
revistas como MDPI (p. ej. Symmetry) y Springer, con configuración headless (`Agg`),
colocación sin colisiones para etiquetas Pareto y una API basada en diccionarios.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# =============================================================================
# 1. PALETAS EDITORIALES, MARCADORES Y TAMAÑOS ESTÁNDAR (EN PULGADAS)
# =============================================================================

PALETTES: dict[str, list[str]] = {
    # Sobria, profesional y amigable con daltonismo (inspirada en Tol Muted / Okabe-Ito)
    "editorial": [
        "#1F4E79",  # Azul marino sobrio
        "#B85042",  # Terracota / carmesí apagado
        "#2E7D5B",  # Verde bosque
        "#C98A2E",  # Ocre / ámbar formal
        "#6B4C7A",  # Ciruela apagado
        "#2A8C82",  # Verde azulado (teal)
        "#5C6B73",  # Gris pizarra
        "#8C564B",  # Marrón neutro
    ],
    # Alternativa ultra-conservadora para MDPI / Springer
    "classic": [
        "#003F5C",
        "#BC5090",
        "#FFA600",
        "#58508D",
        "#FF6361",
        "#2F4B7C",
        "#665191",
    ],
    # Escala de grises para impresión en blanco y negro
    "grayscale": [
        "#1A1A1A",
        "#4D4D4D",
        "#737373",
        "#999999",
        "#BFBFBF",
    ],
}

MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES: list[str] = ["-", "--", "-.", ":"]
HATCHES: list[str] = ["", "//", "\\\\", "xx", "..", "++"]

# Anchos estándar en pulgadas para plantillas LaTeX (MDPI / Springer)
FIG_SIZES: dict[str, tuple[float, float]] = {
    "single": (3.5, 2.6),  # 1 columna (~8.9 cm)
    "medium": (5.2, 3.6),  # Intermedio (muy común en MDPI una columna ancha)
    "double": (6.8, 3.8),  # 2 columnas completas (~17.3 cm)
    "wide": (7.0, 3.2),  # Formato panorámico
    "square": (4.0, 4.0),  # Matrices de confusión / heatmaps cuadrados
}

_ACTIVE_PALETTE: list[str] = list(PALETTES["editorial"])


def get_colors(
    n: int | None = None, palette: str | Sequence[str] | None = None
) -> list[str]:
    """Devuelve una lista de colores de la paleta activa o de una paleta específica."""
    if palette is None:
        base = _ACTIVE_PALETTE
    elif isinstance(palette, str):
        if palette not in PALETTES:
            raise ValueError(
                f"Paleta desconocida '{palette}'. Opciones: {list(PALETTES)}"
            )
        base = PALETTES[palette]
    else:
        base = list(palette)

    if n is None:
        return list(base)
    return [base[i % len(base)] for i in range(n)]


# =============================================================================
# 2. CONFIGURACIÓN GLOBAL DE ESTILO (MDPI / SPRINGER)
# =============================================================================


def apply_style(
    journal: str = "mdpi",
    palette: str = "editorial",
    font_size: float | None = None,
    use_tex: bool = False,
) -> None:
    """
    Aplica la configuración global de Matplotlib para estética de publicación.

    Parámetros:
        journal: "mdpi" (sans-serif limpio, estilo MDPI/Symmetry) o
                 "springer" (serif clásico, ticks hacia adentro, estilo Springer/IEEE).
        palette: "editorial", "classic" o "grayscale".
        font_size: tamaño base de fuente en puntos (opcional; sobrescribe el preset).
        use_tex: si es True, activa el renderizado completo con LaTeX del sistema.
    """
    global _ACTIVE_PALETTE
    if palette not in PALETTES:
        raise ValueError(f"Paleta desconocida '{palette}'. Opciones: {list(PALETTES)}")
    _ACTIVE_PALETTE = list(PALETTES[palette])

    journal_key = journal.lower().strip()
    if journal_key not in {"mdpi", "springer"}:
        raise ValueError("journal debe ser 'mdpi' o 'springer'.")

    is_springer = journal_key == "springer"
    base_fs = font_size if font_size is not None else (9.5 if is_springer else 10.0)

    params: dict[str, Any] = {
        # Tipografía
        "text.usetex": use_tex,
        "font.family": "serif" if is_springer else "sans-serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix" if is_springer else "dejavusans",
        "font.size": base_fs,
        "axes.titlesize": base_fs + 0.5,
        "axes.titleweight": "semibold",
        "axes.titlepad": 8.0,
        "axes.labelsize": base_fs,
        "axes.labelweight": "normal",
        "xtick.labelsize": base_fs - 1.0,
        "ytick.labelsize": base_fs - 1.0,
        "legend.fontsize": base_fs - 1.2,
        "legend.title_fontsize": base_fs - 0.8,
        # Ejes y bordes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#2B2B2B",
        "axes.spines.top": is_springer,
        "axes.spines.right": is_springer,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=_ACTIVE_PALETTE),
        # Ticks
        "xtick.direction": "in" if is_springer else "out",
        "ytick.direction": "in" if is_springer else "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.top": is_springer,
        "ytick.right": is_springer,
        # Rejilla muy sutil
        "grid.color": "#888888",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.22,
        # Líneas y marcadores
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "errorbar.capsize": 3.0,
        # Leyenda limpia
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.45,
        "legend.handlelength": 1.8,
        # Exportación de alta calidad y fuentes embebidas (Type 42 para PDF/PS)
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    mpl.rcParams.update(params)


# =============================================================================
# 3. UTILIDADES INTERNAS, ANOTACIONES SIN COLISIONES Y EXPORTACIÓN
# =============================================================================


def _resolve_figsize(figsize: str | tuple[float, float]) -> tuple[float, float]:
    if isinstance(figsize, str):
        if figsize not in FIG_SIZES:
            raise ValueError(
                f"Tamaño desconocido '{figsize}'. Opciones: {list(FIG_SIZES)}"
            )
        return FIG_SIZES[figsize]
    return figsize


def _prepare_ax(
    ax: Axes | None,
    figsize: str | tuple[float, float],
) -> tuple[Figure, Axes, bool]:
    """Devuelve (fig, ax, created_here). Si ax ya existe, no crea figura nueva."""
    if ax is None:
        fig, new_ax = plt.subplots(figsize=_resolve_figsize(figsize))
        return fig, new_ax, True
    return ax.figure, ax, False


def _boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Devuelve True si dos cajas [x0, x1, y0, y1] se traslapan o tocan."""
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3])


def _overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    dx = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    dy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return dx * dy


def _point_to_segment_dist(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    vx = x2 - x1
    vy = y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / denom))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return math.hypot(px - proj_x, py - proj_y)


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    def _ccw(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = _ccw(b1, b2, a1)
    d2 = _ccw(b1, b2, a2)
    d3 = _ccw(a1, a2, b1)
    d4 = _ccw(a1, a2, b2)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    )


def annotate_points_non_overlapping(
    ax: Axes,
    labels: Sequence[str],
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    xscale: str = "linear",
    yscale: str = "linear",
    fontsize: float | None = None,
    text_color: str = "#222222",
    leader_color: str = "#6B7280",
    reserved_boxes_axes: Sequence[tuple[float, float, float, float]] = (),
    zorder: int = 5,
) -> list[tuple[float, float, float, float]]:
    """
    Coloca anotaciones de puntos en coordenadas de ejes evitando colisiones entre etiquetas,
    marcadores, líneas guía y zonas reservadas (como leyendas), con líneas guía finas.

    Devuelve la lista de cajas `[x0, x1, y0, y1]` en coordenadas normalizadas `[0, 1]`
    en el mismo orden que `labels`.
    """
    if not (len(labels) == len(xs) == len(ys)):
        raise ValueError("labels, xs, and ys must have identical lengths.")
    if not labels:
        return []

    x_lim0, x_lim1 = ax.get_xlim()
    y_lim0, y_lim1 = ax.get_ylim()
    if xscale == "log":
        tx0 = math.log10(max(float(x_lim0), 1e-12))
        tx1 = math.log10(max(float(x_lim1), 1e-12))
    else:
        tx0, tx1 = float(x_lim0), float(x_lim1)
    if yscale == "log":
        ty0 = math.log10(max(float(y_lim0), 1e-12))
        ty1 = math.log10(max(float(y_lim1), 1e-12))
    else:
        ty0, ty1 = float(y_lim0), float(y_lim1)

    tx_span = max(1e-9, tx1 - tx0)
    ty_span = max(1e-9, ty1 - ty0)

    def _to_norm(px: float, py: float) -> tuple[float, float]:
        tx = math.log10(max(float(px), 1e-12)) if xscale == "log" else float(px)
        ty = math.log10(max(float(py), 1e-12)) if yscale == "log" else float(py)
        return ((tx - tx0) / tx_span, (ty - ty0) / ty_span)

    fig = ax.figure
    fig_w_in, fig_h_in = fig.get_size_inches()
    pos = ax.get_position()
    ax_w_pt = max(140.0, float(fig_w_in) * float(pos.width) * 72.0)
    ax_h_pt = max(100.0, float(fig_h_in) * float(pos.height) * 72.0)
    fs = (
        float(fontsize)
        if fontsize is not None
        else max(6.8, float(mpl.rcParams["xtick.labelsize"]) - 0.8)
    )

    pt_norms = [_to_norm(px, py) for px, py in zip(xs, ys)]
    placed_boxes: list[tuple[float, float, float, float]] = list(reserved_boxes_axes)
    placed_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    out_boxes: list[tuple[float, float, float, float] | None] = [None] * len(labels)

    # Multi-ring offset candidates in points with matched alignment (ox, oy, ha, va)
    cand_offsets: list[tuple[float, float, str, str]] = []
    for rad in (8.0, 14.0, 22.0, 32.0, 44.0, 58.0, 74.0, 92.0, 112.0):
        # Eight cardinal/diagonal directions + elongated horizontal spreads
        dirs = (
            (1.0, 0.45, "left", "bottom"),
            (1.0, -0.45, "left", "top"),
            (-1.0, 0.45, "right", "bottom"),
            (-1.0, -0.45, "right", "top"),
            (0.0, 1.0, "center", "bottom"),
            (0.0, -1.0, "center", "top"),
            (1.35, 0.0, "left", "center"),
            (-1.35, 0.0, "right", "center"),
            (0.65, 1.05, "left", "bottom"),
            (-0.65, 1.05, "right", "bottom"),
            (0.65, -1.05, "left", "top"),
            (-0.65, -1.05, "right", "top"),
        )
        for dx_m, dy_m, ha, va in dirs:
            cand_offsets.append((dx_m * rad, dy_m * rad, ha, va))

    # Process crowded points in deterministic left-to-right, bottom-to-top order
    indexed = sorted(range(len(labels)), key=lambda i: (pt_norms[i][0], pt_norms[i][1], labels[i]))

    for idx in indexed:
        lbl = labels[idx]
        ux, vy = pt_norms[idx]
        cx = float(xs[idx])
        cy = float(ys[idx])

        est_w = min(0.36, (0.58 * fs * max(1, len(lbl)) + 6.0) / ax_w_pt)
        est_h = min(0.12, (1.42 * fs + 4.0) / ax_h_pt)

        best_choice: tuple[float, float, str, str] = (8.0, 6.0, "left", "bottom")
        best_box: tuple[float, float, float, float] = (
            ux + 8.0 / ax_w_pt,
            ux + 8.0 / ax_w_pt + est_w,
            vy + 6.0 / ax_h_pt,
            vy + 6.0 / ax_h_pt + est_h,
        )
        best_anchor = (ux + 8.0 / ax_w_pt, vy + 6.0 / ax_h_pt)
        best_score = -1e12

        for ox_pt, oy_pt, ha, va in cand_offsets:
            ax_anchor = ux + ox_pt / ax_w_pt
            ay_anchor = vy + oy_pt / ax_h_pt

            if ha == "left":
                bx0, bx1 = ax_anchor, ax_anchor + est_w
            elif ha == "right":
                bx0, bx1 = ax_anchor - est_w, ax_anchor
            else:
                bx0, bx1 = ax_anchor - 0.5 * est_w, ax_anchor + 0.5 * est_w

            if va == "bottom":
                by0, by1 = ay_anchor, ay_anchor + est_h
            elif va == "top":
                by0, by1 = ay_anchor - est_h, ay_anchor
            else:
                by0, by1 = ay_anchor - 0.5 * est_h, ay_anchor + 0.5 * est_h

            pad_x = 0.004
            pad_y = 0.004
            rect = (bx0 - pad_x, bx1 + pad_x, by0 - pad_y, by1 + pad_y)

            overflow = (
                max(0.0, 0.012 - bx0)
                + max(0.0, bx1 - 0.988)
                + max(0.0, 0.012 - by0)
                + max(0.0, by1 - 0.988)
            )
            box_hits = sum(1 for r in placed_boxes if _boxes_overlap(rect, r))
            box_area = sum(_overlap_area(rect, r) for r in placed_boxes)
            marker_hits = sum(
                1
                for m_idx, (mx, my) in enumerate(pt_norms)
                if (bx0 - 0.014) <= mx <= (bx1 + 0.014)
                and (by0 - 0.016) <= my <= (by1 + 0.016)
                and (m_idx != idx or math.hypot(ox_pt, oy_pt) < 5.0)
            )
            anchor_pt = (
                max(bx0, min(bx1, ux)),
                max(by0, min(by1, vy)),
            )
            seg_near_markers = sum(
                1
                for m_idx, (mx, my) in enumerate(pt_norms)
                if m_idx != idx
                and _point_to_segment_dist(mx, my, ux, vy, anchor_pt[0], anchor_pt[1]) < 0.025
            )
            seg_crossings = sum(
                1
                for s0, s1 in placed_segments
                if _segments_intersect((ux, vy), anchor_pt, s0, s1)
            )
            dist_pen = math.hypot(ox_pt / 32.0, oy_pt / 24.0)

            score = (
                -10000.0 * overflow
                - 2500.0 * box_hits
                - 6000.0 * box_area
                - 350.0 * marker_hits
                - 140.0 * seg_near_markers
                - 180.0 * seg_crossings
                - dist_pen
            )
            if score > best_score:
                best_score = score
                best_choice = (ox_pt, oy_pt, ha, va)
                best_box = (bx0, bx1, by0, by1)
                best_anchor = anchor_pt
                if (
                    overflow == 0.0
                    and box_hits == 0
                    and marker_hits == 0
                    and seg_near_markers == 0
                    and seg_crossings == 0
                    and dist_pen < 0.45
                ):
                    break

        ox_sel, oy_sel, ha_sel, va_sel = best_choice
        placed_boxes.append(
            (best_box[0] - 0.004, best_box[1] + 0.004, best_box[2] - 0.004, best_box[3] + 0.004)
        )
        placed_segments.append(((ux, vy), best_anchor))
        out_boxes[idx] = best_box

        need_leader = math.hypot(ox_sel, oy_sel) > 11.5
        ax.annotate(
            lbl,
            (cx, cy),
            xytext=(ox_sel, oy_sel),
            textcoords="offset points",
            ha=ha_sel,
            va=va_sel,
            fontsize=fs,
            color=text_color,
            arrowprops={
                "arrowstyle": "-",
                "color": leader_color,
                "lw": 0.7,
                "alpha": 0.75,
                "shrinkA": 0,
                "shrinkB": 3,
            }
            if need_leader
            else None,
            zorder=zorder,
        )

    return [b for b in out_boxes if b is not None]


def _finalize_ax(
    fig: Figure,
    ax: Axes,
    created_here: bool,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    grid: bool | str = True,
    show_legend: bool = False,
    legend_loc: str = "best",
    legend_ncol: int = 1,
    legend_title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if grid is True:
        ax.grid(True, which="major")
    elif isinstance(grid, str) and grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, which="major")
    else:
        ax.grid(False)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc=legend_loc,
                ncol=legend_ncol,
                title=legend_title,
                framealpha=0.92,
            )

    if created_here and not fig.get_constrained_layout():
        fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)

    return fig, ax


def save_figure(
    fig: Figure,
    path: str | Path,
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """
    Guarda la figura en PDF, PNG, EPS o SVG creando las carpetas necesarias.
    Recomendación para LaTeX (MDPI/Springer): usar extensión .pdf (vectorial) o .png a 300 DPI.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=transparent,
    )
    return out


def create_grid(
    nrows: int,
    ncols: int,
    *,
    figsize: str | tuple[float, float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    panel_labels: bool = True,
    label_pos: tuple[float, float] = (-0.12, 1.06),
) -> tuple[Figure, np.ndarray]:
    """
    Crea una cuadrícula de subgráficos para combinar varias funciones usando `ax=...`.
    Si `panel_labels=True`, añade automáticamente (a), (b), (c)... en cada panel.
    """
    if figsize is None:
        resolved = (3.4 * ncols, 2.7 * nrows)
    else:
        resolved = _resolve_figsize(figsize)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=resolved,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        constrained_layout=True,
    )

    if panel_labels:
        letters = "abcdefghijklmnopqrstuvwxyz"
        for idx, ax in enumerate(axes.ravel()):
            if idx < len(letters):
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    f"({letters[idx]})",
                    transform=ax.transAxes,
                    fontsize=mpl.rcParams["axes.titlesize"],
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                )

    return fig, axes


# =============================================================================
# 4. FUNCIONES MAESTRAS DE GRAFICACIÓN
# =============================================================================


def plot_lines(
    data: Mapping[str, Any],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    markers: bool = True,
    vary_linestyles: bool = False,
    band_alpha: float = 0.16,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    grid: bool | str = True,
    legend_loc: str = "best",
    legend_ncol: int = 1,
    palette: str | Sequence[str] | None = None,
    figsize: str | tuple[float, float] = "medium",
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """
    Gráfico de curvas / líneas científicas desde un diccionario.

    Formatos aceptados para cada valor del diccionario `data`:
        - `y` (1D): usa índices 0..N-1 en el eje X.
        - `(x, y)`: curva estándar.
        - `(x, y, yerr)`: curva con banda sombreada simétrica `y ± yerr`.
        - `(x, y, y_low, y_high)`: curva con banda asimétrica `[y_low, y_high]`.
    """
    fig, ax, created = _prepare_ax(ax, figsize)
    colors = get_colors(len(data), palette)

    for idx, (label, values) in enumerate(data.items()):
        color = colors[idx]
        ls = LINESTYLES[idx % len(LINESTYLES)] if vary_linestyles else "-"
        mk = MARKERS[idx % len(MARKERS)] if markers else None

        if isinstance(values, tuple):
            if len(values) == 2:
                x, y = np.asarray(values[0], dtype=float), np.asarray(
                    values[1], dtype=float
                )
                y_low = y_high = None
            elif len(values) == 3:
                x = np.asarray(values[0], dtype=float)
                y = np.asarray(values[1], dtype=float)
                err = np.asarray(values[2], dtype=float)
                y_low, y_high = y - err, y + err
            elif len(values) == 4:
                x = np.asarray(values[0], dtype=float)
                y = np.asarray(values[1], dtype=float)
                y_low = np.asarray(values[2], dtype=float)
                y_high = np.asarray(values[3], dtype=float)
            else:
                raise ValueError(
                    f"Formato inválido para '{label}': se esperaban 2, 3 o 4 elementos."
                )
        else:
            y = np.asarray(values, dtype=float)
            x = np.arange(len(y), dtype=float)
            y_low = y_high = None

        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=ls,
            marker=mk,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        if y_low is not None and y_high is not None:
            ax.fill_between(
                x, y_low, y_high, color=color, alpha=band_alpha, linewidth=0
            )

    return _finalize_ax(
        fig,
        ax,
        created,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        grid=grid,
        show_legend=len(data) > 0,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        save_path=save_path,
        dpi=dpi,
    )


def plot_regression(
    data: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    degree: int = 1,
    show_r2: bool = True,
    show_eq: bool = False,
    show_ci: bool = True,
    ci_z: float = 1.96,
    point_alpha: float = 0.72,
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    grid: bool | str = True,
    legend_loc: str = "best",
    palette: str | Sequence[str] | None = None,
    figsize: str | tuple[float, float] = "medium",
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """
    Gráfico de dispersión con regresión lineal o polinomial (puro NumPy) por serie.

    Formato de `data`:
        {"Nombre en paper": (x_array, y_array), ...}

    Incluye cálculo automático de R^2 y banda de confianza aproximada (~95% con z=1.96).
    """
    fig, ax, created = _prepare_ax(ax, figsize)
    colors = get_colors(len(data), palette)

    for idx, (label, (x_raw, y_raw)) in enumerate(data.items()):
        x = np.asarray(x_raw, dtype=float)
        y = np.asarray(y_raw, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        color = colors[idx]
        mk = MARKERS[idx % len(MARKERS)]

        coeffs = np.polyfit(x, y, deg=degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        parts = [label]
        if show_r2:
            parts.append(f"$R^2={r2:.3f}$")
        if show_eq and degree == 1:
            m, b = coeffs
            sign = "+" if b >= 0 else "-"
            parts.append(f"$y={m:.2f}x {sign} {abs(b):.2f}$")
        full_label = " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else ""
        legend_label = f"{label}{full_label}"

        ax.scatter(
            x,
            y,
            color=color,
            marker=mk,
            alpha=point_alpha,
            s=28,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        x_grid = np.linspace(np.min(x), np.max(x), 200)
        y_grid = poly(x_grid)
        ax.plot(
            x_grid,
            y_grid,
            color=color,
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            linewidth=1.7,
            label=legend_label,
            zorder=4,
        )

        if show_ci and len(x) > degree + 2:
            dof = len(x) - (degree + 1)
            s_err = np.sqrt(ss_res / dof)
            x_des = np.vander(x, N=degree + 1)
            x_new = np.vander(x_grid, N=degree + 1)
            cov_inv = np.linalg.pinv(x_des.T @ x_des)
            se_fit = s_err * np.sqrt(
                np.maximum(0.0, np.sum((x_new @ cov_inv) * x_new, axis=1))
            )
            ax.fill_between(
                x_grid,
                y_grid - ci_z * se_fit,
                y_grid + ci_z * se_fit,
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )

    return _finalize_ax(
        fig,
        ax,
        created,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        show_legend=len(data) > 0,
        legend_loc=legend_loc,
        save_path=save_path,
        dpi=dpi,
    )


def plot_scatter(
    data: Mapping[str, tuple[Any, ...]],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    annotate_points: bool = False,
    pareto_direction: tuple[str, str] | None = None,
    point_size: float = 48.0,
    alpha: float = 0.88,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    grid: bool | str = True,
    legend_loc: str = "best",
    legend_ncol: int = 1,
    palette: str | Sequence[str] | None = None,
    figsize: str | tuple[float, float] = "medium",
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """
    Gráfico de dispersión / puntos (ideal para compromisos Pareto: latencia vs F1).

    Formatos aceptados para cada valor de `data`:
        - `(x, y)`: escalares (un modelo = un punto) o arreglos 1D (una familia de puntos).
        - `(x, y, s)`: incluye tamaño del marcador (ej. proporcional a parámetros).

    Parámetro extra útil:
        pareto_direction: tupla opcional como `("min", "max")` para trazar
            automáticamente la frontera de Pareto sobre todos los puntos (ej. minimizar
            latencia en X y maximizar F1 en Y).
    """
    fig, ax, created = _prepare_ax(ax, figsize)
    colors = get_colors(len(data), palette)

    all_x: list[float] = []
    all_y: list[float] = []
    ann_labels: list[str] = []
    ann_xs: list[float] = []
    ann_ys: list[float] = []

    for idx, (label, values) in enumerate(data.items()):
        color = colors[idx]
        mk = MARKERS[idx % len(MARKERS)]

        if len(values) == 2:
            x = np.atleast_1d(np.asarray(values[0], dtype=float))
            y = np.atleast_1d(np.asarray(values[1], dtype=float))
            s: Any = point_size
        elif len(values) == 3:
            x = np.atleast_1d(np.asarray(values[0], dtype=float))
            y = np.atleast_1d(np.asarray(values[1], dtype=float))
            s = np.atleast_1d(np.asarray(values[2], dtype=float))
        else:
            raise ValueError(
                f"Formato inválido para '{label}': use (x, y) o (x, y, size)."
            )

        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

        ax.scatter(
            x,
            y,
            s=s,
            color=color,
            marker=mk,
            alpha=alpha,
            edgecolors="#1A1A1A",
            linewidths=0.6,
            label=label,
            zorder=4,
        )

        if annotate_points and len(x) == 1:
            ann_labels.append(label)
            ann_xs.append(float(x[0]))
            ann_ys.append(float(y[0]))

    if pareto_direction is not None and len(all_x) >= 2:
        dir_x, dir_y = pareto_direction
        pts = np.column_stack([all_x, all_y])
        sx = -1.0 if dir_x == "min" else 1.0
        sy = -1.0 if dir_y == "min" else 1.0
        scored = np.column_stack([pts[:, 0] * sx, pts[:, 1] * sy])

        is_pareto = np.ones(len(pts), dtype=bool)
        for i in range(len(pts)):
            dominated = np.all(scored >= scored[i], axis=1) & np.any(
                scored > scored[i], axis=1
            )
            if np.any(dominated):
                is_pareto[i] = False

        frontier = pts[is_pareto]
        order = np.argsort(frontier[:, 0])
        frontier = frontier[order]
        ax.plot(
            frontier[:, 0],
            frontier[:, 1],
            linestyle="--",
            color="#333333",
            linewidth=1.1,
            alpha=0.75,
            label="Pareto frontier",
            zorder=3,
        )

    show_leg = not annotate_points or pareto_direction is not None
    # Apply scales and limits before placing non-overlapping point annotations
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if annotate_points and ann_labels:
        ax.autoscale_view()
        annotate_points_non_overlapping(
            ax,
            ann_labels,
            ann_xs,
            ann_ys,
            xscale=xscale,
            yscale=yscale,
        )

    return _finalize_ax(
        fig,
        ax,
        created,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        grid=grid,
        show_legend=show_leg,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        save_path=save_path,
        dpi=dpi,
    )


def plot_bars(
    data: Mapping[str, float | Sequence[float]],
    *,
    categories: Sequence[str] | None = None,
    errors: Mapping[str, float | Sequence[float]] | None = None,
    orientation: str = "vertical",
    annotate: bool = True,
    value_fmt: str = "{:.3f}",
    use_hatches: bool = False,
    bar_width: float = 0.72,
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    rotate_xticks: float = 0.0,
    grid: bool | str | None = None,
    legend_loc: str = "best",
    legend_ncol: int = 1,
    palette: str | Sequence[str] | None = None,
    figsize: str | tuple[float, float] = "medium",
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """
    Gráfico de barras simples o agrupadas (vertical u horizontal).

    Dos modos automáticos según `data`:
        1. Barras simples: `{"LR-Dot": 0.80, "MLP-16": 0.78, "MobileNet": 0.76}`
        2. Barras agrupadas: `{"$X_{85}$": [0.71, 0.42], "$X_{94}$": [0.80, 0.58]}`
           junto con `categories=["MVTec-AD Bottle", "KolektorSDD2"]`.
    """
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation debe ser 'vertical' o 'horizontal'.")

    fig, ax, created = _prepare_ax(ax, figsize)
    is_horiz = orientation == "horizontal"

    first_val = next(iter(data.values()))
    is_grouped = (
        isinstance(first_val, (list, tuple, np.ndarray))
        or hasattr(first_val, "__len__")
        and not isinstance(first_val, str)
    )

    if grid is None:
        grid_axis: bool | str = "x" if is_horiz else "y"
    else:
        grid_axis = grid

    if not is_grouped:
        labels = list(data.keys())
        vals = np.asarray([float(v) for v in data.values()], dtype=float)
        errs = (
            np.asarray([float(errors[k]) for k in labels], dtype=float)
            if errors is not None
            else None
        )
        pos = np.arange(len(labels))
        colors = get_colors(len(labels), palette)
        hatches = (
            [HATCHES[i % len(HATCHES)] for i in range(len(labels))]
            if use_hatches
            else None
        )

        if is_horiz:
            bars = ax.barh(
                pos,
                vals,
                height=bar_width,
                xerr=errs,
                color=colors,
                edgecolor="#222222",
                linewidth=0.6,
                hatch=hatches,
            )
            ax.set_yticks(pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
        else:
            bars = ax.bar(
                pos,
                vals,
                width=bar_width,
                yerr=errs,
                color=colors,
                edgecolor="#222222",
                linewidth=0.6,
                hatch=hatches,
            )
            ax.set_xticks(pos)
            ax.set_xticklabels(
                labels,
                rotation=rotate_xticks,
                ha="right" if rotate_xticks else "center",
            )

        if annotate:
            _annotate_bars(ax, bars, errs, value_fmt, is_horiz)

        show_legend = False
    else:
        series_names = list(data.keys())
        n_series = len(series_names)
        n_cats = len(np.asarray(first_val))
        cat_labels = (
            list(categories)
            if categories is not None
            else [f"G{i+1}" for i in range(n_cats)]
        )
        if len(cat_labels) != n_cats:
            raise ValueError(
                "La longitud de `categories` debe coincidir con la longitud de cada serie."
            )

        pos = np.arange(n_cats, dtype=float)
        single_w = bar_width / n_series
        colors = get_colors(n_series, palette)

        for s_idx, s_name in enumerate(series_names):
            vals = np.asarray(data[s_name], dtype=float)
            errs = (
                np.asarray(errors[s_name], dtype=float)
                if (errors and s_name in errors)
                else None
            )
            offset = (s_idx - (n_series - 1) / 2.0) * single_w
            hatch = HATCHES[s_idx % len(HATCHES)] if use_hatches else None

            if is_horiz:
                bars = ax.barh(
                    pos + offset,
                    vals,
                    height=single_w * 0.92,
                    xerr=errs,
                    color=colors[s_idx],
                    edgecolor="#222222",
                    linewidth=0.55,
                    hatch=hatch,
                    label=s_name,
                )
            else:
                bars = ax.bar(
                    pos + offset,
                    vals,
                    width=single_w * 0.92,
                    yerr=errs,
                    color=colors[s_idx],
                    edgecolor="#222222",
                    linewidth=0.55,
                    hatch=hatch,
                    label=s_name,
                )
            if annotate:
                _annotate_bars(ax, bars, errs, value_fmt, is_horiz)

        if is_horiz:
            ax.set_yticks(pos)
            ax.set_yticklabels(cat_labels)
            ax.invert_yaxis()
        else:
            ax.set_xticks(pos)
            ax.set_xticklabels(
                cat_labels,
                rotation=rotate_xticks,
                ha="right" if rotate_xticks else "center",
            )

        show_legend = True

    return _finalize_ax(
        fig,
        ax,
        created,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        xscale=xscale,
        yscale=yscale,
        grid=grid_axis,
        show_legend=show_legend,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        save_path=save_path,
        dpi=dpi,
    )


def _annotate_bars(
    ax: Axes,
    bars: Any,
    errs: np.ndarray | None,
    value_fmt: str,
    is_horiz: bool,
) -> None:
    fs = max(7.0, float(mpl.rcParams["xtick.labelsize"]) - 0.5)
    for i, bar in enumerate(bars):
        err_val = float(errs[i]) if errs is not None else 0.0
        if is_horiz:
            val = bar.get_width()
            x_pos = val + err_val
            y_pos = bar.get_y() + bar.get_height() / 2.0
            ax.annotate(
                value_fmt.format(val),
                xy=(x_pos, y_pos),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=fs,
                color="#222222",
            )
        else:
            val = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2.0
            y_pos = val + err_val
            ax.annotate(
                value_fmt.format(val),
                xy=(x_pos, y_pos),
                xytext=(0, 3),
                textcoords="offset points",
                va="bottom",
                ha="center",
                fontsize=fs,
                color="#222222",
            )


def plot_histogram(
    data: Mapping[str, Sequence[float]],
    *,
    bins: int | Sequence[float] | str = 20,
    orientation: str = "vertical",
    density: bool = False,
    histtype: str = "bar",
    alpha: float = 0.55,
    show_mean: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    grid: bool | str = True,
    legend_loc: str = "best",
    palette: str | Sequence[str] | None = None,
    figsize: str | tuple[float, float] = "medium",
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """
    Histograma científico (vertical u horizontal) para una o varias distribuciones.

    Formato de `data`:
        {"Normal (Good)": arr_good, "Defect": arr_defect}
    """
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation debe ser 'vertical' o 'horizontal'.")

    fig, ax, created = _prepare_ax(ax, figsize)
    colors = get_colors(len(data), palette)

    arrays = {
        k: np.asarray(v, dtype=float)[np.isfinite(np.asarray(v, dtype=float))]
        for k, v in data.items()
    }
    if isinstance(bins, (int, str)) and len(arrays) > 1:
        combined = np.concatenate(list(arrays.values()))
        bin_edges: Any = np.histogram_bin_edges(combined, bins=bins)
    else:
        bin_edges = bins

    for idx, (label, vals) in enumerate(arrays.items()):
        color = colors[idx]
        mean_val = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        lbl = (
            f"{label} ($\\mu={mean_val:.2f}$)"
            if show_mean and np.isfinite(mean_val)
            else label
        )

        ax.hist(
            vals,
            bins=bin_edges,
            density=density,
            orientation=orientation,
            histtype=histtype,
            alpha=alpha,
            color=color,
            edgecolor="#222222",
            linewidth=0.6,
            label=lbl,
        )

        if show_mean and np.isfinite(mean_val):
            if orientation == "vertical":
                ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.3)
            else:
                ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.3)

    if not ylabel and orientation == "vertical":
        ylabel = "Density" if density else "Frequency"
    if not xlabel and orientation == "horizontal":
        xlabel = "Density" if density else "Frequency"

    return _finalize_ax(
        fig,
        ax,
        created,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        show_legend=len(data) > 0,
        legend_loc=legend_loc,
        save_path=save_path,
        dpi=dpi,
    )


def plot_donut(
    data: Mapping[str, float],
    *,
    donut: bool = True,
    center_text: str | None = None,
    pct_threshold: float = 3.5,
    startangle: float = 90.0,
    title: str | None = None,
    legend_loc: str = "center left",
    palette: str | Sequence[str] | None = None,
    figsize: str | tuple[float, float] = "medium",
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """
    Gráfico circular / de dona sobrio para proporciones (ej. participación por escala
    o desglose de latencia por etapa).

    Formato de `data`:
        {"Entropy $8\\times8$": 42.5, "Asymmetry $A_9$": 35.0, "Other scales": 22.5}
    """
    fig, ax, created = _prepare_ax(ax, figsize)
    labels = list(data.keys())
    values = np.asarray([float(v) for v in data.values()], dtype=float)
    colors = get_colors(len(labels), palette)

    def _autopct(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= pct_threshold else ""

    wedgeprops: dict[str, Any] = {"edgecolor": "white", "linewidth": 1.2}
    if donut:
        wedgeprops["width"] = 0.42

    wedges, _, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=startangle,
        autopct=_autopct,
        pctdistance=0.78 if donut else 0.62,
        wedgeprops=wedgeprops,
        counterclock=False,
    )

    for txt in autotexts:
        txt.set_color("white")
        txt.set_fontweight("bold")
        txt.set_fontsize(max(7.5, float(mpl.rcParams["xtick.labelsize"])))

    if donut and center_text:
        ax.text(
            0,
            0,
            center_text,
            ha="center",
            va="center",
            fontsize=mpl.rcParams["axes.labelsize"],
            fontweight="semibold",
            color="#222222",
        )

    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    total = float(np.sum(values))
    legend_labels = [
        f"{lbl} ({v / total * 100:.1f}%)" if total > 0 else lbl
        for lbl, v in zip(labels, values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc=legend_loc,
        bbox_to_anchor=(1.0, 0.5) if "left" in legend_loc else None,
        frameon=True,
    )

    if created and not fig.get_constrained_layout():
        fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)

    return fig, ax


def plot_heatmap(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    cmap: str = "cividis",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    annotate: bool = True,
    value_fmt: str = "{:.2f}",
    cbar_label: str = "",
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    rotate_xticks: float = 0.0,
    figsize: str | tuple[float, float] = "square",
    ax: Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """
    Mapa de calor / mapa de color 2D para explicabilidad espacial (ej. mapas de
    entropía o importancia 2x2, 4x4, 8x8), matrices de confusión o correlaciones.

    Ajusta automáticamente el color del texto anotado (blanco o negro) según la
    luminancia de cada celda para garantizar legibilidad en publicación.
    """
    fig, ax, created = _prepare_ax(ax, figsize)
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("`matrix` debe ser un arreglo 2D.")

    if center is not None:
        max_dev = max(abs(np.nanmin(mat) - center), abs(np.nanmax(mat) - center))
        vmin = center - max_dev if vmin is None else vmin
        vmax = center + max_dev if vmax is None else vmax

    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.7)

    nrows, ncols = mat.shape
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    if xticklabels is not None:
        ax.set_xticklabels(
            xticklabels,
            rotation=rotate_xticks,
            ha="right" if rotate_xticks else "center",
        )
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)

    if annotate:
        norm = im.norm
        cmap_obj = im.cmap
        fs = max(
            6.5,
            float(mpl.rcParams["xtick.labelsize"])
            - (1.0 if max(nrows, ncols) >= 8 else 0.0),
        )
        for r in range(nrows):
            for c in range(ncols):
                val = mat[r, c]
                if not np.isfinite(val):
                    continue
                rgba = cmap_obj(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "#111111" if luminance > 0.52 else "white"
                ax.text(
                    c,
                    r,
                    value_fmt.format(val),
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=fs,
                    fontweight="medium",
                )

    return _finalize_ax(
        fig,
        ax,
        created,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=False,
        show_legend=False,
        save_path=save_path,
        dpi=dpi,
    )
