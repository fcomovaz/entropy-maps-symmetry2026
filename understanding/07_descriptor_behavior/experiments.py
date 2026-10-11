"""Fifteen controlled experiments, grouped here to keep small sweeps readable."""
from dataclasses import dataclass, field
import math
from PIL import Image
from compare_descriptors import extract, entropy_details, partition_boxes, schema
from synthetic_scenes import textured_scene, checkerboard, noise_texture, equal_histogram_pair, cellwise_collision
from perturbations import intensity, noise, blur, insert_defect, translate

EXPERIMENTS = ("global_intensity", "contrast", "noise", "blur", "defect_size", "defect_strength",
               "defect_position", "grid_boundary", "translation", "transformations", "symmetric_defects",
               "entropy_collision", "same_histogram", "resolution", "bins")
FILENAMES = {"grid_boundary": "grid_boundaries", "transformations": "transformations",
             "symmetric_defects": "symmetric_vs_asymmetric", "entropy_collision": "same_entropy_different_structure"}


@dataclass
class Case:
    """One explicit base/modified comparison; series identifies a response curve."""
    series: str
    parameter: float
    base: Image.Image
    image: Image.Image
    parameters: dict = field(default_factory=dict)
    boxes: list[tuple] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    bins: int = 16
    base_bins: int = 16
    consecutive: bool = False


def histogram_checks(a: Image.Image, b: Image.Image, bins: int) -> dict:
    """Measure exact 256-bin equality globally and in every canonical cell."""
    equality = {str(g): [a.crop(q).histogram() == b.crop(q).histogram()
                         for q in partition_boxes(*a.size, g)] for g in schema.LEVELS}
    return {"same_global_histogram": a.histogram() == b.histogram(),
            "equal_cell_histograms": equality,
            "base_global_entropy_bits": entropy_details(a, bins)["entropy_bits"],
            "modified_global_entropy_bits": entropy_details(b, bins)["entropy_bits"],
            "pixel_difference_count": sum(x != y for x, y in zip(a.tobytes(), b.tobytes(), strict=True))}


def cases(name: str, size: int, seed: int, bins: int):
    """Yield deterministic images and all parameters; size >=128 avoids tiny fixtures."""
    scene, regular = textured_scene(size), checkerboard(size)
    unit = lambda px: max(1, round(px*size/256))
    target = noise_texture(unit(64), seed)
    x, y = unit(68), unit(76)

    def case(series, parameter, base, image, **kwargs):
        return Case(series, parameter, base, image, bins=bins, base_bins=bins, **kwargs)

    if name == "global_intensity":
        for value in (-40, -20, -10, 0, 10, 20, 40):
            raw = [v+value for v in scene.tobytes()]
            yield case("offset", value, scene, intensity(scene, offset=value),
                       parameters={"offset": value}, extras={"clipped_pixels": sum(v < 0 or v > 255 for v in raw)})
    elif name == "contrast":
        for value in (.5, .75, 1, 1.25, 1.5, 2):
            raw = [127.5+value*(v-127.5) for v in scene.tobytes()]
            yield case("contrast", value, scene, intensity(scene, factor=value),
                       parameters={"factor": value, "pivot": 127.5},
                       extras={"clipped_pixels": sum(v < 0 or v > 255 for v in raw)})
    elif name == "noise":
        for value in (0, 2, 5, 10, 20, 30):
            yield case("noise_sigma", value, scene, noise(scene, value, seed), parameters={"sigma": value})
    elif name == "blur":
        for value in (0, 1, 2, 4, 8):
            yield case("blur_radius", value, scene, blur(scene, value), parameters={"radius": value})
    elif name in ("defect_size", "defect_strength"):
        values = (4, 8, 16, 24, 32, 48, 64) if name == "defect_size" else (0, .1, .2, .4, .6, .8, 1)
        for value in values:
            side = unit(value if name == "defect_size" else 32)
            strength = 1 if name == "defect_size" else value
            tile = target.crop((0, 0, side, side))
            modified, box = insert_defect(regular, tile, x, y, strength)
            yield case(name, side if name == "defect_size" else value, regular, modified,
                       parameters={"side": side, "strength": strength, "x": x, "y": y,
                                   "target": "fixed seeded noise top-left crop"}, boxes=[box])
    elif name == "defect_position":
        side, step, y = unit(16), unit(4), unit(76)
        tile = target.crop((0, 0, side, side))
        positions = sorted(set(range(0, size-side+1, step)) | {size-side})
        for x in positions:
            modified, box = insert_defect(regular, tile, x, y)
            yield case("horizontal", x, regular, modified, boxes=[box], consecutive=True,
                       parameters={"x": x, "y": y, "side": side, "step": step})
    elif name == "grid_boundary":
        # First internal boundary isolates g=8 from g=4 and g=2, then g=4 from g=2.
        side, y = unit(8), unit(76)
        tile = target.crop((0, 0, side, side))
        for g in (2, 4, 8):
            boundary = partition_boxes(size, size, g)[0][2]
            for offset in range(-8, 9, 2):
                center = boundary + unit(abs(offset))*(1 if offset > 0 else -1) if offset else boundary
                x = center-side//2
                modified, box = insert_defect(regular, tile, x, y)
                yield case(f"grid_{g}", center-boundary, regular, modified, boxes=[box], consecutive=True,
                           parameters={"grid": g, "boundary": boundary, "center_x": center,
                                       "x": x, "y": y, "side": side, "offset": center-boundary,
                                       "all_x_boundaries": [q[0] for q in partition_boxes(size, size, g)[:g]]+[size]})
    elif name == "translation":
        for axis in ("x", "y"):
            for shift in (0, 1, 2, 4, 8, 16):
                dx, dy = (shift, 0) if axis == "x" else (0, shift)
                yield case(axis, shift, scene, translate(scene, dx, dy),
                           parameters={"dx": dx, "dy": dy, "border": "zero padding, overflow discarded"})
    elif name == "transformations":
        for label, transform in (("identity", None), ("lr", Image.Transpose.FLIP_LEFT_RIGHT),
                                 ("tb", Image.Transpose.FLIP_TOP_BOTTOM), ("rot180", Image.Transpose.ROTATE_180)):
            modified = scene.copy() if transform is None else scene.transpose(transform)
            a, b = extract(scene, bins), extract(modified, bins)
            predicted = []
            start = 0
            for g in schema.LEVELS:
                for r in range(g):
                    for c in range(g):
                        rr = g-1-r if label in ("tb", "rot180") else r
                        cc = g-1-c if label in ("lr", "rot180") else c
                        predicted.append(a[start+rr*g+cc])
                start += g*g
            yield case(label, 0, scene, modified, parameters={"transform": label},
                       extras={"permuted_entropy_max_error": max(abs(v-w) for v, w in zip(predicted, b[:85])),
                               "entropy_sum_change": math.fsum(b[:85])-math.fsum(a[:85]),
                               "entropy_l2_norm_change": math.sqrt(math.fsum(v*v for v in b[:85]))-math.sqrt(math.fsum(v*v for v in a[:85])),
                               "partition_reflection_compatible": size % 8 == 0})
    elif name == "symmetric_defects":
        # Pixel-exact LR and TB base, including odd sizes.
        regular = Image.frombytes("L", (size, size), bytes(80+96*((min(xx,size-1-xx)//2+min(yy,size-1-yy)//2)%2)
                                  for yy in range(size) for xx in range(size)))
        side, x, y = unit(16), unit(48), unit(64)
        tile = target.crop((0, 0, side, side))
        first, firstbox = insert_defect(regular, tile, x, y)
        for label, second_x in (("mirrored_pair", size-x-side), ("two_left_equal_area", x+2*side), ("single_left", None)):
            modified, boxes = first, [firstbox]
            if second_x is not None:
                modified, box = insert_defect(first, tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT), second_x, y)
                boxes = [firstbox, box]
            yield case(label, len(boxes)*side*side, regular, modified, boxes=boxes,
                       parameters={"case": label, "side": side, "replacement_area": len(boxes)*side*side,
                                   "symmetry_tested": "LR", "second_tile": "mirrored first tile"},
                       extras={"pixel_lr_symmetric": modified.tobytes() == modified.transpose(Image.Transpose.FLIP_LEFT_RIGHT).tobytes()})
    elif name == "entropy_collision":
        a, b = cellwise_collision(size)
        checks = histogram_checks(a, b, bins)
        assert all(all(v) for v in checks["equal_cell_histograms"].values())
        yield case("cellwise_exact_collision", 0, a, b, extras=checks,
                   parameters={"construction": "alternating columns vs sorted pixels independently inside each 8x8-grid cell"})
    elif name == "same_histogram":
        # Isolate the patch itself, then compare identical-position embeddings.
        side = unit(32)
        a, b = equal_histogram_pair(side, side)
        assert a.histogram() == b.histogram()
        yield case("patch_alone", 0, a, b, extras=histogram_checks(a, b, bins),
                   parameters={"side": side, "construction": "same exact pixel multiset"})
        for label, px, py in (("aligned_r2c2", size//4, size//4),
                              ("aligned_r3c5", 5*size//8, 3*size//8),
                              ("cross_boundary", size//4-side//2, size//4-side//2)):
            first, box = insert_defect(regular, a, px, py)
            second, _ = insert_defect(regular, b, px, py)
            yield case(label, 0, first, second, boxes=[box], extras=histogram_checks(first, second, bins),
                       parameters={"x": px, "y": py, "side": side})
    elif name == "resolution":
        for method in ("native", "resize_nearest", "resize_bilinear"):
            for resolution in (128, 256, 512):
                modified = (textured_scene(resolution) if method == "native" else
                            scene.resize((resolution, resolution), Image.Resampling.NEAREST if method == "resize_nearest" else Image.Resampling.BILINEAR))
                yield case(method, resolution, scene, modified,
                           parameters={"method": method, "reference_resolution": size, "resolution": resolution})
    elif name == "bins":
        collision_a, collision_b = cellwise_collision(size)
        damaged, box = insert_defect(regular, target.crop((0, 0, unit(16), unit(16))), x, y)
        fixtures = {"scene": scene, "regular": regular, "noise": noise_texture(size, seed), "defect": damaged}
        for b in (8, 16, 32, 64, 128, 256):
            for label, image in fixtures.items():
                yield Case(f"stability_{label}", b, image, image, bins=b, base_bins=bins,
                           parameters={"reference_bins": bins, "bins": b, "fixture": label},
                           extras={"max_entropy_bits": math.log2(b), "global_entropy_bits": entropy_details(image, b)["entropy_bits"]})
            for label, a, image in (("defect_discrimination", regular, damaged), ("collision_discrimination", collision_a, collision_b)):
                yield Case(label, b, a, image, bins=b, base_bins=b, parameters={"bins": b, "comparison": "same-bin pair"})
    else:
        raise ValueError(name)
