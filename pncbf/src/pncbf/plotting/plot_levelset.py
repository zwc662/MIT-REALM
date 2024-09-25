import contourpy
import matplotlib.pyplot as plt
import numpy as np
import shapely
from loguru import logger
from matplotlib.colors import to_rgba

from pncbf.plotting.poly_to_patch import poly_to_patch
from pncbf.utils.jax_types import BBFloat


def plot_levelset(
    bb_Xs: BBFloat,
    bb_Ys: BBFloat,
    bb_V: BBFloat,
    color,
    inner_alpha: float,
    ax: plt.Axes,
    lw: float = 1.0,
    ls: str = "-",
    merge: bool = True,
    zorder: int = 4,
    min_area_ratio: float = 0.01,
    **kwargs,
):
    contour_gen = contourpy.contour_generator(bb_Xs, bb_Ys, bb_V)
    # One array for each line. (n_pts, 2)
    paths: list[np.ndarray] = contour_gen.lines(0.0)
    if len(paths) > 1:
        if not merge:
            assert len(paths) == 1, "Should only have 1 path"

        # Remove all paths smaller than n points.
        areas = [shapely.Polygon(path).area for path in paths]
        max_area = max(areas)

        filtered_paths = paths
        if min_area_ratio is not None:
            filtered_paths = []
            for path, area in zip(paths, areas):
                print("Ratio: {:.3f}".format(area / max_area))
                if area / max_area >= min_area_ratio:
                    filtered_paths.append(path)

        # Merge the verts.
        logger.warning("Found {} -> {} paths! Merging...".format(len(paths), len(filtered_paths)))
        verts = np.concatenate([path for path in filtered_paths])
    else:
        verts = paths[0]

    nom_shape = shapely.Polygon(verts)
    facecolor = to_rgba(color, inner_alpha)
    nom_patch = poly_to_patch(
        nom_shape, facecolor=facecolor, edgecolor=color, lw=lw, linestyle=ls, **kwargs, zorder=zorder
    )
    ax.add_patch(nom_patch)


def plot_levelset_multi(
    bb_Xs: BBFloat,
    bb_Ys: BBFloat,
    bb_V: BBFloat,
    color,
    inner_alpha: float,
    ax: plt.Axes,
    lw: float = 1.0,
    ls: str = "-",
    zorder: int = 4,
    min_area_ratio: float = 0.04,
    **kwargs,
):
    contour_gen = contourpy.contour_generator(bb_Xs, bb_Ys, bb_V)

    # One array for each line. (n_pts, 2)
    paths: list[np.ndarray] = contour_gen.lines(0.0)
    polys = [shapely.Polygon(path) for path in paths]
    # Start with the largest polygon.
    poly = max(polys, key=lambda p: p.area)
    max_poly_area = poly.area
    polys = [p for p in polys if p != poly]
    included_polys = [poly]
    # For each other poly, if it is contained, then add as a hole. Otherwise, add it as a union if it is large enough.
    for p2 in polys:
        inside = False
        for ii, p1 in enumerate(included_polys):
            if p1.contains(p2):
                included_polys[ii] = p1.difference(p2)
                inside = True

        if inside:
            continue

        frac = p2.area / max_poly_area
        logger.info("frac: {:.2f}".format(frac))
        if frac > min_area_ratio:
            included_polys.append(p2)

    facecolor = to_rgba(color, inner_alpha)
    for poly in included_polys:
        nom_patch = poly_to_patch(
            poly, facecolor=facecolor, edgecolor=color, lw=lw, linestyle=ls, **kwargs, zorder=zorder
        )
        ax.add_patch(nom_patch)
