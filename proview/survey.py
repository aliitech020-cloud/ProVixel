"""Path of insertion / undercut analysis (dental survey).

For an insertion axis ``d`` (unit vector pointing out of the model, towards where the
restoration comes from) a surface point is in an undercut when something of the model
lies above it along ``d``. Its undercut depth is the horizontal distance (perpendicular
to ``d``) to the nearest place where a vertical analysing rod could reach its height.
"""
from __future__ import annotations

import numpy as np

from .model_builder import _column_crossings, rotation_to_z

# depth limits (mm) and colours, like undercut gauges 0.25 / 0.50 / 0.75
LEVELS = [0.25, 0.50, 0.75]
COLORS = np.array([[255, 221, 60], [255, 150, 40], [235, 60, 50], [150, 45, 165]], dtype=np.uint8)


def undercut_depth(vertices: np.ndarray, faces: np.ndarray, direction, cell: float | None = None,
                   levels: int = 64) -> np.ndarray:
    """Per-face undercut depth in mm (0 = no undercut) for the insertion axis ``direction``."""
    from scipy import ndimage

    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    R = rotation_to_z(d)
    v = vertices @ R.T
    if len(faces) == 0:
        return np.zeros(0)
    lo, hi = v.min(0), v.max(0)
    diag = float(np.linalg.norm(hi[:2] - lo[:2])) or 1.0
    h = float(cell) if cell else float(np.clip(diag / 600.0, 0.08, 0.3))
    pad = 3
    x0 = lo[0] - pad * h + 0.0137 * h
    y0 = lo[1] - pad * h + 0.0291 * h
    nx = int(np.ceil((hi[0] - x0) / h)) + pad
    ny = int(np.ceil((hi[1] - y0) / h)) + pad

    # top height of the model in every column
    cols, z = _column_crossings(v, faces, x0, y0, h, nx, ny)
    hmap = np.full(nx * ny, -np.inf)
    if len(cols):
        np.maximum.at(hmap, cols, z)

    tri = v[faces]
    cen = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0] = 1.0
    nz = n[:, 2] / ln
    ii = np.clip(np.floor((cen[:, 0] - x0) / h).astype(np.int64), 0, nx - 1)
    jj = np.clip(np.floor((cen[:, 1] - y0) / h).astype(np.int64), 0, ny - 1)
    top = hmap[jj * nx + ii]
    # tolerance grows with the slope: a steep visible face changes height a lot inside one cell
    slope = np.sqrt(np.clip(1 - nz ** 2, 0, 1)) / np.maximum(np.abs(nz), 0.05)
    tol = 0.03 + h * np.minimum(slope, 12.0)
    shadowed = cen[:, 2] < top - tol
    under = shadowed | (nz < -0.02)

    depth = np.zeros(len(faces))
    idx = np.nonzero(under)[0]
    if len(idx) == 0:
        return depth
    hm2 = hmap.reshape(ny, nx)
    zc = cen[idx, 2]
    edges = np.linspace(zc.min(), zc.max() + 1e-9, levels + 1)
    which = np.clip(np.searchsorted(edges, zc, side="right") - 1, 0, levels - 1)
    for k in range(levels):
        sel = which == k
        if not sel.any():
            continue
        covered = hm2 >= edges[k]  # material above this height: the rod cannot stand here
        dist = ndimage.distance_transform_edt(covered) * h
        depth[idx[sel]] = dist[jj[idx[sel]], ii[idx[sel]]]
    depth[idx] = np.maximum(depth[idx] - 0.5 * h, 0.01)
    return depth


def depth_colors(depth: np.ndarray, base_rgb) -> np.ndarray:
    """uint8 (m, 3) face colours: base colour where there is no undercut, gauge colours elsewhere."""
    out = np.empty((len(depth), 3), dtype=np.uint8)
    out[:] = np.asarray(base_rgb, dtype=np.uint8)
    cls = np.searchsorted(LEVELS, depth, side="right")
    under = depth > 0
    out[under] = COLORS[cls[under]]
    return out
