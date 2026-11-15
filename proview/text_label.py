"""Engrave or emboss a text label on a model surface.

The text is drawn with Qt (any installed font, Arabic, digits, commas and other marks).
The surface under the label is replaced by a fine height grid that follows the original
surface and is pushed in (engrave) or out (emboss) where the letters are; the grid is then
stitched back into the model, so the result stays one closed surface.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .mesh import Mesh

MARGIN = 1.0  # mm of surface kept flat around the letters


class TextError(Exception):
    pass


def _font(family: str | None = None, bold: bool = True):
    from PySide6.QtGui import QFont

    f = QFont(family or "Arial")
    f.setBold(bold)
    f.setPixelSize(100)  # scaled to the wanted height afterwards
    return f


def _text_path(text: str, family: str | None = None):
    """QPainterPath of the text, 100 px font, y down."""
    from PySide6.QtGui import QPainterPath

    path = QPainterPath()
    for i, line in enumerate(text.split("\n")):
        path.addText(0, i * 115, _font(family), line)
    return path


def text_size(text: str, height: float, family: str | None = None) -> tuple[float, float]:
    """(width, height) in mm of the inked text when its font size is ``height`` mm."""
    br = _text_path(text, family).boundingRect()
    s = height / 100.0
    return br.width() * s, br.height() * s


def text_outlines(text: str, height: float, family: str | None = None) -> list[np.ndarray]:
    """Letter outlines as closed 2D polylines (mm, y up), centred on (0, 0)."""
    path = _text_path(text, family)
    br = path.boundingRect()
    s = height / 100.0
    cx, cy = br.center().x(), br.center().y()
    out = []
    for poly in path.toSubpathPolygons():
        pts = np.array([(p.x(), p.y()) for p in poly], float)
        if len(pts) >= 2:
            out.append(np.stack([(pts[:, 0] - cx) * s, -(pts[:, 1] - cy) * s], 1))
    return out


def text_coverage(text: str, height: float, h: float, margin: float = MARGIN, family: str | None = None):
    """Anti-aliased ink coverage (rows top->bottom, values 0..1) sampled every ``h`` mm on a
    grid centred on the text, plus the grid's (width, height) in mm."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QTransform

    path = _text_path(text, family)
    br = path.boundingRect()
    if br.width() <= 0 or br.height() <= 0:
        raise TextError("The text is empty.")
    s = height / 100.0  # mm per font pixel
    W = br.width() * s + 2 * margin
    H = br.height() * s + 2 * margin
    nu = int(np.ceil(W / h)) + 1
    nv = int(np.ceil(H / h)) + 1
    if nu * nv > 6_000_000:
        raise TextError("The text is too large for this detail. Use a shorter text or a smaller size.")
    img = QImage(nu, nv, QImage.Format_Grayscale8)
    img.fill(0)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)
    # grid vertex (i, j) at pixel centre (i + 0.5, j + 0.5); font px -> mm -> grid px
    t = QTransform()
    t.translate(0.5, 0.5)
    t.scale(s / h, s / h)
    t.translate(-br.center().x() + W / 2 / s, -br.center().y() + H / 2 / s)
    painter.setTransform(t)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.drawPath(path)
    painter.end()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=img.bytesPerLine() * nv).reshape(nv, img.bytesPerLine())[:, :nu]
    return arr.astype(np.float64) / 255.0, (nu - 1) * h, (nv - 1) * h


def text_frame(normal, up_hint):
    """(right, up, normal) for a label facing ``normal`` with its top towards ``up_hint``."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    up = np.asarray(up_hint, float)
    up = up - n * (up @ n)
    if np.linalg.norm(up) < 1e-6:
        up = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(up) < 1e-6:
            up = np.cross(n, [0.0, 1.0, 0.0])
    up = up / np.linalg.norm(up)
    right = np.cross(up, n)
    return right, up, n


def _inside_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point in polygon test (2D)."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    a = poly[None, :, :]
    b = np.roll(poly, -1, 0)[None, :, :]
    ay, by = a[..., 1], b[..., 1]
    cond = (ay > y) != (by > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = a[..., 0] + (y - ay) * (b[..., 0] - a[..., 0]) / (by - ay)
    return (cond & (x < xc)).sum(1) % 2 == 1


class _NoFit(Exception):
    def __init__(self, lost: float):
        super().__init__(f"{lost:.0%} of the text is off the surface")
        self.lost = lost


def _layout(mesh, fn, p, right, up, n, side, text, height, depth, family, h, tw, th, hit_face=None):
    """Lettering grid for the label centred at ``p`` -> (grid mesh, replaced face ids).
    Raises _NoFit when part of the text would be off the surface."""
    from .model_builder import boundary_loops
    from .section import SegmentSet

    W, H = tw + 2 * MARGIN, th + 2 * MARGIN
    v = mesh.vertices
    rel = v - p
    u_, v_, w_ = rel @ right, rel @ up, rel @ n
    reach = max(3.0, abs(depth) * 4)
    in_box = (np.abs(u_) <= W / 2 + 2 * h) & (np.abs(v_) <= H / 2 + 2 * h) & (np.abs(w_) <= reach)
    cand = in_box[mesh.faces].any(1) & (side * (fn @ n) > 0.15)
    if cand.sum() < 4:
        raise _NoFit(1.0)
    # the connected piece of surface under the label
    sub = mesh.keep_faces(cand)
    count, labels = sub.connected_components()
    cand_idx = np.flatnonzero(cand)
    if hit_face is not None and cand[hit_face]:
        centre_face = int(np.searchsorted(cand_idx, hit_face))
    else:
        centre_face = int(np.argmin(np.linalg.norm(sub.face_centroids() - p, axis=1)))
    sub_idx = cand_idx[labels == labels[centre_face]]
    patch = mesh.keep_faces(np.isin(np.arange(mesh.n_faces), sub_idx))
    loops = boundary_loops(patch.faces)
    if not loops:
        raise TextError("The surface under the text is closed. Click on the outside of the model.")
    pr = patch.vertices - p
    pu, pv, pw = pr @ right, pr @ up, pr @ n
    outer = max(loops, key=len)
    poly = np.stack([pu[outer], pv[outer]], 1)

    # the grid must cover every removed triangle (large ones, e.g. on a base wall, reach out of the box)
    need = max(-poly[:, 0].min() - tw / 2, poly[:, 0].max() - tw / 2,
               -poly[:, 1].min() - th / 2, poly[:, 1].max() - th / 2)
    margin = float(np.clip(need + h, MARGIN, 30.0))
    cov, W, H = text_coverage(text, height, h, margin=margin, family=family)
    nv, nu = cov.shape
    gu = -W / 2 + np.arange(nu) * h
    gv = H / 2 - np.arange(nv) * h  # row 0 = top of the text
    GU, GV = np.meshgrid(gu, gv)
    pts2 = np.stack([GU.ravel(), GV.ravel()], 1)
    inside = _inside_polygon(pts2, poly)
    segs = np.stack([poly, np.roll(poly, -1, 0)], 1)
    dist, _ = SegmentSet(segs).nearest(pts2)
    inside &= dist > 0.75 * h
    w = np.full(len(pts2), np.nan)
    if inside.any():
        w[inside] = LinearNDInterpolator(np.stack([pu, pv], 1), pw)(pts2[inside])
    valid = np.isfinite(w)
    ink = cov.ravel()
    if ink.sum() == 0:
        raise TextError("The text is empty.")
    lost = float(ink[~valid].sum() / ink.sum())
    if lost > 0.02:
        raise _NoFit(lost)
    w = np.where(valid, w + depth * ink, 0.0)
    grid3 = p + GU.ravel()[:, None] * right + GV.ravel()[:, None] * up + w[:, None] * n
    ok = valid.reshape(nv, nu)
    idx = np.arange(nv * nu).reshape(nv, nu)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]  # a b / c d, row 0 at top
    cell = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    # counter-clockwise seen from +n: (u right, v up), rows go down
    faces = np.concatenate([np.stack([c[cell], d[cell], a[cell]], 1), np.stack([d[cell], b[cell], a[cell]], 1)])
    grid = Mesh(grid3, faces).keep_faces(np.ones(len(faces), bool))
    cnt, lab = grid.connected_components()
    if cnt > 1:
        grid = grid.keep_faces(lab == np.bincount(lab).argmax())
    return grid, sub_idx


def apply_text(mesh: Mesh, point, normal, up_hint, text: str, height: float = 4.0, depth: float = -0.5,
               family: str | None = None, progress=lambda t: None, face: int | None = None) -> Mesh:
    """Engrave (depth < 0) or emboss (depth > 0) ``text`` on ``mesh`` at ``point``.

    ``normal`` is the surface normal there towards the viewer, ``up_hint`` the direction of the top
    of the letters (usually the screen's up direction) and ``face`` the clicked triangle. All in the
    mesh's own coordinates. When the label does not fit at the point (e.g. close to the edge of a
    base) it is moved onto the surface, as little as needed."""
    from .fuse import fuse_pair

    text = text.strip()
    if not text:
        raise TextError("Type a text first.")
    p = np.asarray(point, float)
    right, up, n = text_frame(normal, up_hint)
    h = float(np.clip(height / 40.0, 0.04, 0.12))  # grid step: ~40 samples over the letter height
    tw, th = text_size(text, height, family)
    if tw <= 0 or th <= 0:
        raise TextError("The text is empty.")
    fn = mesh.face_normals()
    # side of the surface the label goes on (scans can be stored inside-out): the clicked triangle's
    if face is None:
        face = int(np.argmin(np.linalg.norm(mesh.face_centroids() - p, axis=1)))
    side = 1.0 if fn[face] @ n >= 0 else -1.0

    progress("Finding room for the text")
    # first the clicked point, then positions moved along the surface, nearest first
    dv_max, du_max = th / 2 + MARGIN, tw / 2
    steps = []
    for dv in np.arange(0.0, dv_max + 1e-6, 0.5):
        for du in np.arange(0.0, du_max + 1e-6, 1.0):
            for sv in ((1, -1) if dv else (1,)):
                for su in ((1, -1) if du else (1,)):
                    steps.append((float(np.hypot(du, dv)), su * du, sv * dv))
    steps.sort()
    result = None
    best_lost = 1.0
    for k, (_dist, du, dv) in enumerate(steps[:40]):
        q = p + du * right + dv * up
        try:
            result = _layout(mesh, fn, q, right, up, n, side, text, height, depth, family, h, tw, th,
                             hit_face=face)
            break
        except _NoFit as exc:
            best_lost = min(best_lost, exc.lost)
            continue
    if result is None:
        raise TextError("The text does not fit on this part of the surface. Use a smaller text size "
                        "or click on a larger, flatter area.")
    grid, sub_idx = result
    if mesh.colors is not None:
        from scipy.spatial import cKDTree

        _, near = cKDTree(mesh.vertices).query(grid.vertices)
        grid = Mesh(grid.vertices, grid.faces, mesh.colors[near])

    progress("Joining the lettering to the model")
    base = mesh.remove_faces(np.isin(np.arange(mesh.n_faces), sub_idx))
    out, joined = fuse_pair(base, grid, gap=1.0, trim=False)
    if not joined:
        raise TextError("Could not join the lettering to the model here. Try another place.")
    return out
