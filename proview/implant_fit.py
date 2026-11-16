"""Find the implant from a scanned scan body (or an implant hole) and suggest an abutment.

1. Around the clicked point the scan body is measured as a cylinder: axis, radius, top.
2. Library scan bodies of a similar radius are registered onto the scan (several turns about
   the axis, then ICP); each gets a score from how well the scanned surface lies on it and how
   much of its top is covered by the scan.
3. The best match gives the implant: the analog of the library entry is in the same frame as
   its scan body, its top is the implant platform.
4. Angulation to the occlusal axis, gingival height around the implant and the room to the
   opposing jaw lead to a suggestion (straight / angled, angle, GH, abutment height).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .alignment import FixedTarget, icp
from .implant_lib import LibEntry, file_hash, load_geometry
from .mesh import Mesh


# ---------------------------------------------------------------------- cylinder geometry
def _perp_basis(a):
    a = a / np.linalg.norm(a)
    t = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    t /= np.linalg.norm(t)
    return t, np.cross(a, t)


def _circle_fit(xy, w=None):
    """Algebraic circle fit -> (centre, radius, rms residual)."""
    if len(xy) < 3:
        return np.zeros(2), 0.0, np.inf
    w = np.ones(len(xy)) if w is None else w
    A = np.column_stack([xy, np.ones(len(xy))]) * w[:, None]
    b = (xy ** 2).sum(1) * w
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(2), 0.0, np.inf
    c = sol[:2] / 2
    r = float(np.sqrt(max(sol[2] + c @ c, 0.0)))
    res = np.abs(np.linalg.norm(xy - c, axis=1) - r)
    return c, r, float(np.sqrt(np.average(res ** 2, weights=w)))


@dataclass
class Cylinder:
    axis: np.ndarray  # unit, towards the free end (top)
    centre: np.ndarray  # a point on the axis
    radius: float
    top: float  # height of the top along the axis (axis @ point)
    fit: float  # rms of the side fit (mm)
    flipped: bool = False  # the surface's normals point inwards (mesh stored inside-out)

    def height(self, pts):
        return np.asarray(pts) @ self.axis

    def radial(self, pts):
        rel = np.asarray(pts) - self.centre
        return np.linalg.norm(rel - np.outer(rel @ self.axis, self.axis), axis=1)


def fit_cylinder(points, normals, areas, hint_up=None, candidates=None, inward=False,
                 max_tilt: float | None = None) -> Cylinder | None:
    """Cylinder through a patch of surface (points with normals): the side faces are the ones
    perpendicular to the axis. ``inward``: a hole (normals point to the axis)."""
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    w = np.asarray(areas, float)
    M = (normals * w[:, None]).T @ normals
    _, vecs = np.linalg.eigh(M)
    axes = [vecs[:, k] for k in range(3)]
    if hint_up is not None:
        axes.append(np.asarray(hint_up, float))
    if candidates:
        axes += [np.asarray(c, float) for c in candidates]
    best = None
    hint = None if hint_up is None else np.asarray(hint_up, float) / np.linalg.norm(hint_up)
    for a in axes:
        a = a / np.linalg.norm(a)
        for _ in range(3):  # refine: the axis is perpendicular to the side normals
            side = np.abs(normals @ a) < 0.3
            if side.sum() < 10:
                break
            ns = normals[side]
            Ms = (ns * w[side][:, None]).T @ ns
            a_new = np.linalg.eigh(Ms)[1][:, 0]
            a = a_new if a_new @ a >= 0 else -a_new
        side = np.abs(normals @ a) < 0.3
        if side.sum() < 10:
            continue
        if hint is not None and max_tilt is not None and abs(a @ hint) < np.cos(np.radians(max_tilt)):
            continue  # a scan body / implant is never far from the occlusal direction (e.g. the ridge)
        t, u = _perp_basis(a)
        xy = np.column_stack([points[side] @ t, points[side] @ u])
        c, r, res = _circle_fit(xy, w[side])
        if not (0.8 < r < 6.0):
            continue
        # normals must point away from (or, for a hole, towards) the axis
        centre = c[0] * t + c[1] * u
        rel = points[side] - centre
        radial_dir = rel - np.outer(rel @ a, a)
        radial_dir /= np.linalg.norm(radial_dir, axis=1, keepdims=True) + 1e-12
        agree = float(np.average(np.einsum("ij,ij->i", radial_dir, normals[side]), weights=w[side]))
        if abs(agree) < 0.5:  # not a round wall
            continue
        # scans are often stored inside-out: then a scan body's normals point to the axis
        flipped = (agree < 0) != inward
        area = float(w[side][np.abs(np.linalg.norm(xy - c, axis=1) - r) < 0.15].sum())
        score = area * np.exp(-res / 0.08)
        if best is None or score > best[0]:
            best = (score, a, centre, r, res, flipped)
    if best is None:
        return None
    _, a, centre, r, res, flipped = best
    cyl = Cylinder(a, centre, r, 0.0, res, flipped)
    rad = cyl.radial(points)
    near = rad <= r + 0.3
    far = rad > r + 1.0
    h = cyl.height(points)
    # the free end points away from the surrounding tissue (or, for a hole, the opening)
    flip = False
    if far.any() and near.any():
        flip = h[near].mean() < h[far].mean() if not inward else h[near].mean() > h[far].mean()
    elif hint_up is not None:
        flip = a @ hint_up < 0
    if flip:
        cyl.axis = -a
        h = -h
    cyl.top = float(h[near].max()) if near.any() else float(h.max())
    return cyl


# ---------------------------------------------------------------------- shape descriptor
PROFILE_SLICES = np.arange(0.4, 7.0, 0.8)  # depths below the top (mm)


def profile(points, normals, cyl: Cylinder) -> np.ndarray:
    """Outer and inner radius of the side wall in slices below the top: [r90..., r10...]
    (NaN where the slice has no surface, e.g. under the gingiva)."""
    side = np.abs(normals @ cyl.axis) < 0.5
    pts = points[side]
    depth = cyl.top - cyl.height(pts)
    rad = cyl.radial(pts)
    hi, lo = [], []
    for d in PROFILE_SLICES:
        sel = np.abs(depth - d) < 0.4
        if sel.sum() >= 6:
            hi.append(np.percentile(rad[sel], 90))
            lo.append(np.percentile(rad[sel], 10))
        else:
            hi.append(np.nan)
            lo.append(np.nan)
    return np.array(hi + lo)


def profile_distance(a, b) -> float:
    """Mean difference of two profiles; the slices may be shifted by up to 2 (the scanned top is
    not always where the library's top is)."""
    n = len(PROFILE_SLICES)
    best = np.inf
    for sh in range(-2, 3):
        ia = np.arange(max(0, -sh), min(n, n - sh))
        ib = ia + sh
        pa = np.concatenate([a[ia], a[n + ia]])
        pb = np.concatenate([b[ib], b[n + ib]])
        ok = np.isfinite(pa) & np.isfinite(pb)
        if ok.sum() < 3:
            continue
        best = min(best, float(np.mean(np.abs(pa[ok] - pb[ok]))) + 0.02 * abs(sh))
    return best


# ---------------------------------------------------------------------- library shapes
@dataclass
class MarkerShape:
    """A library scan body ready for matching (in its library frame)."""

    path: str
    hash: str
    points: np.ndarray
    normals: np.ndarray
    cyl: Cylinder
    target: FixedTarget
    top_pts: np.ndarray  # points of the top 2 mm (to check that the scan covers them)
    profile: np.ndarray | None = None


_SHAPES: dict = {}


def _marker_cylinder(mesh, axis_hint):
    """(cylinder of the scan body's upper part, centroids, normals, areas) of a library scan body.

    The axis comes from the shape itself (the library's AxisOcclusal is not always given); the top
    is the end farther from the library origin, where the implant sits."""
    fn = mesh.face_normals()
    tri = mesh.vertices[mesh.faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    cen = tri.mean(1)
    hint = np.asarray(axis_hint, float)
    hint /= np.linalg.norm(hint) or 1.0
    whole = fit_cylinder(cen, fn, areas, candidates=[hint])
    a = whole.axis if whole is not None else hint
    hv = mesh.vertices @ a
    lo, hi = float(hv.min()), float(hv.max())
    if abs(abs(hi) - abs(lo)) > 0.5:
        if abs(lo) > abs(hi):
            a = -a
    elif a @ hint < 0:
        a = -a
    h = cen @ a
    upper = h > h.max() - 6.0
    cyl = fit_cylinder(cen[upper], fn[upper], areas[upper], candidates=[a])
    if cyl is None:
        return None, cen, fn, areas
    if cyl.axis @ a < 0:
        cyl.axis = -cyl.axis
    cyl.top = float((mesh.vertices @ cyl.axis).max())
    return cyl, cen, fn, areas


def marker_shape(entry: LibEntry) -> MarkerShape | None:
    key = entry.marker
    if key in _SHAPES:
        return _SHAPES[key]
    mesh = load_geometry(entry.marker)
    shape = None
    if mesh is not None:
        cyl, cen, fn, areas = _marker_cylinder(mesh, entry.axis)
        if cyl is not None:
            pts = np.vstack([mesh.vertices, cen])
            nrm = np.vstack([mesh.vertex_normals(), fn])
            if cyl.flipped:
                nrm, fn = -nrm, -fn
            hp = pts @ cyl.axis
            shape = MarkerShape(entry.marker, file_hash(entry.marker), pts, nrm, cyl,
                                FixedTarget(pts, nrm), pts[hp > cyl.top - 2.0], profile(cen, fn, cyl))
    _SHAPES[key] = shape
    return shape


def describe_marker(path: str, axis) -> dict | None:
    """Small description of a library scan body for fast pre-selection (no ICP data kept)."""
    mesh = load_geometry(path)
    if mesh is None:
        return None
    cyl, cen, fn, areas = _marker_cylinder(mesh, axis)
    if cyl is None:
        return None
    prof = profile(cen, -fn if cyl.flipped else fn, cyl)
    return {"hash": file_hash(path), "radius": cyl.radius, "profile": [None if not np.isfinite(x) else float(x)
                                                                        for x in prof]}


def library_index(entries: list[LibEntry], cache_file: str | None = None, progress=lambda t: None) -> dict:
    """path -> description for every scan body; kept in a cache file between sessions."""
    import json
    import os

    cache = {}
    if cache_file and os.path.isfile(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as fh:
                cache = json.load(fh)
        except (OSError, ValueError):
            cache = {}
    out, changed = {}, False
    todo = sorted({e.marker: e for e in entries}.values(), key=lambda e: e.marker)
    for k, e in enumerate(todo):
        try:
            st = os.stat(e.marker)
        except OSError:
            continue
        key = f"{e.marker}|{st.st_size}|{int(st.st_mtime)}|v2"
        if key not in cache:
            if k % 10 == 0:
                progress(f"Indexing the scan bodies {k + 1}/{len(todo)} (only the first time)")
            cache[key] = describe_marker(e.marker, e.axis)
            changed = True
        d = cache[key]
        if d is not None:
            out[e.marker] = d
    if changed and cache_file:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
            os.replace(cache_file + ".tmp", cache_file)
        except OSError:
            pass
    return out


def platform_in_library(entry: LibEntry, shape: MarkerShape, type_index: int = 0):
    """(point, axis) of the implant platform in the library frame: top of the implant analog
    (all files of an entry share one frame); without an analog, the frame origin on the axis."""
    a = shape.cyl.axis
    analog = None
    if entry.types and 0 <= type_index < len(entry.types) and entry.types[type_index].analog:
        analog = load_geometry(entry.types[type_index].analog)
    if analog is not None:
        hv = analog.vertices @ a
        h_top = float(hv.max())
        # centre of the analog's top (round), not of the scan body (often flattened on one side)
        ring = analog.vertices[hv > h_top - 0.3]
        c = ring.mean(0) if len(ring) >= 3 else shape.cyl.centre
    else:
        c = np.zeros(3)  # libraries put the implant on the frame origin
        h_top = 0.0
    return c + a * (h_top - c @ a), a


# ---------------------------------------------------------------------- scan side
@dataclass
class ScanSite:
    points: np.ndarray  # surface samples around the click (world)
    normals: np.ndarray
    cyl: Cylinder | None
    body_tris: np.ndarray | None = None  # triangles of the scan body (world), for display


def search_cylinder(cen, fn, areas, click, r_range=(1.0, 3.6), inward=False):
    """Cylinder through the clicked point found by trying every axis direction (perpendicular to
    the surface at the click) and every radius: the one most surface lies on, with the surface
    facing away from (or, for a hole, towards) the axis. Independent of how the triangles are
    joined, of noise, of the occlusal direction and of the normals' sign.
    Returns (axis, centre, radius, inlier mask) or None."""
    d = np.linalg.norm(cen - click, axis=1)
    local = d < 0.8
    if local.sum() < 3:
        local = np.argsort(d)[:8]
    n0 = (fn[local] * areas[local][:, None]).sum(0)
    if np.linalg.norm(n0) < 1e-9:
        return None
    n0 /= np.linalg.norm(n0)
    e1, e2 = _perp_basis(n0)
    near = np.flatnonzero(d < 9.0)
    if len(near) > 5000:  # enough samples; keeps the search fast on dense scans
        near = np.random.default_rng(0).choice(near, 5000, replace=False)
    P, N, A = cen[near], fn[near], areas[near]
    def search(thetas, radii, P, N, A):
        best = None
        for th in thetas:
            a = np.cos(th) * e1 + np.sin(th) * e2
            wall = np.abs(N @ a) < 0.3
            if wall.sum() < 10:
                continue
            Pw, Nw, Aw = P[wall], N[wall], A[wall]
            relc = Pw - click
            relc -= np.outer(relc @ a, a)  # perpendicular to the axis
            Nw_perp = Nw - np.outer(Nw @ a, a)
            Nw_perp /= np.linalg.norm(Nw_perp, axis=1, keepdims=True) + 1e-12
            for sgn in (1.0, -1.0):  # centre on either side of the click (normals may be flipped)
                off = sgn * n0  # perpendicular to every tried axis
                q = relc[None, :, :] + radii[:, None, None] * off[None, None, :]
                rad = np.linalg.norm(q, axis=2)
                on = np.abs(rad - radii[:, None]) < 0.08
                rdir = q / (rad[..., None] + 1e-12)
                facing = np.abs(np.einsum("rij,ij->ri", rdir, Nw_perp)) > 0.85
                score = ((on & facing) * Aw[None, :]).sum(1)
                k = int(np.argmax(score))
                if best is None or score[k] > best[0]:
                    best = (float(score[k]), a, click - radii[k] * off, float(radii[k]), th)
        return best

    # coarse (6 degrees, 0.1 mm, fewer points), then fine around the best
    sub = np.random.default_rng(2).choice(len(P), min(len(P), 1800), replace=False)
    coarse = search(np.radians(np.arange(0, 180, 6.0)), np.arange(r_range[0], r_range[1] + 1e-9, 0.1),
                    P[sub], N[sub], A[sub])
    if coarse is None:
        return None
    th0, r0 = coarse[4], coarse[3]
    best = search(th0 + np.radians(np.arange(-6, 6.01, 1.0)),
                  np.arange(max(r_range[0], r0 - 0.15), min(r_range[1], r0 + 0.15) + 1e-9, 0.02), P, N, A)
    if best is None or best[0] < 2.0:  # less than ~2 mm2 of wall on the cylinder
        return None
    score, a, c, r, _th = best
    rel = cen - c
    rad = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)
    inl = (np.abs(rad - r) < 0.15) & (np.abs(fn @ a) < 0.35) & (d < 12.0)
    return a, c, r, inl, score


def _smooth_region(faces, fn, seed, max_angle=20.0):
    """Faces reachable from ``seed`` without crossing an edge sharper than ``max_angle`` degrees
    (the scan body wall ends at its top edge and where it meets the gingiva)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    m = len(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    fid = np.tile(np.arange(m), 3)
    key = np.sort(e, 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, fs = key[order], fid[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    a, b = fs[:-1][same], fs[1:][same]
    smooth = np.einsum("ij,ij->i", fn[a], fn[b]) > np.cos(np.radians(max_angle))
    a, b = a[smooth], b[smooth]
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
    _, lab = connected_components(g, directed=False)
    return lab == lab[seed]


def scan_site(world_vertices, faces, click, radius=7.0, hint_up=None, inward=False) -> ScanSite:
    """Surface around the click and the cylinder (scan body or hole) found in it.

    The wall under the click is found first (smooth surface grown from the clicked triangle up to
    sharp edges), so the round alveolar ridge or a neighbouring tooth is not taken for the scan
    body; without a usable wall the whole neighbourhood is searched, also in a smaller and a larger
    area. Axes far from the occlusal direction are not accepted."""
    v = np.asarray(world_vertices, float)
    click = np.asarray(click, float)
    best = None
    for rad in (radius, 4.5, 10.0):
        near_v = np.linalg.norm(v - click, axis=1) < rad
        f = faces[near_v[faces].any(1)]
        if len(f) <= 20:
            continue
        tri = v[f]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        areas = np.linalg.norm(cr, axis=1) / 2
        fn = cr / (2 * areas[:, None] + 1e-12)
        cen = tri.mean(1)
        cyl = None
        body_tris = None
        found = None if inward else search_cylinder(cen, fn, areas, click)
        if not inward:
            # a click on a flat side gives a poor cylinder: also try points around it where the surface
            # turns, and keep the cylinder with the most wall on it
            dc = np.linalg.norm(cen - click, axis=1)
            n_click = fn[np.argmin(dc)]
            ring = np.flatnonzero((dc > 1.0) & (dc < 2.2) & (np.abs(fn @ n_click) < 0.94))
            for idx in ring[np.linspace(0, len(ring) - 1, min(4, len(ring))).astype(int)] if len(ring) else []:
                alt = search_cylinder(cen, fn, areas, cen[idx])
                if alt is not None and (found is None or alt[4] > found[4] * 1.15):
                    found = alt
        if found is not None:
            a, c, r, inl, _score = found
            refined = fit_cylinder(cen[inl], fn[inl], areas[inl], candidates=[a], inward=inward) \
                if inl.sum() >= 20 else None
            if refined is not None and refined.axis @ a < 0:
                refined.axis = -refined.axis
            if refined is not None and abs(refined.radius - r) < 0.3 and refined.axis @ a > 0.97:
                cyl = refined
            else:
                rel = cen[inl] - c
                agree = np.einsum("ij,ij->i", rel - np.outer(rel @ a, a), fn[inl])
                cyl = Cylinder(a, c, r, 0.0, 0.0, bool(np.sum(agree) < 0))
            # which end is free: away from the surrounding tissue, else towards the occlusal side
            rad_all = cyl.radial(cen)
            h_all = cen @ cyl.axis
            ring = rad_all <= cyl.radius + 0.3
            far = (rad_all > cyl.radius + 1.0) & (np.linalg.norm(cen - click, axis=1) < 8)
            if ring.any() and far.any():
                if h_all[ring].mean() < h_all[far].mean():
                    cyl.axis = -cyl.axis
            elif hint_up is not None and cyl.axis @ hint_up < 0:
                cyl.axis = -cyl.axis
            h_all = cen @ cyl.axis
            cyl.top = float(h_all[ring].max()) if ring.any() else float(h_all.max())
            body = (rad_all <= cyl.radius + 0.2) & (h_all >= cyl.top - 12) & \
                (np.linalg.norm(cen - click, axis=1) < 13)
            body_tris = tri[body]
        seed = int(np.argmin(np.linalg.norm(cen - click, axis=1)))
        wall = _smooth_region(f, fn, seed) if cyl is None else np.zeros(len(f), bool)
        if cyl is None and wall.sum() >= 30:
            cyl = fit_cylinder(cen[wall], fn[wall], areas[wall], hint_up=hint_up, inward=inward, max_tilt=45)
            if cyl is not None:
                # top (or bottom of a hole) from all the surface around the axis
                rad_all = cyl.radial(cen)
                h_all = cen @ cyl.axis
                ring = rad_all <= cyl.radius + 0.3
                if ring.any():
                    cyl.top = float(h_all[ring].max())
        if cyl is None:
            cyl = fit_cylinder(cen, fn, areas, hint_up=hint_up, inward=inward, max_tilt=45)
        if cyl is None:
            continue
        # the cylinder must pass near the click (not a neighbouring tooth)
        if abs(cyl.radial(click[None])[0] - cyl.radius) > 1.0 and not inward:
            continue
        if cyl.flipped:
            fn = -fn  # work with outward normals from here on
        best = ScanSite(cen, fn, cyl, body_tris if found is not None else None)
        break
    if best is None:
        try:
            from .diagnostics import log

            log(f"implant: no cylinder at {np.round(click, 2)} (faces near: "
                f"{int((np.linalg.norm(v - click, axis=1) < radius).sum())} vertices)")
        except Exception:  # noqa: BLE001
            pass
        return ScanSite(np.zeros((0, 3)), np.zeros((0, 3)), None)
    return best


# ---------------------------------------------------------------------- matching
@dataclass
class Match:
    entries: list  # LibEntry objects that share this scan body geometry
    shape: MarkerShape
    T: np.ndarray  # library frame -> world
    rms: float
    inliers: float  # share of scanned scan-body surface lying on the library shape
    coverage: float  # share of the library top covered by the scan
    score: float = 0.0

    @property
    def entry(self) -> LibEntry:
        return self.entries[0]


def _rot_to(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-9:
        t, _ = _perp_basis(a)
        return 2 * np.outer(t, t) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _about(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _zone(site: ScanSite):
    sc = site.cyl
    zone = (sc.radial(site.points) <= sc.radius + 0.6) & (sc.height(site.points) >= sc.top - 9.0)
    src, src_n = site.points[zone], site.normals[zone]
    if len(src) > 2500:
        idx = np.random.default_rng(0).choice(len(src), 2500, replace=False)
        src, src_n = src[idx], src_n[idx]
    return src, src_n


def _quick_starts(shape: MarkerShape, site: ScanSite, src):
    """Start poses (world -> library) scored without ICP: 12 turns x axial shifts of -3..+3 mm.
    Returns (best quick score, starts sorted best first); lower is better."""
    sc, mc = site.cyl, shape.cyl
    quick = src if len(src) <= 1200 else src[np.random.default_rng(1).choice(len(src), 1200, replace=False)]
    R0 = _rot_to(mc.axis, sc.axis)
    top_m = mc.centre + mc.axis * (mc.top - mc.centre @ mc.axis)
    top_s = sc.centre + sc.axis * (sc.top - sc.centre @ sc.axis)
    starts = []
    for k in range(12):
        R = _about(sc.axis, 2 * np.pi * k / 12) @ R0
        for off in np.arange(-3.0, 3.01, 0.75):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = top_s + sc.axis * off - R @ top_m
            W = np.linalg.inv(T)
            d, _ = shape.target.tree.query(quick @ W[:3, :3].T + W[:3, 3])
            starts.append((float(np.mean(np.minimum(d, 0.3))), W, k))
    starts.sort(key=lambda x: x[0])
    # keep the best start of each turn: the ICP tries then cover different turns, not one turn's shifts
    seen, diverse = set(), []
    for st in starts:
        if st[2] not in seen:
            seen.add(st[2])
            diverse.append(st)
    return diverse[0][0], diverse


def _refine(shape: MarkerShape, src, src_n, starts, tries=5):
    """ICP from the best starts -> (T library->world, rms, inliers, coverage)."""
    best = None
    for _, W, _k in starts[:tries]:
        res = icp(src, src_n, shape.target, init=W, iterations=20, trim=0.85, max_distance=0.8)
        res = icp(src, src_n, shape.target, init=res.matrix, iterations=20, trim=0.9, max_distance=0.4)
        Wr = res.matrix
        d, _ = shape.target.tree.query(src @ Wr[:3, :3].T + Wr[:3, 3])
        on = float((d < 0.1).mean())  # share of the scanned surface lying on the library shape
        if best is None or on > best[0]:
            best = (on, res)
    res = icp(src, src_n, shape.target, init=best[1].matrix, iterations=30, trim=0.9, max_distance=0.3)
    W = res.matrix  # world -> library
    p = src @ W[:3, :3].T + W[:3, 3]
    d, _ = shape.target.tree.query(p)
    inl = float((d < 0.1).mean())
    rms = float(np.sqrt(np.mean(d[d < 0.2] ** 2))) if (d < 0.2).any() else float(np.sqrt(np.mean(d ** 2)))
    dc, _ = cKDTree(p).query(shape.top_pts)
    cov = float((dc < 0.15).mean())
    return np.linalg.inv(W), rms, inl, cov


def _match_one(shape: MarkerShape, site: ScanSite) -> tuple[np.ndarray, float, float, float]:
    """Best pose of the library shape on the scan: (T library->world, rms, inliers, coverage)."""
    src, src_n = _zone(site)
    if len(src) < 30:
        return np.eye(4), np.inf, 0.0, 0.0
    _, starts = _quick_starts(shape, site, src)
    return _refine(shape, src, src_n, starts)


def _coarse_score(shape: MarkerShape, site: ScanSite, pts) -> float:
    """Fast shape check without ICP: 12 turns x 5 axial shifts, one KD query for all of them.
    Mean distance (capped at 0.3 mm) of the best start; lower is better."""
    sc, mc = site.cyl, shape.cyl
    R0 = _rot_to(mc.axis, sc.axis)
    top_m = mc.centre + mc.axis * (mc.top - mc.centre @ mc.axis)
    top_s = sc.centre + sc.axis * (sc.top - sc.centre @ sc.axis)
    Ws = []
    for k in range(12):
        R = _about(sc.axis, 2 * np.pi * k / 12) @ R0
        Rt = R.T
        for off in (-2.0, -1.0, 0.0, 1.0, 2.0):
            t = top_s + sc.axis * off - R @ top_m
            W = np.eye(4)
            W[:3, :3] = Rt
            W[:3, 3] = -Rt @ t
            Ws.append(W)
    Ws = np.array(Ws)
    P = np.einsum("sij,nj->sni", Ws[:, :3, :3], pts) + Ws[:, None, :3, 3]
    d, _ = shape.target.tree.query(P.reshape(-1, 3))
    return float(np.minimum(d.reshape(len(Ws), -1), 0.3).mean(1).min())


# ---------------------------------------------------------------------- learning
def memory_file() -> str:
    import os

    from .diagnostics import log_path

    return os.path.join(os.path.dirname(log_path()), "implant_memory.json")


def load_memory() -> list:
    import json
    import os

    path = memory_file()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def remember(site: ScanSite, match: "Match", entry: LibEntry) -> None:
    """The user confirmed this scan body: next time a similar one is always compared in detail and
    systems used more often come first when two fit about equally."""
    import json
    import os

    sc = site.cyl
    near = sc.radial(site.points) <= sc.radius + 1.0
    prof = profile(site.points[near], site.normals[near], sc)
    mem = load_memory()
    for m in mem:
        if m.get("hash") == match.shape.hash and abs(m.get("radius", 0) - sc.radius) < 0.1:
            m["count"] = int(m.get("count", 1)) + 1
            break
    else:
        mem.append({"hash": match.shape.hash, "marker": entry.marker, "library": entry.library,
                    "manufacturer": entry.manufacturer, "radius": float(sc.radius),
                    "profile": [None if not np.isfinite(x) else float(x) for x in prof], "count": 1})
    path = memory_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(mem, fh, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def manufacturer_of(entry: LibEntry) -> str:
    name = entry.manufacturer or entry.library.replace("-", "_").split("_")[0]
    return name.strip() or entry.library


def identify(site: ScanSite, entries: list[LibEntry], progress=lambda t: None, max_candidates=250,
             index: dict | None = None, memory: list | None = None, detailed: int = 8) -> list[Match]:
    """Library scan bodies ranked by how well they fit the scanned one (best first).

    Stage 1: every library scan body of about the same diameter gets a fast shape check (no ICP).
    Stage 2: the most promising ones, and the ones the user confirmed before for a similar scan
    body, are registered precisely."""
    if site.cyl is None:
        return []
    sc = site.cyl
    near = sc.radial(site.points) <= sc.radius + 1.0  # the scan body, not the gingiva around it
    site_prof = profile(site.points[near], site.normals[near], sc)
    if index is None:
        index = library_index(entries, progress=progress)
    memory = load_memory() if memory is None else memory
    groups: dict = {}
    for e in entries:
        d = index.get(e.marker)
        if d is None:
            continue
        g = groups.setdefault(d["hash"], [d, []])
        g[1].append(e)
    # confirmed before for a similar scan body
    forced, used = set(), {}
    for m in memory:
        used[m.get("manufacturer") or m.get("library", "")] = used.get(m.get("manufacturer") or m.get("library", ""),
                                                                       0) + int(m.get("count", 1))
        prof = np.array([np.nan if x is None else x for x in m.get("profile", [])], float)
        if abs(m.get("radius", 0) - sc.radius) < 0.2 and len(prof) == len(site_prof) and \
                profile_distance(site_prof, prof) < 0.3:
            forced.add(m.get("hash"))
    ranked = []
    for h, (d, es) in groups.items():
        dr = abs(d["radius"] - sc.radius)
        if dr > 0.6 and h not in forced:
            continue
        prof = np.array([np.nan if x is None else x for x in d["profile"]], float)
        dist = profile_distance(site_prof, prof)
        ranked.append((0 if h in forced else 1, (dist if np.isfinite(dist) else 1.0) + dr, h, es))
    ranked.sort(key=lambda r: (r[0], r[1]))
    shortlist = ranked[:max_candidates]
    src, src_n = _zone(site)
    if len(src) < 30:
        return []
    pts = src if len(src) <= 400 else src[np.random.default_rng(3).choice(len(src), 400, replace=False)]
    coarse = []
    for k, (f, _dist, h, es) in enumerate(shortlist):
        if k % 5 == 0:
            progress(f"Checking the library {k + 1}/{len(shortlist)}")
        shape = marker_shape(es[0])
        if shape is None:
            continue
        coarse.append((f, _coarse_score(shape, site, pts), shape, es))
    coarse.sort(key=lambda x: (x[0], x[1]))
    chosen = [c for c in coarse if c[0] == 0] + [c for c in coarse if c[0] == 1][:detailed]
    out = []
    for k, (_f, _q, shape, es) in enumerate(chosen):
        progress(f"Fitting {k + 1}/{len(chosen)}: {es[0].title}")
        _, starts = _quick_starts(shape, site, src)
        T, rms, inl, cov = _refine(shape, src, src_n, starts)
        if not np.isfinite(rms):
            continue
        score = inl * (0.75 + 0.25 * cov) * np.exp(-max(rms - 0.03, 0) / 0.08)
        out.append(Match(es, shape, T, rms, inl, cov, float(score)))
    # systems used more often first when two fit about equally
    out.sort(key=lambda m: -(m.score + 0.01 * np.log1p(used.get(m.entry.manufacturer or m.entry.library, 0))))
    return out


def confidence(matches: list[Match]) -> str:
    """'sure', 'likely' or 'unsure' for the best match."""
    if not matches:
        return "unsure"
    best = matches[0].score
    second = matches[1].score if len(matches) > 1 else 0.0
    if best > 0.75 and best - second > 0.08:
        return "sure"
    if best > 0.55 and best - second > 0.03:
        return "likely"
    return "unsure"


# ---------------------------------------------------------------------- measurements / suggestion
@dataclass
class Rules:
    straight_max: float = 10.0  # up to this angulation a straight abutment is fine
    angled: tuple = (15.0, 25.0)  # angled abutments that are available
    margin_depth: float = 0.5  # crown margin below the gingiva (mm)
    crown_min: float = 1.5  # room the crown needs above the abutment (mm)


@dataclass
class Measures:
    platform: np.ndarray
    axis: np.ndarray  # implant axis (towards occlusal)
    occlusal: np.ndarray | None
    angulation: float | None
    gh_min: float | None
    gh_max: float | None
    gh_mean: float | None
    space: float | None  # platform to the opposing jaw along the implant axis
    approximate: bool = False
    notes: list = field(default_factory=list)


def ray_hit(origin, direction, vertices, faces, max_dist=40.0):
    """Distance along a ray to the first triangle (or None)."""
    v = np.asarray(vertices, float)
    rel = v - origin
    along = rel @ direction
    perp = np.linalg.norm(rel - np.outer(along, direction), axis=1)
    keep = (perp[faces] < 6.0).any(1) & (along[faces] > -1).any(1) & (along[faces] < max_dist).any(1)
    tri = v[faces[keep]]
    if not len(tri):
        return None
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    t0 = origin - tri[:, 0]
    u = np.einsum("ij,ij->i", t0, p) * inv
    q = np.cross(t0, e1)
    w = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 0.05)
    return float(t[hit].min()) if hit.any() else None


def measure(site: ScanSite, platform, axis, occlusal=None, opposing=(), radius=None, approximate=False) -> Measures:
    """Angulation, gingival height around the implant and room to the opposing jaw."""
    axis = axis / np.linalg.norm(axis)
    m = Measures(np.asarray(platform, float), axis, None, None, None, None, None, None, approximate)
    if occlusal is not None:
        occ = np.asarray(occlusal, float)
        occ = occ / np.linalg.norm(occ)
        if occ @ axis < 0:
            occ = -occ
        m.occlusal = occ
        m.angulation = float(np.degrees(np.arccos(np.clip(occ @ axis, -1, 1))))
    # gingiva: the surface just around the scan body / implant, per sector around the axis
    r = radius if radius is not None else (site.cyl.radius if site.cyl else 2.0)
    rel = site.points - platform
    h = rel @ axis
    lat = rel - np.outer(h, axis)
    dist = np.linalg.norm(lat, axis=1)
    ring = (dist > r + 0.25) & (dist < r + 1.8) & (h > -2.0) & (h < 10.0) & (np.abs(site.normals @ axis) > 0.2)
    if ring.sum() >= 8:
        t, u = _perp_basis(axis)
        ang = np.arctan2(lat[ring] @ u, lat[ring] @ t)
        sector = ((ang + np.pi) / (2 * np.pi) * 12).astype(int) % 12
        levels = [float(np.median(h[ring][sector == s])) for s in range(12) if (sector == s).sum() >= 2]
        if levels:
            m.gh_min, m.gh_max, m.gh_mean = min(levels), max(levels), float(np.mean(levels))
            if len(levels) < 8:
                m.notes.append("The gingiva around the implant is only partly scanned.")
    else:
        m.notes.append("No gingiva found around the implant.")
    for verts, faces in opposing:
        d = ray_hit(np.asarray(platform, float), axis, verts, faces)
        if d is not None and (m.space is None or d < m.space):
            m.space = d
    return m


@dataclass
class Suggestion:
    kind: str  # "Straight" / "Angled 15°" / ...
    gh: float | None
    height: float | None
    height_choice: str | None  # matching ti-base of the library
    lines: list


def suggest(meas: Measures, rules: Rules, entry: LibEntry | None = None, type_index: int = 0) -> Suggestion:
    lines = []
    a = meas.angulation
    if a is None:
        kind = "Straight (angulation unknown)"
        lines.append("Occlusal direction unknown: angulation not measured.")
    elif a <= rules.straight_max:
        kind = "Straight"
        lines.append(f"Angulation {a:.1f}° ≤ {rules.straight_max:.0f}°: a straight abutment fits.")
    elif entry is not None and entry.asc_max and a <= entry.asc_max:
        kind = f"Ti-base with angled screw channel ({a:.0f}°)"
        lines.append(f"Angulation {a:.1f}°: within the library's angled screw channel range "
                     f"(up to {entry.asc_max:.0f}°).")
    else:
        best = min(rules.angled, key=lambda x: abs(x - a)) if rules.angled else None
        kind = f"Angled {best:.0f}°" if best is not None else "Angled"
        lines.append(f"Angulation {a:.1f}°: an angled abutment ({best:.0f}°) corrects the axis.")
        if best is not None and abs(best - a) > 7:
            lines.append(f"The nearest available angle differs by {abs(best - a):.0f}°: check the design.")
    gh = None
    if meas.gh_min is not None:
        gh = max(0.5, round((meas.gh_min - rules.margin_depth) * 2) / 2)
        lines.append(f"Gingiva {meas.gh_min:.1f}–{meas.gh_max:.1f} mm above the platform: GH ≈ {gh:.1f} mm "
                     f"(margin {rules.margin_depth:.1f} mm below the lowest gingiva).")
    height = None
    choice = None
    if meas.space is not None and gh is not None:
        avail = meas.space - gh - rules.crown_min
        height = max(avail, 0.0)
        lines.append(f"Room to the opposing jaw {meas.space:.1f} mm: abutment above the margin ≤ "
                     f"{height:.1f} mm (crown needs {rules.crown_min:.1f} mm).")
        if entry is not None and entry.types:
            subs = [s for s in entry.types[min(type_index, len(entry.types) - 1)].subtypes if s.height]
            fit = [s for s in subs if s.height <= height]
            if fit:
                pick = max(fit, key=lambda s: s.height)
                choice = pick.name
                lines.append(f"Library ti-base: {pick.name} (height {pick.height:g} mm).")
            elif subs:
                pick = min(subs, key=lambda s: s.height)
                choice = pick.name
                lines.append(f"Even the lowest library ti-base ({pick.name}, {pick.height:g} mm) is higher "
                             "than the room: shorten it or check the bite.")
    elif meas.space is None:
        lines.append("No opposing jaw found along the implant axis: height not suggested.")
    if meas.approximate:
        lines.append("Position from the implant hole: approximate, check it.")
    lines += meas.notes
    return Suggestion(kind, gh, height, choice, lines)
