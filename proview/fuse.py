"""Fuse overlapping scans into one connected surface.

Two scans that were aligned on each other (e.g. a jaw scan and a scan-body scan) overlap
and leave a small gap where one ends and the other begins. Fusing:

1. removes the part of the larger scan that the smaller one covers (the smaller scan is
   the detail that was scanned on purpose, so it wins),
2. joins the facing open edges of both scans with a strip of triangles (a "zipper"),
3. closes the small notches left where the joined edges end.

No vertex of either scan is moved, so the scanned detail is kept exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh, merge_meshes
from .model_builder import boundary_loops
from .repair import fill_hole, remove_dangling_faces

TRIM = 0.3  # mm: surface of the larger scan this close to the smaller one is always removed


def _area(mesh: Mesh) -> float:
    tri = mesh.vertices[mesh.faces]
    return float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)


def _drop_small_pieces(mesh: Mesh, min_share=0.01, min_faces=30) -> Mesh:
    count, labels = mesh.connected_components()
    if count <= 1:
        return mesh
    sizes = np.bincount(labels, minlength=count)
    keep = sizes >= max(min_faces, min_share * sizes.max())
    keep[np.argmax(sizes)] = True
    return mesh.keep_faces(keep[labels])


def _has_edge(faces: np.ndarray, a: int, b: int) -> bool:
    return bool(np.any(((faces[:, 0] == a) & (faces[:, 1] == b)) | ((faces[:, 1] == a) & (faces[:, 2] == b)) |
                       ((faces[:, 2] == a) & (faces[:, 0] == b))))


def _monotone(q: np.ndarray, n: int) -> np.ndarray:
    """Positions along a loop of n points: unwrapped across the loop's start, never going back."""
    q = np.asarray(q, float).copy()
    d = np.diff(q)
    d[d < -n / 2] += n
    d[d > n / 2] -= n
    q = q[0] + np.concatenate([[0.0], np.cumsum(d)])
    return np.maximum.accumulate(q)


def _zipper(P: np.ndarray, Q: np.ndarray, pv: np.ndarray, qv: np.ndarray, qpar=None) -> list[tuple[int, int, int]]:
    """Triangle strip between two polylines that run the same way (vertex ids P, Q; points pv, qv).

    P must be an edge loop as stored by ``boundary_loops`` of its surface (a->b in its faces)
    and Q the reverse of the other surface's loop; then the strip is oriented like both."""
    faces = []
    n, m = len(P), len(Q)

    def frac(pts):  # position along the polyline, 0 at the start, 1 at the end
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        c = np.concatenate([[0.0], np.cumsum(seg)])
        return c / c[-1] if c[-1] > 0 else np.linspace(0, 1, len(pts))

    if qpar is not None:  # where each Q point lies along P (index units): exact pairing
        fp = np.arange(n, dtype=float)
        fq = np.asarray(qpar, float)
    else:
        fp, fq = frac(pv), frac(qv)
    i = j = 0
    while i < n - 1 or j < m - 1:
        # advance along the edge whose next point comes first (both edges run side by side,
        # so the share of their length walked so far matches; robust to different point spacing)
        if i == n - 1:
            adv_p = False
        elif j == m - 1:
            adv_p = True
        else:
            adv_p = fp[i + 1] <= fq[j + 1]
        if adv_p:
            faces.append((P[i + 1], P[i], Q[j]))
            i += 1
        else:
            faces.append((Q[j], Q[j + 1], P[i]))
            j += 1
    return faces


def _runs(mask: np.ndarray, closed: bool) -> list[tuple[int, int]]:
    """Index ranges [start, end] (inclusive, may wrap) where mask is True along a loop."""
    n = len(mask)
    if mask.all():
        return [(0, n - 1)] if not closed else [(-1, -1)]
    if not mask.any():
        return []
    start = int(np.argmin(mask))  # begin at a False so runs do not wrap past the start
    out = []
    k = 0
    while k < n:
        idx = (start + k) % n
        if mask[idx]:
            s = idx
            length = 0
            while k < n and mask[(start + k) % n]:
                k += 1
                length += 1
            out.append((s, (s + length - 1) % n))
        else:
            k += 1
    return out


def fuse_pair(base: Mesh, insert: Mesh, gap: float = 1.0, progress=lambda text: None,
              region=None, trim: bool = True) -> tuple[Mesh, int]:
    """Fuse ``insert`` into ``base`` (both in the same space). Returns (mesh, joined edge length in
    vertices; 0 means the scans did not meet and were only combined)."""
    if base.n_faces == 0 or insert.n_faces == 0:
        return merge_meshes([base, insert]), 0
    bv = insert.vertices
    # dense samples of the insert surface, and how far each is from the insert's open edge
    samples = np.vstack([bv, insert.face_centroids()])
    b_loops = boundary_loops(insert.faces)
    b_edge = np.concatenate([np.asarray(lp) for lp in b_loops]) if b_loops else np.zeros(0, int)
    tree = cKDTree(samples)
    if len(b_edge):
        edge_dist = cKDTree(bv[b_edge]).query(samples)[0]
    else:
        edge_dist = np.full(len(samples), np.inf)

    if trim:
        progress("Removing the covered surface")
        d, idx = tree.query(base.vertices, distance_upper_bound=gap + 1.0)
        near = np.isfinite(d)
        covered = np.zeros(base.n_vertices, bool)
        covered[near] = (d[near] < TRIM) | ((d[near] < gap) & (edge_dist[idx[near]] > d[near] + TRIM))
        if not covered.any() and not near.any():
            return merge_meshes([base, insert]), 0
        trimmed = base.remove_faces(covered[base.faces].any(1))
        trimmed = _drop_small_pieces(remove_dangling_faces(trimmed)) if trimmed.n_faces else trimmed
        if trimmed.n_faces == 0:
            return insert.copy(), 0
    else:
        trimmed = base
    progress("Joining the edges")
    ins = insert
    joined, extra_faces = _join(trimmed, ins, gap, region)
    if joined == -1:  # the insert is stored the other way round: turn it and try again
        ins = Mesh(insert.vertices, insert.faces[:, ::-1], insert.colors)
        joined, extra_faces = _join(trimmed, ins, gap, region)
        if joined == -1:
            joined, extra_faces = 0, []
    out = merge_meshes([trimmed, ins])
    if not extra_faces:
        return out, 0
    faces = np.vstack([out.faces, np.array(extra_faces, dtype=np.int64)])
    out = Mesh(out.vertices, faces, out.colors)

    progress("Closing the seam")
    out = _close_seam_notches(out, trimmed.n_vertices, gap)
    return out, joined


def _dist_to_polyline(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest segment of an open polyline (3D)."""
    if len(line) < 2:
        return np.linalg.norm(pts - line[:1], axis=1) if len(line) else np.full(len(pts), np.inf)
    a, b = line[:-1], line[1:]
    tree = cKDTree((a + b) / 2)
    k = min(8, len(a))
    _, cand = tree.query(pts, k=k)
    cand = np.asarray(cand).reshape(len(pts), k)
    a0, ab = a[cand], b[cand] - a[cand]
    den = np.einsum("nkj,nkj->nk", ab, ab)
    den[den == 0] = 1.0
    t = np.clip(np.einsum("nkj,nkj->nk", pts[:, None] - a0, ab) / den, 0, 1)
    return np.linalg.norm(a0 + ab * t[..., None] - pts[:, None], axis=2).min(1)


def _short_faces(faces, allv, limit: float, P=None, Q=None) -> list:
    """The triangles of a joining strip whose corners all lie within ``limit`` of the other
    surface's edge line; elsewhere (where the edges part) the strip is left open. Thin fans
    along a long edge (e.g. the tall triangles of a model's base wall) are kept."""
    if not faces:
        return []
    f = np.asarray(faces)
    if P is None or Q is None:
        t = allv[f]
        far = np.linalg.norm(t - np.roll(t, 1, axis=1), axis=2).max(1)
    else:
        d = {}
        for ids, other in ((np.asarray(P), np.asarray(Q)), (np.asarray(Q), np.asarray(P))):
            dist = _dist_to_polyline(allv[ids], allv[other])
            for v_id, dv in zip(ids.tolist(), dist.tolist()):
                d[v_id] = min(dv, d.get(v_id, np.inf))
        far = np.array([max(d.get(int(x), 0.0) for x in face) for face in f])
    keep = far <= limit
    return [tuple(x) for x, k in zip(faces, keep) if k]


def _join(a: Mesh, b: Mesh, gap: float, region=None):
    """Zipper faces between facing open edges of a and b (b's vertex ids offset by a.n_vertices).

    Only edge stretches that run side by side within the gap are joined; a stretch whose strip
    would contain a long triangle is left open. ``region`` (points) limits joining to edges near
    those points (the user's brush selection). Returns (number of joined b edge vertices, faces),
    or (-1, []) when b is oriented against a."""
    a_loops = boundary_loops(a.faces)
    b_loops = boundary_loops(b.faces)
    if not a_loops or not b_loops:
        return 0, []
    av, bv = a.vertices, b.vertices
    a_ids = np.concatenate([np.asarray(lp) for lp in a_loops])
    a_loop_of = np.concatenate([np.full(len(lp), k) for k, lp in enumerate(a_loops)])
    a_pos = np.concatenate([np.arange(len(lp)) for lp in a_loops])
    reach = gap + TRIM + 0.3
    a_ok = np.ones(len(a_ids), bool)
    region_tree = cKDTree(region) if region is not None and len(region) else None
    if region_tree is not None:
        a_ok = region_tree.query(av[a_ids], distance_upper_bound=1.0)[0] < np.inf
    if not a_ok.any():
        return 0, []
    sel = np.flatnonzero(a_ok)
    # a's boundary as segments (start vertex -> next vertex of its loop): points are matched to the
    # nearest segment, not the nearest vertex, so long edges (e.g. a base wall) are found too
    nxt = np.concatenate([np.roll(np.asarray(lp), -1) for lp in a_loops])
    seg_a, seg_b = av[a_ids[sel]], av[nxt[sel]]
    seg_tree = cKDTree((seg_a + seg_b) / 2)
    seg_half = float(np.linalg.norm(seg_b - seg_a, axis=1).max()) / 2 if len(sel) else 0.0

    def nearest_seg(pts):
        """(distance, index into sel) of the nearest a-boundary segment for each point."""
        kk = min(16, len(sel))
        _, cand = seg_tree.query(pts, k=kk, distance_upper_bound=reach + seg_half + 1e-9)
        cand = np.asarray(cand).reshape(len(pts), kk)
        valid = cand < len(sel)
        cs = np.where(valid, cand, 0)
        a0, ab = seg_a[cs], seg_b[cs] - seg_a[cs]
        den = np.einsum("nkj,nkj->nk", ab, ab)
        den[den == 0] = 1.0
        t = np.clip(np.einsum("nkj,nkj->nk", pts[:, None] - a0, ab) / den, 0, 1)
        d = np.linalg.norm(a0 + ab * t[..., None] - pts[:, None], axis=2)
        d[~valid] = np.inf
        best = np.argmin(d, 1)
        rows = np.arange(len(pts))
        return d[rows, best], cs[rows, best], t[rows, best]

    off = a.n_vertices
    allv = np.vstack([av, bv])
    limit = 2.0 * reach + 0.5
    faces: list = []
    joined = 0
    votes_same = votes_other = 0
    plans = []
    for blp in b_loops:
        Q = np.asarray(blp)[::-1]  # reversed: runs the same way round as a's facing edge
        dq, k, tq = nearest_seg(bv[Q])
        ok = dq <= reach
        if region_tree is not None:
            ok &= region_tree.query(bv[Q], distance_upper_bound=1.0)[0] < np.inf
        if ok.sum() < 3:
            continue
        nq = sel[np.minimum(k, len(sel) - 1)]
        loop_id = np.bincount(a_loop_of[nq[ok]]).argmax()
        ok &= a_loop_of[nq] == loop_id
        P_full = np.asarray(a_loops[loop_id])
        pos = a_pos[nq]
        n_a = len(P_full)
        steps = (np.diff(pos[ok]) + n_a // 2) % n_a - n_a // 2  # signed step along a's loop
        near = np.abs(steps) <= max(2, n_a // 8)
        votes_same += np.sum(near & (steps > 0))
        votes_other += np.sum(near & (steps < 0))
        plans.append((Q, ok, P_full, pos + tq))
    if votes_other > votes_same:
        return -1, []
    for Q, ok, P_full, par in plans:
        n = len(P_full)
        if ok.all():  # the whole edge loop meets one edge loop of the other scan: close the ring
            k0 = int(np.floor(par[0])) % n
            P = np.roll(P_full, -k0)
            qp = _monotone((par - k0) % n, n)
            if qp[-1] - qp[0] < n:
                P_ext = np.append(P, P[0])
                Q_ext = np.append(Q, Q[0])
                qp_ext = np.append(qp, qp[0] + n)
                ring = _zipper(P_ext, Q_ext + off, av[P_ext], bv[Q_ext], qp_ext)
                short = _short_faces(ring, allv, limit, P_ext, Q_ext + off)
                if len(short) > 0.9 * len(ring):
                    faces += short
                    joined += len(Q)
                    continue
        for s, e in _runs(ok, False):
            qi = np.arange(s, s + ((e - s) % len(Q)) + 1) % len(Q)
            # split where the matching point on a's edge jumps (the edges part and meet again)
            step = (np.diff(par[qi]) + n / 2) % n - n / 2  # signed step along a's loop
            cuts = np.flatnonzero((step < -2) | (step > 20)) + 1
            for part in np.split(qi, cuts):
                if len(part) < 3:
                    continue
                p0 = int(np.floor(par[part[0]])) % n
                qp = _monotone((par[part] - p0) % n, n)
                span = int(np.ceil(qp[-1]))
                if span == 0 or span > 4 * len(part) + 20 or span >= n:
                    continue
                P = P_full[(p0 + np.arange(span + 1)) % n]
                strip = _zipper(P, Q[part] + off, av[P], bv[Q[part]], qp)
                short = _short_faces(strip, allv, limit, P, Q[part] + off)
                if len(short) >= 2:
                    faces += short
                    joined += len(part)
    return joined, faces


def _close_seam_notches(mesh: Mesh, n_a: int, gap: float) -> Mesh:
    """Fill the small holes that touch both scans (left where a joined edge ends)."""
    from .repair import loop_size

    v, f, c = mesh.vertices, mesh.faces, mesh.colors
    loops = boundary_loops(f)
    new_v, new_f, new_c = [v], [f], [c] if c is not None else None
    limit = max(3.0 * gap, 2.0)
    for lp in loops:
        arr = np.asarray(lp)
        if not ((arr < n_a).any() and (arr >= n_a).any()):
            continue
        if loop_size(v, lp) > limit:
            continue
        allv = np.vstack(new_v)
        nv, nf, nc = fill_hole(allv, np.vstack(new_f), lp, None if c is None else np.vstack(new_c))
        new_v.append(nv)
        new_f.append(nf)
        if new_c is not None:
            new_c.append(nc)
    if len(new_v) == 1:
        return mesh
    return Mesh(np.vstack(new_v), np.vstack(new_f), np.vstack(new_c) if new_c is not None else None)


def fuse_meshes(meshes: list[Mesh], gap: float = 1.0, progress=lambda text: None,
                fallback_colors=None, region=None) -> tuple[Mesh, int]:
    """Fuse several aligned scans: the largest is the base, smaller ones are joined into it.
    ``region``: points; when given, only open edges near them are joined.
    Returns (mesh, number of scans that were joined to the others)."""
    if fallback_colors is not None and any(m.colors is not None for m in meshes):
        meshes = [m if m.colors is not None else
                  Mesh(m.vertices, m.faces,
                       np.tile((np.array(fallback_colors[i]) * 255).round().astype(np.uint8), (m.n_vertices, 1)))
                  for i, m in enumerate(meshes)]
    order = sorted(range(len(meshes)), key=lambda i: -_area(meshes[i]))
    out = meshes[order[0]]
    n_joined = 0
    for k, i in enumerate(order[1:], 1):
        prefix = f"{k}/{len(order) - 1}: " if len(order) > 2 else ""
        out, joined = fuse_pair(out, meshes[i], gap, lambda t, p=prefix: progress(p + t), region)
        n_joined += joined > 0
    return out, n_joined
