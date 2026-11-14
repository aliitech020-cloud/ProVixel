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


def _zipper(P: np.ndarray, Q: np.ndarray, pv: np.ndarray, qv: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangle strip between two polylines that run the same way (vertex ids P, Q; points pv, qv).

    P must be an edge loop as stored by ``boundary_loops`` of its surface (a->b in its faces)
    and Q the reverse of the other surface's loop; then the strip is oriented like both."""
    faces = []
    i = j = 0
    n, m = len(P), len(Q)
    while i < n - 1 or j < m - 1:
        if i == n - 1:
            adv_p = False
        elif j == m - 1:
            adv_p = True
        else:
            adv_p = np.linalg.norm(pv[i + 1] - qv[j]) <= np.linalg.norm(pv[i] - qv[j + 1])
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


def fuse_pair(base: Mesh, insert: Mesh, gap: float = 1.0, progress=lambda text: None) -> tuple[Mesh, int]:
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

    progress("Joining the edges")
    ins = insert
    joined, extra_faces = _join(trimmed, ins, gap)
    if joined == -1:  # the insert is stored the other way round: turn it and try again
        ins = Mesh(insert.vertices, insert.faces[:, ::-1], insert.colors)
        joined, extra_faces = _join(trimmed, ins, gap)
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


def _join(a: Mesh, b: Mesh, gap: float):
    """Zipper faces between facing open edges of a and b (b's vertex ids offset by a.n_vertices).
    Returns (number of joined b edge vertices, faces) or (-1, []) when b is oriented against a."""
    a_loops = boundary_loops(a.faces)
    b_loops = boundary_loops(b.faces)
    if not a_loops or not b_loops:
        return 0, []
    av, bv = a.vertices, b.vertices
    a_ids = np.concatenate([np.asarray(lp) for lp in a_loops])
    a_loop_of = np.concatenate([np.full(len(lp), k) for k, lp in enumerate(a_loops)])
    a_pos = np.concatenate([np.arange(len(lp)) for lp in a_loops])
    a_tree = cKDTree(av[a_ids])
    reach = gap + TRIM + 0.3
    off = a.n_vertices
    faces: list = []
    joined = 0
    votes_same = votes_other = 0
    plans = []
    for blp in b_loops:
        Q = np.asarray(blp)[::-1]  # reversed: runs the same way round as a's facing edge
        dq, nq = a_tree.query(bv[Q], distance_upper_bound=reach)
        ok = np.isfinite(dq)
        if ok.sum() < 3:
            continue
        loop_id = np.bincount(a_loop_of[nq[ok]]).argmax()
        ok &= np.isfinite(dq) & (a_loop_of[np.minimum(nq, len(a_ids) - 1)] == loop_id)
        P_full = np.asarray(a_loops[loop_id])
        pos = a_pos[np.minimum(nq, len(a_ids) - 1)]
        # orientation check: along matched b points, a's matching positions should move forward
        steps = np.diff(pos[ok]) % len(P_full)
        fwd = np.sum((steps > 0) & (steps < len(P_full) / 2))
        bwd = np.sum(steps > len(P_full) / 2)
        votes_same += fwd
        votes_other += bwd
        plans.append((Q, ok, P_full, pos))
    if votes_other > votes_same:
        return -1, []
    for Q, ok, P_full, pos in plans:
        n = len(P_full)
        closed = ok.mean() > 0.9
        if closed:
            ok = np.ones(len(Q), bool)
        for s, e in _runs(ok, closed):
            if (s, e) == (-1, -1):  # whole loop: close the ring
                k0 = int(np.argmin(np.linalg.norm(av[P_full] - bv[Q[0]], axis=1)))
                P = np.roll(P_full, -k0)
                P = np.append(P, P[0])
                Qr = np.append(Q, Q[0])
            else:
                qi = np.arange(s, s + ((e - s) % len(Q)) + 1) % len(Q)
                if len(qi) < 3:
                    continue
                Qr = Q[qi]
                p0, p1 = pos[qi[0]], pos[qi[-1]]
                span = (p1 - p0) % n
                if span == 0 or span > n / 2:
                    continue
                P = P_full[(p0 + np.arange(span + 1)) % n]
            faces += [(f[0], f[1], f[2]) for f in _zipper_ids(P, Qr + off, av, bv, off)]
            joined += len(Qr)
    return joined, faces


def _zipper_ids(P, Q, av, bv, off):
    return _zipper(P, Q, av[P], bv[Q - off])


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
                fallback_colors=None) -> tuple[Mesh, int]:
    """Fuse several aligned scans: the largest is the base, smaller ones are joined into it.
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
        out, joined = fuse_pair(out, meshes[i], gap, lambda t, p=prefix: progress(p + t))
        n_joined += joined > 0
    return out, n_joined
