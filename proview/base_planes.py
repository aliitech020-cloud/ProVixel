"""Base planes for the model builder.

Every selected scan (upper and/or lower jaw) gets a flat base plane. All planes share one
axis, so the bases of an upper and a lower model are parallel. A plane can be dragged with
the mouse along that axis to make the base taller or shorter; the base itself (the walls
from the scan edge down to the plane) is shown transparent and follows the plane at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_builder import (
    BuildError,
    _loop_length,
    _smooth_loop_xy,
    _triangulate_planar_loops,
    boundary_loops,
    detect_up,
    rotation_to_z,
)

MIN_BASE = -30.0  # negative: the plane is inside the scan, which is cut there
MAX_BASE = 40.0


@dataclass
class Jaw:
    """One scan and its base, in the scan's teeth-up frame (frame = world @ R.T)."""

    model: object
    up: np.ndarray  # world direction of the teeth (+Z of the frame)
    R: np.ndarray
    xy: np.ndarray  # open edge of the scan seen from above (smoothed), frame x/y
    top: np.ndarray  # height where the base walls start, per edge point
    ref: float  # base height is measured from here: the trim level or the lowest scan point
    z_cut: float | None  # trim level (frame z) or None
    lo: np.ndarray  # footprint bounds (frame x/y)
    hi: np.ndarray
    base: float  # base height in mm

    @property
    def level(self) -> float:
        return self.ref - self.base

    def to_world(self, pts):
        return np.asarray(pts, float) @ self.R

    def plane_corners(self, z):
        pad = 0.12 * float(np.linalg.norm(self.hi - self.lo)) + 2.0
        (x0, y0), (x1, y1) = self.lo - pad, self.hi + pad
        return self.to_world([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])


class JawCache:
    """Edge loop and detected teeth direction per scan (the slow part), kept between updates."""

    def __init__(self):
        self._data = {}

    def get(self, model):
        key = (id(model.mesh), model.matrix.tobytes())
        if key not in self._data:
            if len(self._data) > 16:
                self._data.clear()
            scan = model.mesh.transformed(model.matrix)
            loops = boundary_loops(scan.faces) if scan.n_faces else []
            if not loops:
                self._data[key] = None
            else:
                v = scan.vertices
                main = max(loops, key=lambda lp: _loop_length(v, lp))
                self._data[key] = (v, main, detect_up(v, main, scan.faces))
        return self._data[key]


def make_jaws(models, cache: JawCache, auto: bool, view_up, trim: bool, trim_offset: float,
              bases: dict, default_base: float) -> list[Jaw]:
    """Jaws for the selected scans with one common axis (closed models are skipped)."""
    infos = [(m, cache.get(m)) for m in models]
    infos = [(m, i) for m, i in infos if i is not None]
    if not infos:
        return []
    detected = [i[2] for _, i in infos]
    if view_up is not None:
        n = np.asarray(view_up, float)
        n = n / np.linalg.norm(n)
        if len(infos) == 1:
            ups = [n]
        else:  # upper and lower jaw: each keeps its own side of the view axis
            ups = [n if d @ n >= 0 else -n for d in detected]
    elif auto:
        acc = np.zeros(3)
        for d in detected:
            acc += d if d @ detected[0] >= 0 else -d
        n = acc / (np.linalg.norm(acc) or 1.0)
        ups = [n if d @ n >= 0 else -n for d in detected]
    else:
        z = np.array([0.0, 0.0, 1.0])
        if len(infos) == 1:
            ups = [z]
        else:  # the model's own Z axis, but the upper jaw keeps its teeth pointing down
            ups = [z if d @ z >= 0 else -z for d in detected]
    jaws = []
    for (m, (v, main, _d)), up in zip(infos, ups):
        R = rotation_to_z(up)
        vl = v @ R.T
        edge = vl[main]
        z_cut = float(np.quantile(edge[:, 2], 0.2)) + trim_offset if trim else None
        if trim:
            top = np.maximum(edge[:, 2], z_cut)
            ref = z_cut
        else:
            top = edge[:, 2]
            ref = float(vl[:, 2].min())
        xy = _smooth_loop_xy(edge[:, :2].copy())
        jaws.append(Jaw(m, up, R, xy, top, ref, z_cut, vl[:, :2].min(0), vl[:, :2].max(0),
                        float(np.clip(bases.get(id(m), default_base), MIN_BASE, MAX_BASE))))
    return jaws


def _polydata(points, polys=None, lines=None):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    pd = vtkPolyData()
    vp = vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    pd.SetPoints(vp)

    def cells(arr):
        ca = vtkCellArray()
        if isinstance(arr, np.ndarray) and arr.ndim == 2:  # many cells of one size: no Python loop
            flat = np.hstack([np.full((len(arr), 1), arr.shape[1]), arr]).astype(np.int64).ravel()
            ca.SetCells(len(arr), numpy_to_vtkIdTypeArray(flat, deep=True))
            return ca
        flat = []
        for c in arr:
            flat.append(len(c))
            flat.extend(int(i) for i in c)
        ca.SetCells(len(arr), numpy_to_vtkIdTypeArray(np.asarray(flat, dtype=np.int64), deep=True))
        return ca

    if polys is not None and len(polys):
        pd.SetPolys(cells(polys))
    if lines is not None and len(lines):
        pd.SetLines(cells(lines))
    return pd


class BasePlanes:
    """Draws the base planes, the transparent bases and the trim planes; lets the user drag
    the base planes (it is a viewer tool)."""

    PLANE = (0.2, 0.75, 0.85)
    PLANE_HOT = (0.35, 0.9, 1.0)
    BASE = (0.95, 0.75, 0.35)
    CUT = (0.9, 0.12, 0.1)
    TRIM = (1.0, 0.55, 0.15)

    def __init__(self, viewer, on_drag=None, on_release=None):
        self.viewer = viewer
        self._drag_cb = on_drag  # (jaw) while dragging
        self._release_cb = on_release  # (jaw) after a drag
        self.jaws: list[Jaw] = []
        self.actors = []
        self.hot = None  # jaw index under the mouse
        self.drag = None  # (jaw index, s0, base0)
        self.show_bases = True

    # ------------------------------------------------------------------ drawing
    def _actor(self, pd, color, opacity, lines=False, width=2):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(pd)
        if lines:
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-2, -2)
        a = vtkActor()
        a.SetMapper(mapper)
        p = a.GetProperty()
        p.SetColor(*color)
        p.SetOpacity(opacity)
        if lines:
            p.SetLineWidth(width)
            p.SetLighting(False)
        else:
            p.SetAmbient(0.25)
        a.PickableOff()
        return a

    def show(self, jaws: list[Jaw], show_bases=True):
        self.jaws = jaws
        self.show_bases = show_bases
        self._rebuild()

    def _rebuild(self):
        show_bases = self.show_bases
        ren = self.viewer.ren
        for a in self.actors:
            ren.RemoveActor(a)
        self.actors = []
        for k, j in enumerate(self.jaws):
            hot = k == self.hot or (self.drag is not None and self.drag[0] == k)
            z = j.level
            c = j.plane_corners(z)
            self.actors.append(self._actor(_polydata(c, [[0, 1, 2, 3]]), self.PLANE_HOT if hot else self.PLANE,
                                           0.42 if hot else 0.22))
            self.actors.append(self._actor(_polydata(c, lines=[[0, 1, 2, 3, 0]]), self.PLANE_HOT if hot else self.PLANE,
                                           1.0, lines=True, width=3 if hot else 2))
            if j.z_cut is not None:
                t = j.plane_corners(j.z_cut)
                self.actors.append(self._actor(_polydata(t, [[0, 1, 2, 3]]), self.TRIM, 0.18))
            if show_bases:
                n = len(j.xy)
                top = np.column_stack([j.xy, np.maximum(j.top, z)])  # plane inside: no walls above it
                bottom = np.column_stack([j.xy, np.full(n, z)])
                pts = j.to_world(np.vstack([top, bottom]))
                i = np.arange(n)
                quads = np.stack([i, (i + 1) % n, (i + 1) % n + n, i + n], 1)
                self.actors.append(self._actor(_polydata(pts, quads), self.BASE, 0.38))
                try:
                    cap = _triangulate_planar_loops([j.xy], z)
                    if cap.n_faces:
                        self.actors.append(self._actor(_polydata(j.to_world(cap.vertices), cap.faces),
                                                       self.BASE, 0.25))
                except Exception:  # noqa: BLE001 - the cap is only a hint
                    pass
        for j in self.jaws:  # the part of the scan the plane cuts away is shown red
            red = self._cut_part(j)
            if red is not None:
                a = self._actor(red, self.CUT, 1.0)
                a.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
                a.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -2)
                self.actors.append(a)
        for a in self.actors:
            ren.AddActor(a)
        self.viewer.render()

    def _cut_part(self, j):
        m = j.model
        if not m.visible:
            return None
        wv, cen, _ = m.world_geometry()
        below = (cen @ j.up) < j.level
        if not below.any():
            return None
        f = m.mesh.faces[below]
        used = np.unique(f)
        remap = np.zeros(len(wv), np.int64)
        remap[used] = np.arange(len(used))
        return _polydata(wv[used], remap[f])

    def hide(self):
        for a in self.actors:
            self.viewer.ren.RemoveActor(a)
        self.actors = []
        self.jaws = []
        self.hot = None
        self.drag = None
        self.viewer.render()

    # ------------------------------------------------------------------ picking / dragging
    def _ray(self, x, y):
        from .selection import display_ray

        return display_ray(self.viewer.ren, x, y)

    def _hit(self, x, y):
        o, d = self._ray(x, y)
        best = None
        for k, j in enumerate(self.jaws):
            c = j.plane_corners(j.level)
            centre = c.mean(0)
            den = d @ j.up
            if abs(den) < 1e-6:
                continue
            t = ((centre - o) @ j.up) / den
            if t <= 0:
                continue
            p = (o + d * t) @ j.R.T
            pad = 0.12 * float(np.linalg.norm(j.hi - j.lo)) + 2.0
            if np.all(p[:2] >= j.lo - pad) and np.all(p[:2] <= j.hi + pad) and (best is None or t < best[0]):
                best = (t, k)
        return None if best is None else best[1]

    def _axis_param(self, k, x, y):
        """Position along the jaw's axis closest to the mouse ray (None when looking along it)."""
        j = self.jaws[k]
        o, d = self._ray(x, y)
        c = j.plane_corners(j.level).mean(0)
        u = j.up
        w0 = c - o
        b = u @ d
        den = 1.0 - b * b
        if den < 1e-4:
            return None
        return (b * (d @ w0) - (u @ w0)) / den + (c @ u)

    def on_press(self, x, y, ctrl, shift, rep):
        k = self._hit(x, y)
        if k is None:
            return False
        s0 = self._axis_param(k, x, y)
        if s0 is None:
            return False
        self.drag = (k, s0, self.jaws[k].base)
        self.show_bases = True
        self._rebuild()
        if self._drag_cb is not None:
            self._drag_cb(self.jaws[k])
        return True

    def on_move(self, x, y, ctrl, shift):
        if self.drag is None:
            k = self._hit(x, y)
            if k != self.hot:
                self.hot = k
                self._rebuild()
            return False
        k, s0, base0 = self.drag
        s = self._axis_param(k, x, y)
        if s is None:
            return True
        j = self.jaws[k]
        # moving the plane along the teeth direction (+up) makes the base shorter
        j.base = float(np.clip(base0 - (s - s0), MIN_BASE, MAX_BASE))
        self._rebuild()
        if self._drag_cb is not None:
            self._drag_cb(j)
        return True

    def on_release(self, x, y, ctrl, shift):
        if self.drag is None:
            return False
        k = self.drag[0]
        self.drag = None
        self._rebuild()
        if self._release_cb is not None:
            self._release_cb(self.jaws[k])
        return True

    def on_wheel(self, direction):
        return False


__all__ = ["BasePlanes", "BuildError", "Jaw", "JawCache", "make_jaws"]
