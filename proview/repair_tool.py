"""Repair tab: cut, fill holes and sculpt brushes (build up / carve, smooth)."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from scipy.sparse import coo_matrix, diags
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

from .i18n import tr
from .icons import icon
from .mesh import Mesh, transform_points
from .model import numpy_to_vtk_matrix
from .model_builder import boundary_loops, rotation_to_z
from .repair import fill_holes, loop_size, vertex_adjacency
from .section import iso_segments
from .selection import pick_mesh

BUILD, SMOOTH = "build", "smooth"
# brush colours: build up, carve, smooth, strong smooth
COLOR_ADD = (0.30, 0.90, 0.45)
COLOR_CARVE = (1.00, 0.38, 0.30)
COLOR_SMOOTH = (0.35, 0.68, 1.00)
COLOR_SMOOTH_STRONG = (0.62, 0.45, 1.00)


class BrushCursor:
    """Brush drawn on the surface itself: the brush edge follows the surface (where the
    surface meets the brush sphere), the brushed area is tinted (more = stronger) and an
    arrow shows what the brush does."""

    def __init__(self, renderer):
        self.ren = renderer
        self.ring_pd = vtkPolyData()
        self.tint_pd = vtkPolyData()
        self.ring = self._actor(self.ring_pd, lines=True)
        self.tint = self._actor(self.tint_pd)
        rp = self.ring.GetProperty()
        rp.SetLineWidth(3.0)
        rp.SetLighting(False)
        rp.SetRenderLinesAsTubes(True)
        self.tint.GetProperty().SetLighting(False)
        # unit arrow along +Z, starting at the origin
        arrow = vtkArrowSource()
        arrow.SetShaftRadius(0.06)
        arrow.SetTipRadius(0.16)
        arrow.SetTipLength(0.35)
        tr_ = vtkTransform()
        tr_.RotateY(-90)
        f = vtkTransformPolyDataFilter()
        f.SetInputConnection(arrow.GetOutputPort())
        f.SetTransform(tr_)
        f.Update()
        self.arrow = self._actor(f.GetOutput())
        self.shown = False

    @staticmethod
    def _actor(pd, lines=False):
        m = vtkPolyDataMapper()
        m.SetInputData(pd)
        m.ScalarVisibilityOff()
        m.SetResolveCoincidentTopologyToPolygonOffset()
        if lines:
            m.SetRelativeCoincidentTopologyLineOffsetParameters(-4, -4)
        else:
            m.SetRelativeCoincidentTopologyPolygonOffsetParameters(-3, -3)
        a = vtkActor()
        a.SetMapper(m)
        a.PickableOff()
        return a

    def place(self, center, normal, radius, strength, color, arrow_sign, lv=None, lf=None):
        """``lv``/``lf``: world vertices and faces of the surface around the brush."""
        n = np.asarray(normal, float)
        n = n / (np.linalg.norm(n) or 1.0)
        center = np.asarray(center, float)
        lift = n * min(0.02, 0.01 * radius)
        if lv is not None and len(lf):
            d = np.linalg.norm(lv - center, axis=1)
            segs = iso_segments(lv, lf, d - radius) + lift
            self._set_lines(segs)
            inside = (d[lf] < radius).any(1)
            self._set_tris(lv + lift, lf[inside])
        # arrow: out of the surface (build up) or into it (carve); none for smoothing
        if arrow_sign == 0:
            self.arrow.SetVisibility(False)
        else:
            Ra = rotation_to_z(n * arrow_sign).T
            A = np.eye(4)
            A[:3, :3] = Ra * radius * 0.9
            A[:3, 3] = center + (n * radius * 0.95 if arrow_sign < 0 else n * 0.05 * radius)
            self.arrow.SetUserMatrix(numpy_to_vtk_matrix(A))
            self.arrow.SetVisibility(True)
            self.arrow.GetProperty().SetColor(*color)
        self.ring.GetProperty().SetColor(*color)
        tp = self.tint.GetProperty()
        tp.SetColor(*color)
        tp.SetOpacity(0.18 + 0.4 * float(strength))
        if not self.shown:
            for a in (self.tint, self.ring, self.arrow):
                self.ren.AddActor(a)
            self.shown = True

    def _set_lines(self, segs):
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        k = len(segs)
        pts = vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(segs.reshape(-1, 3)), deep=True))
        cells = vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 2 * k + 1, 2, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(np.arange(2 * k, dtype=np.int64), deep=True))
        self.ring_pd.SetPoints(pts)
        self.ring_pd.SetLines(cells)
        self.ring_pd.Modified()

    def _set_tris(self, v, f):
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        pts = vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(v), deep=True))
        cells = vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(f) + 1, 3, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(np.ascontiguousarray(f.ravel(), dtype=np.int64), deep=True))
        self.tint_pd.SetPoints(pts)
        self.tint_pd.SetPolys(cells)
        self.tint_pd.Modified()

    def hide(self):
        if self.shown:
            for a in (self.tint, self.ring, self.arrow):
                self.ren.RemoveActor(a)
            self.shown = False


class SculptTool:
    """Left drag sculpts the active model.
    Build mode: build up; hold Shift while dragging to carve. Smooth mode: Shift = strong smoothing.
    Shift + wheel = size, Ctrl + wheel = strength."""

    def __init__(self, viewer, get_model, push_undo, on_change):
        self.viewer = viewer
        self.get_model = get_model
        self.push_undo = push_undo
        self.on_change = on_change
        self.mode = BUILD
        self.radius = 2.0  # mm
        self.strength = 0.5  # 0..1
        self.active = False
        self.cursor = BrushCursor(viewer.ren)
        self._s = None  # stroke state
        self._topo = {}
        self._local = None
        self._mouse = None
        self._last_full_pick = 0.0
        self._last_hit = None  # (point, normal)
        self._shift = False

    # ------------------------------------------------------------ lifecycle
    def activate(self, mode=None):
        if mode is not None:
            self.mode = mode
        self.active = True
        self.viewer.style.tool = self
        if all(m.opacity >= 0.999 for m in self.viewer.models):
            self.viewer.ren.SetUseDepthPeeling(0)  # much faster; the brush tint blends fine without it
        self.viewer.render()

    def deactivate(self):
        self._end()
        self.active = False
        if self.viewer.style.tool is self:
            self.viewer.style.tool = None
        self.cursor.hide()
        self._topo = {}
        self._local = None
        self.viewer.ren.SetUseDepthPeeling(1)
        self.viewer.render()

    def set_radius(self, r):
        self.radius = float(np.clip(r, 0.2, 30.0))
        self._refresh_cursor()
        self.on_change()

    def set_strength(self, s):
        self.strength = float(np.clip(s, 0.02, 1.0))
        self._refresh_cursor()
        self.on_change()

    def _style(self, shift):
        """(colour, arrow sign, label) of the current action."""
        if self.mode == BUILD:
            return (COLOR_CARVE, -1) if shift else (COLOR_ADD, 1)
        return (COLOR_SMOOTH_STRONG, 0) if shift else (COLOR_SMOOTH, 0)

    def _refresh_cursor(self):
        if self._last_hit is not None and self.active:
            self._show(self._last_hit, self._shift)
            self.viewer.render()

    # ------------------------------------------------------------ helpers
    def _topology(self, mesh: Mesh, need_w: bool):
        key = id(mesh.faces)
        t = self._topo.get(key)
        if t is None:
            f = mesh.faces
            m, n = len(f), mesh.n_vertices
            inc = coo_matrix((np.ones(3 * m, np.float32), (f.ravel(), np.repeat(np.arange(m), 3))),
                             shape=(n, m)).tocsr()
            t = {"inc": inc, "w": None}
            self._topo = {key: t}
        if need_w and t["w"] is None:
            a = vertex_adjacency(mesh.n_vertices, mesh.faces)
            deg = np.asarray(a.sum(1)).ravel()
            deg[deg == 0] = 1
            t["w"] = (diags(1.0 / deg) @ a).tocsr()
        return t

    def _pick(self, wv, faces, x, y, allow_full=True):
        """(point, unit normal facing the camera) under the cursor. Only triangles near the last
        hit are tested when possible, so it stays fast on large scans."""
        ren = self.viewer.ren
        loc = self._local
        hit = tri = None
        if loc is not None and loc["faces_id"] == id(faces):
            lv = wv[loc["vidx"]]
            hit = pick_mesh(ren, x, y, lv, loc["faces"])
            if hit is not None:
                tri = lv[loc["faces"][hit[2]]]
                if np.linalg.norm(hit[1] - loc["center"]) < loc["reach"] - self.radius:
                    return hit[1], self._normal(tri)
        if hit is None:
            if not allow_full:
                return None
            hit = pick_mesh(ren, x, y, wv, faces)
            if hit is None:
                return None
            tri = wv[faces[hit[2]]]
        c = hit[1]
        reach = max(2.5 * self.radius, 2.0)
        near_v = np.einsum("ij,ij->i", wv - c, wv - c) < reach ** 2
        sub = faces[near_v[faces[:, 0]] | near_v[faces[:, 1]] | near_v[faces[:, 2]]]
        vidx, inv = np.unique(sub.ravel(), return_inverse=True)
        self._local = {"faces_id": id(faces), "vidx": vidx, "faces": inv.reshape(-1, 3), "center": c,
                       "reach": reach}
        return c, self._normal(tri)

    def _normal(self, tri):
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n /= np.linalg.norm(n) or 1.0
        cam = np.array(self.viewer.ren.GetActiveCamera().GetPosition())
        if n @ (cam - tri[0]) < 0:
            n = -n
        return n

    def _begin(self, model):
        self.push_undo(model)
        model.begin_vertex_edit()
        mesh = model.mesh
        wv = transform_points(model.matrix, mesh.vertices)
        tri = wv[mesh.faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        topo = self._topology(mesh, self.mode == SMOOTH)
        self._local = None
        self._s = {"model": model, "wv": wv, "fn": fn, "topo": topo,
                   "Minv": np.linalg.inv(model.matrix), "R": model.matrix[:3, :3].copy()}

    def _end(self):
        if self._s is not None:
            self._s["model"].end_vertex_edit()
            self._s = None
            self.on_change()

    def _apply(self, x, y, shift):
        s = self._s
        model = s["model"]
        wv, faces = s["wv"], model.mesh.faces
        res = self._pick(wv, faces, x, y)
        if res is None:
            return None
        c, nrm = res
        loc = self._local
        if loc is None or np.linalg.norm(c - loc["center"]) > loc["reach"] - self.radius:
            self._recentre(c)
            loc = self._local
        cand = loc["vidx"]  # only vertices near the brush (the patch covers 2.5 x the radius)
        dv = wv[cand] - c
        d2c = np.einsum("ij,ij->i", dv, dv)
        inside = d2c < self.radius ** 2
        idx = cand[inside]
        if len(idx) == 0:
            return c, nrm
        fall = (1.0 - d2c[inside] / self.radius ** 2) ** 2
        topo = s["topo"]
        inc = topo["inc"]
        if self.mode == SMOOTH:
            strength = min(1.0, self.strength * (2.5 if shift else 1.0))
            for _ in range(3 if shift else 1):  # Shift: strong smoothing
                target = topo["w"][idx] @ wv
                wv[idx] += (strength * fall)[:, None] * (target - wv[idx])
        else:
            n = inc[idx] @ s["fn"]
            n = (n * fall[:, None]).sum(0)
            ln = np.linalg.norm(n)
            if ln > 0:
                n /= ln
                sign = -1.0 if shift else 1.0  # Shift: carve
                wv[idx] += (sign * self.strength * self.radius * 0.06 * fall)[:, None] * n
        touched = np.unique(inc[idx].indices)
        tri = wv[faces[touched]]
        s["fn"][touched] = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ring = np.unique(faces[touched].ravel())
        vn = inc[ring] @ s["fn"]
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        ln[ln == 0] = 1.0
        vn = vn / ln
        model.set_vertices(idx, transform_points(s["Minv"], wv[idx]), ring, vn @ s["R"])
        # cursor follows the average normal of the brushed area
        an = vn.mean(0) if len(vn) else nrm
        if np.linalg.norm(an) > 0 and an @ nrm > 0:
            nrm = an / np.linalg.norm(an)
        return c, nrm

    def _patch(self):
        """World vertices / faces of the surface around the last hit (for drawing the brush)."""
        loc = self._local
        if loc is None:
            return None, None
        if self._s is not None:
            wv = self._s["wv"]
        else:
            m = self.get_model()
            if m is None:
                return None, None
            wv = m.world_geometry()[0]
        if len(loc["vidx"]) and loc["vidx"].max() >= len(wv):
            return None, None
        return wv[loc["vidx"]], loc["faces"]

    def _show(self, res, shift):
        self._shift = shift
        if res is None:
            self._last_hit = None
            self.cursor.hide()
            return
        self._last_hit = res
        color, sign = self._style(shift)
        # the patch must cover the brush: re-centre it when the brush grew or moved away
        loc = self._local
        if loc is None or np.linalg.norm(res[0] - loc["center"]) > loc["reach"] - self.radius:
            self._recentre(res[0])
        lv, lf = self._patch()
        self.cursor.place(res[0], res[1], self.radius, self.strength, color, sign, lv, lf)

    def _recentre(self, c):
        m = self._s["model"] if self._s is not None else self.get_model()
        if m is None:
            return
        wv = self._s["wv"] if self._s is not None else m.world_geometry()[0]
        faces = m.mesh.faces
        reach = max(2.5 * self.radius, 2.0)
        near_v = np.einsum("ij,ij->i", wv - c, wv - c) < reach ** 2
        sub = faces[near_v[faces[:, 0]] | near_v[faces[:, 1]] | near_v[faces[:, 2]]]
        vidx, inv = np.unique(sub.ravel(), return_inverse=True)
        self._local = {"faces_id": id(faces), "vidx": vidx, "faces": inv.reshape(-1, 3), "center": c,
                       "reach": reach}

    # ------------------------------------------------------------ events
    def on_press(self, x, y, ctrl, shift, repeat):
        model = self.get_model()
        if model is None or not model.visible:
            return False
        if self._pick(model.world_geometry()[0], model.mesh.faces, x, y) is None:
            return False  # beside the model: rotate the camera
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._begin(model)
        finally:
            QApplication.restoreOverrideCursor()
        self._show(self._apply(x, y, shift), shift)
        self.viewer.render()
        return True

    def on_move(self, x, y, ctrl, shift):
        self._mouse = (x, y)
        if self._s is not None:
            self._show(self._apply(x, y, shift), shift)
            self.viewer.render()
            return True
        model = self.get_model()
        res = None
        if model is not None and model.visible:
            now = time.monotonic()
            allow_full = now - self._last_full_pick > 0.12  # a full pick at most ~8x per second
            res = self._pick(model.world_geometry()[0], model.mesh.faces, x, y, allow_full=allow_full)
            if allow_full:
                self._last_full_pick = now
            if res is None and not allow_full and self._last_hit is not None:
                res = self._last_hit
        self._show(res, shift)
        self.viewer.render()
        return False

    def on_release(self, x, y, ctrl, shift):
        if self._s is None:
            return False
        self._end()
        self.viewer.render()
        return True

    def on_wheel(self, direction):
        return False

    def on_wheel_mod(self, direction, ctrl, shift):
        """Shift + wheel: brush size. Ctrl + wheel: strength."""
        if shift:
            self.set_radius(self.radius * (1.12 if direction > 0 else 1 / 1.12))
        elif ctrl:
            self.set_strength(self.strength + (0.05 if direction > 0 else -0.05))
        else:
            return False
        return True


class RepairPanel(QWidget):
    """Cut, hole filling and sculpting in one tab."""

    TOOLS = ("cut", BUILD, SMOOTH)

    def __init__(self, sculpt: SculptTool, cut, cut_panel, get_models, push_undo, on_change, status,
                 settings=None, parent=None):
        super().__init__(parent)
        self.sculpt = sculpt
        self.cut = cut
        self.cut_panel = cut_panel
        self.get_models = get_models
        self.push_undo = push_undo
        self.on_change = on_change
        self.status = status
        self.settings = settings
        self.active = False
        self.tool = "cut"
        lay = QVBoxLayout(self)
        self.model_lbl = QLabel("")
        self.model_lbl.setWordWrap(True)
        self.model_lbl.setStyleSheet("font-weight:bold;")
        lay.addWidget(self.model_lbl)

        # ---------------- tool selector
        row = QHBoxLayout()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.tool_buttons = {}
        for name, key, ic in (("cut", "repair_tool_cut", "cut"), (BUILD, "repair_tool_build", "sculpt_build"),
                              (SMOOTH, "repair_tool_smooth", "sculpt_smooth")):
            b = QToolButton()
            b.setText(tr(key))
            b.setIcon(icon(ic))
            b.setIconSize(QSize(26, 26))
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setCheckable(True)
            b.setMinimumSize(92, 58)
            b.setStyleSheet("QToolButton { background:#25282f; border:1px solid #30343c; border-radius:9px;"
                            " padding:4px; } QToolButton:hover { border-color:#4a90e2; }"
                            " QToolButton:checked { background:#1d3a5e; border:1px solid #4aa3ff; }")
            b.toggled.connect(lambda on, n=name: on and self.set_tool(n))
            self.group.addButton(b)
            self.tool_buttons[name] = b
            row.addWidget(b)
        lay.addLayout(row)

        # ---------------- cut (the cut tool's own panel)
        lay.addWidget(cut_panel)

        # ---------------- sculpt
        self.sculpt_box = QGroupBox(tr("sculpt_title"))
        bl = QVBoxLayout(self.sculpt_box)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("sculpt_size")))
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(2, 300)  # 0.2 .. 30 mm
        self.size.valueChanged.connect(lambda v: self.sculpt.set_radius(v / 10.0))
        row.addWidget(self.size, 1)
        self.size_lbl = QLabel("")
        self.size_lbl.setFixedWidth(60)
        row.addWidget(self.size_lbl)
        bl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("sculpt_strength")))
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(2, 100)
        self.strength.valueChanged.connect(lambda v: self.sculpt.set_strength(v / 100.0))
        row.addWidget(self.strength, 1)
        self.strength_lbl = QLabel("")
        self.strength_lbl.setFixedWidth(60)
        row.addWidget(self.strength_lbl)
        bl.addLayout(row)
        self.sculpt_help = QLabel("")
        self.sculpt_help.setWordWrap(True)
        self.sculpt_help.setStyleSheet("color: gray; font-size: 11px;")
        bl.addWidget(self.sculpt_help)
        lay.addWidget(self.sculpt_box)

        # ---------------- holes
        rep = QGroupBox(tr("repair_title"))
        rl = QVBoxLayout(rep)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("repair_max")))
        self.hole_max = QDoubleSpinBox()
        self.hole_max.setRange(0.5, 200)
        self.hole_max.setSingleStep(1.0)
        self.hole_max.setDecimals(1)
        self.hole_max.setSuffix(" mm")
        saved = settings.value("builder/hole_max", 10.0) if settings is not None else 10.0
        self.hole_max.setValue(float(saved))
        row.addWidget(self.hole_max)
        rl.addLayout(row)
        self.holes_lbl = QLabel("")
        self.holes_lbl.setStyleSheet("color:#aab0bb; font-size:11px;")
        self.holes_lbl.setWordWrap(True)
        rl.addWidget(self.holes_lbl)
        b = QPushButton(tr("repair_go"))
        b.clicked.connect(lambda _=False: self.fill())
        rl.addWidget(b)
        self.hole_max.valueChanged.connect(lambda _v: self.refresh())
        lay.addWidget(rep)
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        lay.addWidget(self.result_lbl)
        lay.addStretch(1)
        self.tool_buttons["cut"].setChecked(True)
        self._show_sections()
        self.sync()

    # ------------------------------------------------------------ tool switching
    def set_active(self, on):
        self.active = on
        if on:
            self._activate_tool()
        else:
            self.cut.deactivate() if self.cut.active else None
            self.sculpt.deactivate()

    def set_tool(self, name):
        self.tool = name
        if self.tool_buttons[name].isChecked() is False:
            self.tool_buttons[name].setChecked(True)
            return
        self._show_sections()
        if self.active:
            self._activate_tool()

    def _activate_tool(self):
        if self.tool == "cut":
            self.sculpt.deactivate()
            if not self.cut.active:
                self.cut.activate()
        else:
            if self.cut.active:
                self.cut.deactivate()
            self.sculpt.activate(self.tool)
        self.on_change()

    def _show_sections(self):
        cut = self.tool == "cut"
        self.cut_panel.setVisible(cut)
        self.sculpt_box.setVisible(not cut)
        self.sculpt_help.setText(tr("sculpt_help_build" if self.tool == BUILD else "sculpt_help_smooth"))

    # ------------------------------------------------------------ display
    def sync(self):
        for slider, value in ((self.size, int(round(self.sculpt.radius * 10))),
                              (self.strength, int(round(self.sculpt.strength * 100)))):
            if slider.value() != value:
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
        self.size_lbl.setText(f"{self.sculpt.radius:.1f} mm")
        self.strength_lbl.setText(f"{self.sculpt.strength * 100:.0f} %")

    def refresh(self):
        self.sync()
        self.cut_panel.refresh()
        models = self.get_models()
        if not models:
            self.model_lbl.setText(tr("select_model_first"))
            self.holes_lbl.setText("")
            return
        self.model_lbl.setText(tr("orient_model", name=", ".join(m.name for m in models)))
        small = big = 0
        for m in models:
            loops = boundary_loops(m.mesh.faces) if m.mesh.n_faces else []
            if loops:
                main = max(loops, key=len)
                for lp in loops:
                    if lp is main:
                        continue
                    if loop_size(m.mesh.vertices, lp) <= self.hole_max.value():
                        small += 1
                    else:
                        big += 1
        self.holes_lbl.setText(tr("repair_found", small=small, big=big))

    def fill(self):
        models = self.get_models()
        if not models:
            return
        if self.settings is not None:
            self.settings.setValue("builder/hole_max", self.hole_max.value())
        QApplication.setOverrideCursor(Qt.WaitCursor)
        total = 0
        try:
            for m in models:
                loops = boundary_loops(m.mesh.faces)
                keep = max(loops, key=len) if loops else None
                fixed, filled, _ = fill_holes(m.mesh, self.hole_max.value(), keep=keep)
                if filled:
                    self.push_undo(m)
                    m.set_mesh(fixed)
                    total += filled
        finally:
            QApplication.restoreOverrideCursor()
        self.result_lbl.setText(tr("repair_done", n=total))
        self.status(tr("repair_done", n=total))
        self.on_change()
        self.refresh()

