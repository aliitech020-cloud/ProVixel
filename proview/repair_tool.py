"""Repair tab: fill holes and sculpt brushes (add / remove / smooth)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.sparse import coo_matrix, diags

from .i18n import tr
from .mesh import Mesh, transform_points
from .model_builder import boundary_loops
from .repair import fill_holes, loop_size, vertex_adjacency
from .selection import pick_mesh, project
from .viewer import Overlay2D

ADD, REMOVE, SMOOTH = "add", "remove", "smooth"


class SculptTool:
    """Left drag sculpts the active model; Shift + wheel = size, Ctrl + wheel = strength."""

    def __init__(self, viewer, get_model, push_undo, on_change):
        self.viewer = viewer
        self.get_model = get_model
        self.push_undo = push_undo
        self.on_change = on_change
        self.mode = SMOOTH
        self.radius = 2.0  # mm
        self.strength = 0.5  # 0..1
        self.active = False
        self.cursor = Overlay2D(viewer.ren, color=(1.0, 1.0, 1.0), width=1.5)
        self.cursor.set_visible(False)
        self._s = None  # stroke state
        self._topo = {}  # cached per-topology matrices
        self._mouse = None

    # ------------------------------------------------------------ lifecycle
    def activate(self):
        self.active = True
        self.viewer.style.tool = self
        self.cursor.set_visible(True)

    def deactivate(self):
        self.active = False
        self._end()
        if self.viewer.style.tool is self:
            self.viewer.style.tool = None
        self.cursor.clear()
        self.cursor.set_visible(False)
        self._topo = {}
        self._local = None
        self.viewer.render()

    def set_radius(self, r):
        self.radius = float(np.clip(r, 0.2, 30.0))
        self.on_change()

    def set_strength(self, s):
        self.strength = float(np.clip(s, 0.02, 1.0))
        self.on_change()

    # ------------------------------------------------------------ helpers
    def _topology(self, mesh: Mesh, need_w: bool):
        key = id(mesh.faces)
        t = self._topo.get(key)
        if t is None:
            f = mesh.faces
            m, n = len(f), mesh.n_vertices
            rows = f.ravel()
            cols = np.repeat(np.arange(m), 3)
            inc = coo_matrix((np.ones(3 * m, np.float32), (rows, cols)), shape=(n, m)).tocsr()
            t = {"inc": inc, "w": None, "faces": f}
            self._topo = {key: t}  # keep one
        if need_w and t["w"] is None:
            a = vertex_adjacency(mesh.n_vertices, mesh.faces)
            deg = np.asarray(a.sum(1)).ravel()
            deg[deg == 0] = 1
            t["w"] = (diags(1.0 / deg) @ a).tocsr()
        return t

    def _pick(self, wv, faces, x, y):
        """Ray pick that only tests the triangles near the last hit (fast on large scans)."""
        ren = self.viewer.ren
        loc = getattr(self, "_local", None)
        hit = None
        if loc is not None and loc["faces_id"] == id(faces):
            hit = pick_mesh(ren, x, y, wv[loc["vidx"]], loc["faces"])
            if hit is not None and np.linalg.norm(hit[1] - loc["center"]) < loc["reach"] - self.radius:
                return hit
        if hit is None:  # outside the local patch: test the whole model
            hit = pick_mesh(ren, x, y, wv, faces)
            if hit is None:
                return None
        # (re)centre the local patch on the hit
        c = hit[1]
        reach = max(2.5 * self.radius, 2.0)
        near_v = np.einsum("ij,ij->i", wv - c, wv - c) < reach ** 2
        sub = faces[near_v[faces[:, 0]] | near_v[faces[:, 1]] | near_v[faces[:, 2]]]
        vidx, inv = np.unique(sub.ravel(), return_inverse=True)  # compact: only nearby vertices
        self._local = {"faces_id": id(faces), "vidx": vidx, "faces": inv.reshape(-1, 3), "center": c,
                       "reach": reach}
        return hit

    def _begin(self, model):
        self.push_undo(model)
        model.begin_vertex_edit()
        mesh = model.mesh
        wv = transform_points(model.matrix, mesh.vertices)
        tri = wv[mesh.faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        topo = self._topology(mesh, self.mode == SMOOTH)
        self._s = {"model": model, "wv": wv, "fn": fn, "topo": topo,
                   "Minv": np.linalg.inv(model.matrix), "R": model.matrix[:3, :3].copy()}

    def _end(self):
        if self._s is not None:
            self._s["model"].end_vertex_edit()
            self._s = None
            self.on_change()

    def _pixel_radius(self, center):
        cam = self.viewer.ren.GetActiveCamera()
        up = np.array(cam.GetViewUp(), float)
        up /= np.linalg.norm(up)
        p = project(self.viewer.ren, np.array([center, center + up * self.radius]))[:, :2]
        return float(np.linalg.norm(p[1] - p[0]))

    def _draw_cursor(self, x, y, center=None):
        r = self._pixel_radius(center) if center is not None else getattr(self, "_last_px", 25.0)
        self._last_px = r
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        self.cursor.set(np.stack([x + r * np.cos(t), y + r * np.sin(t)], 1), closed=True)

    def _apply(self, x, y):
        s = self._s
        model = s["model"]
        wv, faces = s["wv"], model.mesh.faces
        hit = self._pick(wv, faces, x, y)
        if hit is None:
            return None
        c = hit[1]
        d2 = np.einsum("ij,ij->i", wv - c, wv - c)
        idx = np.nonzero(d2 < self.radius ** 2)[0]
        if len(idx) == 0:
            return c
        fall = (1.0 - d2[idx] / self.radius ** 2) ** 2
        inc = s["topo"]["inc"]
        if self.mode == SMOOTH:
            target = s["topo"]["w"][idx] @ wv
            wv[idx] += (self.strength * fall)[:, None] * (target - wv[idx])
        else:
            n = inc[idx] @ s["fn"]
            n = (n * fall[:, None]).sum(0)
            ln = np.linalg.norm(n)
            if ln > 0:
                n /= ln
                sign = 1.0 if self.mode == ADD else -1.0
                step = sign * self.strength * self.radius * 0.06
                wv[idx] += (step * fall)[:, None] * n
        # refresh normals around the edited vertices
        touched = np.unique(inc[idx].indices)
        tri = wv[faces[touched]]
        s["fn"][touched] = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ring = np.unique(faces[touched].ravel())
        vn = inc[ring] @ s["fn"]
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        ln[ln == 0] = 1.0
        vn = vn / ln
        local = transform_points(s["Minv"], wv[idx])
        model.set_vertices(idx, local, ring, vn @ s["R"])
        return c

    # ------------------------------------------------------------ events
    def on_press(self, x, y, ctrl, shift, repeat):
        model = self.get_model()
        if model is None or not model.visible:
            return False
        if self._pick(model.world_geometry()[0], model.mesh.faces, x, y) is None:
            return False  # clicked beside the model: rotate the camera
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._begin(model)
        finally:
            QApplication.restoreOverrideCursor()
        c = self._apply(x, y)
        self._draw_cursor(x, y, c)
        self.viewer.render()
        return True

    def on_move(self, x, y, ctrl, shift):
        self._mouse = (x, y)
        if self._s is not None:
            c = self._apply(x, y)
            self._draw_cursor(x, y, c)
            self.viewer.render()
            return True
        model = self.get_model()
        c = None
        if model is not None and model.visible:
            loc = getattr(self, "_local", None)
            if loc is not None and loc["faces_id"] == id(model.mesh.faces):  # cheap: near the last hit only
                hit = pick_mesh(self.viewer.ren, x, y, model.world_geometry()[0][loc["vidx"]], loc["faces"])
                c = hit[1] if hit else None
        self._draw_cursor(x, y, c)
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
        if self._mouse:
            self._last_px = getattr(self, "_last_px", 25.0) * (1.12 if shift and direction > 0 else
                                                              1 / 1.12 if shift else 1.0)
            self._draw_cursor(*self._mouse)
            self.viewer.render()
        return True


class RepairPanel(QWidget):
    def __init__(self, sculpt: SculptTool, get_models, push_undo, on_change, status, settings=None, parent=None):
        super().__init__(parent)
        self.sculpt = sculpt
        self.get_models = get_models
        self.push_undo = push_undo
        self.on_change = on_change
        self.status = status
        self.settings = settings
        lay = QVBoxLayout(self)
        self.model_lbl = QLabel("")
        self.model_lbl.setWordWrap(True)
        self.model_lbl.setStyleSheet("font-weight:bold;")
        lay.addWidget(self.model_lbl)

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

        # ---------------- sculpt
        box = QGroupBox(tr("sculpt_title"))
        bl = QVBoxLayout(box)
        self.group = QButtonGroup(self)
        self.mode_buttons = {}
        for mode, key in ((ADD, "sculpt_add"), (REMOVE, "sculpt_remove"), (SMOOTH, "sculpt_smooth")):
            rb = QRadioButton(tr(key))
            self.group.addButton(rb)
            self.mode_buttons[mode] = rb
            rb.toggled.connect(lambda on, m=mode: on and setattr(self.sculpt, "mode", m))
            bl.addWidget(rb)
        self.mode_buttons[sculpt.mode].setChecked(True)
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
        help_lbl = QLabel(tr("sculpt_help"))
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: gray; font-size: 11px;")
        bl.addWidget(help_lbl)
        lay.addWidget(box)
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        lay.addWidget(self.result_lbl)
        lay.addStretch(1)
        self.sync()

    def sync(self):
        """Show the tool's size / strength (they also change with Shift / Ctrl + wheel)."""
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
