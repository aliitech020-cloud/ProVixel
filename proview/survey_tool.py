"""Path of insertion tool: choose the insertion axis, see it as an arrow and colour the undercuts."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .i18n import tr
from .model_builder import boundary_loops, detect_up
from .orient_tool import _arrow_actor, rotation_about
from .survey import COLORS, LEVELS, depth_colors, undercut_depth

ARROW_COLOR = (0.2, 0.85, 1.0)


class SurveyController:
    def __init__(self, viewer, get_models, on_change):
        self.viewer = viewer
        self.get_models = get_models
        self.on_change = on_change
        self.active = False
        self.direction = None  # unit vector, out of the model (towards where the restoration comes from)
        self.show_arrow = True
        self.models = []
        self.stats = None  # (area mm2 per class, max depth)
        self._arrows = []

    # ------------------------------------------------------------ lifecycle
    def activate(self):
        self.active = True
        if self.direction is None:
            self.auto_direction(update=False)
        self.update()

    def deactivate(self):
        self.active = False
        self._clear()
        self.viewer.render()

    def _clear(self):
        for m in self.models:
            if m in self.viewer.models:
                m.set_overlay(None)
        self.models = []
        for a in self._arrows:
            self.viewer.ren.RemoveActor(a)
        self._arrows = []

    # ------------------------------------------------------------ direction
    def set_from_view(self):
        """Insertion axis = the current viewing direction (you look along the path)."""
        cam = self.viewer.ren.GetActiveCamera()
        self.direction = -np.array(cam.GetDirectionOfProjection())
        self.update()

    def auto_direction(self, update=True):
        """Perpendicular to the occlusal plane of the first selected model (or the view direction)."""
        d = None
        for m in self.get_models():
            try:
                wv = m.world_geometry()[0]
                loops = boundary_loops(m.mesh.faces)
                if loops:
                    d = detect_up(wv, max(loops, key=len), m.mesh.faces)
                    break
            except Exception:  # noqa: BLE001
                d = None
        if d is None:
            d = -np.array(self.viewer.ren.GetActiveCamera().GetDirectionOfProjection())
        self.direction = d / np.linalg.norm(d)
        if update:
            self.update()

    def tilt(self, axis_name: str, degrees: float):
        """Tilt the axis around the screen's horizontal ('x') or vertical ('y') direction."""
        if self.direction is None:
            return
        cam = self.viewer.ren.GetActiveCamera()
        up = np.array(cam.GetViewUp(), float)
        dop = np.array(cam.GetDirectionOfProjection(), float)
        right = np.cross(dop, up)
        axis = right if axis_name == "x" else up
        R = rotation_about(axis, np.radians(degrees), np.zeros(3))[:3, :3]
        self.direction = R @ self.direction
        self.direction /= np.linalg.norm(self.direction)
        self.update()

    def look_along(self):
        if self.direction is None:
            return
        ren = self.viewer.ren
        cam = ren.GetActiveCamera()
        fp = np.array(cam.GetFocalPoint())
        dist = cam.GetDistance()
        d = self.direction
        up = np.array(cam.GetViewUp(), float)
        up = up - d * (up @ d)
        if np.linalg.norm(up) < 1e-6:
            up = np.cross(d, [1.0, 0.0, 0.0])
        cam.SetPosition(*(fp + d * dist))
        cam.SetViewUp(*(up / np.linalg.norm(up)))
        self.viewer.render()

    # ------------------------------------------------------------ analysis
    def update(self):
        self._clear()
        if not self.active or self.direction is None:
            self.viewer.render()
            self.on_change()
            return
        models = [m for m in self.get_models() if m.visible and m.mesh.n_faces]
        area = np.zeros(len(LEVELS) + 1)
        max_depth = 0.0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for m in models:
                wv = m.world_geometry()[0]
                depth = undercut_depth(wv, m.mesh.faces, self.direction)
                m.set_overlay(depth_colors(depth, (np.array(m.color) * 255).round()))
                tri = wv[m.mesh.faces]
                fa = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                under = depth > 0
                cls = np.searchsorted(LEVELS, depth[under], side="right")
                area += np.bincount(cls, weights=fa[under], minlength=len(LEVELS) + 1)
                if under.any():
                    max_depth = max(max_depth, float(np.percentile(depth[under], 99.5)))
        finally:
            QApplication.restoreOverrideCursor()
        self.models = models
        self.stats = (area, max_depth)
        if self.show_arrow and models:
            lo = np.min([m.world_bounds()[0] for m in models], axis=0)
            hi = np.max([m.world_bounds()[1] for m in models], axis=0)
            c = (lo + hi) / 2
            size = float(np.linalg.norm(hi - lo))
            d = self.direction
            # arrow comes down onto the model along the path of insertion
            start = c + d * size * 0.95
            arrow = _arrow_actor(start, -d, size * 0.5, ARROW_COLOR)
            self._arrows = [arrow]
            self.viewer.ren.AddActor(arrow)
        self.viewer.render()
        self.on_change()

    def set_show_arrow(self, on):
        self.show_arrow = on
        self.update()


class SurveyPanel(QWidget):
    def __init__(self, ctrl: SurveyController, parent=None):
        super().__init__(parent)
        self.c = ctrl
        lay = QVBoxLayout(self)
        self.model_lbl = QLabel("")
        self.model_lbl.setWordWrap(True)
        self.model_lbl.setStyleSheet("font-weight:bold;")
        lay.addWidget(self.model_lbl)
        help_lbl = QLabel(tr("survey_help"))
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: gray; font-size: 11px;")
        lay.addWidget(help_lbl)

        b = QPushButton(tr("survey_from_view"))
        b.setStyleSheet("QPushButton { background:#2c6ed5; color:white; font-weight:bold; padding:8px; }")
        b.setToolTip(tr("survey_from_view_tip"))
        b.clicked.connect(lambda _=False: self.c.set_from_view())
        lay.addWidget(b)
        row = QHBoxLayout()
        b = QPushButton(tr("survey_auto"))
        b.setToolTip(tr("survey_auto_tip"))
        b.clicked.connect(lambda _=False: self.c.auto_direction())
        row.addWidget(b)
        b = QPushButton(tr("survey_look"))
        b.clicked.connect(lambda _=False: self.c.look_along())
        row.addWidget(b)
        lay.addLayout(row)

        grid = QGridLayout()
        grid.addWidget(QLabel(tr("survey_tilt")), 0, 0)
        self.step = QSpinBox()
        self.step.setRange(1, 30)
        self.step.setValue(2)
        self.step.setSuffix("°")
        grid.addWidget(self.step, 0, 1)
        for col, (text, axis, sign) in enumerate((("←", "y", -1), ("→", "y", 1), ("↑", "x", -1), ("↓", "x", 1))):
            b = QPushButton(text)
            b.setFixedWidth(40)
            b.clicked.connect(lambda _=False, a=axis, s=sign: self.c.tilt(a, s * self.step.value()))
            grid.addWidget(b, 0, col + 2)
        lay.addLayout(grid)

        self.arrow_cb = QCheckBox(tr("survey_arrow"))
        self.arrow_cb.setChecked(True)
        self.arrow_cb.toggled.connect(lambda on: self.c.set_show_arrow(on))
        lay.addWidget(self.arrow_cb)

        lay.addWidget(QLabel(tr("survey_legend")))
        self.legend = []
        names = [f"0 – {LEVELS[0]:.2f} mm"] + [f"{LEVELS[i]:.2f} – {LEVELS[i + 1]:.2f} mm" for i in
                                              range(len(LEVELS) - 1)] + [f"> {LEVELS[-1]:.2f} mm"]
        for name, rgb in zip(names, COLORS):
            row = QHBoxLayout()
            sw = QLabel()
            sw.setFixedSize(18, 14)
            sw.setStyleSheet(f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border:1px solid #111;")
            row.addWidget(sw)
            row.addWidget(QLabel(name))
            val = QLabel("")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setStyleSheet("color:#aab0bb;")
            row.addWidget(val, 1)
            self.legend.append(val)
            lay.addLayout(row)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet("font-weight:bold;")
        lay.addWidget(self.stats)
        lay.addStretch(1)

    def refresh(self):
        models = self.c.get_models()
        self.model_lbl.setText(tr("orient_model", name=", ".join(m.name for m in models)) if models
                               else tr("select_model_first"))
        if self.c.stats is None or not self.c.models:
            for v in self.legend:
                v.setText("")
            self.stats.setText("")
            return
        area, max_depth = self.c.stats
        for v, a in zip(self.legend, area):
            v.setText(f"{a:,.1f} mm²")
        self.stats.setText(tr("survey_stats", depth=max_depth) if area.sum() > 0 else tr("survey_none"))
