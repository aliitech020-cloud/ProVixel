"""Panel for the model builder (scan -> printable, optionally hollow honeycomb model) and scan repair."""
from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .i18n import tr
from .base_planes import BasePlanes, JawCache, make_jaws
from .model_builder import BuildError, BuildOptions, boundary_loops, build_models
from .repair import fill_holes, loop_size

QUALITY = [("quality_fine", 0.15), ("quality_normal", 0.2), ("quality_fast", 0.3)]

# defaults (the values the lab uses); the last used values are remembered
DEFAULTS = dict(hole_max=10.0, auto=True, trim=True, trim_offset=0.0, base=3.0, edge=1.0, hollow=True,
                wall=2.5, inner=5.0, honey=False, cell=6.0, rib=1.0, quality=0, live=True,
                text_size=4.0, text_depth=0.5, text_mode=0)
PREVIEW_VOXEL = 0.3  # the live preview is built at this (faster) detail


class ModelPreview:
    """Live preview: the model is rebuilt in the background (at a faster detail) after each
    change and shown in place of the scan."""

    def __init__(self, viewer):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        self.viewer = viewer
        self.mapper = vtkPolyDataMapper()
        self.actor = vtkActor()
        self.actor.SetMapper(self.mapper)
        p = self.actor.GetProperty()
        p.SetColor(0.93, 0.82, 0.48)
        p.SetSpecular(0.15)
        p.SetSpecularPower(20)
        self.actor.PickableOff()
        self.shown = False
        self.hidden_models = []
        self.gen = 0
        self.running = False
        self.pending = None
        self.result = None

    def request(self, models, opts, on_done):
        """Start (or queue) a background build; ``opts``: one BuildOptions per model.
        on_done(text) is called on the GUI thread."""
        self.gen += 1
        scans = [m.mesh.transformed(m.matrix) for m in models]
        job = (self.gen, models, scans, opts, on_done)
        if self.running:
            self.pending = job
            return
        self._start(job)

    def _start(self, job):
        gen, models, scans, opts, on_done = job
        self.running = True
        self.result = None

        def work():
            try:
                out = []
                for scan, o in zip(scans, opts):
                    out += build_models(scan, o)
                self.result = (gen, models, out, None)
            except Exception as exc:  # noqa: BLE001
                self.result = (gen, models, None, exc)

        threading.Thread(target=work, daemon=True).start()
        self._poll(on_done)

    def _poll(self, on_done):
        if self.result is None:
            QTimer.singleShot(80, lambda: self._poll(on_done))
            return
        gen, models, meshes, exc = self.result
        self.running = False
        if self.pending is not None:  # parameters changed meanwhile: build again
            job, self.pending = self.pending, None
            self._start(job)
            return
        if gen != self.gen:  # cancelled
            return
        if exc is not None or not meshes:
            self.hide()
            on_done(str(exc) if isinstance(exc, BuildError) else (f"{type(exc).__name__}: {exc}" if exc else ""))
            return
        from .mesh import merge_meshes, to_polydata

        self.mapper.SetInputData(to_polydata(merge_meshes(meshes)))
        self.mapper.Modified()
        self._restore_models()
        for m in models:
            if m.actor.GetVisibility():
                m.actor.SetVisibility(False)
                self.hidden_models.append(m)
        if not self.shown:
            self.viewer.ren.AddActor(self.actor)
            self.shown = True
        self.viewer.render()
        on_done(None)

    def _restore_models(self):
        for m in self.hidden_models:
            m.actor.SetVisibility(m.visible)
        self.hidden_models = []

    def cancel(self):
        self.gen += 1
        self.pending = None
        self.hide()

    def hide(self):
        self._restore_models()
        if self.shown:
            self.viewer.ren.RemoveActor(self.actor)
            self.shown = False
        self.viewer.render()


class TextPlacer:
    """Viewer tool: shows the text outline under the cursor; a click puts it on the model."""

    def __init__(self, panel):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        self.panel = panel
        self.viewer = panel.viewer
        self.mapper = vtkPolyDataMapper()
        self.actor = vtkActor()
        self.actor.SetMapper(self.mapper)
        p = self.actor.GetProperty()
        p.SetColor(1.0, 0.45, 0.1)
        p.SetLineWidth(2)
        p.SetLighting(False)
        self.actor.PickableOff()
        self.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-4, -4)
        self.shown = False
        self._outline_key = None
        self._outlines = []
        self.hit = None

    def _frame_at(self, x, y):
        from .selection import pick_mesh

        best = None
        for m in self.viewer.models:
            if not m.visible or not m.actor.GetVisibility():  # hidden, or replaced by the preview
                continue
            wv, _, fn = m.world_geometry()
            h = pick_mesh(self.viewer.ren, x, y, wv, m.mesh.faces)
            if h is not None and (best is None or h[0] < best[0]):
                best = (h[0], h[1], fn[h[2]], m)
        if best is None:
            return None
        cam = self.viewer.ren.GetActiveCamera()
        n = np.asarray(best[2], float)
        if n @ (np.array(cam.GetPosition(), float) - best[1]) < 0:  # scans can face inwards: towards the viewer
            n = -n
        return best[1], n, np.array(cam.GetViewUp(), float), best[3]

    def _draw(self, point, normal, up, model=None):
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

        from .text_label import text_frame, text_outlines

        key = (self.panel.text.text(), self.panel.text_size.value())
        if key != self._outline_key:
            self._outline_key = key
            try:
                self._outlines = text_outlines(*key)
            except Exception:  # noqa: BLE001
                self._outlines = []
        if not self._outlines:
            self.hide()
            return
        right, up, n = text_frame(normal, up)
        surf = None
        if model is not None:  # lay the outline on the surface (it is curved): height along n of
            # the front-facing surface nearest in the text plane
            from scipy.spatial import cKDTree

            wv, _, fn = model.world_geometry()
            ext = max(max(np.abs(o).max() for o in self._outlines) + 1.0, 2.0)
            rel = wv - point
            near = np.abs(rel @ right) < ext
            near &= np.abs(rel @ up) < ext
            near &= np.abs(rel @ n) < 5.0
            f = model.mesh.faces
            front = near[f].any(1) & (np.abs(fn @ n) > 0.15)
            vid = np.unique(f[front])
            if len(vid) >= 3:
                surf = (cKDTree(np.stack([rel[vid] @ right, rel[vid] @ up], 1)), rel[vid] @ n)
        pts, cells, k = [], [], 0
        for poly in self._outlines:
            q = point + poly[:, :1] * right + poly[:, 1:] * up
            if surf is not None:
                _, idx = surf[0].query(poly)
                q = q + surf[1][idx][:, None] * n
            q = q + 0.08 * n
            q = np.vstack([q, q[:1]])
            pts.append(q)
            cells.append(np.concatenate([[len(q)], np.arange(k, k + len(q))]))
            k += len(q)
        vp = vtkPoints()
        vp.SetData(numpy_to_vtk(np.vstack(pts), deep=True))
        ca = vtkCellArray()
        ca.SetCells(len(cells), numpy_to_vtkIdTypeArray(np.concatenate(cells).astype(np.int64), deep=True))
        pd = vtkPolyData()
        pd.SetPoints(vp)
        pd.SetLines(ca)
        self.mapper.SetInputData(pd)
        if not self.shown:
            self.viewer.ren.AddActor(self.actor)
            self.shown = True
        self.viewer.render()

    def hide(self):
        if self.shown:
            self.viewer.ren.RemoveActor(self.actor)
            self.shown = False
            self.viewer.render()

    # viewer tool interface
    def on_move(self, x, y, ctrl, shift):
        self.hit = self._frame_at(x, y)
        if self.hit is None:
            self.hide()
            return False
        self._draw(*self.hit)
        return False

    def on_press(self, x, y, ctrl, shift, rep):
        hit = self._frame_at(x, y)
        if hit is None:
            return False
        self.panel.place_text(*hit)
        return True

    def on_release(self, x, y, ctrl, shift):
        return False

    def on_wheel(self, direction):
        return False


def _spin(value, lo, hi, step, suffix=" mm", decimals=1):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(value)
    sb.setSuffix(suffix)
    return sb


class BuilderPanel(QWidget):
    def __init__(self, get_models, add_result, status, push_undo, on_change, viewer=None, settings=None,
                 parent=None):
        super().__init__(parent)
        self.settings = settings
        self.model_preview = ModelPreview(viewer) if viewer is not None else None
        self.planes = (BasePlanes(viewer, on_drag=lambda j: self._plane_dragged(j),
                                  on_release=lambda j: self._plane_released(j)) if viewer is not None else None)
        self.jaw_cache = JawCache()
        self.jaws = []
        self.bases = {}  # id(model) -> base height set by dragging its plane
        self.viewer = viewer
        self.view_up = None  # teeth direction taken from the view (world), None = auto
        self.placer = None
        self.active = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(lambda: self.update_preview())
        self.get_models = get_models  # selected models (or the active one)
        self.add_result = add_result
        self.status = status
        self.push_undo = push_undo
        self.on_change = on_change
        lay = QVBoxLayout(self)
        self.model_lbl = QLabel("")
        self.model_lbl.setWordWrap(True)
        self.model_lbl.setStyleSheet("font-weight:bold;")
        lay.addWidget(self.model_lbl)

        # ---------------- builder
        box = QGroupBox(tr("builder_title"))
        bl = QVBoxLayout(box)
        help_lbl = QLabel(tr("builder_help"))
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: gray; font-size: 11px;")
        bl.addWidget(help_lbl)
        self.live = QCheckBox(tr("builder_live"))
        self.live.setToolTip(tr("builder_live_tip"))
        self.live.setChecked(self._saved("live"))
        self.live.toggled.connect(lambda on: self._live_toggled(on))
        bl.addWidget(self.live)
        form = QFormLayout()
        self.auto = QCheckBox(tr("builder_auto"))
        self.auto.setChecked(self._saved("auto"))
        form.addRow(self.auto)
        row = QHBoxLayout()
        self.from_view = QPushButton(tr("builder_from_view"))
        self.from_view.setToolTip(tr("builder_from_view_tip"))
        self.from_view.clicked.connect(lambda _=False: self.set_from_view())
        row.addWidget(self.from_view)
        self.dir_lbl = QLabel("")
        self.dir_lbl.setStyleSheet("color: #9aa0a6;")
        row.addWidget(self.dir_lbl, 1)
        form.addRow(row)
        self.trim = QCheckBox(tr("builder_trim"))
        self.trim.setChecked(self._saved("trim"))
        self.trim.setToolTip(tr("builder_trim_tip"))
        form.addRow(self.trim)
        self.trim_off = _spin(self._saved("trim_offset"), -20, 30, 0.5)
        self.trim_off.setToolTip(tr("builder_trim_tip"))
        form.addRow(tr("builder_trim_level"), self.trim_off)
        self.trim.toggled.connect(lambda on: (self.trim_off.setEnabled(on), self._changed()))
        self.trim_off.setEnabled(self.trim.isChecked())
        self.trim_off.valueChanged.connect(lambda _v: self._changed())
        self.auto.toggled.connect(lambda on: self._auto_toggled(on))
        self.base = _spin(self._saved("base"), 0.5, 40, 0.5)
        form.addRow(tr("builder_base"), self.base)
        self.edge = _spin(self._saved("edge"), 0.0, 10, 0.5)
        form.addRow(tr("builder_edge"), self.edge)
        self.hollow = QCheckBox(tr("builder_hollow"))
        self.hollow.setChecked(self._saved("hollow"))
        form.addRow(self.hollow)
        self.wall = _spin(self._saved("wall"), 0.8, 10, 0.1)
        form.addRow(tr("builder_wall"), self.wall)
        self.inner = _spin(self._saved("inner"), 0.0, 15, 0.5)
        form.addRow(tr("builder_inner"), self.inner)
        self.honey = QCheckBox(tr("builder_honeycomb"))
        self.honey.setChecked(self._saved("honey"))
        form.addRow(self.honey)
        self.cell = _spin(self._saved("cell"), 2, 30, 0.5)
        form.addRow(tr("builder_cell"), self.cell)
        self.rib = _spin(self._saved("rib"), 0.4, 5, 0.1)
        form.addRow(tr("builder_rib"), self.rib)
        self.quality = QComboBox()
        for key, h in QUALITY:
            self.quality.addItem(f"{tr(key)}  ({h:g} mm)", h)
        self.quality.setCurrentIndex(int(self._saved("quality")))
        form.addRow(tr("builder_quality"), self.quality)
        bl.addLayout(form)
        self.hollow.toggled.connect(lambda _on: (self._enable(), self._changed()))
        self.honey.toggled.connect(lambda _on: (self._enable(), self._changed()))
        self.base.setToolTip(tr("builder_base_tip"))
        self.base.valueChanged.connect(lambda v: self._base_typed(v))
        for sb in (self.edge, self.wall, self.inner, self.cell, self.rib):
            sb.valueChanged.connect(lambda _v: self._changed())
        self.quality.currentIndexChanged.connect(lambda _i: self._changed())
        self._enable()
        self._update_dir_label()

        self.go = QPushButton(tr("builder_go"))
        self.go.setStyleSheet("QPushButton { background:#2c6ed5; color:white; font-weight:bold; padding:8px; }")
        self.go.clicked.connect(lambda _=False: self.build())
        bl.addWidget(self.go)
        lay.addWidget(box)

        # ---------------- text label
        tbox = QGroupBox(tr("text_title"))
        tl = QVBoxLayout(tbox)
        tform = QFormLayout()
        self.text = QLineEdit()
        self.text.setPlaceholderText(tr("text_placeholder"))
        if self.settings is not None:
            self.text.setText(str(self.settings.value("builder/text", "")))
        tform.addRow(tr("text_text"), self.text)
        self.text_size = _spin(self._saved("text_size"), 1.0, 20.0, 0.5)
        tform.addRow(tr("text_size"), self.text_size)
        self.text_mode = QComboBox()
        self.text_mode.addItem(tr("text_engrave"), -1)
        self.text_mode.addItem(tr("text_emboss"), 1)
        self.text_mode.setCurrentIndex(int(self._saved("text_mode")))
        tform.addRow(tr("text_mode"), self.text_mode)
        self.text_depth = _spin(self._saved("text_depth"), 0.1, 3.0, 0.1)
        tform.addRow(tr("text_depth"), self.text_depth)
        tl.addLayout(tform)
        self.place_btn = QPushButton(tr("text_place"))
        self.place_btn.setCheckable(True)
        self.place_btn.setToolTip(tr("text_place_tip"))
        self.place_btn.toggled.connect(lambda on: self._placing(on))
        tl.addWidget(self.place_btn)
        thelp = QLabel(tr("text_help"))
        thelp.setWordWrap(True)
        thelp.setStyleSheet("color: gray; font-size: 11px;")
        tl.addWidget(thelp)
        lay.addWidget(tbox)

        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        lay.addWidget(self.result_lbl)
        lay.addStretch(1)

    # ------------------------------------------------------------------ settings
    def _saved(self, key):
        default = DEFAULTS[key]
        if self.settings is None:
            return default
        value = self.settings.value(f"builder/{key}", default)
        if isinstance(default, bool):
            return value in (True, "true", "1", 1)
        if isinstance(default, int):
            return int(value)
        return float(value)

    def _save(self):
        if self.settings is None:
            return
        values = dict(auto=self.auto.isChecked(), trim=self.trim.isChecked(),
                      trim_offset=self.trim_off.value(), base=self.base.value(), edge=self.edge.value(),
                      hollow=self.hollow.isChecked(), wall=self.wall.value(), inner=self.inner.value(),
                      honey=self.honey.isChecked(), cell=self.cell.value(), rib=self.rib.value(),
                      quality=self.quality.currentIndex(), live=self.live.isChecked(),
                      text_size=self.text_size.value(), text_depth=self.text_depth.value(),
                      text_mode=self.text_mode.currentIndex())
        for k, v in values.items():
            self.settings.setValue(f"builder/{k}", v)

    # ------------------------------------------------------------------ preview
    def set_active(self, on):
        self.active = on
        if on:
            self._use_planes_tool()
            self._schedule_preview()
        else:
            self.place_btn.setChecked(False)
            if self.planes is not None:
                self.planes.hide()
                if self.viewer.style.tool is self.planes:
                    self.viewer.style.tool = None
            if self.model_preview is not None:
                self.model_preview.cancel()

    def _schedule_preview(self, force=False):
        if self.active and (force or not getattr(self, "_just_built", False)):
            self._preview_timer.start(250)

    def _changed(self):
        self._just_built = False
        self._schedule_preview(force=True)

    def selection_changed(self):
        """The user picked other models: show the trim preview again."""
        self._just_built = False
        self.refresh()

    def update_preview(self):
        if self.planes is None or not self.active:
            return
        models = self.get_models()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.jaws = make_jaws(models, self.jaw_cache, self.auto.isChecked(), self.view_up,
                                  self.trim.isChecked(), self.trim_off.value(), self.bases, self.base.value())
        except Exception:  # noqa: BLE001
            import traceback

            from .diagnostics import log

            log("base planes failed\n" + traceback.format_exc())
            self.jaws = []
        finally:
            QApplication.restoreOverrideCursor()
        self.planes.show_bases = not (self.model_preview is not None and self.model_preview.shown)
        if self.jaws:
            self.planes.show(self.jaws, self.planes.show_bases)
        else:
            self.planes.hide()
        self._update_model_preview(models)

    def _use_planes_tool(self):
        if self.planes is not None and self.viewer is not None and not self.place_btn.isChecked():
            self.viewer.style.tool = self.planes

    def _base_typed(self, value):
        for j in self.jaws:
            self.bases[id(j.model)] = value
        self._changed()

    def _plane_dragged(self, jaw):
        self.bases[id(jaw.model)] = jaw.base
        self.base.blockSignals(True)
        self.base.setValue(jaw.base)
        self.base.blockSignals(False)
        if self.model_preview is not None and self.model_preview.shown:
            self.model_preview.cancel()  # show the scan and the transparent base while dragging
        self.planes.show_bases = True

    def _plane_released(self, jaw):
        self._plane_dragged(jaw)
        self._changed()

    def options_for(self, model) -> BuildOptions:
        """Build options of one model: its base ends on its (dragged) base plane."""
        opts = self.options()
        for j in self.jaws:
            if j.model is model:
                opts.up = tuple(float(x) for x in j.up)
                opts.base_height = j.base
                opts.base_level = j.level
                break
        return opts

    def _update_model_preview(self, models):
        mp = self.model_preview
        if mp is None:
            return
        models = [m for m in models if self._is_open(m)]  # built (closed) models have nothing to preview
        if not (self.live.isChecked() and models and self.active):
            mp.cancel()
            return
        opts = [self.options_for(m) for m in models]
        for o in opts:
            o.voxel = max(o.voxel, PREVIEW_VOXEL)
        self.result_lbl.setText(tr("builder_preview_busy"))

        def done(error):
            if error is None:
                self.result_lbl.setText(tr("builder_preview_ready"))
                if self.planes is not None and self.planes.jaws:
                    self.planes.show_bases = False  # the previewed model shows the base itself
                    self.planes.show(self.planes.jaws, False)
            else:
                self.result_lbl.setText(tr("builder_preview_failed", err=error) if error else "")

        mp.request(models, opts, done)

    _open_cache: dict = {}

    def _is_open(self, model) -> bool:
        key = id(model.mesh)
        if key not in self._open_cache:
            if len(self._open_cache) > 16:
                self._open_cache.clear()
            self._open_cache[key] = bool(boundary_loops(model.mesh.faces)) if model.mesh.n_faces else False
        return self._open_cache[key]

    def _live_toggled(self, on):
        self._save()
        if on:
            self._changed()
        elif self.model_preview is not None:
            self.model_preview.cancel()
            self.result_lbl.setText("")

    # ------------------------------------------------------------------ direction
    def set_from_view(self):
        """The base is made perpendicular to the view direction: the teeth face the viewer."""
        if self.viewer is None:
            return
        cam = self.viewer.ren.GetActiveCamera()
        d = np.array(cam.GetPosition(), float) - np.array(cam.GetFocalPoint(), float)
        if np.linalg.norm(d) == 0:
            return
        self.view_up = d / np.linalg.norm(d)
        self.auto.blockSignals(True)
        self.auto.setChecked(False)
        self.auto.blockSignals(False)
        self._update_dir_label()
        self._changed()

    def _auto_toggled(self, on):
        if on:
            self.view_up = None
        self._update_dir_label()
        self._changed()

    def _update_dir_label(self):
        if self.view_up is not None:
            self.dir_lbl.setText(tr("builder_dir_view"))
        elif self.auto.isChecked():
            self.dir_lbl.setText(tr("builder_dir_auto"))
        else:
            self.dir_lbl.setText(tr("builder_dir_model"))

    # ------------------------------------------------------------------ text
    def _placing(self, on):
        if self.viewer is None:
            return
        if on:
            if not self.text.text().strip():
                self.place_btn.setChecked(False)
                QMessageBox.information(self, tr("text_title"), tr("text_empty"))
                return
            if self.placer is None:
                self.placer = TextPlacer(self)
            self.viewer.style.tool = self.placer
            self.status(tr("text_place_status"))
        else:
            if self.placer is not None:
                self.placer.hide()
                if self.viewer.style.tool is self.placer:
                    self.viewer.style.tool = None
            if self.active:
                self._use_planes_tool()

    def place_text(self, point, normal, up, model):
        """Put the text on ``model`` at the world point (called by the click tool)."""
        from .mesh import transform_points
        from .text_label import TextError, apply_text

        inv = np.linalg.inv(model.matrix)
        p = transform_points(inv, np.asarray(point, float)[None])[0]
        n = inv[:3, :3] @ np.asarray(normal, float)
        u = inv[:3, :3] @ np.asarray(up, float)
        depth = self.text_depth.value() * float(self.text_mode.currentData())
        if self.settings is not None:
            self.settings.setValue("builder/text", self.text.text())
        self._save()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mesh = apply_text(model.mesh, p, n, u, self.text.text(), self.text_size.value(), depth,
                              progress=lambda t: (self.status(f"{tr('text_title')}: {t}..."),
                                                  QApplication.processEvents()))
        except TextError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, tr("text_title"), str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            import traceback

            from .diagnostics import log

            log("text label failed\n" + traceback.format_exc())
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, tr("text_title"), f"{type(exc).__name__}: {exc}")
            return
        QApplication.restoreOverrideCursor()
        self.push_undo(model)
        model.set_mesh(mesh)
        self.place_btn.setChecked(False)
        self.status(tr("text_done"))
        self.on_change()
        if self.viewer is not None:
            self.viewer.render()

    def _enable(self):
        h = self.hollow.isChecked()
        for w in (self.wall, self.inner, self.honey):
            w.setEnabled(h)
        for w in (self.cell, self.rib):
            w.setEnabled(h and self.honey.isChecked())

    def refresh(self):
        models = self.get_models()
        self.model_lbl.setText(tr("orient_model", name=", ".join(m.name for m in models)) if models
                               else tr("select_model_first"))
        self.go.setEnabled(bool(models))
        self._schedule_preview()

    def options(self) -> BuildOptions:
        return BuildOptions(
            base_height=self.base.value(), hollow=self.hollow.isChecked(), wall=self.wall.value(),
            honeycomb=self.honey.isChecked(), cell=self.cell.value(), rib=self.rib.value(),
            voxel=float(self.quality.currentData()), edge_smooth=self.edge.value(),
            inner_smooth=self.inner.value(), auto_orient=self.auto.isChecked(),
            trim=self.trim.isChecked(), trim_offset=self.trim_off.value(),
            up=tuple(self.view_up) if self.view_up is not None else None)

    # ------------------------------------------------------------------ build
    def build(self):
        models = self.get_models()
        if not models:
            return

        warnings = []

        def progress(text):
            if text.startswith("!"):
                warnings.append(text[1:].strip())
                return
            self.status(f"{tr('builder_title')}: {text}...")
            self.result_lbl.setText(f"{text}...")
            QApplication.processEvents()

        self._save()
        if self.planes is not None:
            self.planes.hide()
        if self.model_preview is not None:
            self.model_preview.cancel()
        per_model = {id(m): self.options_for(m) for m in models}

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.go.setEnabled(False)
        made = 0
        faces = 0
        try:
            for m in models:
                scan = m.mesh.transformed(m.matrix)
                prefix = f"{m.name}: " if len(models) > 1 else ""
                results = build_models(scan, per_model[id(m)], lambda t, p=prefix: progress(p + t))
                for i, r in enumerate(results):
                    self.add_result(m, r, i + 1 if len(results) > 1 else None, made == 0)
                    made += 1
                    faces += r.n_faces
        except BuildError as exc:
            self.result_lbl.setText("")
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, tr("builder_title"), str(exc))
            QApplication.setOverrideCursor(Qt.WaitCursor)
        except MemoryError:
            self.result_lbl.setText("")
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, tr("builder_title"), tr("builder_memory"))
            QApplication.setOverrideCursor(Qt.WaitCursor)
        except Exception as exc:  # noqa: BLE001
            import traceback

            from .diagnostics import log

            log("model builder failed\n" + traceback.format_exc())
            self.result_lbl.setText("")
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, tr("builder_title"), f"{type(exc).__name__}: {exc}")
            QApplication.setOverrideCursor(Qt.WaitCursor)
        finally:
            QApplication.restoreOverrideCursor()
            self.go.setEnabled(True)
        self._just_built = made > 0
        if made:
            text = tr("builder_done", n=made, faces=faces)
            if warnings:
                text += "\n\n⚠ " + "\n⚠ ".join(dict.fromkeys(warnings))
            self.result_lbl.setText(text)
            self.status(tr("builder_done", n=made, faces=faces))
            if warnings:
                QMessageBox.warning(self, tr("builder_title"), "\n".join(dict.fromkeys(warnings)))
        self.refresh()
