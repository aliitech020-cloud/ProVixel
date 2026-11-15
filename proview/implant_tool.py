"""Implant tool: identify the implant system from a scanned scan body (or an implant hole),
show the implant axis and suggest an abutment."""
from __future__ import annotations

import glob
import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .implant_fit import (
    Rules,
    confidence,
    identify,
    marker_shape,
    measure,
    platform_in_library,
    scan_site,
    suggest,
)
from .implant_lib import load_geometry, scan_library


def default_library() -> str:
    for pattern in ("C:/exocad*/DentalCADApp/library/implant", "C:/Program Files/exocad*/DentalCADApp/library/implant",
                    "D:/exocad*/DentalCADApp/library/implant"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    return ""


class _Picker:
    """Viewer tool: a click on a model starts the search."""

    def __init__(self, panel):
        self.panel = panel

    def on_press(self, x, y, ctrl, shift, rep):
        from .selection import pick_mesh

        best = None
        for m in self.panel.viewer.models:
            if not m.visible:
                continue
            h = pick_mesh(self.panel.viewer.ren, x, y, m.world_geometry()[0], m.mesh.faces)
            if h is not None and (best is None or h[0] < best[0]):
                best = (h[0], h[1], m)
        if best is None:
            return False
        self.panel.clicked(best[2], best[1])
        return True

    def on_move(self, x, y, ctrl, shift):
        return False

    def on_release(self, x, y, ctrl, shift):
        return False

    def on_wheel(self, direction):
        return False


class Overlay:
    """Registered scan body, implant axis, occlusal axis and platform in the view."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.actors = []

    def clear(self):
        for a in self.actors:
            self.viewer.ren.RemoveActor(a)
        self.actors = []
        self.viewer.render()

    def _add(self, pd, color, opacity=1.0, width=None):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        mp = vtkPolyDataMapper()
        mp.SetInputData(pd)
        a = vtkActor()
        a.SetMapper(mp)
        p = a.GetProperty()
        p.SetColor(*color)
        p.SetOpacity(opacity)
        if width:
            p.SetLineWidth(width)
            p.SetLighting(False)
            p.SetRenderLinesAsTubes(True)
        a.PickableOff()
        self.viewer.ren.AddActor(a)
        self.actors.append(a)

    def show(self, marker_world=None, platform=None, axis=None, occlusal=None, radius=2.0):
        from .base_planes import _polydata
        from .implant_fit import _perp_basis

        self.clear()
        if marker_world is not None:
            self._add(_polydata(marker_world.vertices, marker_world.faces), (0.35, 0.85, 0.55), 0.45)
        if platform is not None and axis is not None:
            self._add(_polydata(np.array([platform - axis * 6, platform + axis * 16]), lines=[[0, 1]]),
                      (0.35, 1.0, 0.5), width=4)
            t, u = _perp_basis(axis)
            ang = np.linspace(0, 2 * np.pi, 49)
            ring = platform + radius * (np.outer(np.cos(ang), t) + np.outer(np.sin(ang), u))
            self._add(_polydata(ring, lines=[list(range(len(ring)))]), (1.0, 0.85, 0.2), width=3)
            if occlusal is not None:
                self._add(_polydata(np.array([platform, platform + occlusal * 16]), lines=[[0, 1]]),
                          (0.35, 0.6, 1.0), width=3)
        self.viewer.render()


class ImplantPanel(QWidget):
    def __init__(self, viewer, status, settings=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.status = status
        self.settings = settings
        self.entries = None  # library entries, read on first use
        self.index = None
        self.library_root = None
        self.matches = []
        self.site = None
        self.model = None
        self.mode = None
        self.picker = _Picker(self)
        self.overlay = Overlay(viewer)
        self.active = False
        self.occlusal = None
        self.occlusal_override = None  # set from the view by the user

        lay = QVBoxLayout(self)
        # ---- library
        box = QGroupBox("Implant library")
        bl = QVBoxLayout(box)
        self.lib_lbl = QLabel("")
        self.lib_lbl.setWordWrap(True)
        self.lib_lbl.setStyleSheet("color:#8c929c;")
        bl.addWidget(self.lib_lbl)
        row = QHBoxLayout()
        b = QPushButton("Choose folder...")
        b.setToolTip("The folder with the implant libraries, e.g. ...\\DentalCADApp\\library\\implant")
        b.clicked.connect(lambda _=False: self.choose_library())
        row.addWidget(b)
        b = QPushButton("Check library")
        b.clicked.connect(lambda _=False: self.check_library())
        row.addWidget(b)
        bl.addLayout(row)
        lay.addWidget(box)

        # ---- find
        box = QGroupBox("Find the implant")
        fl = QVBoxLayout(box)
        self.btn_body = QPushButton("Click the scan body")
        self.btn_body.setCheckable(True)
        self.btn_body.setProperty("primary", True)
        self.btn_body.toggled.connect(lambda on: self._pick_mode("body" if on else None))
        fl.addWidget(self.btn_body)
        self.btn_hole = QPushButton("Click the implant hole (no scan body)")
        self.btn_hole.setCheckable(True)
        self.btn_hole.toggled.connect(lambda on: self._pick_mode("hole" if on else None))
        fl.addWidget(self.btn_hole)
        self.find_lbl = QLabel("Turn the view so the scan body is well visible, press the button and click "
                               "on the side of the scan body near its top.")
        self.find_lbl.setWordWrap(True)
        self.find_lbl.setStyleSheet("color:#8c929c; font-size:11px;")
        fl.addWidget(self.find_lbl)
        self.results = QListWidget()
        self.results.setMinimumHeight(120)
        self.results.currentRowChanged.connect(lambda r: self._show_match(r))
        fl.addWidget(self.results)
        form = QFormLayout()
        self.entry_combo = QComboBox()
        self.entry_combo.currentIndexChanged.connect(lambda _i: self._entry_changed())
        form.addRow("Library", self.entry_combo)
        self.type_combo = QComboBox()
        self.type_combo.currentIndexChanged.connect(lambda _i: self._update_result())
        form.addRow("Implant", self.type_combo)
        fl.addLayout(form)
        row = QHBoxLayout()
        b = QPushButton("Occlusal axis from view")
        b.setToolTip("Look straight onto the occlusal plane (teeth towards you) and press: the angulation is "
                     "measured to this direction instead of the automatic one.")
        b.clicked.connect(lambda _=False: self.occlusal_from_view())
        row.addWidget(b)
        b = QPushButton("Auto")
        b.setToolTip("Occlusal axis found automatically from the jaw")
        b.clicked.connect(lambda _=False: self.occlusal_auto())
        row.addWidget(b)
        fl.addLayout(row)
        self.occ_lbl = QLabel("Occlusal axis: automatic")
        self.occ_lbl.setStyleSheet("color:#8c929c; font-size:11px;")
        fl.addWidget(self.occ_lbl)
        lay.addWidget(box)

        # ---- result
        box = QGroupBox("Measurements and suggestion")
        rl = QVBoxLayout(box)
        self.result_lbl = QLabel("—")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setTextFormat(Qt.RichText)
        self.result_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rl.addWidget(self.result_lbl)
        lay.addWidget(box)

        # ---- rules
        box = QGroupBox("Rules")
        form = QFormLayout(box)
        r = Rules()
        self.r_straight = self._spin(self._get("straight_max", r.straight_max), 0, 30, 1, " °")
        form.addRow("Straight up to", self.r_straight)
        self.r_angles = QLineEdit(self._get("angled", "15, 25"))
        self.r_angles.setToolTip("Angled abutments that are available, in degrees")
        form.addRow("Angled options", self.r_angles)
        self.r_margin = self._spin(self._get("margin_depth", r.margin_depth), 0, 3, 0.1, " mm")
        form.addRow("Margin below gingiva", self.r_margin)
        self.r_crown = self._spin(self._get("crown_min", r.crown_min), 0.5, 5, 0.1, " mm")
        form.addRow("Room for the crown", self.r_crown)
        for w in (self.r_straight, self.r_margin, self.r_crown):
            w.valueChanged.connect(lambda _v: self._rules_changed())
        self.r_angles.editingFinished.connect(lambda: self._rules_changed())
        lay.addWidget(box)
        note = QLabel("A suggestion to help: the final choice is the technician's / dentist's.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#7d838c; font-size:11px;")
        lay.addWidget(note)
        lay.addStretch(1)
        self._update_lib_label()

    # ------------------------------------------------------------------ settings
    @staticmethod
    def _spin(v, lo, hi, step, suffix):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(1)
        sb.setSuffix(suffix)
        sb.setValue(float(v))
        return sb

    def _get(self, key, default):
        if self.settings is None:
            return default
        v = self.settings.value(f"implant/{key}", default)
        return type(default)(v) if not isinstance(default, str) else str(v)

    def rules(self) -> Rules:
        try:
            angles = tuple(float(x) for x in self.r_angles.text().replace(";", ",").split(",") if x.strip())
        except ValueError:
            angles = (15.0, 25.0)
        return Rules(self.r_straight.value(), angles, self.r_margin.value(), self.r_crown.value())

    def _rules_changed(self):
        if self.settings is not None:
            self.settings.setValue("implant/straight_max", self.r_straight.value())
            self.settings.setValue("implant/angled", self.r_angles.text())
            self.settings.setValue("implant/margin_depth", self.r_margin.value())
            self.settings.setValue("implant/crown_min", self.r_crown.value())
        self._update_result()

    # ------------------------------------------------------------------ library
    def library_path(self) -> str:
        path = str(self.settings.value("implant/library", "")) if self.settings is not None else ""
        return path or default_library()

    def _update_lib_label(self):
        p = self.library_path()
        if not p:
            self.lib_lbl.setText("No library folder chosen yet.")
        elif self.entries is not None and self.library_root == p:
            self.lib_lbl.setText(f"{p}\n{len(self.entries)} readable scan bodies")
        else:
            self.lib_lbl.setText(p)

    def choose_library(self):
        path = QFileDialog.getExistingDirectory(self, "Implant library folder", self.library_path())
        if path:
            if self.settings is not None:
                self.settings.setValue("implant/library", path)
            self.entries = None
            self._update_lib_label()

    def load_library(self, root=None):
        root = root or self.library_path()
        if not root or not os.path.isdir(root):
            return None, []
        if self.entries is not None and self.library_root == root:
            return self.entries, self._report
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.entries, self._report = scan_library(
                root, lambda t: (self.status(t), QApplication.processEvents()))
            self.library_root = root
            from .diagnostics import log_path
            from .implant_fit import library_index

            cache = os.path.join(os.path.dirname(log_path()), "implant_index.json")
            self.index = library_index(self.entries, cache, lambda t: (self.status(t), QApplication.processEvents()))
        finally:
            QApplication.restoreOverrideCursor()
        self._update_lib_label()
        return self.entries, self._report

    def check_library(self):
        entries, report = self.load_library()
        if entries is None:
            QMessageBox.information(self, "Implant library", "Choose the library folder first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Implant library")
        dlg.resize(640, 480)
        v = QVBoxLayout(dlg)
        total = sum(r["entries"] for r in report)
        v.addWidget(QLabel(f"{len(report)} libraries, {total} scan bodies, {len(entries)} readable "
                           f"(used for matching)."))
        t = QTableWidget(len(report), 4)
        t.setHorizontalHeaderLabels(["Library", "Scan bodies", "Readable", "Implant types"])
        for i, r in enumerate(report):
            for k, val in enumerate([r["library"], r["entries"], r["readable"], r["types"]]):
                t.setItem(i, k, QTableWidgetItem(str(val)))
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        t.verticalHeader().setVisible(False)
        v.addWidget(t)
        b = QPushButton("Close")
        b.clicked.connect(lambda _=False: dlg.accept())
        v.addWidget(b)
        dlg.exec()

    # ------------------------------------------------------------------ picking
    def set_active(self, on):
        self.active = on
        if not on:
            self.btn_body.setChecked(False)
            self.btn_hole.setChecked(False)
            self.overlay.clear()

    def _pick_mode(self, mode):
        if mode is not None:
            other = self.btn_hole if mode == "body" else self.btn_body
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
            self.mode = mode
            self.viewer.style.tool = self.picker
            self.status("Click on the scan body" if mode == "body" else "Click inside the implant hole")
        elif not (self.btn_body.isChecked() or self.btn_hole.isChecked()):
            self.mode = None
            if self.viewer.style.tool is self.picker:
                self.viewer.style.tool = None

    def occlusal_from_view(self):
        cam = self.viewer.ren.GetActiveCamera()
        d = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        self.occlusal_override = d / (np.linalg.norm(d) or 1.0)
        self.occ_lbl.setText("Occlusal axis: from the view")
        self.occlusal = self.occlusal_override
        self._update_result()

    def occlusal_auto(self):
        self.occlusal_override = None
        self.occ_lbl.setText("Occlusal axis: automatic")
        if self.model is not None:
            self.occlusal = self._occlusal(self.model)
        self._update_result()

    def _occlusal(self, model):
        if self.occlusal_override is not None:
            return self.occlusal_override
        try:
            from .base_planes import JawCache

            info = JawCache().get(model)
            if info is not None:
                return info[2]
        except Exception:  # noqa: BLE001
            pass
        cam = self.viewer.ren.GetActiveCamera()
        d = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        return d / (np.linalg.norm(d) or 1.0)

    def clicked(self, model, point):
        mode = self.mode
        self.btn_body.setChecked(False)
        self.btn_hole.setChecked(False)
        self.model = model
        self.results.clear()
        self.matches = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            wv = model.world_geometry()[0]
            self.occlusal = self._occlusal(model)
            self.site = scan_site(wv, model.mesh.faces, np.asarray(point, float), hint_up=self.occlusal,
                                  inward=(mode == "hole"))
            if self.site.cyl is None:
                QApplication.restoreOverrideCursor()
                QMessageBox.information(self, "Implant", "No round scan body / hole found there. Click on the "
                                        "side of the scan body near its top (or inside the hole).")
                return
            if mode == "hole":
                self._hole_result()
                return
            entries, _ = self.load_library()
            if not entries:
                QApplication.restoreOverrideCursor()
                QMessageBox.information(self, "Implant", "No readable implant library. Choose the library folder "
                                        "(e.g. ...\\DentalCADApp\\library\\implant) and check it.")
                return
            self.matches = identify(self.site, entries,
                                    lambda t: (self.status(t), QApplication.processEvents()), index=self.index)
        finally:
            QApplication.restoreOverrideCursor()
        self._fill_results()

    def _fill_results(self):
        self.results.blockSignals(True)
        self.results.clear()
        conf = confidence(self.matches)
        if not self.matches:
            self.find_lbl.setText(f"No library scan body fits (scan body Ø {2 * self.site.cyl.radius:.2f} mm). "
                                  "Check the library folder.")
        else:
            head = {"sure": "Identified", "likely": "Probably", "unsure": "Not sure — closest systems"}[conf]
            self.find_lbl.setText(f"{head}. Scan body Ø {2 * self.site.cyl.radius:.2f} mm.")
        for m in self.matches[:6]:
            extra = f"  (+{len(m.entries) - 1} with the same scan body)" if len(m.entries) > 1 else ""
            it = QListWidgetItem(f"{m.score * 100:3.0f}%   {m.entry.title}{extra}\n"
                                 f"        fit {m.rms * 1000:.0f} µm · surface {m.inliers * 100:.0f}% · top "
                                 f"{m.coverage * 100:.0f}%")
            self.results.addItem(it)
        self.results.blockSignals(False)
        if self.matches:
            self.results.setCurrentRow(0)
            self._show_match(0)

    def _show_match(self, row):
        if not (0 <= row < len(self.matches)):
            return
        m = self.matches[row]
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        for e in m.entries:
            self.entry_combo.addItem(f"{e.library}  ({e.manufacturer})" if e.manufacturer else e.library)
        self.entry_combo.blockSignals(False)
        self._entry_changed()

    def _entry_changed(self):
        row = self.results.currentRow()
        if not (0 <= row < len(self.matches)):
            return
        m = self.matches[row]
        e = m.entries[max(self.entry_combo.currentIndex(), 0)]
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for t in e.types:
            self.type_combo.addItem(t.name or "implant")
        self.type_combo.blockSignals(False)
        self._update_result()

    def _update_result(self):
        if self.site is None:
            return
        if self.mode == "hole" and not self.matches:
            self._hole_result()
            return
        row = self.results.currentRow()
        if not (0 <= row < len(self.matches)):
            return
        m = self.matches[row]
        e = m.entries[max(self.entry_combo.currentIndex(), 0)]
        ti = max(self.type_combo.currentIndex(), 0)
        p_lib, a_lib = platform_in_library(e, m.shape, ti)
        platform = m.T[:3, :3] @ p_lib + m.T[:3, 3]
        axis = m.T[:3, :3] @ a_lib
        meas = measure(self.site, platform, axis, self.occlusal, self._opposing(),
                       radius=m.shape.cyl.radius)
        if not e.types or not e.types[min(ti, len(e.types) - 1)].analog:
            meas.notes.append("This library names no implant analog: the implant platform is taken at the library's "
                              "origin. Check that the yellow ring sits on the implant.")
        sg = suggest(meas, self.rules(), e, ti)
        marker = load_geometry(e.marker)
        if marker is not None:
            from .mesh import Mesh

            marker = Mesh(marker.vertices @ m.T[:3, :3].T + m.T[:3, 3], marker.faces)
        self.overlay.show(marker, platform, axis, meas.occlusal, radius=m.shape.cyl.radius)
        self._show_text(meas, sg, e)

    def _hole_result(self):
        cyl = self.site.cyl
        pts = self.site.points
        near = cyl.radial(pts) <= cyl.radius + 0.2
        h_bottom = float(cyl.height(pts[near]).min()) if near.any() else cyl.top
        platform = cyl.centre + cyl.axis * (h_bottom - cyl.centre @ cyl.axis)
        meas = measure(self.site, platform, cyl.axis, self.occlusal, self._opposing(), radius=cyl.radius,
                       approximate=True)
        sg = suggest(meas, self.rules())
        self.mode = "hole"
        self.find_lbl.setText(f"Implant hole Ø {2 * cyl.radius:.2f} mm (approximate; the platform is taken at the "
                              "bottom of the hole).")
        self.overlay.show(None, platform, cyl.axis, meas.occlusal, radius=cyl.radius)
        self._show_text(meas, sg, None)

    def _opposing(self):
        out = []
        for mm in self.viewer.models:
            if mm is self.model or not mm.visible:
                continue
            out.append((mm.world_geometry()[0], mm.mesh.faces))
        return out

    def _show_text(self, meas, sg, entry):
        def fmt(v, unit=" mm"):
            return "—" if v is None else f"{v:.1f}{unit}"

        rows = [
            ("Angulation", fmt(meas.angulation, "°")),
            ("Gingival height", "—" if meas.gh_min is None else
             f"{meas.gh_min:.1f} – {meas.gh_max:.1f} mm (mean {meas.gh_mean:.1f})"),
            ("Room to opposing", fmt(meas.space)),
        ]
        html = "<table cellspacing='4'>" + "".join(
            f"<tr><td style='color:#8c929c'>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows) + "</table>"
        html += (f"<p style='margin-top:8px'><span style='color:#8c929c'>Suggested abutment</span><br>"
                 f"<span style='font-size:14px; color:#7fd3a0'><b>{sg.kind}</b></span>")
        parts = []
        if sg.gh is not None:
            parts.append(f"GH {sg.gh:.1f} mm")
        if sg.height_choice:
            parts.append(sg.height_choice)
        elif sg.height is not None:
            parts.append(f"height ≤ {sg.height:.1f} mm")
        if parts:
            html += "<br><b>" + " · ".join(parts) + "</b>"
        html += "</p><ul style='margin-left:-20px'>" + "".join(f"<li>{line}</li>" for line in sg.lines) + "</ul>"
        self.result_lbl.setText(html)
        self.status(f"Implant: {sg.kind}" + (f", GH {sg.gh:.1f} mm" if sg.gh is not None else ""))
