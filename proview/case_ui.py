"""Dialogs for cases: start screen, case information, case browser, step history."""
from __future__ import annotations

import datetime as _dt
import os

from PySide6.QtCore import QDate, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .cases import WORK_TYPES, Case, CaseInfo, list_cases

ACCENT = "#4c8dff"


def open_folder(path: str) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _big_button(title: str, subtitle: str) -> QPushButton:
    b = QPushButton(f"{title}\n{subtitle}")
    b.setMinimumHeight(64)
    b.setStyleSheet("QPushButton { text-align: left; padding: 10px 14px; font-size: 13px; border-radius: 8px;"
                    " background: #2e333b; border: 1px solid #3c424c; }"
                    f"QPushButton:hover {{ border-color: {ACCENT}; background: #333a44; }}")
    return b


class CaseInfoDialog(QDialog):
    """New case / edit the information of a case."""

    def __init__(self, parent=None, info: CaseInfo | None = None, title="New case"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(440)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        info = info or CaseInfo(patient="")
        self.patient = QLineEdit(info.patient)
        self.patient.setPlaceholderText("Required")
        form.addRow("Patient name", self.patient)
        self.doctor = QLineEdit(info.doctor)
        form.addRow("Doctor", self.doctor)
        self.date = QDateEdit()
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("yyyy-MM-dd")
        d = QDate.fromString(info.date, "yyyy-MM-dd")
        self.date.setDate(d if d.isValid() else QDate.currentDate())
        form.addRow("Date", self.date)
        self.case_no = QLineEdit(info.case_no)
        form.addRow("Case number", self.case_no)
        self.work = QComboBox()
        self.work.setEditable(True)
        self.work.addItems([""] + WORK_TYPES)
        self.work.setCurrentText(info.work_type)
        form.addRow("Work type", self.work)
        self.teeth = QLineEdit(info.teeth)
        self.teeth.setPlaceholderText("e.g. 14, 15, 36  (FDI)")
        form.addRow("Teeth", self.teeth)
        self.notes = QPlainTextEdit(info.notes)
        self.notes.setPlaceholderText("Notes")
        self.notes.setMinimumHeight(90)
        form.addRow("Notes", self.notes)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self._accept())
        buttons.rejected.connect(lambda: self.reject())
        lay.addWidget(buttons)
        self.patient.setFocus()

    def _accept(self):
        if not self.patient.text().strip():
            QMessageBox.information(self, self.windowTitle(), "Type the patient's name.")
            return
        self.accept()

    def info(self) -> CaseInfo:
        return CaseInfo(patient=self.patient.text().strip(), doctor=self.doctor.text().strip(),
                        date=self.date.date().toString("yyyy-MM-dd"), work_type=self.work.currentText().strip(),
                        case_no=self.case_no.text().strip(), teeth=self.teeth.text().strip(),
                        notes=self.notes.toPlainText().strip())


class CaseBrowser(QDialog):
    """List of the cases in the cases folder, with search and a doctor filter."""

    COLS = ["Date", "Patient", "Doctor", "Case no.", "Work type", "Teeth", "Last change"]

    def __init__(self, root: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open case")
        self.resize(900, 520)
        self.root = root
        self.chosen: str | None = None
        self.new_requested = False
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search patient, doctor, case number, notes...")
        self.search.textChanged.connect(lambda _t: self._fill())
        top.addWidget(self.search, 1)
        self.doctor = QComboBox()
        self.doctor.currentIndexChanged.connect(lambda _i: self._fill())
        top.addWidget(QLabel("Doctor"))
        top.addWidget(self.doctor)
        lay.addLayout(top)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(lambda _r, _c: self._open())
        lay.addWidget(self.table, 1)
        self.where = QLabel(f"Cases folder: {root}")
        self.where.setStyleSheet("color: #9aa0a6;")
        lay.addWidget(self.where)
        row = QHBoxLayout()
        b = QPushButton("New case...")
        b.clicked.connect(lambda _=False: self._new())
        row.addWidget(b)
        b = QPushButton("Open folder")
        b.setToolTip("Show the selected case folder in Windows Explorer")
        b.clicked.connect(lambda _=False: self._open_folder())
        row.addWidget(b)
        row.addStretch(1)
        b = QPushButton("Cancel")
        b.clicked.connect(lambda _=False: self.reject())
        row.addWidget(b)
        self.open_btn = QPushButton("Open")
        self.open_btn.setDefault(True)
        self.open_btn.setStyleSheet(f"QPushButton {{ background:{ACCENT}; color:white; padding:6px 18px; }}")
        self.open_btn.clicked.connect(lambda _=False: self._open())
        row.addWidget(self.open_btn)
        lay.addLayout(row)
        self.cases = list_cases(root)
        doctors = sorted({c.info.doctor for c in self.cases if c.info.doctor}, key=str.lower)
        self.doctor.blockSignals(True)
        self.doctor.addItem("All", "")
        for d in doctors:
            self.doctor.addItem(d, d)
        self.doctor.blockSignals(False)
        self._fill()

    def _fill(self):
        q = self.search.text().strip().lower()
        doc = self.doctor.currentData() or ""
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for c in self.cases:
            i = c.info
            if doc and i.doctor != doc:
                continue
            if q and q not in " ".join([i.patient, i.doctor, i.case_no, i.work_type, i.teeth, i.notes,
                                         i.date]).lower():
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [i.date, i.patient, i.doctor, i.case_no, i.work_type, i.teeth, c.modified.replace("T", " ")]
            for k, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setData(Qt.UserRole, c.path)
                if k == 1 and i.notes:
                    item.setToolTip(i.notes)
                self.table.setItem(r, k, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        if self.table.rowCount():
            self.table.selectRow(0)

    def _selected(self) -> str | None:
        items = self.table.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def _open(self):
        self.chosen = self._selected()
        if self.chosen:
            self.accept()

    def _open_folder(self):
        path = self._selected() or self.root
        if os.path.isdir(path):
            open_folder(path)

    def _new(self):
        self.new_requested = True
        self.accept()


class StartDialog(QDialog):
    """Shown when ProVixel starts: quick view, new case or open a case."""

    QUICK, NEW, OPEN, RECENT = range(4)

    def __init__(self, root: str, recent: list[str], parent=None, files=None):
        super().__init__(parent)
        self.setWindowTitle("ProVixel")
        self.setMinimumWidth(520)
        self.choice = self.QUICK
        self.recent_path: str | None = None
        lay = QVBoxLayout(self)
        head = QLabel("<span style='font-size:20px; font-weight:600;'>ProVixel</span>"
                      "<br><span style='color:#9aa0a6;'>How do you want to work?</span>")
        lay.addWidget(head)
        if files:
            names = ", ".join(os.path.basename(f) for f in files[:3]) + (" ..." if len(files) > 3 else "")
            info = QLabel(f"Files to open: {names}")
            info.setWordWrap(True)
            info.setStyleSheet("color: #c9ced6;")
            lay.addWidget(info)
        b = _big_button("Quick view", "Look at and edit models without a case (nothing is saved automatically)")
        b.clicked.connect(lambda _=False: self._choose(self.QUICK))
        lay.addWidget(b)
        b = _big_button("New case", "Patient, doctor, notes; every step is saved in the case folder")
        b.clicked.connect(lambda _=False: self._choose(self.NEW))
        lay.addWidget(b)
        b = _big_button("Open case", "Continue a saved case")
        b.clicked.connect(lambda _=False: self._choose(self.OPEN))
        lay.addWidget(b)
        cases = []
        for p in recent:
            try:
                cases.append(Case.load(p))
            except Exception:  # noqa: BLE001
                continue
        if cases:
            lay.addWidget(QLabel("Recent cases"))
            self.list = QListWidget()
            for c in cases[:8]:
                it = QListWidgetItem(f"{c.info.date}   {c.info.title()}"
                                     + (f"   ({c.info.work_type})" if c.info.work_type else ""))
                it.setData(Qt.UserRole, c.path)
                self.list.addItem(it)
            self.list.setMaximumHeight(170)
            self.list.itemDoubleClicked.connect(lambda it: self._recent(it))
            self.list.itemActivated.connect(lambda it: self._recent(it))
            lay.addWidget(self.list)
        where = QLabel(f"Cases folder: {root}")
        where.setStyleSheet("color: #7d838c; font-size: 11px;")
        where.setWordWrap(True)
        lay.addWidget(where)

    def _choose(self, c):
        self.choice = c
        self.accept()

    def _recent(self, item):
        self.recent_path = item.data(Qt.UserRole)
        self.choice = self.RECENT
        self.accept()


class HistoryDialog(QDialog):
    """Steps saved in a case; any of them can be restored."""

    def __init__(self, case: Case, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"History - {case.info.title()}")
        self.resize(460, 480)
        self.case = case
        self.chosen: int | None = None
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Every step is saved. Restoring a step adds it as the newest step, so "
                             "nothing is lost."))
        self.list = QListWidget()
        for s in reversed(case.steps):
            t = s.get("time", "").replace("T", " ")
            try:
                t = _dt.datetime.fromisoformat(s["time"]).strftime("%Y-%m-%d  %H:%M")
            except (KeyError, ValueError):
                pass
            it = QListWidgetItem(f"#{s['n']}   {s.get('label', '')}   ·   {t}   ·   {s.get('models', 0)} model(s)")
            it.setData(Qt.UserRole, s["n"])
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(lambda _it: self._restore())
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        b = QPushButton("Close")
        b.clicked.connect(lambda _=False: self.reject())
        row.addWidget(b)
        b = QPushButton("Restore this step")
        b.setStyleSheet(f"QPushButton {{ background:{ACCENT}; color:white; padding:6px 14px; }}")
        b.clicked.connect(lambda _=False: self._restore())
        row.addWidget(b)
        lay.addLayout(row)

    def _restore(self):
        it = self.list.currentItem()
        if it is None:
            return
        self.chosen = int(it.data(Qt.UserRole))
        self.accept()
