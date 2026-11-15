"""ProVixel look: one dark theme for the whole application (palette + style sheet)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette

# colours
BG = "#15171c"  # window background
SURFACE = "#1c1f25"  # panels
CARD = "#22262d"  # group boxes, inputs
CARD_HI = "#2a2f37"
BORDER = "#2e333b"
BORDER_HI = "#3b414b"
TEXT = "#e7e9ee"
MUTED = "#8c929c"
ACCENT = "#4c8dff"
ACCENT_DK = "#2f6fe0"
ACCENT_BG = "#1b3156"
DANGER = "#e5534b"

QSS = f"""
* {{ outline: none; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ color: {TEXT}; }}
QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER_HI}; padding: 5px 7px; border-radius: 6px; }}

/* menus */
QMenuBar {{ background: {BG}; border: none; padding: 2px 6px; }}
QMenuBar::item {{ padding: 4px 9px; border-radius: 5px; background: transparent; }}
QMenuBar::item:selected {{ background: {CARD_HI}; }}
QMenu {{ background: {CARD}; border: 1px solid {BORDER_HI}; border-radius: 8px; padding: 5px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT_BG}; }}
QMenu::item:disabled {{ color: #5d636c; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}

/* status bar */
QStatusBar {{ background: {BG}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}

/* tool panel on the right */
QDockWidget {{ color: {MUTED}; font-weight: 600; }}
QDockWidget::title {{ background: {SURFACE}; padding: 8px 10px; border-bottom: 1px solid {BORDER}; text-align: left; }}
QDockWidget > QWidget, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {SURFACE}; }}

/* cards */
QGroupBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; margin-top: 22px;
            padding: 12px 10px 10px 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 4px; top: 2px;
                   color: {MUTED}; padding: 0 4px; }}
QGroupBox QLabel {{ font-weight: normal; }}

/* buttons */
QPushButton {{ background: {CARD_HI}; border: 1px solid {BORDER_HI}; border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ background: #323843; border-color: #4a515d; }}
QPushButton:pressed {{ background: #262b33; }}
QPushButton:checked {{ background: {ACCENT_BG}; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #5d636c; background: {CARD}; border-color: {BORDER}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; padding: 8px 14px; }}
QPushButton[primary="true"]:hover {{ background: #6aa0ff; }}
QPushButton[primary="true"]:pressed {{ background: {ACCENT_DK}; }}
QPushButton[danger="true"] {{ background: {DANGER}; border-color: {DANGER}; color: white; font-weight: 600; }}
QPushButton[danger="true"]:hover {{ background: #f06a62; }}
QToolButton {{ border-radius: 7px; }}

/* inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit, QPlainTextEdit, QTextEdit {{
    background: {BG}; border: 1px solid {BORDER_HI}; border-radius: 6px; padding: 4px 7px;
    selection-background-color: {ACCENT}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: #5d636c; border-color: {BORDER}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button,
QDateEdit::up-button, QDateEdit::down-button {{ width: 16px; border: none; background: transparent; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {BORDER_HI}; selection-background-color: {ACCENT_BG}; }}

QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}
QCheckBox::indicator {{ border: 1px solid #555c67; border-radius: 4px; background: {BG}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QRadioButton::indicator {{ border: 1px solid #555c67; border-radius: 8px; background: {BG}; }}
QRadioButton::indicator:checked {{ background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                   stop:0 white, stop:0.35 white, stop:0.45 {ACCENT}, stop:1 {ACCENT}); border-color: {ACCENT}; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: #5d636c; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER_HI}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: white; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}

/* lists and tables */
QListWidget, QTableWidget, QTreeWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px;
    alternate-background-color: {SURFACE}; gridline-color: {BORDER}; }}
QListWidget::item {{ padding: 6px 8px; border-radius: 6px; }}
QListWidget::item:selected, QTableWidget::item:selected {{ background: {ACCENT_BG}; color: {TEXT}; }}
QListWidget::item:hover {{ background: {CARD}; }}
QHeaderView::section {{ background: {SURFACE}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 6px 8px; font-weight: 600; }}

/* scroll bars */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #3a404a; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #4b525e; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #3a404a; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* top bar */
QToolBar#actionBar {{ background: {SURFACE}; border: none; border-bottom: 1px solid {BORDER}; padding: 6px 12px; spacing: 6px; }}
QToolBar#actionBar QToolButton {{ color: {TEXT}; border: 1px solid transparent; border-radius: 8px; padding: 5px; }}
QToolBar#actionBar QToolButton:hover {{ background: {CARD_HI}; border-color: {BORDER_HI}; }}
QToolBar#actionBar QToolButton[tile="true"] {{ background: transparent; border: 1px solid transparent; border-radius: 10px;
    min-width: 62px; max-width: 62px; min-height: 54px; font-size: 11px; padding: 4px 2px 3px 2px; }}
QToolBar#actionBar QToolButton[tile="true"]:hover {{ background: {CARD_HI}; border-color: {BORDER_HI}; }}
QToolBar#actionBar QToolButton[tile="true"]:checked {{ background: {ACCENT_BG}; border-color: {ACCENT}; }}
QToolBar#actionBar QToolButton[tile="true"]:pressed {{ background: #183052; }}
QLabel[groupCaption="true"] {{ color: #6f7681; font-size: 9px; font-weight: 700; letter-spacing: 1px; }}
QFrame[groupSep="true"] {{ background: {BORDER}; }}
QToolButton#caseButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 4px 10px;
    text-align: left; font-size: 12px; }}
QToolButton#caseButton:hover {{ border-color: {ACCENT}; }}
QToolButton#caseButton::menu-indicator {{ subcontrol-position: right center; right: 6px; }}
QLabel#savePill {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 11px; padding: 3px 10px;
    color: {MUTED}; font-size: 11px; }}
QLabel#savePill[state="saved"] {{ color: #7fd3a0; border-color: #2c5a41; }}
QLabel#savePill[state="saving"] {{ color: #f0c674; border-color: #5c4b23; }}
QLabel#savePill[state="error"] {{ color: #ff8a80; border-color: #6a2a26; }}
"""


_SVGS = {
    "check": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" fill="none" '
             'stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "up": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M2 6.5 5 3.5l3 3" fill="none" '
          'stroke="#aab0ba" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "down": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M2 3.5 5 6.5l3-3" fill="none" '
            'stroke="#aab0ba" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
}


def _assets() -> dict:
    """Small images the style sheet needs (written once to a cache folder)."""
    import os
    import tempfile

    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    folder = os.path.join(tempfile.gettempdir(), "provixel_theme_v2")
    os.makedirs(folder, exist_ok=True)
    out = {}
    for name, svg in _SVGS.items():
        path = os.path.join(folder, name + ".png")  # PNG: needs no image plug-in in the packed exe
        if not os.path.exists(path):
            img = QImage(48, 48, QImage.Format_ARGB32)
            img.fill(Qt.transparent)
            painter = QPainter(img)
            QSvgRenderer(QByteArray(svg.encode())).render(painter)
            painter.end()
            img.save(path)
        out[name] = path.replace("\\", "/")
    return out


def _asset_qss(a: dict) -> str:
    return f"""
QCheckBox::indicator:checked {{{{ image: url("{a['check']}"); }}}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow, QDateEdit::up-arrow {{{{ image: url("{a['up']}"); width: 9px; height: 9px; }}}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow, QDateEdit::down-arrow {{{{ image: url("{a['down']}"); width: 9px; height: 9px; }}}}
QComboBox::down-arrow {{{{ image: url("{a['down']}"); width: 10px; height: 10px; }}}}
"""


def apply_theme(app) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(BG))
    pal.setColor(QPalette.AlternateBase, QColor(SURFACE))
    pal.setColor(QPalette.ToolTipBase, QColor(CARD))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(CARD_HI))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.PlaceholderText, QColor("#636a74"))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        pal.setColor(QPalette.Disabled, role, QColor("#5d636c"))
    app.setPalette(pal)
    f = QFont(app.font())
    f.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Inter", f.family()])
    f.setPointSizeF(max(f.pointSizeF(), 9.5))
    app.setFont(f)
    try:
        extra = _asset_qss(_assets()).replace("{{", "{").replace("}}", "}")
    except OSError:
        extra = ""
    app.setStyleSheet(QSS + extra)


def set_primary(button, on=True) -> None:
    """Mark a QPushButton as the main action of its panel (accent colour)."""
    button.setProperty("primary", on)
    button.style().unpolish(button)
    button.style().polish(button)


def set_state(widget, state: str) -> None:
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
