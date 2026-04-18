"""
PyQt5 GUI for manually adding a single replication entry to the database.
Launches data_ingestor.py to process and merge the entry.

Usage:
    python add_entry_gui.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import threading

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QTextEdit, QComboBox, QCheckBox,
    QPushButton, QGroupBox, QScrollArea, QSizePolicy, QFrame, QMessageBox,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ONTOLOGY_PATH = os.path.join(PROJECT_ROOT, "data", "metascience_observatory_topic_ontology.json")


def _load_ontology():
    """Load ontology and return (disciplines, discipline_to_subdisciplines)."""
    with open(ONTOLOGY_PATH, encoding="utf-8") as f:
        ontology = json.load(f)
    disciplines = []
    discipline_to_subdisciplines = {}
    for _domain, disciplines_dict in ontology.items():
        for discipline, subdisciplines in disciplines_dict.items():
            disciplines.append(discipline)
            discipline_to_subdisciplines[discipline] = subdisciplines
    disciplines.sort(key=str.lower)
    return disciplines, discipline_to_subdisciplines


DISCIPLINE_OPTIONS, DISCIPLINE_TO_SUBDISCIPLINES = _load_ontology()
# Prepend empty option for optional field
DISCIPLINE_OPTIONS = [""] + DISCIPLINE_OPTIONS

RESULT_OPTIONS = [
    "", "success", "failure", "replicated", "not replicated",
    "mixed", "inconclusive", "reversal",
]

VALIDATED_OPTIONS = ["", "no", "partial", "partially", "yes"]

VALIDATED_PERSON_OPTIONS = [
    "", "Dan Elton", "Curate Science team", "FReD_API_team",
]

ES_TYPE_OPTIONS = [
    "", "r", "d", "cohen's d", "hedges' g", "smd",
    "or", "odds ratio", "hr", "hazard ratio",
    "eta2", "eta-squared", "partial eta-squared",
    "f", "cohen's f", "f2", "r2",
    "test statistic",
]

FONT_SIZE = 9
LABEL_FONT_SIZE = 9


def _make_font(size=FONT_SIZE, bold=False):
    f = QFont()
    f.setPointSize(size)
    f.setBold(bold)
    return f


def _styled_combo(options: list, editable: bool = True) -> QComboBox:
    c = QComboBox()
    c.setEditable(editable)
    c.addItems(options)
    c.setFont(_make_font(FONT_SIZE))
    c.setMinimumHeight(24)
    return c


def _group(title: str) -> tuple[QGroupBox, QFormLayout]:
    box = QGroupBox(title)
    box.setFont(_make_font(FONT_SIZE, bold=True))
    box.setStyleSheet(
        "QGroupBox { border: 1px solid #d0d0d0; border-radius: 6px; "
        "margin-top: 10px; padding-top: 6px; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
        "font-weight: bold; color: #1e40af; }"
    )
    form = QFormLayout(box)
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setSpacing(10)
    form.setContentsMargins(16, 20, 16, 12)
    return box, form


def _label(text: str, required: bool = False) -> QLabel:
    lbl = QLabel(text + ("  <span style='color:#dc2626'>*</span>" if required else ""))
    lbl.setTextFormat(Qt.RichText)
    lbl.setFont(_make_font(LABEL_FONT_SIZE))
    return lbl


def _line_edit(placeholder: str = "") -> QLineEdit:
    e = QLineEdit()
    e.setPlaceholderText(placeholder)
    e.setFont(_make_font(FONT_SIZE))
    e.setMinimumHeight(24)
    return e


# ── Signal helper for thread → GUI communication ──────────────────────────────

class _Signals(QObject):
    append_text    = pyqtSignal(str)
    show_info      = pyqtSignal(str)
    show_warn      = pyqtSignal(str)
    reenable_submit = pyqtSignal()


# ── Main window ───────────────────────────────────────────────────────────────

class AddEntryWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add Replication Entry")
        self.setMinimumSize(640, 580)
        self.resize(720, 780)

        self._signals = _Signals()
        self._signals.append_text.connect(self._append_log)
        self._signals.show_info.connect(lambda m: QMessageBox.information(self, "Done", m))
        self._signals.show_warn.connect(lambda m: QMessageBox.warning(self, "Warning", m))
        self._signals.reenable_submit.connect(lambda: self.submit_btn.setEnabled(True))

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── scrollable form area ──────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        form_container = QWidget()
        form_container.setStyleSheet("background: #ffffff;")
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(24, 20, 24, 16)
        form_layout.setSpacing(14)

        # Title
        title = QLabel("Add Replication Entry")
        title.setFont(_make_font(13, bold=True))
        title.setStyleSheet("color: #111827;")
        form_layout.addWidget(title)

        subtitle = QLabel("Fields marked <span style='color:#dc2626'>*</span> are required.")
        subtitle.setTextFormat(Qt.RichText)
        subtitle.setFont(_make_font(LABEL_FONT_SIZE))
        subtitle.setStyleSheet("color: #6b7280; margin-bottom: 6px;")
        form_layout.addWidget(subtitle)

        # ── Papers ────────────────────────────────────────────────────
        box, form = _group("Papers")
        self.original_url    = _line_edit("https://doi.org/10.xxxx/...")
        self.replication_url = _line_edit("https://doi.org/10.xxxx/...")
        form.addRow(_label("Original DOI / URL",    required=True), self.original_url)
        form.addRow(_label("Replication DOI / URL", required=True), self.replication_url)
        form_layout.addWidget(box)

        # ── Description ───────────────────────────────────────────────
        box, form = _group("Description")
        self.description = QTextEdit()
        self.description.setFont(_make_font(FONT_SIZE))
        self.description.document().setDefaultFont(_make_font(FONT_SIZE))
        self.description.setMinimumHeight(60)
        self.description.setMaximumHeight(90)
        self.description.setPlaceholderText(
            "Brief summary of the original finding and how the replication was conducted…")
        form.addRow(_label("Description", required=True), self.description)
        form_layout.addWidget(box)

        # ── Classification ────────────────────────────────────────────
        box, form = _group("Classification")
        self.result            = _styled_combo(RESULT_OPTIONS)
        self.discipline        = _styled_combo(DISCIPLINE_OPTIONS, editable=False)
        self.subdiscipline     = _styled_combo([""], editable=False)  # Populated from ontology when discipline is selected
        self.discipline.currentTextChanged.connect(self._on_discipline_changed)
        self._on_discipline_changed()  # Initial population
        self.initiative_tag    = _line_edit("e.g. RP:P, ML1, XPHIR")
        self.validated         = _styled_combo(VALIDATED_OPTIONS, editable=False)
        self.validated_person  = _styled_combo(VALIDATED_PERSON_OPTIONS)
        form.addRow(_label("Result"),              self.result)
        form.addRow(_label("Discipline"),          self.discipline)
        form.addRow(_label("Subdiscipline"),       self.subdiscipline)
        form.addRow(_label("Initiative Tag"),      self.initiative_tag)
        form.addRow(_label("Validated"),           self.validated)
        form.addRow(_label("Validated by"),        self.validated_person)
        form_layout.addWidget(box)

        # ── Effect Size — Original ────────────────────────────────────
        box, form = _group("Effect Size — Original Study")
        self.original_es      = _line_edit("numeric value, e.g. 0.35")
        self.original_es_type = _styled_combo(ES_TYPE_OPTIONS)
        self.original_n       = _line_edit("total N")
        form.addRow(_label("ES Value"),   self.original_es)
        form.addRow(_label("ES Type"),    self.original_es_type)
        form.addRow(_label("N"),          self.original_n)
        form_layout.addWidget(box)

        # ── Effect Size — Replication ─────────────────────────────────
        box, form = _group("Effect Size — Replication")
        self.replication_es      = _line_edit("numeric value, e.g. 0.12")
        self.replication_es_type = _styled_combo(ES_TYPE_OPTIONS)
        self.replication_n       = _line_edit("total N")
        form.addRow(_label("ES Value"),   self.replication_es)
        form.addRow(_label("ES Type"),    self.replication_es_type)
        form.addRow(_label("N"),          self.replication_n)
        form_layout.addWidget(box)

        # ── Options ───────────────────────────────────────────────────
        box, form = _group("Options")
        self.skip_api_cb = QCheckBox("Skip API calls  (no metadata enrichment — faster for testing)")
        self.skip_api_cb.setFont(_make_font(FONT_SIZE))
        form.addRow("", self.skip_api_cb)
        form_layout.addWidget(box)

        form_layout.addStretch()
        scroll.setWidget(form_container)
        root.addWidget(scroll, stretch=3)

        # ── Log output ────────────────────────────────────────────────
        log_container = QWidget()
        log_container.setStyleSheet("background: #f8fafc;")
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(24, 8, 24, 8)
        log_layout.setSpacing(4)

        log_header = QLabel("Output")
        log_header.setFont(_make_font(LABEL_FONT_SIZE, bold=True))
        log_header.setStyleSheet("color: #374151;")
        log_layout.addWidget(log_header)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Courier", 10))
        self.log.setStyleSheet(
            "QTextEdit { background: #1e293b; color: #e2e8f0; "
            "border: none; border-radius: 4px; padding: 8px; }"
        )
        self.log.setMinimumHeight(100)
        log_layout.addWidget(self.log)
        root.addWidget(log_container, stretch=1)

        # ── Button bar ────────────────────────────────────────────────
        btn_bar = QWidget()
        btn_bar.setStyleSheet("background: #f1f5f9; border-top: 1px solid #e2e8f0;")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(24, 12, 24, 12)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setFont(_make_font(FONT_SIZE, bold=True))
        self.submit_btn.setMinimumHeight(30)
        self.submit_btn.setMinimumWidth(140)
        self.submit_btn.setStyleSheet(
            "QPushButton { background: #1e40af; color: white; border-radius: 5px; padding: 0 20px; } "
            "QPushButton:hover { background: #1d4ed8; } "
            "QPushButton:disabled { background: #93c5fd; }"
        )
        self.submit_btn.clicked.connect(self.submit)

        clear_btn = QPushButton("Clear")
        clear_btn.setFont(_make_font(FONT_SIZE))
        clear_btn.setMinimumHeight(30)
        clear_btn.setMinimumWidth(100)
        clear_btn.setStyleSheet(
            "QPushButton { border: 1px solid #d1d5db; border-radius: 5px; padding: 0 16px; } "
            "QPushButton:hover { background: #f3f4f6; }"
        )
        clear_btn.clicked.connect(self.clear)

        btn_layout.addWidget(self.submit_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        root.addWidget(btn_bar)

    # ── helpers ───────────────────────────────────────────────────────

    def _on_discipline_changed(self):
        """Update subdiscipline dropdown to show only options for the selected discipline."""
        discipline = self.discipline.currentText().strip()
        subdisciplines = DISCIPLINE_TO_SUBDISCIPLINES.get(discipline, [])
        options = [""] + sorted(subdisciplines, key=str.lower)
        current = self.subdiscipline.currentText().strip()
        self.subdiscipline.clear()
        self.subdiscipline.addItems(options)
        # Restore selection if still valid
        idx = self.subdiscipline.findText(current)
        if idx >= 0:
            self.subdiscipline.setCurrentIndex(idx)
        else:
            self.subdiscipline.setCurrentIndex(0)

    def _get(self, widget) -> str:
        if isinstance(widget, QTextEdit):
            return widget.toPlainText().strip()
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        return widget.text().strip()

    def _append_log(self, text: str):
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    # ── actions ───────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        errs = []
        if not self._get(self.original_url):
            errs.append("Original DOI / URL is required.")
        if not self._get(self.replication_url):
            errs.append("Replication DOI / URL is required.")
        if not self._get(self.description):
            errs.append("Description is required.")
        return errs

    def clear(self):
        for w in (
            self.original_url, self.replication_url,
            self.initiative_tag, self.original_es, self.original_n,
            self.replication_es, self.replication_n,
        ):
            w.clear()
        for w in (
            self.result, self.discipline, self.subdiscipline,
            self.original_es_type, self.replication_es_type,
            self.validated, self.validated_person,
        ):
            w.setCurrentIndex(0)
        self.description.clear()
        self.log.clear()

    def submit(self):
        errs = self.validate()
        if errs:
            QMessageBox.critical(self, "Validation Error", "\n".join(errs))
            return

        row_data = {
            "original_url":               self._get(self.original_url),
            "replication_url":            self._get(self.replication_url),
            "description":                self._get(self.description),
            "result":                     self._get(self.result),
            "discipline":                 self._get(self.discipline),
            "subdiscipline":              self._get(self.subdiscipline),
            "replication_initiative_tag": self._get(self.initiative_tag),
            "original_es":                self._get(self.original_es),
            "original_es_type":           self._get(self.original_es_type),
            "original_n":                 self._get(self.original_n),
            "replication_es":             self._get(self.replication_es),
            "replication_es_type":        self._get(self.replication_es_type),
            "replication_n":              self._get(self.replication_n),
            "validated":                  self._get(self.validated),
            "validated_person":           self._get(self.validated_person),
        }

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="add_entry_",
            delete=False, newline="", dir=SCRIPT_DIR,
        )
        writer = csv.DictWriter(tmp, fieldnames=list(row_data.keys()))
        writer.writeheader()
        writer.writerow(row_data)
        tmp.close()

        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "data_ingestor.py"),
            tmp.name,
            "--workers", "1",
        ]
        if self.skip_api_cb.isChecked():
            cmd.append("--skip-api-calls")

        self.log.clear()
        self._append_log(f"$ {' '.join(cmd)}\n\n")
        self.submit_btn.setEnabled(False)

        signals = self._signals

        def run():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=SCRIPT_DIR,
                )
                for line in iter(proc.stdout.readline, ""):
                    signals.append_text.emit(line)
                proc.wait()
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
                if proc.returncode == 0:
                    signals.show_info.emit("Entry processed successfully!")
                else:
                    signals.show_warn.emit(
                        f"Ingestor exited with code {proc.returncode}.\nSee output log."
                    )
            except Exception as ex:
                signals.append_text.emit(f"\nError: {ex}\n")
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            finally:
                signals.append_text.emit("\n[done]\n")
                signals.reenable_submit.emit()

        threading.Thread(target=run, daemon=True).start()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    # Global font
    font = QFont()
    font.setPointSize(FONT_SIZE)
    app.setFont(font)

    window = AddEntryWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
