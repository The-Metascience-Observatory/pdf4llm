"""
PyQt5 GUI for reviewing duplicate rows during ingestion.

Launched by data_ingestor.py STEP 5 when potential duplicates or auto-skipped rows exist.
Returns user decisions (skip/add/merge/replace) back to the ingestor.
"""

import sys
import math
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QScrollArea, QTabWidget,
    QFrame, QCheckBox, QSizePolicy, QGroupBox, QStackedWidget, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette


# Columns to display in the compact view (excludes verbose reference columns)
DISPLAY_COLS = [
    'description', 'result', 'discipline', 'subdiscipline',
    'original_es', 'original_es_type', 'original_es_r', 'original_n',
    'replication_es', 'replication_es_type', 'replication_es_r', 'replication_n',
    'validated',
]

# Verbose columns excluded from display
EXCLUDED_COLS = {
    'original_authors', 'original_title', 'original_journal',
    'original_volume', 'original_issue', 'original_pages', 'original_year',
    'replication_authors', 'replication_title', 'replication_journal',
    'replication_volume', 'replication_issue', 'replication_pages', 'replication_year',
    'original_citation_html', 'replication_citation_html',
    'original_url', 'replication_url',
}


def _norm(v):
    """Normalize a value for display/comparison."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = str(v).strip()
    return '' if s.lower() == 'nan' else s


class FieldLabel(QLabel):
    """A label for a field value, with optional background highlight."""

    def __init__(self, text, bg_color=None):
        super().__init__(text)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if bg_color:
            self.setAutoFillBackground(True)
            pal = self.palette()
            pal.setColor(QPalette.Window, QColor(bg_color))
            self.setPalette(pal)


class MatchCard(QFrame):
    """Card showing one existing master row that matches an incoming row."""

    def __init__(self, master_row_idx, master_row, incoming_row, parent=None):
        super().__init__(parent)
        self.master_row_idx = master_row_idx
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("MatchCard { background: #f8f8f8; border: 1px solid #ccc; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        # Header with row index and action dropdown
        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>Master row {master_row_idx}</b>"))
        header.addStretch()

        self.action_combo = QComboBox()
        self.action_combo.addItems(["No action", "Merge into this row", "Replace this row"])
        self.action_combo.setMinimumWidth(180)
        header.addWidget(self.action_combo)
        layout.addLayout(header)

        # Description (full, wrapped)
        desc = _norm(master_row.get('description'))
        if desc:
            desc_label = QLabel(f"<i>{desc}</i>")
            desc_label.setWordWrap(True)
            desc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(desc_label)

        # Field grid
        grid = QHBoxLayout()
        grid.setSpacing(12)

        self.replace_result_cb = None
        for col in DISPLAY_COLS:
            if col == 'description':
                continue  # already shown above
            existing_val = _norm(master_row.get(col))
            incoming_val = _norm(incoming_row.get(col))

            # Determine highlight
            bg = None
            if not existing_val and incoming_val:
                bg = '#d4edda'  # green — merge would fill
            elif existing_val and incoming_val and existing_val != incoming_val:
                bg = '#fff3cd'  # yellow — differs

            val_display = existing_val if existing_val else '—'

            if col == 'result' and existing_val and incoming_val and existing_val != incoming_val:
                # Add replace checkbox for result field when values differ
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(2)
                label = FieldLabel(f"<small><b>{col}</b></small><br>{val_display}", bg)
                label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
                container_layout.addWidget(label)
                self.replace_result_cb = QCheckBox("replace")
                self.replace_result_cb.setStyleSheet("QCheckBox { font-size: 10px; }")
                container_layout.addWidget(self.replace_result_cb)
                grid.addWidget(container)
            else:
                label = FieldLabel(f"<small><b>{col}</b></small><br>{val_display}", bg)
                label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
                grid.addWidget(label)

        grid.addStretch()
        layout.addLayout(grid)

    def get_action(self):
        """Return action string, master row index, and force-replace fields."""
        text = self.action_combo.currentText()
        force_replace = set()
        if self.replace_result_cb and self.replace_result_cb.isChecked():
            force_replace.add('result')
        if text == "Merge into this row":
            return 'merge', self.master_row_idx, force_replace
        elif text == "Replace this row":
            return 'replace', self.master_row_idx, force_replace
        return None, None, force_replace


class DuplicateCard(QWidget):
    """Card for one incoming potential-duplicate row + its existing matches."""

    def __init__(self, incoming_idx, incoming_row, match_indices, master_df, parent=None):
        super().__init__(parent)
        self.incoming_idx = incoming_idx

        # Collect all master rows for this match set (used for green highlight logic)
        master_rows = [master_df.loc[mi] for mi in match_indices]

        outer_layout = QVBoxLayout(self)

        # Scrollable area for this card's content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)

        # ── Incoming row card (same format as match cards) ──
        incoming_frame = QFrame()
        incoming_frame.setFrameShape(QFrame.StyledPanel)
        incoming_frame.setStyleSheet("QFrame { background: #e8f0fe; border: 1px solid #4a90d9; }")
        incoming_layout = QVBoxLayout(incoming_frame)
        incoming_layout.setContentsMargins(8, 6, 8, 6)

        # Header with label and action dropdown
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>INCOMING ROW</b>"))
        header.addStretch()
        self.incoming_action = QComboBox()
        self.incoming_action.addItems(["Skip (don't import)", "Add as new row"])
        self.incoming_action.setMinimumWidth(180)
        header.addWidget(self.incoming_action)
        incoming_layout.addLayout(header)

        # Description (full, wrapped)
        desc = _norm(incoming_row.get('description'))
        if desc:
            desc_label = QLabel(f"<i>{desc}</i>")
            desc_label.setWordWrap(True)
            desc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            incoming_layout.addWidget(desc_label)

        # Field grid — same columns as MatchCard, green if no existing row has data
        grid = QHBoxLayout()
        grid.setSpacing(12)
        for col in DISPLAY_COLS:
            if col == 'description':
                continue
            incoming_val = _norm(incoming_row.get(col))

            # Green: incoming has data and NO existing match row has data for this field
            bg = None
            if incoming_val:
                any_existing_has = any(_norm(mr.get(col)) for mr in master_rows)
                if not any_existing_has:
                    bg = '#d4edda'  # green — new data not in any existing row

            val_display = incoming_val if incoming_val else '—'
            label = FieldLabel(f"<small><b>{col}</b></small><br>{val_display}", bg)
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            grid.addWidget(label)

        grid.addStretch()
        incoming_layout.addLayout(grid)
        layout.addWidget(incoming_frame)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #999;")
        layout.addWidget(sep)

        # Existing matches header
        layout.addWidget(QLabel(f"<b>Existing matches in master database ({len(match_indices)} row{'s' if len(match_indices) > 1 else ''}):</b>"))

        # Match cards
        self.match_cards = []
        for mi in match_indices:
            card = MatchCard(mi, master_df.loc[mi], incoming_row)
            self.match_cards.append(card)
            layout.addWidget(card)

        layout.addStretch()
        scroll.setWidget(scroll_widget)
        outer_layout.addWidget(scroll)

    def get_decision(self):
        """Return the user's decision for this incoming row."""
        # Check match cards first for merge/replace
        for card in self.match_cards:
            action, target, force_replace = card.get_action()
            if action:
                return {
                    'incoming_idx': self.incoming_idx,
                    'action': action,
                    'target_master_idx': target,
                    'force_replace_fields': force_replace
                }

        # Fall back to incoming action
        incoming_text = self.incoming_action.currentText()
        if incoming_text == "Add as new row":
            action = 'add'
        else:
            action = 'skip'

        return {
            'incoming_idx': self.incoming_idx,
            'action': action,
            'target_master_idx': None
        }


class AutoSkipCard(QFrame):
    """Compact card for an auto-skipped row with override checkbox."""

    def __init__(self, incoming_idx, incoming_row, parent=None):
        super().__init__(parent)
        self.incoming_idx = incoming_idx
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("AutoSkipCard { border: 1px solid #ddd; background: #fafafa; }")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)

        self.override_cb = QCheckBox("Import anyway")
        layout.addWidget(self.override_cb)

        desc = _norm(incoming_row.get('description'))
        result = _norm(incoming_row.get('result'))
        rep_es = _norm(incoming_row.get('replication_es'))
        rep_es_type = _norm(incoming_row.get('replication_es_type'))

        info = f"Row {incoming_idx + 1}: {desc[:100]}{'...' if len(desc) > 100 else ''}"
        if result:
            info += f"  | result: {result}"
        if rep_es:
            info += f"  | rep_es: {rep_es} ({rep_es_type})"

        label = QLabel(info)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(label, stretch=1)


class IdenticalTitleCard(QFrame):
    """Card for an incoming row where original_title == replication_title."""

    def __init__(self, incoming_idx, incoming_row, parent=None):
        super().__init__(parent)
        self.incoming_idx = incoming_idx
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "IdenticalTitleCard { border: 2px solid #e07000; background: #fff8f0; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Warning header
        warning = QLabel(
            "⚠  original_title == replication_title — likely self-paired or within-paper error"
        )
        warning.setStyleSheet("color: #b35900; font-weight: bold;")
        layout.addWidget(warning)

        # Shared title
        title = _norm(incoming_row.get('replication_title', ''))
        title_label = QLabel(f"<b>Shared title:</b> {title}")
        title_label.setWordWrap(True)
        title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(title_label)

        # URLs
        orig_url = _norm(incoming_row.get('original_url', ''))
        repl_url = _norm(incoming_row.get('replication_url', ''))
        url_label = QLabel(
            f"<b>original_url:</b> {orig_url or '—'}<br>"
            f"<b>replication_url:</b> {repl_url or '—'}"
        )
        url_label.setWordWrap(True)
        url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(url_label)

        # Description (abbreviated)
        desc = _norm(incoming_row.get('description', ''))
        if desc:
            desc_label = QLabel(f"<i>{desc[:200]}{'…' if len(desc) > 200 else ''}</i>")
            desc_label.setWordWrap(True)
            desc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(desc_label)

        # Discipline / field
        disc = _norm(incoming_row.get('discipline', ''))
        field = _norm(incoming_row.get('field', ''))
        if disc or field:
            layout.addWidget(QLabel(f"<b>discipline:</b> {disc or '—'}  |  <b>field:</b> {field or '—'}"))

        # Action row
        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("<b>Action:</b>"))
        self.action_combo = QComboBox()
        self.action_combo.addItems(["Add as-is", "Skip (don't import)", "Correct and import"])
        self.action_combo.setMinimumWidth(200)
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)
        action_row.addWidget(self.action_combo)
        action_row.addStretch()
        layout.addLayout(action_row)

        # Correction fields (shown only when "Correct and import" is selected)
        self.correction_widget = QWidget()
        corr_layout = QVBoxLayout(self.correction_widget)
        corr_layout.setContentsMargins(0, 4, 0, 0)
        corr_layout.setSpacing(4)

        corr_layout.addWidget(QLabel("Corrected <b>original_title</b>:"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Enter corrected original title…")
        corr_layout.addWidget(self.title_edit)

        corr_layout.addWidget(QLabel("Corrected <b>original_url</b>:"))
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("e.g. https://doi.org/10.xxxx/…")
        corr_layout.addWidget(self.url_edit)

        self.correction_widget.setVisible(False)
        layout.addWidget(self.correction_widget)

    def _on_action_changed(self, index):
        # Show correction fields only for "Correct and import" (index 2)
        self.correction_widget.setVisible(index == 2)

    def get_decision(self):
        text = self.action_combo.currentText()
        if "Skip" in text:
            return {'incoming_idx': self.incoming_idx, 'action': 'skip'}
        elif "Correct" in text:
            return {
                'incoming_idx': self.incoming_idx,
                'action': 'correct',
                'corrected_original_title': self.title_edit.text().strip() or None,
                'corrected_original_url': self.url_edit.text().strip() or None,
            }
        else:
            return {'incoming_idx': self.incoming_idx, 'action': 'add'}


class DuplicateReviewWindow(QMainWindow):
    """Main window for reviewing duplicates during ingestion."""

    def __init__(self, potential_dups, auto_skipped, processed_df, master_df,
                 identical_title_list=None):
        """
        Args:
            potential_dups: list of (incoming_idx, match_indices) tuples
            auto_skipped: list of (incoming_idx, match_indices) tuples
            processed_df: the processed incoming DataFrame
            master_df: the current master DataFrame
        """
        super().__init__()
        self.setWindowTitle("Duplicate Review")
        self.setMinimumSize(1200, 800)
        self.results = None
        self.current_dup_idx = 0
        self.visited_pages = {0}  # page 0 is visible on open
        identical_title_list = identical_title_list or []

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Summary bar
        parts = [
            f"<b>{len(potential_dups)}</b> potential duplicate(s) to review",
            f"<b>{len(auto_skipped)}</b> auto-skipped (identical)",
        ]
        if identical_title_list:
            parts.append(
                f"<b style='color:#b35900'>{len(identical_title_list)}</b> identical-title pair(s) flagged"
            )
        summary = QLabel("  |  ".join(parts))
        summary.setStyleSheet("font-size: 14px; padding: 6px;")
        main_layout.addWidget(summary)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # ── Tab 1: Potential Duplicates (paginated) ──
        dup_tab = QWidget()
        dup_layout = QVBoxLayout(dup_tab)

        # Navigation bar
        nav_bar = QHBoxLayout()

        self.prev_btn = QPushButton("\u25C0  Prev")
        self.prev_btn.setMinimumHeight(32)
        self.prev_btn.setMinimumWidth(100)
        self.prev_btn.clicked.connect(self._prev_dup)
        nav_bar.addWidget(self.prev_btn)

        nav_bar.addStretch()

        self.page_label = QLabel()
        self.page_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.page_label.setAlignment(Qt.AlignCenter)
        nav_bar.addWidget(self.page_label)

        nav_bar.addStretch()

        self.next_btn = QPushButton("Next  \u25B6")
        self.next_btn.setMinimumHeight(32)
        self.next_btn.setMinimumWidth(100)
        self.next_btn.clicked.connect(self._next_dup)
        nav_bar.addWidget(self.next_btn)

        dup_layout.addLayout(nav_bar)

        # Stacked widget — one page per duplicate
        self.dup_stack = QStackedWidget()
        self.dup_cards = []

        for idx, match_indices in potential_dups:
            card = DuplicateCard(idx, processed_df.loc[idx], match_indices, master_df)
            self.dup_cards.append(card)
            self.dup_stack.addWidget(card)

        dup_layout.addWidget(self.dup_stack)
        self.tabs.addTab(dup_tab, f"Potential Duplicates ({len(potential_dups)})")

        self._update_nav()

        # ── Tab 2: Auto-Skipped ──
        skip_tab = QWidget()
        skip_layout = QVBoxLayout(skip_tab)

        skip_scroll = QScrollArea()
        skip_scroll.setWidgetResizable(True)
        skip_scroll_widget = QWidget()
        skip_scroll_layout = QVBoxLayout(skip_scroll_widget)

        self.skip_cards = []
        for idx, match_indices in auto_skipped:
            card = AutoSkipCard(idx, processed_df.loc[idx])
            self.skip_cards.append(card)
            skip_scroll_layout.addWidget(card)

        skip_scroll_layout.addStretch()
        skip_scroll.setWidget(skip_scroll_widget)
        skip_layout.addWidget(skip_scroll)
        self.tabs.addTab(skip_tab, f"Auto-Skipped ({len(auto_skipped)})")

        # ── Tab 3: Identical Titles ──
        self.identical_title_cards = []
        if identical_title_list:
            ident_tab = QWidget()
            ident_layout = QVBoxLayout(ident_tab)

            ident_header = QLabel(
                "<b>⚠ Identical title pairs</b> — incoming rows where original_title == "
                "replication_title. These are likely self-paired or within-paper errors. "
                "Choose an action for each row."
            )
            ident_header.setWordWrap(True)
            ident_header.setStyleSheet("color: #b35900; padding: 4px;")
            ident_layout.addWidget(ident_header)

            ident_scroll = QScrollArea()
            ident_scroll.setWidgetResizable(True)
            ident_scroll_widget = QWidget()
            ident_scroll_layout = QVBoxLayout(ident_scroll_widget)

            for idx in identical_title_list:
                card = IdenticalTitleCard(idx, processed_df.loc[idx])
                self.identical_title_cards.append(card)
                ident_scroll_layout.addWidget(card)

            ident_scroll_layout.addStretch()
            ident_scroll.setWidget(ident_scroll_widget)
            ident_layout.addWidget(ident_scroll)
            self.tabs.addTab(ident_tab, f"⚠ Identical Titles ({len(identical_title_list)})")

        # ── Apply button ──
        button_bar = QHBoxLayout()
        button_bar.addStretch()
        apply_btn = QPushButton("Apply All Decisions")
        apply_btn.setMinimumHeight(40)
        apply_btn.setMinimumWidth(200)
        apply_btn.setStyleSheet(
            "QPushButton { background: #4a90d9; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 4px; } "
            "QPushButton:hover { background: #357abd; }")
        apply_btn.clicked.connect(self._apply)
        button_bar.addWidget(apply_btn)
        button_bar.addStretch()
        main_layout.addLayout(button_bar)

    def _update_nav(self):
        """Update navigation buttons and page label."""
        total = len(self.dup_cards)
        if total == 0:
            self.page_label.setText("No duplicates to review")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return

        self.visited_pages.add(self.current_dup_idx)
        reviewed = len(self.visited_pages)
        self.page_label.setText(f"{self.current_dup_idx + 1} / {total}  ({reviewed} reviewed)")
        self.prev_btn.setEnabled(self.current_dup_idx > 0)
        self.next_btn.setEnabled(self.current_dup_idx < total - 1)
        self.dup_stack.setCurrentIndex(self.current_dup_idx)

    def _prev_dup(self):
        if self.current_dup_idx > 0:
            self.current_dup_idx -= 1
            self._update_nav()

    def _next_dup(self):
        if self.current_dup_idx < len(self.dup_cards) - 1:
            self.current_dup_idx += 1
            self._update_nav()

    def _apply(self):
        """Collect decisions for reviewed cards only and close."""
        reviewed_decisions = []
        for i, card in enumerate(self.dup_cards):
            if i in self.visited_pages:
                reviewed_decisions.append(card.get_decision())
        self.results = {
            'potential_dups': reviewed_decisions,
            'auto_skip_overrides': [
                card.incoming_idx for card in self.skip_cards
                if card.override_cb.isChecked()
            ],
            'identical_titles': [
                card.get_decision() for card in self.identical_title_cards
            ],
        }
        self.close()

    def get_results(self):
        return self.results


def launch_duplicate_review(potential_dups, auto_skipped, processed_df, master_df,
                            identical_title_list=None):
    """Launch the GUI and return user decisions. Blocks until window is closed."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = DuplicateReviewWindow(
        potential_dups, auto_skipped, processed_df, master_df,
        identical_title_list=identical_title_list or []
    )
    window.show()
    app.exec_()

    return window.get_results()
