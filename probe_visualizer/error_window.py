"""Machine error / message monitoring window.

Opens as a non-modal dialog next to the main window. Polls the currently
connected `CNCClient` every 500 ms for:
    - controller status (program name, run/hold/alarm, mode)
    - active alarms (FOCAS cnc_rdalmmsg2)
    - current operator messages (FOCAS cnc_rdopmsg3)

Each message / alarm is translated through `cnc_messages.json` which ships
next to this module. The rules dictionary can be freely edited by the user;
entries support exact, substring, and regex matching and can supply a
friendly description plus a list of possible causes. Untranslated items are
still shown with their raw text.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QGroupBox, QSplitter, QPushButton, QWidget, QSizePolicy,
)

from probe_visualizer.cnc_client import (
    CNCAlarm, CNCOperatorMessage, CNCStatus,
)


# ---------------------------------------------------------------------------
# Translation rules
# ---------------------------------------------------------------------------


@dataclass
class TranslationRule:
    kind: str               # "alarm" or "operator"
    match: str              # "exact" | "contains" | "regex"
    pattern: str            # raw text or regex
    friendly: str           # display template (may contain {1},{2},...)
    severity: str = "info"  # info | waiting | warning | error
    causes: List[str] = field(default_factory=list)
    _compiled: Optional[re.Pattern] = None

    def matches(self, text: str):
        """Return (is_match, groups_tuple) — groups is empty for non-regex."""
        if self.match == "exact":
            return (text == self.pattern, ())
        if self.match == "contains":
            return (self.pattern in text, ())
        if self.match == "regex":
            if self._compiled is None:
                try:
                    self._compiled = re.compile(self.pattern)
                except re.error:
                    self._compiled = re.compile(r"(?!x)x")  # never matches
            m = self._compiled.search(text)
            if not m:
                return (False, ())
            return (True, m.groups())
        return (False, ())


class MessageDictionary:
    """Wraps `cnc_messages.json` and translates raw strings."""

    def __init__(self, path: str):
        self.path = path
        self.rules: List[TranslationRule] = []
        self.load()

    def load(self) -> None:
        self.rules.clear()
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return
        for raw in data.get("rules", []):
            try:
                self.rules.append(TranslationRule(
                    kind=raw.get("type", "operator"),
                    match=raw.get("match", "contains"),
                    pattern=raw.get("pattern", ""),
                    friendly=raw.get("friendly", ""),
                    severity=raw.get("severity", "info"),
                    causes=list(raw.get("causes", [])),
                ))
            except Exception:
                continue

    def translate_operator(self, msg: CNCOperatorMessage):
        """Return (friendly_text, severity, causes) for an operator message."""
        return self._translate(msg.text, kind="operator")

    def translate_alarm(self, alarm: CNCAlarm):
        """Return (friendly_text, severity, causes) for an alarm.

        The alarm's short code (e.g. SV1026) is matched first; if that misses
        the raw message text is matched. `{axis}` in the friendly template is
        substituted with the alarm's axis number.
        """
        code = alarm.code
        for rule in self.rules:
            if rule.kind != "alarm":
                continue
            ok, groups = rule.matches(code)
            if not ok:
                ok, groups = rule.matches(alarm.message)
            if ok:
                return (
                    self._fill(rule.friendly, groups, axis=alarm.axis),
                    rule.severity,
                    list(rule.causes),
                )
        return (alarm.message, "error", [])

    def _translate(self, text, kind):
        for rule in self.rules:
            if rule.kind != kind:
                continue
            ok, groups = rule.matches(text)
            if ok:
                return (
                    self._fill(rule.friendly, groups),
                    rule.severity,
                    list(rule.causes),
                )
        return (text, "info", [])

    @staticmethod
    def _fill(template: str, groups: tuple, **extra) -> str:
        out = template
        for i, g in enumerate(groups, start=1):
            out = out.replace("{" + str(i) + "}", str(g))
        for k, v in extra.items():
            out = out.replace("{" + k + "}", str(v))
        return out


# ---------------------------------------------------------------------------
# Error window
# ---------------------------------------------------------------------------


SEVERITY_COLORS = {
    "info":    ("#d0e8ff", "#0b3d6b"),   # bg, fg
    "waiting": ("#fff4c4", "#5a4700"),
    "warning": ("#ffd8a8", "#7a3c00"),
    "error":   ("#ffc4c4", "#6b0000"),
}


def _default_dict_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cnc_messages.json")


class ErrorWindow(QDialog):
    """Non-modal window for Fanuc alarms and operator messages."""

    POLL_MS = 500
    HISTORY_LIMIT = 100

    def __init__(self, parent, get_client_callback):
        """`get_client_callback()` must return the currently-active
        CNCClient or None — we don't hold a direct reference so the main
        app can swap connections freely."""
        super().__init__(parent)
        self.setWindowTitle("Fanuc Machine Messages")
        self.resize(760, 620)
        # Non-modal: user can keep using the main window.
        self.setModal(False)

        self._get_client = get_client_callback
        self._dict = MessageDictionary(_default_dict_path())
        self._seen_keys = set()
        self._last_status: Optional[CNCStatus] = None

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        # Prime the window immediately
        self._poll()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Status strip ---
        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet(
            "padding: 6px; background-color: #eef1f5; "
            "border: 1px solid #c8d0da; border-radius: 4px; "
            "font-family: monospace; font-size: 12px;"
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Active alarms + operator messages in a splitter ---
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter, stretch=1)

        alarm_box = QGroupBox("Active alarms")
        alarm_layout = QVBoxLayout(alarm_box)
        self.alarm_list = QListWidget()
        self.alarm_list.setStyleSheet("font-family: monospace;")
        alarm_layout.addWidget(self.alarm_list)
        splitter.addWidget(alarm_box)

        op_box = QGroupBox("Operator messages")
        op_layout = QVBoxLayout(op_box)
        self.op_list = QListWidget()
        self.op_list.setStyleSheet("font-family: monospace;")
        op_layout.addWidget(self.op_list)
        splitter.addWidget(op_box)

        hist_box = QGroupBox("History (newest first)")
        hist_layout = QVBoxLayout(hist_box)
        self.history_list = QListWidget()
        self.history_list.setStyleSheet(
            "font-family: monospace; font-size: 11px;"
        )
        hist_layout.addWidget(self.history_list)
        splitter.addWidget(hist_box)

        splitter.setSizes([180, 180, 260])

        # --- Buttons row ---
        btn_row = QHBoxLayout()
        reload_btn = QPushButton("Reload translations")
        reload_btn.setToolTip(
            "Re-read cnc_messages.json without restarting the app"
        )
        reload_btn.clicked.connect(self._reload_dict)
        btn_row.addWidget(reload_btn)

        clear_btn = QPushButton("Clear history")
        clear_btn.clicked.connect(self._clear_history)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # -- Behaviour -----------------------------------------------------------

    def _reload_dict(self):
        self._dict.load()
        self._poll()

    def _clear_history(self):
        self.history_list.clear()
        self._seen_keys.clear()

    def _poll(self):
        client = self._get_client()
        if client is None or not client.is_connected():
            self.status_label.setText(
                "CNC not connected — click Connect CNC in the main window."
            )
            self.status_label.setStyleSheet(
                "padding: 6px; background-color: #f2f2f2; "
                "border: 1px solid #c8d0da; border-radius: 4px; "
                "font-family: monospace; font-size: 12px; color: #666;"
            )
            return

        # ---- Status ----
        try:
            status = client.read_status()
        except Exception as e:
            status = CNCStatus()
            self.status_label.setText(f"Status read error: {e}")
        else:
            parts = [
                f"Machine: {getattr(client, 'description', 'CNC')}",
                f"Program: {status.program_name or '-'}"
                f" (O{status.program_o_number})"
                if status.program_o_number else f"Program: {status.program_name or '-'}",
                f"Line: N{status.current_line}",
                f"Mode: {status.mode_text}",
                f"Status: {status.status_text}",
            ]
            self.status_label.setText("   ".join(parts))
            sev = "error" if (status.emergency or status.alarm) else "info"
            bg, fg = SEVERITY_COLORS.get(sev, ("#eef1f5", "#0b3d6b"))
            self.status_label.setStyleSheet(
                f"padding: 6px; background-color: {bg}; color: {fg}; "
                "border: 1px solid #c8d0da; border-radius: 4px; "
                "font-family: monospace; font-size: 12px;"
            )

        # ---- Alarms ----
        try:
            alarms = client.read_alarms()
        except Exception:
            alarms = []
        self.alarm_list.clear()
        for al in alarms:
            friendly, sev, causes = self._dict.translate_alarm(al)
            item_text = f"[{al.code}]  {friendly}"
            if al.message and friendly != al.message:
                item_text += f"\n         raw: {al.message}"
            for c in causes:
                item_text += f"\n         → {c}"
            item = QListWidgetItem(item_text)
            bg, fg = SEVERITY_COLORS.get(sev, ("#ffffff", "#000000"))
            item.setBackground(self._color(bg))
            item.setForeground(self._color(fg))
            self.alarm_list.addItem(item)
            self._remember(f"A:{al.code}:{al.message}", item_text, sev)
        if not alarms:
            self.alarm_list.addItem("(no active alarms)")

        # ---- Operator messages ----
        try:
            opmsgs = client.read_operator_messages()
        except Exception:
            opmsgs = []
        self.op_list.clear()
        for m in opmsgs:
            friendly, sev, causes = self._dict.translate_operator(m)
            item_text = f"[#{m.number}]  {friendly}"
            if m.text and friendly != m.text:
                item_text += f"\n         raw: {m.text}"
            for c in causes:
                item_text += f"\n         → {c}"
            item = QListWidgetItem(item_text)
            bg, fg = SEVERITY_COLORS.get(sev, ("#ffffff", "#000000"))
            item.setBackground(self._color(bg))
            item.setForeground(self._color(fg))
            self.op_list.addItem(item)
            self._remember(f"O:{m.number}:{m.text}", item_text, sev)
        if not opmsgs:
            self.op_list.addItem("(no operator messages)")

    def _remember(self, key: str, pretty_text: str, severity: str):
        if key in self._seen_keys:
            return
        self._seen_keys.add(key)
        stamp = time.strftime("%H:%M:%S")
        first_line = pretty_text.split("\n", 1)[0]
        item = QListWidgetItem(f"{stamp}  {first_line}")
        bg, fg = SEVERITY_COLORS.get(severity, ("#ffffff", "#000000"))
        item.setBackground(self._color(bg))
        item.setForeground(self._color(fg))
        self.history_list.insertItem(0, item)
        while self.history_list.count() > self.HISTORY_LIMIT:
            self.history_list.takeItem(self.history_list.count() - 1)

    @staticmethod
    def _color(hex_str: str):
        from PyQt5.QtGui import QColor
        return QColor(hex_str)

    # -- lifetime ------------------------------------------------------------

    def closeEvent(self, ev):
        self._timer.stop()
        super().closeEvent(ev)
