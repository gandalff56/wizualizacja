import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QComboBox, QCheckBox, QGroupBox, QPushButton, QFileDialog,
    QMessageBox, QStatusBar, QAction, QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

from probe_visualizer.data_loader import ProbeData
from probe_visualizer.colors import SIDE_COLORS, SIDE_ORDER
from probe_visualizer.view_2d import View2D
from probe_visualizer.view_3d import View3D


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Probe Data Visualizer")
        self.resize(1200, 800)

        self.data = None
        self.visible_sides = {name: True for name in SIDE_ORDER}
        self.color_mode = "side"

        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()

    def _setup_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Toolbar row
        toolbar = QHBoxLayout()

        open_btn = QPushButton("Open File")
        open_btn.clicked.connect(self._open_file)
        toolbar.addWidget(open_btn)

        toolbar.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["2D", "3D"])
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(self.view_combo)

        toolbar.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["By Side", "By Z Value"])
        self.color_combo.currentIndexChanged.connect(self._on_color_changed)
        toolbar.addWidget(self.color_combo)

        # Side checkboxes
        sides_group = QGroupBox("Sides")
        sides_layout = QHBoxLayout(sides_group)
        sides_layout.setContentsMargins(6, 2, 6, 2)
        self.side_checks = {}
        for name in SIDE_ORDER:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_side_toggled)
            sides_layout.addWidget(cb)
            self.side_checks[name] = cb
        toolbar.addWidget(sides_group)

        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # Stacked views
        self.stack = QStackedWidget()
        self.view_2d = View2D()
        self.view_3d = View3D()
        self.stack.addWidget(self.view_2d)
        self.stack.addWidget(self.view_3d)
        main_layout.addWidget(self.stack, stretch=1)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - open a JSON file to start")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Probe Data", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            self.data = ProbeData.from_json_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            return

        filename = os.path.basename(path)
        n_sides = len(self.data.sides)
        n_pts = self.data.total_point_count()
        z_min, z_max = self.data.z_range()
        self.setWindowTitle(f"Probe Data Visualizer - {filename}")
        self.status_bar.showMessage(
            f"{filename} | {n_sides} sides | {n_pts} points | "
            f"Z range: {z_min:.1f} - {z_max:.1f}"
        )

        # Update side checkboxes for sides present in file
        known = {s.name for s in self.data.sides}
        for name, cb in self.side_checks.items():
            cb.setEnabled(name in known)
            if name in known:
                cb.setChecked(True)
                self.visible_sides[name] = True

        self._refresh_views()

    def _on_view_changed(self, index):
        self.stack.setCurrentIndex(index)

    def _on_color_changed(self, index):
        self.color_mode = "side" if index == 0 else "z"
        self._refresh_views()

    def _on_side_toggled(self):
        for name, cb in self.side_checks.items():
            self.visible_sides[name] = cb.isChecked()
        self._refresh_views()

    def _refresh_views(self):
        if self.data is None:
            return
        self.view_2d.update_plot(self.data, self.visible_sides, self.color_mode)
        self.view_3d.update_plot(self.data, self.visible_sides, self.color_mode)
