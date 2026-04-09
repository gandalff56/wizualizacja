import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QComboBox, QCheckBox, QGroupBox, QPushButton, QFileDialog,
    QMessageBox, QStatusBar, QAction, QLabel, QScrollArea,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

from probe_visualizer.data_loader import ProbeData
from probe_visualizer.view_2d import View2D
from probe_visualizer.view_3d import View3D


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Probe Data Visualizer")
        self.resize(1200, 800)

        self.data = None
        self.visible_sides = {}
        self.color_mode = "side"
        self._current_path = None

        # Selected point info
        self._sel_side_idx = -1
        self._sel_pt_idx = -1
        self._editing = False  # prevent feedback loop

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

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save_file)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._save_file_as)
        file_menu.addAction(save_as_action)

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

        # All/None buttons
        all_btn = QPushButton("All")
        all_btn.setFixedWidth(40)
        all_btn.clicked.connect(self._select_all_sides)
        none_btn = QPushButton("None")
        none_btn.setFixedWidth(45)
        none_btn.clicked.connect(self._select_no_sides)
        toolbar.addWidget(all_btn)
        toolbar.addWidget(none_btn)

        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # Dynamic side checkboxes in a scrollable area
        self.sides_group = QGroupBox("Sides / Sessions")
        self.sides_layout = QHBoxLayout(self.sides_group)
        self.sides_layout.setContentsMargins(6, 2, 6, 2)

        scroll = QScrollArea()
        scroll.setWidget(self.sides_group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(60)
        main_layout.addWidget(scroll)

        self.side_checks = {}

        # Main content: views + edit panel
        content_layout = QHBoxLayout()

        # Stacked views
        self.stack = QStackedWidget()
        self.view_2d = View2D()
        self.view_3d = View3D()
        self.stack.addWidget(self.view_2d)
        self.stack.addWidget(self.view_3d)
        content_layout.addWidget(self.stack, stretch=1)

        # Connect point selection signal
        self.view_3d.point_selected.connect(self._on_point_selected)

        # Edit panel (right side)
        self.edit_panel = QGroupBox("Edit Point")
        edit_layout = QVBoxLayout(self.edit_panel)

        self.edit_info_label = QLabel("Click a point in 3D view\nto select it")
        self.edit_info_label.setAlignment(Qt.AlignCenter)
        edit_layout.addWidget(self.edit_info_label)

        # X spinbox
        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("X:"))
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setDecimals(3)
        self.spin_x.setRange(-999999, 999999)
        self.spin_x.valueChanged.connect(self._on_spin_changed)
        x_row.addWidget(self.spin_x)
        edit_layout.addLayout(x_row)

        # Y spinbox
        y_row = QHBoxLayout()
        y_row.addWidget(QLabel("Y:"))
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setDecimals(3)
        self.spin_y.setRange(-999999, 999999)
        self.spin_y.valueChanged.connect(self._on_spin_changed)
        y_row.addWidget(self.spin_y)
        edit_layout.addLayout(y_row)

        # Z spinbox
        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z:"))
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setDecimals(3)
        self.spin_z.setRange(-999999, 999999)
        self.spin_z.valueChanged.connect(self._on_spin_changed)
        z_row.addWidget(self.spin_z)
        edit_layout.addLayout(z_row)

        # Save button
        save_btn = QPushButton("Save (Ctrl+S)")
        save_btn.clicked.connect(self._save_file)
        edit_layout.addWidget(save_btn)

        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self._save_file_as)
        edit_layout.addWidget(save_as_btn)

        edit_layout.addStretch()

        self.edit_panel.setFixedWidth(200)
        self.edit_panel.setEnabled(False)
        content_layout.addWidget(self.edit_panel)

        main_layout.addLayout(content_layout, stretch=1)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - open a JSON file to start")

    def _rebuild_side_checks(self):
        for cb in self.side_checks.values():
            self.sides_layout.removeWidget(cb)
            cb.deleteLater()
        self.side_checks.clear()
        self.visible_sides.clear()

        if self.data is None:
            return

        for side in self.data.sides:
            cb = QCheckBox(side.name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_side_toggled)
            self.sides_layout.addWidget(cb)
            self.side_checks[side.name] = cb
            self.visible_sides[side.name] = True

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

        self._current_path = path
        filename = os.path.basename(path)
        n_sides = len(self.data.sides)
        n_pts = self.data.total_point_count()
        z_min, z_max = self.data.z_range()
        fmt = self.data.format_name
        self.setWindowTitle(f"Probe Data Visualizer - {filename}")
        self.status_bar.showMessage(
            f"{filename} | Format: {fmt} | {n_sides} groups | "
            f"{n_pts} points | Z: {z_min:.1f} - {z_max:.1f}"
        )

        self._sel_side_idx = -1
        self._sel_pt_idx = -1
        self.edit_panel.setEnabled(False)
        self.edit_info_label.setText("Click a point in 3D view\nto select it")

        self._rebuild_side_checks()
        self._refresh_views()

    def _save_file(self):
        if self.data is None:
            return
        try:
            self.data.save_to_json_file()
            self.status_bar.showMessage(f"Saved to {os.path.basename(self.data.source_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _save_file_as(self):
        if self.data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Probe Data As", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            self.data.save_to_json_file(path)
            self._current_path = path
            self.setWindowTitle(f"Probe Data Visualizer - {os.path.basename(path)}")
            self.status_bar.showMessage(f"Saved to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _on_point_selected(self, side_idx, pt_idx, side_name, x, y, z):
        self._sel_side_idx = side_idx
        self._sel_pt_idx = pt_idx
        self.edit_panel.setEnabled(True)
        self.edit_info_label.setText(f"{side_name}\nPoint #{pt_idx + 1}")

        self._editing = True
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_z.setValue(z)
        self._editing = False

    def _on_spin_changed(self):
        if self._editing or self._sel_side_idx < 0 or self.data is None:
            return

        x = self.spin_x.value()
        y = self.spin_y.value()
        z = self.spin_z.value()

        # Update data model
        self.data.sides[self._sel_side_idx].points[self._sel_pt_idx] = [x, y, z]

        # Update 3D view in real-time
        self.view_3d.update_selected_point(self._sel_side_idx, self._sel_pt_idx, x, y, z)

        # Update 2D view too
        self.view_2d.update_plot(self.data, self.visible_sides, self.color_mode)

    def _on_view_changed(self, index):
        self.stack.setCurrentIndex(index)

    def _on_color_changed(self, index):
        self.color_mode = "side" if index == 0 else "z"
        self._refresh_views()

    def _on_side_toggled(self):
        for name, cb in self.side_checks.items():
            self.visible_sides[name] = cb.isChecked()
        self._refresh_views()

    def _select_all_sides(self):
        for cb in self.side_checks.values():
            cb.setChecked(True)

    def _select_no_sides(self):
        for cb in self.side_checks.values():
            cb.setChecked(False)

    def _refresh_views(self):
        if self.data is None:
            return
        self.view_2d.update_plot(self.data, self.visible_sides, self.color_mode)
        self.view_3d.update_plot(self.data, self.visible_sides, self.color_mode)
