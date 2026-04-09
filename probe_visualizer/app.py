import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QComboBox, QCheckBox, QGroupBox, QPushButton, QFileDialog,
    QMessageBox, QStatusBar, QAction, QLabel, QScrollArea,
    QDoubleSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView,
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
        self.resize(1300, 800)

        self.data = None
        self.visible_sides = {}
        self.color_mode = "side"
        self._current_path = None
        self._sel_side_idx = -1
        self._sel_pt_idx = -1
        self._editing = False

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

        # Side checkboxes
        self.sides_group = QGroupBox("Sides / Sessions")
        self.sides_layout = QHBoxLayout(self.sides_group)
        self.sides_layout.setContentsMargins(6, 2, 6, 2)
        scroll = QScrollArea()
        scroll.setWidget(self.sides_group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(60)
        main_layout.addWidget(scroll)
        self.side_checks = {}

        # Main content: views + right panel
        content_layout = QHBoxLayout()

        self.stack = QStackedWidget()
        self.view_2d = View2D()
        self.view_3d = View3D()
        self.stack.addWidget(self.view_2d)
        self.stack.addWidget(self.view_3d)
        content_layout.addWidget(self.stack, stretch=1)

        self.view_3d.point_selected.connect(self._on_point_selected)

        # Right panel with tabs
        self.right_tabs = QTabWidget()
        self.right_tabs.setFixedWidth(300)

        # Tab 1: Edit Point
        edit_tab = QWidget()
        edit_layout = QVBoxLayout(edit_tab)

        self.edit_info_label = QLabel("Click a point in 3D view\nto select it")
        self.edit_info_label.setAlignment(Qt.AlignCenter)
        edit_layout.addWidget(self.edit_info_label)

        for label_text, attr in [("X:", "spin_x"), ("Y:", "spin_y"), ("Z:", "spin_z")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-999999, 999999)
            spin.valueChanged.connect(self._on_spin_changed)
            row.addWidget(spin)
            edit_layout.addLayout(row)
            setattr(self, attr, spin)

        save_btn = QPushButton("Save (Ctrl+S)")
        save_btn.clicked.connect(self._save_file)
        edit_layout.addWidget(save_btn)

        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self._save_file_as)
        edit_layout.addWidget(save_as_btn)

        edit_layout.addStretch()
        self.right_tabs.addTab(edit_tab, "Edit Point")

        # Tab 2: Statistics
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        # Z stats table
        stats_layout.addWidget(QLabel("Z Statistics per Side:"))
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(7)
        self.stats_table.setHorizontalHeaderLabels(
            ["Side", "Pts", "Z min", "Z max", "Z avg", "Z std", "Length"]
        )
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setAlternatingRowColors(True)
        stats_layout.addWidget(self.stats_table)

        # Plate dimensions
        stats_layout.addWidget(QLabel("Plate Dimensions:"))
        self.plate_info = QLabel("No data loaded")
        self.plate_info.setWordWrap(True)
        self.plate_info.setStyleSheet(
            "background-color: #2a2a2a; color: #ddd; padding: 8px; "
            "border-radius: 4px; font-family: monospace; font-size: 11px;"
        )
        stats_layout.addWidget(self.plate_info)
        stats_layout.addStretch()
        self.right_tabs.addTab(stats_tab, "Statistics")

        content_layout.addWidget(self.right_tabs)
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

    def _update_statistics(self):
        if self.data is None:
            self.stats_table.setRowCount(0)
            self.plate_info.setText("No data loaded")
            return

        # Z stats table
        stats = self.data.side_stats()
        self.stats_table.setRowCount(len(stats))
        for row, s in enumerate(stats):
            self.stats_table.setItem(row, 0, QTableWidgetItem(s["name"]))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(s["count"])))
            self.stats_table.setItem(row, 2, QTableWidgetItem(f"{s['z_min']:.2f}"))
            self.stats_table.setItem(row, 3, QTableWidgetItem(f"{s['z_max']:.2f}"))
            self.stats_table.setItem(row, 4, QTableWidgetItem(f"{s['z_avg']:.2f}"))
            self.stats_table.setItem(row, 5, QTableWidgetItem(f"{s['z_std']:.2f}"))
            self.stats_table.setItem(row, 6, QTableWidgetItem(f"{s['span']:.1f}"))

        # Plate dimensions
        dims = self.data.plate_dimensions()
        if dims is None:
            self.plate_info.setText("Not enough sides for plate calculation")
            return

        lines = []
        lines.append(f"Width:      {dims['width']:.1f} mm")
        lines.append(f"Height:     {dims['height']:.1f} mm")
        lines.append(f"Angle:      {dims['angle_deg']:.2f} deg")
        lines.append("")
        lines.append("Side lengths:")
        side_names = [s.name for s in self.data.sides]
        for i, length in enumerate(dims["side_lengths"]):
            name = side_names[i] if i < len(side_names) else f"Side {i}"
            lines.append(f"  {name}: {length:.1f} mm")

        if dims["diagonal_1"] > 0:
            lines.append("")
            lines.append(f"Diagonal 1: {dims['diagonal_1']:.1f} mm")
            lines.append(f"Diagonal 2: {dims['diagonal_2']:.1f} mm")
            diff = abs(dims["diagonal_1"] - dims["diagonal_2"])
            lines.append(f"Diag diff:  {diff:.1f} mm")

        self.plate_info.setText("\n".join(lines))

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
        self.edit_info_label.setText("Click a point in 3D view\nto select it")

        self._rebuild_side_checks()
        self._update_statistics()
        self._refresh_views()

    def _save_file(self):
        if self.data is None:
            return
        try:
            self.data.save_to_json_file()
            self.status_bar.showMessage(
                f"Saved to {os.path.basename(self.data.source_path)}"
            )
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
            self.setWindowTitle(
                f"Probe Data Visualizer - {os.path.basename(path)}"
            )
            self.status_bar.showMessage(f"Saved to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _on_point_selected(self, side_idx, pt_idx, side_name, x, y, z):
        self._sel_side_idx = side_idx
        self._sel_pt_idx = pt_idx
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
        self.data.sides[self._sel_side_idx].points[self._sel_pt_idx] = [x, y, z]
        self.view_3d.update_selected_point(
            self._sel_side_idx, self._sel_pt_idx, x, y, z
        )
        self.view_2d.update_plot(self.data, self.visible_sides, self.color_mode)
        self._update_statistics()

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
