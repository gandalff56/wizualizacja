import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QEvent
from PyQt5.QtGui import QMatrix4x4, QVector4D

import pyqtgraph.opengl as gl
import matplotlib.cm as cm

from probe_visualizer.colors import get_color_for_side


def _hex_to_gl(hex_color, alpha=1.0):
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (r, g, b, alpha)


class View3D(QWidget):
    # Signal: (side_index, point_index, side_name, x, y, z)
    point_selected = pyqtSignal(int, int, str, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setMouseTracking(True)

        # Tooltip overlay
        self.tooltip_label = QLabel(self.gl_widget)
        self.tooltip_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180); color: white; "
            "padding: 4px 8px; border-radius: 4px; font-size: 12px;"
        )
        self.tooltip_label.hide()

        # Install event filter for mouse tracking and clicks
        self.gl_widget.installEventFilter(self)
        self._mouse_press_pos = None

        self.z_scale_slider = QSlider(Qt.Horizontal)
        self.z_scale_slider.setMinimum(1)
        self.z_scale_slider.setMaximum(200)
        self.z_scale_slider.setValue(10)
        self.z_scale_slider.setTickInterval(10)
        self.z_scale_slider.valueChanged.connect(self._on_z_scale_changed)

        self.z_label = QLabel("Z scale: 1.0x")

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Z Scale:"))
        slider_layout.addWidget(self.z_scale_slider)
        slider_layout.addWidget(self.z_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_widget, stretch=1)
        layout.addLayout(slider_layout)

        self.data = None
        self.visible_sides = {}
        self.color_mode = "side"
        self._items = []

        # For hover/click: store original and transformed points
        # Each entry: (side_index_in_data, side_name, orig_points, trans_points)
        self._point_map = []

        # Selected point highlight
        self._selected_side_idx = -1
        self._selected_pt_idx = -1
        self._highlight_item = None

    def eventFilter(self, obj, event):
        if obj is not self.gl_widget:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.MouseMove:
            self._on_mouse_move(event.pos())
        elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._mouse_press_pos = event.pos()
        elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._mouse_press_pos is not None:
                delta = event.pos() - self._mouse_press_pos
                if abs(delta.x()) < 5 and abs(delta.y()) < 5:
                    self._on_click(event.pos())
            self._mouse_press_pos = None

        return super().eventFilter(obj, event)

    def _get_matrices(self):
        w = self.gl_widget.width()
        h = self.gl_widget.height()
        view_matrix = QMatrix4x4(np.array(self.gl_widget.viewMatrix().glData(), dtype=np.float32))
        proj_matrix = QMatrix4x4(np.array(self.gl_widget.projectionMatrix().glData(), dtype=np.float32))
        return view_matrix, proj_matrix, w, h

    def _project_point(self, pt_3d, view_matrix, proj_matrix, width, height):
        v = QVector4D(pt_3d[0], pt_3d[1], pt_3d[2], 1.0)
        v = view_matrix * v
        v = proj_matrix * v
        if abs(v.w()) < 1e-10:
            return None
        ndc_x = v.x() / v.w()
        ndc_y = v.y() / v.w()
        screen_x = (ndc_x + 1.0) * 0.5 * width
        screen_y = (1.0 - ndc_y) * 0.5 * height
        return (screen_x, screen_y)

    def _find_nearest_point(self, pos, max_dist=15.0):
        """Find nearest point to screen position. Returns (side_idx, pt_idx, side_name, orig_pt) or None."""
        mx, my = pos.x(), pos.y()
        view_matrix, proj_matrix, w, h = self._get_matrices()

        best_dist = max_dist
        best = None

        for side_data_idx, side_name, orig_pts, trans_pts in self._point_map:
            for j in range(len(trans_pts)):
                screen = self._project_point(trans_pts[j], view_matrix, proj_matrix, w, h)
                if screen is None:
                    continue
                dx = screen[0] - mx
                dy = screen[1] - my
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = (side_data_idx, j, side_name, orig_pts[j])

        return best

    def _on_mouse_move(self, pos):
        if self.data is None or not self._point_map:
            self.tooltip_label.hide()
            return

        result = self._find_nearest_point(pos)

        if result is not None:
            _, _, side_name, orig_pt = result
            self.tooltip_label.setText(
                f"{side_name}\n"
                f"X: {orig_pt[0]:.3f}\n"
                f"Y: {orig_pt[1]:.3f}\n"
                f"Z: {orig_pt[2]:.3f}"
            )
            w = self.gl_widget.width()
            mx, my = pos.x(), pos.y()
            tx = min(mx + 15, w - self.tooltip_label.sizeHint().width() - 5)
            ty = max(my - self.tooltip_label.sizeHint().height() - 5, 5)
            self.tooltip_label.move(tx, ty)
            self.tooltip_label.adjustSize()
            self.tooltip_label.show()
        else:
            self.tooltip_label.hide()

    def _on_click(self, pos):
        if self.data is None or not self._point_map:
            return

        result = self._find_nearest_point(pos)

        if result is not None:
            side_idx, pt_idx, side_name, orig_pt = result
            self._selected_side_idx = side_idx
            self._selected_pt_idx = pt_idx
            self._update_highlight()
            self.point_selected.emit(
                side_idx, pt_idx, side_name,
                float(orig_pt[0]), float(orig_pt[1]), float(orig_pt[2])
            )

    def _update_highlight(self):
        if self._highlight_item is not None:
            self.gl_widget.removeItem(self._highlight_item)
            self._highlight_item = None

        if self._selected_side_idx < 0 or self._selected_pt_idx < 0:
            return

        for side_data_idx, _, _, trans_pts in self._point_map:
            if side_data_idx == self._selected_side_idx:
                if self._selected_pt_idx < len(trans_pts):
                    pt = trans_pts[self._selected_pt_idx:self._selected_pt_idx + 1]
                    self._highlight_item = gl.GLScatterPlotItem(
                        pos=pt,
                        color=np.array([[1.0, 1.0, 1.0, 1.0]]),
                        size=12.0, pxMode=True
                    )
                    self.gl_widget.addItem(self._highlight_item)
                break

    def update_selected_point(self, side_idx, pt_idx, x, y, z):
        """Called when user edits point values - update data and redraw."""
        if self.data is None:
            return
        self.data.sides[side_idx].points[pt_idx] = [x, y, z]
        self._selected_side_idx = side_idx
        self._selected_pt_idx = pt_idx
        self._draw()
        self._update_highlight()

    def _get_z_scale(self):
        return self.z_scale_slider.value() / 10.0

    def _on_z_scale_changed(self, value):
        scale = value / 10.0
        self.z_label.setText(f"Z scale: {scale:.1f}x")
        if self.data is not None:
            self._draw()
            self._update_highlight()

    def update_plot(self, data, visible_sides, color_mode):
        self.data = data
        self.visible_sides = visible_sides
        self.color_mode = color_mode
        self._selected_side_idx = -1
        self._selected_pt_idx = -1
        self._draw()

    def _draw(self):
        for item in self._items:
            self.gl_widget.removeItem(item)
        self._items.clear()
        self._point_map.clear()
        if self._highlight_item is not None:
            self.gl_widget.removeItem(self._highlight_item)
            self._highlight_item = None
        self.tooltip_label.hide()

        if self.data is None:
            return

        all_pts = self.data.all_points()
        z_scale = self._get_z_scale()

        center_x = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2.0
        center_y = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2.0
        center_z = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2.0
        span_x = all_pts[:, 0].max() - all_pts[:, 0].min()
        span_y = all_pts[:, 1].max() - all_pts[:, 1].min()
        max_span = max(span_x, span_y)

        z_min = all_pts[:, 2].min()
        z_max = all_pts[:, 2].max()
        z_range = z_max - z_min if z_max - z_min > 1e-9 else 1.0

        cmap = cm.get_cmap("viridis")

        for i, side in enumerate(self.data.sides):
            if not self.visible_sides.get(side.name, True):
                continue

            pts = side.points.copy()
            pts[:, 0] -= center_x
            pts[:, 1] -= center_y
            pts[:, 2] = (pts[:, 2] - center_z) * z_scale

            # Store for hover/click lookup
            self._point_map.append((i, side.name, side.points.copy(), pts.copy()))

            if self.color_mode == "side":
                color = _hex_to_gl(get_color_for_side(side.name, i))
                line = gl.GLLinePlotItem(
                    pos=pts, color=color, width=2.0, antialias=True
                )
                self._items.append(line)
                self.gl_widget.addItem(line)

                n = len(pts)
                colors = np.tile(np.array(color), (n, 1))
                scatter = gl.GLScatterPlotItem(
                    pos=pts, color=colors, size=4.0, pxMode=True
                )
                self._items.append(scatter)
                self.gl_widget.addItem(scatter)
            else:
                z_orig = side.points[:, 2]
                z_norm = (z_orig - z_min) / z_range
                colors = cmap(z_norm)

                line_color = np.array([0.7, 0.7, 0.7, 0.5])
                line = gl.GLLinePlotItem(
                    pos=pts, color=line_color, width=1.0, antialias=True
                )
                self._items.append(line)
                self.gl_widget.addItem(line)

                scatter = gl.GLScatterPlotItem(
                    pos=pts, color=colors, size=5.0, pxMode=True
                )
                self._items.append(scatter)
                self.gl_widget.addItem(scatter)

        grid = gl.GLGridItem()
        grid.setSize(max_span * 1.2, max_span * 1.2, 0)
        grid.setSpacing(max_span / 20, max_span / 20, 1)
        grid_z = (all_pts[:, 2].min() - center_z) * z_scale - 10
        grid.translate(0, 0, grid_z)
        self._items.append(grid)
        self.gl_widget.addItem(grid)

        self.gl_widget.setCameraPosition(distance=max_span * 0.8)
