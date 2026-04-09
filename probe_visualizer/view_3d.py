import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt

import pyqtgraph.opengl as gl
import matplotlib.cm as cm

from probe_visualizer.colors import SIDE_COLORS


def _hex_to_gl(hex_color, alpha=1.0):
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (r, g, b, alpha)


class View3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gl_widget = gl.GLViewWidget()

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

    def _get_z_scale(self):
        return self.z_scale_slider.value() / 10.0

    def _on_z_scale_changed(self, value):
        scale = value / 10.0
        self.z_label.setText(f"Z scale: {scale:.1f}x")
        if self.data is not None:
            self._draw()

    def update_plot(self, data, visible_sides, color_mode):
        self.data = data
        self.visible_sides = visible_sides
        self.color_mode = color_mode
        self._draw()

    def _draw(self):
        for item in self._items:
            self.gl_widget.removeItem(item)
        self._items.clear()

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

        for side in self.data.sides:
            if not self.visible_sides.get(side.name, True):
                continue

            pts = side.points.copy()
            pts[:, 0] -= center_x
            pts[:, 1] -= center_y
            pts[:, 2] = (pts[:, 2] - center_z) * z_scale

            if self.color_mode == "side":
                color = _hex_to_gl(SIDE_COLORS.get(side.name, "#888888"))
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
