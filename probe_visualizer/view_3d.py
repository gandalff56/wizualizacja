import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QMatrix4x4, QVector4D

import pyqtgraph.opengl as gl
import matplotlib.cm as cm

from probe_visualizer.colors import get_color_for_side

try:
    from scipy.spatial import Delaunay
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _hex_to_gl(hex_color, alpha=1.0):
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (r, g, b, alpha)


class InteractiveGLWidget(gl.GLViewWidget):
    """GLViewWidget subclass with mouse tracking for hover and click detection."""

    hover_moved = pyqtSignal(object)  # QPoint or None
    point_clicked = pyqtSignal(object)  # QPoint

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._press_pos = None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._press_pos = ev.pos()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        self.hover_moved.emit(ev.pos())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._press_pos is not None:
            delta = ev.pos() - self._press_pos
            if abs(delta.x()) < 5 and abs(delta.y()) < 5:
                self.point_clicked.emit(ev.pos())
        self._press_pos = None
        super().mouseReleaseEvent(ev)


class View3D(QWidget):
    point_selected = pyqtSignal(int, int, str, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gl_widget = InteractiveGLWidget()
        self.gl_widget.hover_moved.connect(self._on_mouse_move)
        self.gl_widget.point_clicked.connect(self._on_click)

        # Tooltip overlay
        self.tooltip_label = QLabel(self.gl_widget)
        self.tooltip_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180); color: white; "
            "padding: 4px 8px; border-radius: 4px; font-size: 12px;"
        )
        self.tooltip_label.hide()

        # Z scale slider
        self.z_scale_slider = QSlider(Qt.Horizontal)
        self.z_scale_slider.setMinimum(1)
        self.z_scale_slider.setMaximum(200)
        self.z_scale_slider.setValue(10)
        self.z_scale_slider.valueChanged.connect(self._on_z_scale_changed)
        self.z_label = QLabel("1.0x")

        # Point size slider
        self.pt_size_slider = QSlider(Qt.Horizontal)
        self.pt_size_slider.setMinimum(1)
        self.pt_size_slider.setMaximum(20)
        self.pt_size_slider.setValue(4)
        self.pt_size_slider.valueChanged.connect(self._on_pt_size_changed)
        self.pt_label = QLabel("4")

        # Show surface checkbox
        self.surface_check = QCheckBox("Surface")
        self.surface_check.setChecked(False)
        self.surface_check.stateChanged.connect(self._on_surface_toggled)

        # Show arrows checkbox
        self.arrows_check = QCheckBox("Arrows")
        self.arrows_check.setChecked(True)
        self.arrows_check.stateChanged.connect(self._on_arrows_toggled)

        # Controls layout
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Z Scale:"))
        controls.addWidget(self.z_scale_slider)
        controls.addWidget(self.z_label)
        controls.addSpacing(15)
        controls.addWidget(QLabel("Point Size:"))
        controls.addWidget(self.pt_size_slider)
        controls.addWidget(self.pt_label)
        controls.addSpacing(15)
        controls.addWidget(self.arrows_check)
        controls.addWidget(self.surface_check)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_widget, stretch=1)
        layout.addLayout(controls)

        self.data = None
        self.visible_sides = {}
        self.color_mode = "side"
        self._items = []
        self._point_map = []
        self._selected_side_idx = -1
        self._selected_pt_idx = -1
        self._highlight_item = None

    # --- Projection & hit detection ---

    def _get_matrices(self):
        w = self.gl_widget.width()
        h = self.gl_widget.height()
        view_matrix = QMatrix4x4(
            np.array(self.gl_widget.viewMatrix().glData(), dtype=np.float32)
        )
        proj_matrix = QMatrix4x4(
            np.array(self.gl_widget.projectionMatrix().glData(), dtype=np.float32)
        )
        return view_matrix, proj_matrix, w, h

    def _project_point(self, pt_3d, view_matrix, proj_matrix, width, height):
        v = QVector4D(float(pt_3d[0]), float(pt_3d[1]), float(pt_3d[2]), 1.0)
        v = view_matrix * v
        v = proj_matrix * v
        if abs(v.w()) < 1e-10:
            return None
        screen_x = (v.x() / v.w() + 1.0) * 0.5 * width
        screen_y = (1.0 - v.y() / v.w()) * 0.5 * height
        return (screen_x, screen_y)

    def _find_nearest_point(self, pos, max_dist=15.0):
        mx, my = pos.x(), pos.y()
        view_matrix, proj_matrix, w, h = self._get_matrices()
        best_dist = max_dist
        best = None
        for side_data_idx, side_name, orig_pts, trans_pts in self._point_map:
            for j in range(len(trans_pts)):
                screen = self._project_point(
                    trans_pts[j], view_matrix, proj_matrix, w, h
                )
                if screen is None:
                    continue
                dist = ((screen[0] - mx) ** 2 + (screen[1] - my) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = (side_data_idx, j, side_name, orig_pts[j])
        return best

    # --- Mouse events ---

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
            hint = self.tooltip_label.sizeHint()
            tx = min(mx + 15, w - hint.width() - 5)
            ty = max(my - hint.height() - 5, 5)
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

    # --- Slider/checkbox callbacks ---

    def _on_z_scale_changed(self, value):
        self.z_label.setText(f"{value / 10.0:.1f}x")
        if self.data is not None:
            self._draw()
            self._update_highlight()

    def _on_pt_size_changed(self, value):
        self.pt_label.setText(str(value))
        if self.data is not None:
            self._draw()
            self._update_highlight()

    def _on_surface_toggled(self):
        if self.data is not None:
            self._draw()
            self._update_highlight()

    def _on_arrows_toggled(self):
        if self.data is not None:
            self._draw()
            self._update_highlight()

    # --- Public API ---

    def update_plot(self, data, visible_sides, color_mode):
        self.data = data
        self.visible_sides = visible_sides
        self.color_mode = color_mode
        self._selected_side_idx = -1
        self._selected_pt_idx = -1
        self._draw()

    def update_selected_point(self, side_idx, pt_idx, x, y, z):
        if self.data is None:
            return
        self.data.sides[side_idx].points[pt_idx] = [x, y, z]
        self._selected_side_idx = side_idx
        self._selected_pt_idx = pt_idx
        self._draw()
        self._update_highlight()

    # --- Highlight ---

    def _update_highlight(self):
        if self._highlight_item is not None:
            self.gl_widget.removeItem(self._highlight_item)
            self._highlight_item = None
        if self._selected_side_idx < 0:
            return
        for side_data_idx, _, _, trans_pts in self._point_map:
            if side_data_idx == self._selected_side_idx:
                if self._selected_pt_idx < len(trans_pts):
                    pt = trans_pts[self._selected_pt_idx : self._selected_pt_idx + 1]
                    self._highlight_item = gl.GLScatterPlotItem(
                        pos=pt,
                        color=np.array([[1.0, 1.0, 1.0, 1.0]]),
                        size=14.0,
                        pxMode=True,
                    )
                    self.gl_widget.addItem(self._highlight_item)
                break

    # --- Main draw ---

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
        z_scale = self.z_scale_slider.value() / 10.0
        pt_size = self.pt_size_slider.value()
        show_arrows = self.arrows_check.isChecked()
        show_surface = self.surface_check.isChecked() and HAS_SCIPY

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

        # Collect all transformed points for surface
        all_trans = []

        for i, side in enumerate(self.data.sides):
            if not self.visible_sides.get(side.name, True):
                continue

            pts = side.points.copy()
            pts[:, 0] -= center_x
            pts[:, 1] -= center_y
            pts[:, 2] = (pts[:, 2] - center_z) * z_scale

            self._point_map.append((i, side.name, side.points.copy(), pts.copy()))
            all_trans.append(pts)

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
                    pos=pts, color=colors, size=pt_size, pxMode=True
                )
                self._items.append(scatter)
                self.gl_widget.addItem(scatter)
            else:
                z_orig = side.points[:, 2]
                z_norm = (z_orig - z_min) / z_range
                colors = cmap(z_norm)

                line = gl.GLLinePlotItem(
                    pos=pts,
                    color=np.array([0.7, 0.7, 0.7, 0.5]),
                    width=1.0,
                    antialias=True,
                )
                self._items.append(line)
                self.gl_widget.addItem(line)

                scatter = gl.GLScatterPlotItem(
                    pos=pts, color=colors, size=pt_size, pxMode=True
                )
                self._items.append(scatter)
                self.gl_widget.addItem(scatter)

            # Direction arrows
            if show_arrows and len(pts) > 1:
                self._draw_arrows(pts, i)

        # Mesh surface
        if show_surface and all_trans:
            self._draw_surface(all_trans, z_min, z_max, z_scale, center_z, cmap)

        # Grid
        grid = gl.GLGridItem()
        grid.setSize(max_span * 1.2, max_span * 1.2, 0)
        grid.setSpacing(max_span / 20, max_span / 20, 1)
        grid_z = (all_pts[:, 2].min() - center_z) * z_scale - 10
        grid.translate(0, 0, grid_z)
        self._items.append(grid)
        self.gl_widget.addItem(grid)

        self.gl_widget.setCameraPosition(distance=max_span * 0.8)

    def _draw_arrows(self, pts, side_idx):
        """Draw direction arrows every ~10 points."""
        step = max(1, len(pts) // 15)
        arrow_len = 0.02 * max(
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            1.0,
        )

        for j in range(0, len(pts) - 1, step):
            p0 = pts[j]
            p1 = pts[min(j + 1, len(pts) - 1)]
            direction = p1 - p0
            length = np.linalg.norm(direction)
            if length < 1e-10:
                continue
            direction = direction / length

            # Arrow shaft: short line from point in direction
            tip = p0 + direction * arrow_len * 3
            # Arrow head: two angled lines
            perp = np.array([-direction[1], direction[0], 0.0])
            head_size = arrow_len * 1.5
            h1 = tip - direction * head_size + perp * head_size * 0.5
            h2 = tip - direction * head_size - perp * head_size * 0.5

            arrow_pts = np.array([p0, tip, h1, tip, h2, tip])
            arrow = gl.GLLinePlotItem(
                pos=arrow_pts,
                color=(1.0, 1.0, 1.0, 0.7),
                width=1.5,
                antialias=True,
                mode="lines",
            )
            self._items.append(arrow)
            self.gl_widget.addItem(arrow)

    def _draw_surface(self, all_trans, z_min, z_max, z_scale, center_z, cmap):
        """Draw interpolated mesh surface from all visible points."""
        combined = np.vstack(all_trans)
        if len(combined) < 4:
            return

        try:
            tri = Delaunay(combined[:, :2])
        except Exception:
            return

        z_range = z_max - z_min if z_max - z_min > 1e-9 else 1.0

        # Compute face colors from average Z of vertices
        faces = tri.simplices
        verts = combined

        # Filter out very large triangles (artifacts from Delaunay)
        edge_threshold = np.median(np.linalg.norm(
            np.diff(combined, axis=0), axis=1
        )) * 5.0

        good_faces = []
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            e1 = np.linalg.norm(v1 - v0)
            e2 = np.linalg.norm(v2 - v1)
            e3 = np.linalg.norm(v0 - v2)
            if max(e1, e2, e3) < edge_threshold:
                good_faces.append(face)

        if not good_faces:
            return

        good_faces = np.array(good_faces)

        # Face colors based on average Z
        face_colors = np.zeros((len(good_faces), 4))
        for fi, face in enumerate(good_faces):
            avg_z = np.mean(verts[face, 2])
            # Convert back to original Z for coloring
            orig_z = avg_z / z_scale + center_z
            z_norm = (orig_z - z_min) / z_range
            z_norm = np.clip(z_norm, 0, 1)
            c = cmap(z_norm)
            face_colors[fi] = [c[0], c[1], c[2], 0.4]

        md = gl.MeshData(vertexes=verts, faces=good_faces, faceColors=face_colors)
        mesh_item = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            shader="balloon",
            glOptions="translucent",
        )
        self._items.append(mesh_item)
        self.gl_widget.addItem(mesh_item)
