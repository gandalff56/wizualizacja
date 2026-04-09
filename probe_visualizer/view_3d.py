import time
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

import pyqtgraph.opengl as gl
import matplotlib.cm as cm

from probe_visualizer.colors import get_color_for_side

try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _hex_to_gl(hex_color, alpha=1.0):
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (r, g, b, alpha)


class InteractiveGLWidget(gl.GLViewWidget):
    """GLViewWidget with mouse tracking for hover and click."""

    hover_moved = pyqtSignal(object)
    point_clicked = pyqtSignal(object)

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
        self.gl_widget.hover_moved.connect(self._on_mouse_move_raw)
        self.gl_widget.point_clicked.connect(self._on_click)

        # Tooltip
        self.tooltip_label = QLabel(self.gl_widget)
        self.tooltip_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180); color: white; "
            "padding: 4px 8px; border-radius: 4px; font-size: 12px;"
        )
        self.tooltip_label.hide()

        # Throttled hover timer (30fps max)
        self._hover_timer = QTimer()
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(33)  # ~30fps
        self._hover_timer.timeout.connect(self._on_hover_throttled)
        self._pending_hover_pos = None

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

        # Checkboxes
        self.surface_check = QCheckBox("Surface")
        self.surface_check.setChecked(False)
        self.surface_check.stateChanged.connect(self._on_surface_toggled)

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

        # State
        self.data = None
        self.visible_sides = {}
        self.color_mode = "side"
        self._items = []

        # Point map: list of (side_data_idx, side_name, orig_pts, trans_pts)
        self._point_map = []
        # Flat arrays for vectorized hover lookup
        self._all_trans_flat = None   # (N, 3) all transformed points
        self._all_orig_flat = None    # (N, 3) all original points
        self._all_side_idx = None     # (N,) side index per point
        self._all_pt_idx = None       # (N,) point index within side
        self._all_side_names = None   # (N,) side name per point

        # Selection
        self._selected_side_idx = -1
        self._selected_pt_idx = -1
        self._highlight_item = None

        # Cached geometry (survives z_scale / pt_size changes)
        self._cached_delaunay = None  # Delaunay triangulation
        self._cached_good_faces = None
        self._cached_center = None    # (cx, cy, cz)
        self._cached_max_span = None
        self._cached_z_range = None   # (z_min, z_max)
        self._data_version = 0        # incremented on data/visibility change

    # === VECTORIZED HOVER/CLICK (no Python loops) ===

    def _project_all_points(self, trans_pts):
        """Project N x 3 points to screen coords using numpy vectorization."""
        w = self.gl_widget.width()
        h = self.gl_widget.height()
        if w == 0 or h == 0:
            return None

        # Get 4x4 matrices as numpy arrays
        vm = np.array(self.gl_widget.viewMatrix().glData(), dtype=np.float64).reshape(4, 4)
        pm = np.array(self.gl_widget.projectionMatrix().glData(), dtype=np.float64).reshape(4, 4)
        mvp = pm @ vm  # combined model-view-projection

        # Homogeneous coords: (N, 4)
        n = len(trans_pts)
        ones = np.ones((n, 1), dtype=np.float64)
        pts4 = np.hstack([trans_pts, ones])  # (N, 4)

        # Transform all at once
        clip = (mvp @ pts4.T).T  # (N, 4)

        # Perspective divide
        w_clip = clip[:, 3]
        valid = np.abs(w_clip) > 1e-10
        ndc = np.zeros((n, 2), dtype=np.float64)
        ndc[valid, 0] = clip[valid, 0] / w_clip[valid]
        ndc[valid, 1] = clip[valid, 1] / w_clip[valid]

        # NDC to screen
        screen_x = (ndc[:, 0] + 1.0) * 0.5 * w
        screen_y = (1.0 - ndc[:, 1]) * 0.5 * h

        return screen_x, screen_y, valid

    def _find_nearest_point(self, pos, max_dist=15.0):
        """Vectorized nearest-point search."""
        if self._all_trans_flat is None or len(self._all_trans_flat) == 0:
            return None

        mx, my = pos.x(), pos.y()
        result = self._project_all_points(self._all_trans_flat)
        if result is None:
            return None

        sx, sy, valid = result
        dist_sq = (sx - mx) ** 2 + (sy - my) ** 2
        dist_sq[~valid] = 1e18  # invalidate

        max_dist_sq = max_dist * max_dist
        idx = np.argmin(dist_sq)
        if dist_sq[idx] > max_dist_sq:
            return None

        return (
            int(self._all_side_idx[idx]),
            int(self._all_pt_idx[idx]),
            self._all_side_names[idx],
            self._all_orig_flat[idx],
        )

    # === MOUSE EVENTS ===

    def _on_mouse_move_raw(self, pos):
        """Throttle hover events to max ~30fps."""
        self._pending_hover_pos = pos
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _on_hover_throttled(self):
        pos = self._pending_hover_pos
        if pos is None or self.data is None:
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
        if self.data is None:
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

    # === SLIDER/CHECKBOX CALLBACKS ===

    def _on_z_scale_changed(self, value):
        self.z_label.setText(f"{value / 10.0:.1f}x")
        if self.data is not None:
            self._draw_fast()  # reuse cached geometry

    def _on_pt_size_changed(self, value):
        self.pt_label.setText(str(value))
        if self.data is not None:
            self._draw_fast()

    def _on_surface_toggled(self):
        if self.data is not None:
            self._draw_fast()

    def _on_arrows_toggled(self):
        if self.data is not None:
            self._draw_fast()

    # === PUBLIC API ===

    def update_plot(self, data, visible_sides, color_mode):
        self.data = data
        self.visible_sides = visible_sides
        self.color_mode = color_mode
        self._selected_side_idx = -1
        self._selected_pt_idx = -1
        self._invalidate_cache()
        self._draw_full()

    def update_selected_point(self, side_idx, pt_idx, x, y, z):
        if self.data is None:
            return
        self.data.sides[side_idx].points[pt_idx] = [x, y, z]
        self._selected_side_idx = side_idx
        self._selected_pt_idx = pt_idx
        self._invalidate_cache()
        self._draw_full()
        self._update_highlight()

    # === CACHE ===

    def _invalidate_cache(self):
        self._cached_delaunay = None
        self._cached_good_faces = None
        self._cached_center = None
        self._data_version += 1

    def _compute_geometry_cache(self):
        """Pre-compute center, spans, Delaunay - called once per data change."""
        all_pts = self.data.all_points()
        cx = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2.0
        cy = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2.0
        cz = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2.0
        span_x = all_pts[:, 0].max() - all_pts[:, 0].min()
        span_y = all_pts[:, 1].max() - all_pts[:, 1].min()
        max_span = max(span_x, span_y)
        z_min = float(all_pts[:, 2].min())
        z_max = float(all_pts[:, 2].max())

        self._cached_center = (cx, cy, cz)
        self._cached_max_span = max_span
        self._cached_z_range = (z_min, z_max)

    def _compute_delaunay_cache(self, all_centered_xy):
        """Pre-compute Delaunay triangulation with face filtering."""
        if not HAS_SCIPY or len(all_centered_xy) < 4:
            self._cached_delaunay = None
            self._cached_good_faces = None
            return

        try:
            tri = Delaunay(all_centered_xy)
        except Exception:
            self._cached_delaunay = None
            self._cached_good_faces = None
            return

        self._cached_delaunay = tri

        # Vectorized edge length filtering
        faces = tri.simplices
        verts_xy = all_centered_xy

        v0 = verts_xy[faces[:, 0]]
        v1 = verts_xy[faces[:, 1]]
        v2 = verts_xy[faces[:, 2]]

        e1 = np.linalg.norm(v1 - v0, axis=1)
        e2 = np.linalg.norm(v2 - v1, axis=1)
        e3 = np.linalg.norm(v0 - v2, axis=1)
        max_edge = np.maximum(np.maximum(e1, e2), e3)

        # Threshold: 5x median consecutive-point distance
        diffs = np.diff(all_centered_xy, axis=0)
        median_dist = np.median(np.linalg.norm(diffs, axis=1))
        threshold = median_dist * 5.0

        mask = max_edge < threshold
        self._cached_good_faces = faces[mask]

    # === HIGHLIGHT ===

    def _update_highlight(self):
        if self._highlight_item is not None:
            self.gl_widget.removeItem(self._highlight_item)
            self._highlight_item = None
        if self._selected_side_idx < 0:
            return
        for side_data_idx, _, _, trans_pts in self._point_map:
            if side_data_idx == self._selected_side_idx:
                if self._selected_pt_idx < len(trans_pts):
                    pt = trans_pts[self._selected_pt_idx:self._selected_pt_idx + 1]
                    self._highlight_item = gl.GLScatterPlotItem(
                        pos=pt,
                        color=np.array([[1.0, 1.0, 1.0, 1.0]]),
                        size=14.0, pxMode=True,
                    )
                    self.gl_widget.addItem(self._highlight_item)
                break

    # === DRAWING ===

    def _clear_items(self):
        for item in self._items:
            self.gl_widget.removeItem(item)
        self._items.clear()
        if self._highlight_item is not None:
            self.gl_widget.removeItem(self._highlight_item)
            self._highlight_item = None
        self.tooltip_label.hide()

    def _draw_full(self):
        """Full draw: recompute geometry cache + render."""
        if self.data is None:
            self._clear_items()
            return
        self._compute_geometry_cache()
        self._draw_fast()

    def _draw_fast(self):
        """Fast draw: use cached geometry, only recompute z_scale transforms."""
        self._clear_items()
        self._point_map.clear()

        if self.data is None or self._cached_center is None:
            return

        cx, cy, cz = self._cached_center
        max_span = self._cached_max_span
        z_min, z_max = self._cached_z_range
        z_scale = self.z_scale_slider.value() / 10.0
        pt_size = self.pt_size_slider.value()
        show_arrows = self.arrows_check.isChecked()
        show_surface = self.surface_check.isChecked() and HAS_SCIPY
        z_range = z_max - z_min if z_max - z_min > 1e-9 else 1.0
        cmap = cm.get_cmap("viridis")

        # Build transformed points and flat arrays for hover
        all_trans_list = []
        flat_trans = []
        flat_orig = []
        flat_side_idx = []
        flat_pt_idx = []
        flat_side_names = []

        for i, side in enumerate(self.data.sides):
            if not self.visible_sides.get(side.name, True):
                continue

            pts = side.points.copy()
            pts[:, 0] -= cx
            pts[:, 1] -= cy
            pts[:, 2] = (pts[:, 2] - cz) * z_scale

            self._point_map.append((i, side.name, side.points.copy(), pts.copy()))
            all_trans_list.append(pts)

            n = len(pts)
            flat_trans.append(pts)
            flat_orig.append(side.points.copy())
            flat_side_idx.append(np.full(n, i, dtype=np.int32))
            flat_pt_idx.append(np.arange(n, dtype=np.int32))
            flat_side_names.extend([side.name] * n)

            # Draw lines + scatter
            if self.color_mode == "side":
                color = _hex_to_gl(get_color_for_side(side.name, i))
                self._add_item(gl.GLLinePlotItem(
                    pos=pts, color=color, width=2.0, antialias=True
                ))
                colors_arr = np.tile(np.array(color), (n, 1))
                self._add_item(gl.GLScatterPlotItem(
                    pos=pts, color=colors_arr, size=pt_size, pxMode=True
                ))
            else:
                z_norm = (side.points[:, 2] - z_min) / z_range
                colors_arr = cmap(z_norm)
                self._add_item(gl.GLLinePlotItem(
                    pos=pts, color=np.array([0.7, 0.7, 0.7, 0.5]),
                    width=1.0, antialias=True,
                ))
                self._add_item(gl.GLScatterPlotItem(
                    pos=pts, color=colors_arr, size=pt_size, pxMode=True
                ))

            # Direction arrows (batched into single GLLinePlotItem)
            if show_arrows and n > 1:
                self._draw_arrows_batched(pts)

        # Build flat arrays for hover
        if flat_trans:
            self._all_trans_flat = np.vstack(flat_trans)
            self._all_orig_flat = np.vstack(flat_orig)
            self._all_side_idx = np.concatenate(flat_side_idx)
            self._all_pt_idx = np.concatenate(flat_pt_idx)
            self._all_side_names = flat_side_names
        else:
            self._all_trans_flat = None
            self._all_orig_flat = None

        # Mesh surface (compute Delaunay only if cache invalid)
        if show_surface and all_trans_list:
            combined = np.vstack(all_trans_list)
            if self._cached_delaunay is None:
                self._compute_delaunay_cache(combined[:, :2])
            self._draw_surface_cached(combined, z_min, z_max, z_scale, cz, cmap)

        # Grid
        grid = gl.GLGridItem()
        grid.setSize(max_span * 1.2, max_span * 1.2, 0)
        grid.setSpacing(max_span / 20, max_span / 20, 1)
        grid_z = (z_min - cz) * z_scale - 10
        grid.translate(0, 0, grid_z)
        self._add_item(grid)

        self.gl_widget.setCameraPosition(distance=max_span * 0.8)
        self._update_highlight()

    def _add_item(self, item):
        self._items.append(item)
        self.gl_widget.addItem(item)

    def _draw_arrows_batched(self, pts):
        """Draw all arrows for a side in a single GLLinePlotItem."""
        n = len(pts)
        step = max(1, n // 15)
        arrow_len = 0.02 * max(
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            1.0,
        )

        lines = []
        for j in range(0, n - 1, step):
            p0 = pts[j]
            p1 = pts[min(j + 1, n - 1)]
            d = p1 - p0
            length = np.linalg.norm(d)
            if length < 1e-10:
                continue
            d = d / length
            tip = p0 + d * arrow_len * 3
            perp = np.array([-d[1], d[0], 0.0])
            hs = arrow_len * 1.5
            h1 = tip - d * hs + perp * hs * 0.5
            h2 = tip - d * hs - perp * hs * 0.5
            lines.extend([p0, tip, h1, tip, h2, tip])

        if lines:
            self._add_item(gl.GLLinePlotItem(
                pos=np.array(lines),
                color=(1.0, 1.0, 1.0, 0.7),
                width=1.5, antialias=True, mode="lines",
            ))

    def _draw_surface_cached(self, combined, z_min, z_max, z_scale, cz, cmap):
        """Draw mesh using cached Delaunay faces."""
        if self._cached_good_faces is None or len(self._cached_good_faces) == 0:
            return

        z_range = z_max - z_min if z_max - z_min > 1e-9 else 1.0
        faces = self._cached_good_faces
        verts = combined

        # Vectorized face color computation
        avg_z = np.mean(verts[faces, 2], axis=1)
        orig_z = avg_z / max(z_scale, 1e-10) + cz
        z_norm = np.clip((orig_z - z_min) / z_range, 0, 1)
        rgba = cmap(z_norm)
        rgba[:, 3] = 0.4

        md = gl.MeshData(vertexes=verts, faces=faces, faceColors=rgba)
        self._add_item(gl.GLMeshItem(
            meshdata=md, smooth=True,
            shader="balloon", glOptions="translucent",
        ))
