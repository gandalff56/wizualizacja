import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

import pyqtgraph.opengl as gl

from probe_visualizer.colors import get_color_for_side

# Lazy imports
_cmap_cache = None
_Delaunay = None


def _get_cmap():
    global _cmap_cache
    if _cmap_cache is None:
        import matplotlib.cm as cm
        _cmap_cache = cm.get_cmap("viridis")
    return _cmap_cache


def _get_delaunay():
    global _Delaunay
    if _Delaunay is None:
        try:
            from scipy.spatial import Delaunay
            _Delaunay = Delaunay
        except ImportError:
            _Delaunay = False
    return _Delaunay if _Delaunay is not False else None


def _hex_to_gl(hex_color, alpha=1.0):
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (r, g, b, alpha)


def _ev_pos(ev):
    if hasattr(ev, 'position'):
        return ev.position().toPoint()
    return ev.pos()


class InteractiveGLWidget(gl.GLViewWidget):
    point_clicked = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default: full pyqtgraph navigation (left = orbit, middle = pan,
        # wheel = zoom). When pan_mode is enabled via the "Change view"
        # checkbox, left-drag acts like middle-drag (pan only, no rotation).
        self._pan_mode = False
        self._pan_last_pos = None

    def set_pan_mode(self, enabled: bool):
        self._pan_mode = enabled
        self._pan_last_pos = None

    def mousePressEvent(self, ev):
        if self._pan_mode and ev.button() == Qt.LeftButton:
            self._pan_last_pos = _ev_pos(ev)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if (
            self._pan_mode
            and (ev.buttons() & Qt.LeftButton)
            and self._pan_last_pos is not None
        ):
            pos = _ev_pos(ev)
            diff = pos - self._pan_last_pos
            self._pan_last_pos = pos
            # Same call pyqtgraph's GLViewWidget uses for middle-button drag.
            self.pan(diff.x(), diff.y(), 0, relative="view-upright")
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.point_clicked.emit(_ev_pos(ev))
        super().mouseDoubleClickEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._pan_last_pos = None
        if ev.button() == Qt.RightButton:
            self.point_clicked.emit(_ev_pos(ev))
        super().mouseReleaseEvent(ev)


class View3D(QWidget):
    point_selected = pyqtSignal(int, int, str, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gl_widget = InteractiveGLWidget()
        self.gl_widget.point_clicked.connect(self._on_click)

        # Tooltip
        self.tooltip_label = QLabel(self.gl_widget)
        self.tooltip_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180); color: white; "
            "padding: 4px 8px; border-radius: 4px; font-size: 12px;"
        )
        self.tooltip_label.hide()

        # Hover: poll cursor every 50ms
        self._last_hover_x = -1
        self._last_hover_y = -1
        self._hover_timer = QTimer()
        self._hover_timer.setInterval(50)
        self._hover_timer.timeout.connect(self._poll_cursor)
        self._hover_timer.start()

        # Z scale slider
        self.z_scale_slider = QSlider(Qt.Horizontal)
        self.z_scale_slider.setMinimum(1)
        self.z_scale_slider.setMaximum(400)
        self.z_scale_slider.setValue(10)
        self.z_scale_slider.valueChanged.connect(self._on_z_scale_changed)
        self.z_label = QLabel("1.0x")

        # Y scale slider (1.0x - 5.0x)
        self.y_scale_slider = QSlider(Qt.Horizontal)
        self.y_scale_slider.setMinimum(10)
        self.y_scale_slider.setMaximum(50)
        self.y_scale_slider.setValue(10)
        self.y_scale_slider.valueChanged.connect(self._on_y_scale_changed)
        self.y_label = QLabel("1.0x")

        # Point size slider
        self.pt_size_slider = QSlider(Qt.Horizontal)
        self.pt_size_slider.setMinimum(1)
        self.pt_size_slider.setMaximum(20)
        self.pt_size_slider.setValue(4)
        self.pt_size_slider.valueChanged.connect(self._on_pt_size_changed)
        self.pt_label = QLabel("4")

        # Checkboxes
        self.arrows_check = QCheckBox("Arrows")
        self.arrows_check.setChecked(False)
        self.arrows_check.stateChanged.connect(self._on_arrows_toggled)
        self.change_view_check = QCheckBox("Change view")
        self.change_view_check.setChecked(False)
        self.change_view_check.stateChanged.connect(self._on_change_view_toggled)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Z Scale:"))
        controls.addWidget(self.z_scale_slider)
        controls.addWidget(self.z_label)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Y Scale:"))
        controls.addWidget(self.y_scale_slider)
        controls.addWidget(self.y_label)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Point Size:"))
        controls.addWidget(self.pt_size_slider)
        controls.addWidget(self.pt_label)
        controls.addSpacing(10)
        controls.addWidget(self.arrows_check)
        controls.addWidget(self.change_view_check)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_widget, stretch=1)
        layout.addLayout(controls)

        # State
        self.data = None
        self.visible_sides = {}
        self.color_mode = "side"
        self._items = []
        self._point_map = []

        # Flat arrays for picking
        self._all_trans_flat = None
        self._all_orig_flat = None
        self._all_side_idx = None
        self._all_pt_idx = None
        self._all_side_names = None

        # Selection
        self._selected_side_idx = -1
        self._selected_pt_idx = -1
        self._highlight_item = None

        # Cached geometry
        self._cached_delaunay = None
        self._cached_good_faces = None
        self._cached_center = None
        self._cached_max_span = None
        self._cached_z_range = None
        self._data_version = 0

    # === PROJECTION (using pyqtgraph's own matrix methods) ===

    @staticmethod
    def _qmatrix_to_numpy(m):
        """Convert QMatrix4x4 to numpy 4x4 array (row-major)."""
        arr = np.zeros((4, 4), dtype=np.float64)
        for r in range(4):
            for c in range(4):
                arr[r, c] = m.row(r)[c]
        return arr

    def _project_all_points(self, trans_pts):
        """Project 3D points to 2D screen coords using pyqtgraph's matrices."""
        w = self.gl_widget.width()
        h = self.gl_widget.height()
        if w == 0 or h == 0 or len(trans_pts) == 0:
            return None

        try:
            vm = self.gl_widget.viewMatrix()
            pm = self.gl_widget.projectionMatrix(
                region=(0, 0, w, h), viewport=(0, 0, w, h)
            )
        except Exception:
            return None

        vm_np = self._qmatrix_to_numpy(vm)
        pm_np = self._qmatrix_to_numpy(pm)
        mvp = pm_np @ vm_np

        n = len(trans_pts)
        pts4 = np.hstack([trans_pts, np.ones((n, 1), dtype=np.float64)])
        clip = (mvp @ pts4.T).T

        w_clip = clip[:, 3]
        valid = np.abs(w_clip) > 1e-10

        screen_x = np.zeros(n, dtype=np.float64)
        screen_y = np.zeros(n, dtype=np.float64)
        screen_x[valid] = (clip[valid, 0] / w_clip[valid] + 1.0) * 0.5 * w
        screen_y[valid] = (1.0 - clip[valid, 1] / w_clip[valid]) * 0.5 * h

        return screen_x, screen_y, valid

    def _find_nearest_point(self, mx, my, max_dist=20.0):
        if self._all_trans_flat is None or len(self._all_trans_flat) == 0:
            return None

        result = self._project_all_points(self._all_trans_flat)
        if result is None:
            return None

        sx, sy, valid = result
        dist_sq = (sx - mx) ** 2 + (sy - my) ** 2
        dist_sq[~valid] = 1e18

        idx = np.argmin(dist_sq)
        if dist_sq[idx] > max_dist * max_dist:
            return None

        return (
            int(self._all_side_idx[idx]),
            int(self._all_pt_idx[idx]),
            self._all_side_names[idx],
            self._all_orig_flat[idx],
        )

    # === HOVER + CLICK ===

    def _poll_cursor(self):
        if self.data is None or self._all_trans_flat is None:
            if self.tooltip_label.isVisible():
                self.tooltip_label.hide()
            return

        from PyQt5.QtGui import QCursor
        gpos = QCursor.pos()
        lpos = self.gl_widget.mapFromGlobal(gpos)
        mx, my = lpos.x(), lpos.y()

        if mx < 0 or my < 0 or mx >= self.gl_widget.width() or my >= self.gl_widget.height():
            if self.tooltip_label.isVisible():
                self.tooltip_label.hide()
            return

        if mx == self._last_hover_x and my == self._last_hover_y:
            return
        self._last_hover_x = mx
        self._last_hover_y = my

        try:
            result = self._find_nearest_point(mx, my)
        except Exception:
            self.tooltip_label.hide()
            return

        if result is not None:
            _, _, side_name, orig_pt = result
            self.tooltip_label.setText(
                f"{side_name}\n"
                f"X: {orig_pt[0]:.3f}\n"
                f"Y: {orig_pt[1]:.3f}\n"
                f"Z: {orig_pt[2]:.3f}"
            )
            w = self.gl_widget.width()
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
        try:
            result = self._find_nearest_point(pos.x(), pos.y())
        except Exception:
            return
        if result is not None:
            side_idx, pt_idx, side_name, orig_pt = result
            self._selected_side_idx = side_idx
            self._selected_pt_idx = pt_idx
            self._update_highlight()
            self.point_selected.emit(
                side_idx, pt_idx, side_name,
                float(orig_pt[0]), float(orig_pt[1]), float(orig_pt[2])
            )

    # === CONTROLS ===

    def _on_z_scale_changed(self, value):
        self.z_label.setText(f"{value / 10.0:.1f}x")
        if self.data is not None:
            self._draw_fast()

    def _on_y_scale_changed(self, value):
        self.y_label.setText(f"{value / 10.0:.1f}x")
        if self.data is not None:
            # Y affects the XY Delaunay triangulation, invalidate cache
            self._invalidate_cache()
            self._draw_full()

    def _on_pt_size_changed(self, value):
        self.pt_label.setText(str(value))
        if self.data is not None:
            self._draw_fast()

    def _on_arrows_toggled(self):
        if self.data is not None:
            self._draw_fast()

    def _on_change_view_toggled(self):
        self.gl_widget.set_pan_mode(self.change_view_check.isChecked())

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

    def highlight_point(self, side_idx, pt_idx):
        self._selected_side_idx = side_idx
        self._selected_pt_idx = pt_idx
        self._update_highlight()

    # === CACHE ===

    def _invalidate_cache(self):
        self._cached_delaunay = None
        self._cached_good_faces = None
        self._cached_center = None
        self._data_version += 1

    def _compute_geometry_cache(self):
        all_pts = self.data.all_points()
        cx = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2.0
        cy = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2.0
        cz = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2.0
        span_x = all_pts[:, 0].max() - all_pts[:, 0].min()
        span_y = all_pts[:, 1].max() - all_pts[:, 1].min()
        self._cached_center = (cx, cy, cz)
        self._cached_max_span = max(span_x, span_y)
        self._cached_z_range = (float(all_pts[:, 2].min()), float(all_pts[:, 2].max()))

    def _compute_delaunay_cache(self, xy):
        Cls = _get_delaunay()
        if Cls is None or len(xy) < 4:
            self._cached_good_faces = None
            return
        try:
            tri = Cls(xy)
        except Exception:
            self._cached_good_faces = None
            return
        self._cached_delaunay = tri
        faces = tri.simplices
        v0, v1, v2 = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
        e1 = np.linalg.norm(v1 - v0, axis=1)
        e2 = np.linalg.norm(v2 - v1, axis=1)
        e3 = np.linalg.norm(v0 - v2, axis=1)
        mx_e = np.maximum(np.maximum(e1, e2), e3)
        threshold = np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1)) * 5.0
        self._cached_good_faces = faces[mx_e < threshold]

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
                        pos=pt, color=np.array([[1, 1, 1, 1.0]]),
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

    def _add_item(self, item):
        self._items.append(item)
        self.gl_widget.addItem(item)

    def _draw_full(self):
        if self.data is None:
            self._clear_items()
            return
        self._compute_geometry_cache()
        self._draw_fast()

    def _draw_fast(self):
        self._clear_items()
        self._point_map.clear()
        if self.data is None or self._cached_center is None:
            return

        cx, cy, cz = self._cached_center
        max_span = self._cached_max_span
        z_min, z_max = self._cached_z_range
        z_scale = self.z_scale_slider.value() / 10.0
        y_scale = self.y_scale_slider.value() / 10.0
        pt_size = self.pt_size_slider.value()
        show_arrows = self.arrows_check.isChecked()
        z_range = z_max - z_min if z_max - z_min > 1e-9 else 1.0
        cmap = _get_cmap()

        all_trans_list = []
        flat_trans, flat_orig = [], []
        flat_side_idx, flat_pt_idx = [], []
        flat_side_names = []
        visible_order = []  # (side_idx, transformed_pts, orig_span) in draw order

        for i, side in enumerate(self.data.sides):
            if not self.visible_sides.get(side.name, True):
                continue
            pts = side.points.copy()
            pts[:, 0] -= cx
            pts[:, 1] = (pts[:, 1] - cy) * y_scale
            pts[:, 2] = (pts[:, 2] - cz) * z_scale

            self._point_map.append((i, side.name, side.points.copy(), pts.copy()))
            all_trans_list.append(pts)
            n = len(pts)
            flat_trans.append(pts)
            flat_orig.append(side.points.copy())
            flat_side_idx.append(np.full(n, i, dtype=np.int32))
            flat_pt_idx.append(np.arange(n, dtype=np.int32))
            flat_side_names.extend([side.name] * n)

            if self.color_mode == "side":
                color = _hex_to_gl(get_color_for_side(side.name, i))
                self._add_item(gl.GLLinePlotItem(pos=pts, color=color, width=2.0, antialias=True))
                self._add_item(gl.GLScatterPlotItem(
                    pos=pts, color=np.tile(np.array(color), (n, 1)),
                    size=pt_size, pxMode=True
                ))
            else:
                z_norm = (side.points[:, 2] - z_min) / z_range
                self._add_item(gl.GLLinePlotItem(
                    pos=pts, color=np.array([0.7, 0.7, 0.7, 0.5]), width=1.0, antialias=True
                ))
                self._add_item(gl.GLScatterPlotItem(
                    pos=pts, color=cmap(z_norm), size=pt_size, pxMode=True
                ))

            # Record the side's own span (first → last point in original XY)
            # for long/short classification when drawing arrows.
            orig = side.points
            span = float(np.linalg.norm(orig[-1, :2] - orig[0, :2])) if n > 1 else 0.0
            visible_order.append((i, pts, span))

        if show_arrows and visible_order:
            # Classify visible sides by span: top half = long, bottom half = short.
            spans_sorted = sorted((s for _, _, s in visible_order), reverse=True)
            half = max(1, len(spans_sorted) // 2)
            long_threshold = spans_sorted[half - 1]
            long_drawn = False
            short_drawn = False
            for _, pts, span in visible_order:
                if len(pts) < 2:
                    continue
                is_long = span >= long_threshold
                if is_long and not long_drawn:
                    self._draw_arrow(pts)
                    long_drawn = True
                elif not is_long and not short_drawn:
                    self._draw_arrow(pts)
                    short_drawn = True
                if long_drawn and short_drawn:
                    break

        if flat_trans:
            self._all_trans_flat = np.vstack(flat_trans)
            self._all_orig_flat = np.vstack(flat_orig)
            self._all_side_idx = np.concatenate(flat_side_idx)
            self._all_pt_idx = np.concatenate(flat_pt_idx)
            self._all_side_names = flat_side_names
        else:
            self._all_trans_flat = None

        grid = gl.GLGridItem()
        grid.setSize(max_span * 1.2, max_span * 1.2, 0)
        grid.setSpacing(max_span / 20, max_span / 20, 1)
        grid.translate(0, 0, (z_min - cz) * z_scale - 10)
        self._add_item(grid)
        self.gl_widget.setCameraPosition(distance=max_span * 0.8)
        self._update_highlight()

    def _draw_arrow(self, pts):
        n = len(pts)
        if n < 3:
            return
        span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min(), 1.0)
        al = span * 0.04
        j = max(0, min(n // 2, n - 2))
        d = pts[j + 1] - pts[j]
        ln = np.linalg.norm(d)
        if ln < 1e-10:
            return
        d /= ln
        tip = pts[j] + d * al * 3
        perp = np.array([-d[1], d[0], 0.0])
        hs = al * 1.5
        lines = np.array([pts[j], tip, tip - d * hs + perp * hs * 0.5, tip, tip - d * hs - perp * hs * 0.5, tip])
        self._add_item(gl.GLLinePlotItem(pos=lines, color=(1, 1, 1, 0.8), width=2.0, antialias=True, mode="lines"))

