import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from probe_visualizer.colors import get_color_for_side


class View2D(QWidget):
    # Signal: (side_index, point_index, side_name, x, y, z)
    point_selected = pyqtSignal(int, int, str, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Hover info label (fixed position, not tooltip)
        self.hover_label = QLabel("")
        self.hover_label.setStyleSheet(
            "background-color: #1a1a2e; color: #eee; padding: 4px 8px; "
            "font-family: monospace; font-size: 11px;"
        )
        self.hover_label.setFixedHeight(22)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.hover_label)

        self.data = None
        self.visible_sides = {}
        self.color_mode = "side"

        # For pick/hover: flat arrays of all visible points
        self._scatter_artists = []  # list of (side_idx, side_name, scatter_artist)
        self._side_points = []  # list of (side_idx, side_name, points_array)

        # Highlight artist for selected point
        self._highlight_artist = None
        self._selected_side_idx = -1
        self._selected_pt_idx = -1

        # Connect matplotlib events
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        # Debounce timer
        self._draw_timer = QTimer()
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(50)
        self._draw_timer.timeout.connect(self._draw)

    def update_plot(self, data, visible_sides, color_mode):
        self.data = data
        self.visible_sides = visible_sides
        self.color_mode = color_mode
        if not self._draw_timer.isActive():
            self._draw_timer.start()

    def highlight_point(self, side_idx, pt_idx):
        """Highlight a specific point (called from app when 3D or table selects)."""
        self._selected_side_idx = side_idx
        self._selected_pt_idx = pt_idx
        self._update_highlight()

    def _on_pick(self, event):
        """Handle matplotlib pick event - user clicked on a scatter point."""
        if self.data is None:
            return
        artist = event.artist
        ind = event.ind[0]  # index within this scatter artist

        for side_idx, side_name, sc in self._scatter_artists:
            if sc is artist:
                pts = self.data.sides[side_idx].points
                if ind < len(pts):
                    x, y, z = pts[ind]
                    self._selected_side_idx = side_idx
                    self._selected_pt_idx = ind
                    self._update_highlight()
                    self.point_selected.emit(
                        side_idx, ind, side_name,
                        float(x), float(y), float(z)
                    )
                break

    def _on_motion(self, event):
        """Handle matplotlib motion event - show hover info."""
        if self.data is None or event.inaxes is None:
            self.hover_label.setText("")
            return

        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            self.hover_label.setText("")
            return

        # Find nearest visible point
        best_dist = float("inf")
        best_info = None

        for side_idx, side_name, pts in self._side_points:
            dx = pts[:, 0] - mx
            dy = pts[:, 1] - my
            dists = dx * dx + dy * dy
            idx = np.argmin(dists)
            if dists[idx] < best_dist:
                best_dist = dists[idx]
                best_info = (side_name, pts[idx])

        if best_info is not None:
            name, pt = best_info
            self.hover_label.setText(
                f"  {name}  |  X: {pt[0]:.3f}  Y: {pt[1]:.3f}  Z: {pt[2]:.3f}"
            )
        else:
            self.hover_label.setText("")

    def _update_highlight(self):
        """Update the highlight marker on selected point."""
        if self._highlight_artist is not None:
            try:
                self._highlight_artist.remove()
            except ValueError:
                pass
            self._highlight_artist = None

        if self._selected_side_idx < 0 or self.data is None:
            self.canvas.draw_idle()
            return

        ax = self.figure.axes[0] if self.figure.axes else None
        if ax is None:
            return

        side = self.data.sides[self._selected_side_idx]
        if self._selected_pt_idx < len(side.points):
            pt = side.points[self._selected_pt_idx]
            self._highlight_artist = ax.scatter(
                [pt[0]], [pt[1]], s=100, facecolors="none",
                edgecolors="white", linewidths=2, zorder=10
            )
        self.canvas.draw_idle()

    def _draw(self):
        self.figure.clear()
        self._scatter_artists.clear()
        self._side_points.clear()
        self._highlight_artist = None

        if self.data is None:
            self.canvas.draw_idle()
            return

        ax = self.figure.add_subplot(111)
        z_min, z_max = self.data.z_range()

        for i, side in enumerate(self.data.sides):
            if not self.visible_sides.get(side.name, True):
                continue

            x = side.points[:, 0]
            y = side.points[:, 1]
            z = side.points[:, 2]

            if self.color_mode == "side":
                color = get_color_for_side(side.name, i)
                ax.plot(x, y, "-", color=color, linewidth=1.5, label=side.name)
                sc = ax.scatter(x, y, c=color, s=12, zorder=5, picker=5)
            else:
                ax.plot(x, y, "-", color="#cccccc", linewidth=0.8)
                norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
                sc = ax.scatter(
                    x, y, c=z, cmap="viridis", norm=norm, s=12, zorder=5, picker=5
                )

            self._scatter_artists.append((i, side.name, sc))
            self._side_points.append((i, side.name, side.points.copy()))

        if self.color_mode == "z":
            visible_any = any(
                self.visible_sides.get(s.name, True) for s in self.data.sides
            )
            if visible_any:
                norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
                sm = cm.ScalarMappable(cmap="viridis", norm=norm)
                sm.set_array([])
                self.figure.colorbar(sm, ax=ax, label="Z [mm]", shrink=0.8)
        else:
            n_sides = sum(
                1 for s in self.data.sides
                if self.visible_sides.get(s.name, True)
            )
            if n_sides <= 20:
                ax.legend(loc="best", fontsize=8)

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_title("Probe Data - 2D View (click point to select, hover for info)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()

        # Re-highlight if needed
        if self._selected_side_idx >= 0:
            self._update_highlight()
        else:
            self.canvas.draw_idle()
