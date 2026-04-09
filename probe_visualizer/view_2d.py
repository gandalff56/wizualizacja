import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from probe_visualizer.colors import get_color_for_side


class View2D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.data = None
        self.visible_sides = {}
        self.color_mode = "side"

    def update_plot(self, data, visible_sides, color_mode):
        self.data = data
        self.visible_sides = visible_sides
        self.color_mode = color_mode
        self._draw()

    def _draw(self):
        self.figure.clear()
        if self.data is None:
            self.canvas.draw_idle()
            return

        ax = self.figure.add_subplot(111)

        all_pts = self.data.all_points()
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
                ax.scatter(x, y, c=color, s=8, zorder=5)
            else:
                ax.plot(x, y, "-", color="#cccccc", linewidth=0.8)
                norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
                sc = ax.scatter(x, y, c=z, cmap="viridis", norm=norm, s=12, zorder=5)

        if self.color_mode == "z":
            visible_any = any(self.visible_sides.get(s.name, True) for s in self.data.sides)
            if visible_any:
                norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
                sm = cm.ScalarMappable(cmap="viridis", norm=norm)
                sm.set_array([])
                self.figure.colorbar(sm, ax=ax, label="Z [mm]", shrink=0.8)
        else:
            n_sides = sum(1 for s in self.data.sides if self.visible_sides.get(s.name, True))
            if n_sides <= 20:
                ax.legend(loc="best", fontsize=8)

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_title("Probe Data - 2D View")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()
