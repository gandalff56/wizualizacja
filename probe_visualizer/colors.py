import numpy as np
import matplotlib.cm as cm

SIDE_COLORS = {
    "Left": "#e74c3c",
    "Bottom": "#3498db",
    "Right": "#2ecc71",
    "Top": "#f39c12",
}

SIDE_ORDER = ["Left", "Bottom", "Right", "Top"]


def side_color_rgba(side_name):
    hex_color = SIDE_COLORS.get(side_name, "#888888")
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (r, g, b, 1.0)


def z_to_rgba(z_values, cmap_name="viridis"):
    z = np.asarray(z_values, dtype=float)
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-9:
        norm = np.zeros_like(z)
    else:
        norm = (z - z_min) / (z_max - z_min)
    cmap = cm.get_cmap(cmap_name)
    return cmap(norm)
