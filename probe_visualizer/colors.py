import numpy as np
import matplotlib.cm as cm

SIDE_COLORS = {
    "Left": "#e74c3c",
    "Bottom": "#3498db",
    "Right": "#2ecc71",
    "Top": "#f39c12",
}

# Extended palette for sessions in Raw format
_EXTRA_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    "#8bc34a", "#ff5722", "#607d8b", "#795548", "#673ab7",
    "#009688", "#ff9800", "#03a9f4", "#cddc39", "#f44336",
]


def get_color_for_side(name, index=0):
    if name in SIDE_COLORS:
        return SIDE_COLORS[name]
    return _EXTRA_COLORS[index % len(_EXTRA_COLORS)]


def z_to_rgba(z_values, cmap_name="viridis"):
    z = np.asarray(z_values, dtype=float)
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-9:
        norm = np.zeros_like(z)
    else:
        norm = (z - z_min) / (z_max - z_min)
    cmap = cm.get_cmap(cmap_name)
    return cmap(norm)
