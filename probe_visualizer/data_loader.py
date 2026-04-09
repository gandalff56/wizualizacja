import json
from dataclasses import dataclass
import numpy as np


@dataclass
class SideData:
    name: str
    points: np.ndarray  # shape (N, 3), columns X, Y, Z


class ProbeData:
    def __init__(self, sides):
        self.sides = sides

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError("JSON root must be an array")

        sides = []
        for entry in raw:
            name = entry.get("Side", "Unknown")
            pts_raw = entry.get("ProbeDataPointsSys", [])
            if not pts_raw:
                continue
            pts = np.array([[p["X"], p["Y"], p["Z"]] for p in pts_raw])
            sides.append(SideData(name=name, points=pts))

        if not sides:
            raise ValueError("No valid sides found in JSON")

        return cls(sides)

    def all_points(self):
        return np.vstack([s.points for s in self.sides])

    def z_range(self):
        all_pts = self.all_points()
        return float(all_pts[:, 2].min()), float(all_pts[:, 2].max())

    def total_point_count(self):
        return sum(len(s.points) for s in self.sides)
