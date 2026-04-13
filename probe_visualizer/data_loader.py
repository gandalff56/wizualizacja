import json
from dataclasses import dataclass
import numpy as np


@dataclass
class SideData:
    name: str
    points: np.ndarray  # shape (N, 3), columns X, Y, Z


class ProbeData:
    def __init__(self, sides, format_name="unknown", source_path=None, raw_json=None):
        self.sides = sides
        self.format_name = format_name
        self.source_path = source_path
        self._raw_json = raw_json  # keep original for save

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Format 1: SYS format - array of {Side, ProbeDataPointsSys}
        if isinstance(raw, list) and raw and "Side" in raw[0]:
            result = cls._parse_sys_format(raw)
            result.source_path = path
            result._raw_json = raw
            return result

        # Format 2: Raw format - {ProbeDataSessionList: [...]}
        if isinstance(raw, dict) and "ProbeDataSessionList" in raw:
            result = cls._parse_raw_format(raw)
            result.source_path = path
            result._raw_json = raw
            return result

        # Format 3: Array but without Side key - try ProbeDataPoints
        if isinstance(raw, list) and raw and "ProbeDataPoints" in raw[0]:
            result = cls._parse_session_list(raw, "raw")
            result.source_path = path
            result._raw_json = raw
            return result

        raise ValueError(
            "Unrecognized JSON format. Expected SYS format (array with Side) "
            "or Raw format (object with ProbeDataSessionList)."
        )

    @classmethod
    def _parse_sys_format(cls, raw):
        sides = []
        for entry in raw:
            name = entry.get("Side", "Unknown")
            pts_raw = entry.get("ProbeDataPointsSys", [])
            if not pts_raw:
                continue
            pts = np.array([[p["X"], p["Y"], p["Z"]] for p in pts_raw])
            sides.append(SideData(name=name, points=pts))

        if not sides:
            raise ValueError("No valid sides found in SYS JSON")
        return cls(sides, format_name="SYS")

    @classmethod
    def _parse_raw_format(cls, raw):
        sessions = raw["ProbeDataSessionList"]
        if not isinstance(sessions, list):
            raise ValueError("ProbeDataSessionList must be an array")
        return cls._parse_session_list(sessions, "Raw")

    @classmethod
    def _parse_session_list(cls, sessions, fmt_name):
        sides = []
        for i, session in enumerate(sessions):
            pts_raw = session.get("ProbeDataPoints", [])
            if not pts_raw:
                continue
            pts = np.array([[p["X"], p["Y"], p["Z"]] for p in pts_raw])
            line_num = session.get("ProgramLineNumber", i)
            name = f"Session {i+1} (L{line_num})"
            sides.append(SideData(name=name, points=pts))

        if not sides:
            raise ValueError("No valid sessions found in Raw JSON")
        return cls(sides, format_name=fmt_name)

    def save_to_json_file(self, path=None):
        """Save current point data back to JSON, preserving original format."""
        path = path or self.source_path
        if not path:
            raise ValueError("No file path specified")

        if self.format_name == "SYS":
            output = []
            for side in self.sides:
                pts = [{"X": float(p[0]), "Y": float(p[1]), "Z": float(p[2])}
                       for p in side.points]
                output.append({"Side": side.name, "ProbeDataPointsSys": pts})
            with open(path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

        elif self.format_name == "Raw" and self._raw_json is not None:
            # Update points in original structure to preserve extra fields
            if isinstance(self._raw_json, dict):
                sessions = self._raw_json["ProbeDataSessionList"]
            else:
                sessions = self._raw_json

            side_idx = 0
            for session in sessions:
                pts_raw = session.get("ProbeDataPoints", [])
                if not pts_raw:
                    continue
                if side_idx >= len(self.sides):
                    break
                new_pts = self.sides[side_idx].points
                for j, pt in enumerate(pts_raw):
                    if j < len(new_pts):
                        pt["X"] = float(new_pts[j, 0])
                        pt["Y"] = float(new_pts[j, 1])
                        pt["Z"] = float(new_pts[j, 2])
                side_idx += 1

            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._raw_json, f, indent=2)
        else:
            raise ValueError(f"Cannot save format: {self.format_name}")

        self.source_path = path

    def all_points(self):
        return np.vstack([s.points for s in self.sides])

    @classmethod
    def empty_live_dataset(cls):
        """Return an empty dataset containing a single writable 'Live' side."""
        sides = [SideData(name="Live", points=np.zeros((0, 3), dtype=float))]
        return cls(sides, format_name="Live")

    def append_live_point(self, x, y, z):
        """Append a single sample to the 'Live' side, creating it if needed."""
        new_pt = np.array([[float(x), float(y), float(z)]], dtype=float)
        for side in self.sides:
            if side.name == "Live":
                if len(side.points):
                    side.points = np.vstack([side.points, new_pt])
                else:
                    side.points = new_pt
                return
        self.sides.append(SideData(name="Live", points=new_pt))

    def clear_live_side(self):
        """Drop all points from the 'Live' side without removing it."""
        for side in self.sides:
            if side.name == "Live":
                side.points = np.zeros((0, 3), dtype=float)
                return

    def z_range(self):
        all_pts = self.all_points()
        if len(all_pts) == 0:
            return 0.0, 0.0
        return float(all_pts[:, 2].min()), float(all_pts[:, 2].max())

    def total_point_count(self):
        return sum(len(s.points) for s in self.sides)

    def side_stats(self):
        """Return per-side statistics: Z min/max/avg/std, point count, XY path length."""
        stats = []
        for side in self.sides:
            if len(side.points) == 0:
                stats.append({
                    "name": side.name,
                    "count": 0,
                    "z_min": 0.0,
                    "z_max": 0.0,
                    "z_avg": 0.0,
                    "z_std": 0.0,
                    "path_length": 0.0,
                    "span": 0.0,
                })
                continue
            z = side.points[:, 2]
            diffs = np.diff(side.points[:, :2], axis=0)
            path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
            span = float(np.linalg.norm(
                side.points[-1, :2] - side.points[0, :2]
            ))
            stats.append({
                "name": side.name,
                "count": len(side.points),
                "z_min": float(z.min()),
                "z_max": float(z.max()),
                "z_avg": float(z.mean()),
                "z_std": float(z.std()),
                "path_length": path_len,
                "span": span,
            })
        return stats

    @staticmethod
    def _is_side_curved(side):
        """Return True if the side's points deviate noticeably from a straight
        line between its first and last point (i.e. the side is an arc)."""
        pts = side.points[:, :2]
        n = len(pts)
        if n < 4:
            return False
        p0 = pts[0]
        chord = pts[-1] - p0
        chord_len = float(np.linalg.norm(chord))
        if chord_len < 1e-9:
            return False
        diffs = pts - p0
        # Perpendicular distance from each point to the chord line.
        cross = np.abs(diffs[:, 0] * chord[1] - diffs[:, 1] * chord[0])
        max_dev = float(cross.max() / chord_len)
        # Curved if any point deviates > 1% of chord length AND > 2 mm.
        return max_dev > max(chord_len * 0.01, 2.0)

    def plate_dimensions(self):
        """Calculate plate dimensions and rotation angle.

        Uses corner points (first point of each side) to determine
        plate geometry. The plate may be rotated relative to machine axes.

        Returns a dict with `shape` = "rectangle" or "cone" and the
        appropriate fields for that shape.
        """
        if len(self.sides) < 2:
            return None

        # Corners = first point of each side
        corners = []
        for side in self.sides:
            corners.append(side.points[0, :2].copy())

        corners = np.array(corners)

        # Side lengths between consecutive corners
        n = len(corners)
        side_lengths = []
        for i in range(n):
            j = (i + 1) % n
            side_lengths.append(float(np.linalg.norm(corners[j] - corners[i])))

        # Rotation angle: angle of first side (index 0 -> index 1) relative to X axis
        if n >= 2:
            vec = corners[1] - corners[0]
            angle_rad = float(np.arctan2(vec[1], vec[0]))
            angle_deg = float(np.degrees(angle_rad))
        else:
            angle_deg = 0.0

        # Cone detection: 4 sides, and the two longest are curved arcs.
        if n == 4:
            # Span of each side = distance between its first and last point
            spans = [
                float(np.linalg.norm(s.points[-1, :2] - s.points[0, :2]))
                for s in self.sides
            ]
            order = sorted(range(4), key=lambda i: -spans[i])
            a1, a2 = order[0], order[1]
            s1, s2 = order[2], order[3]
            if (
                self._is_side_curved(self.sides[a1])
                and self._is_side_curved(self.sides[a2])
            ):
                chord_w = (spans[s1] + spans[s2]) / 2.0
                return {
                    "shape": "cone",
                    "chord1": spans[a1],
                    "chord2": spans[a2],
                    "chord_w": chord_w,
                    "chord1_side": self.sides[a1].name,
                    "chord2_side": self.sides[a2].name,
                    "w_side_1": self.sides[s1].name,
                    "w_side_2": self.sides[s2].name,
                    "angle_deg": angle_deg,
                    "corners": corners,
                    "side_lengths": side_lengths,
                }

        # Rectangle plate
        if n == 4:
            width = (side_lengths[0] + side_lengths[2]) / 2.0
            length = (side_lengths[1] + side_lengths[3]) / 2.0
            diag1 = float(np.linalg.norm(corners[2] - corners[0]))
            diag2 = float(np.linalg.norm(corners[3] - corners[1]))
        else:
            width = side_lengths[0] if side_lengths else 0
            length = side_lengths[1] if len(side_lengths) > 1 else 0
            diag1 = diag2 = 0

        return {
            "shape": "rectangle",
            "corners": corners,
            "side_lengths": side_lengths,
            "width": width,
            "length": length,
            "angle_deg": angle_deg,
            "diagonal_1": diag1,
            "diagonal_2": diag2,
        }
