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

    def z_range(self):
        all_pts = self.all_points()
        return float(all_pts[:, 2].min()), float(all_pts[:, 2].max())

    def total_point_count(self):
        return sum(len(s.points) for s in self.sides)
