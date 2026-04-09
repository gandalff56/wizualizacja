import json
from dataclasses import dataclass
import numpy as np


@dataclass
class SideData:
    name: str
    points: np.ndarray  # shape (N, 3), columns X, Y, Z


class ProbeData:
    def __init__(self, sides, format_name="unknown"):
        self.sides = sides
        self.format_name = format_name

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Format 1: SYS format - array of {Side, ProbeDataPointsSys}
        if isinstance(raw, list) and raw and "Side" in raw[0]:
            return cls._parse_sys_format(raw)

        # Format 2: Raw format - {ProbeDataSessionList: [...]}
        if isinstance(raw, dict) and "ProbeDataSessionList" in raw:
            return cls._parse_raw_format(raw)

        # Format 3: Array but without Side key - try ProbeDataPoints
        if isinstance(raw, list) and raw and "ProbeDataPoints" in raw[0]:
            return cls._parse_session_list(raw, "raw")

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

    def all_points(self):
        return np.vstack([s.points for s in self.sides])

    def z_range(self):
        all_pts = self.all_points()
        return float(all_pts[:, 2].min()), float(all_pts[:, 2].max())

    def total_point_count(self):
        return sum(len(s.points) for s in self.sides)
