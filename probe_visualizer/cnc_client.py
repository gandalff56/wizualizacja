"""CNC data source abstraction for live probe visualization.

Three implementations are provided:
- `CNCClient`: abstract base class
- `MockCNCClient`: local simulator, useful without a real machine
- `FanucFocasClient`: reads X/Y from axis position and Z from macro #10111
  via the FANUC FOCAS2 library (Fwlib32.dll on Windows / libfwlib32.so on
  Linux). The DLL must be provided by the user (not redistributed by Fanuc).
"""

from __future__ import annotations

import math
import os
import platform
import time
from ctypes import (
    Structure,
    byref,
    c_char,
    c_long,
    c_short,
    c_ushort,
    sizeof,
)
from dataclasses import dataclass
from typing import Optional


@dataclass
class CNCSample:
    """Single probe sample: machine X/Y + laser Z at a moment in time."""

    x: float
    y: float
    z: float
    timestamp: float


class CNCClient:
    """Abstract CNC data source. Subclasses implement connect/read/disconnect."""

    def connect(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def disconnect(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def is_connected(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def read_sample(self) -> Optional[CNCSample]:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def description(self) -> str:  # pragma: no cover - interface
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Mock client - used for development / testing without a real machine.
# ---------------------------------------------------------------------------


class MockCNCClient(CNCClient):
    """Generate synthetic samples so the UI can be tested without a CNC."""

    def __init__(self) -> None:
        self._t = 0.0
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        self._t = 0.0

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_sample(self) -> Optional[CNCSample]:
        if not self._connected:
            return None
        self._t += 0.1
        # Move along a slowly-expanding spiral so new points are visible.
        r = 50.0 + self._t * 0.4
        x = 100.0 + r * math.cos(self._t * 0.5)
        y = 100.0 + r * math.sin(self._t * 0.5)
        z = 10.0 + math.sin(self._t * 2.0) * 0.5
        return CNCSample(x=x, y=y, z=z, timestamp=time.time())

    @property
    def description(self) -> str:
        return "Mock CNC (local simulation)"


# ---------------------------------------------------------------------------
# Fanuc FOCAS2 client
# ---------------------------------------------------------------------------


class _ODBM(Structure):
    """Macro variable readout. See fwlib32.h `ODBM`."""

    _fields_ = [
        ("prm_type", c_short),
        ("reserve", c_short),
        ("mcr_val", c_long),
        ("dec_val", c_short),
    ]


class _POSELM(Structure):
    """Single position element. See fwlib32.h `POSELM`."""

    _fields_ = [
        ("data", c_long),
        ("dec", c_short),
        ("unit", c_short),
        ("disp", c_short),
        ("name", c_char),
        ("suff", c_char),
    ]


class _ODBPOS(Structure):
    """Position readout for a single axis. See fwlib32.h `ODBPOS`."""

    _fields_ = [
        ("abs", _POSELM),
        ("mach", _POSELM),
        ("rel", _POSELM),
        ("dist", _POSELM),
    ]


LASER_MACRO_NUMBER = 10111


class FanucFocasClient(CNCClient):
    """Read X/Y from the machine position and Z from macro #10111 via FOCAS2.

    The user must supply `Fwlib32.dll` (64-bit) next to this file, in the
    application's working directory, or anywhere on PATH. Fanuc does not
    publicly redistribute this library - it ships on the FOCAS2 CD from the
    machine service.
    """

    def __init__(
        self,
        ip: str,
        port: int = 8193,
        timeout: int = 10,
        macro_number: int = LASER_MACRO_NUMBER,
    ) -> None:
        self._ip = ip
        self._port = port
        self._timeout = timeout
        self._macro_number = macro_number
        self._fw = None
        self._handle = c_ushort(0)
        self._connected = False

    # -- connection management ------------------------------------------------

    def _load_library(self):
        here = os.path.dirname(os.path.abspath(__file__))
        if platform.system() == "Windows":
            import ctypes

            candidates = [
                os.path.join(here, "Fwlib32.dll"),
                "Fwlib32.dll",  # search PATH
            ]
            last_err = None
            for cand in candidates:
                try:
                    return ctypes.windll.LoadLibrary(cand)
                except OSError as e:
                    last_err = e
            raise RuntimeError(
                "Fwlib32.dll not found. Copy it next to cnc_client.py or "
                f"onto PATH. Last loader error: {last_err}"
            )
        # Linux / mac (Fanuc ships libfwlib32.so for some platforms)
        import ctypes

        for name in ("libfwlib32.so", "libfwlib32.so.1"):
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise RuntimeError(
            "libfwlib32.so not found. This backend only works on Windows "
            "unless you have a Linux build of the FANUC FOCAS library."
        )

    def connect(self) -> None:
        if self._connected:
            return
        self._fw = self._load_library()
        ret = self._fw.cnc_allclibhndl3(
            self._ip.encode("ascii"),
            c_ushort(self._port),
            c_long(self._timeout),
            byref(self._handle),
        )
        if ret != 0:
            raise RuntimeError(
                f"cnc_allclibhndl3 failed (code {ret}). Check IP, port, "
                f"and that FOCAS2 is enabled on the control."
            )
        self._connected = True

    def disconnect(self) -> None:
        if self._fw is not None and self._connected:
            try:
                self._fw.cnc_freelibhndl(self._handle)
            except Exception:
                pass
        self._connected = False
        self._handle = c_ushort(0)

    def is_connected(self) -> bool:
        return self._connected

    # -- sampling -------------------------------------------------------------

    def read_sample(self) -> Optional[CNCSample]:
        if not self._connected or self._fw is None:
            return None

        # First 2 axes = X, Y (absolute workpiece coordinates).
        posbuf = (_ODBPOS * 2)()
        axis_cnt = c_short(2)
        ret = self._fw.cnc_rdposition(
            self._handle,
            c_short(0),
            byref(axis_cnt),
            byref(posbuf),
        )
        if ret != 0:
            return None
        x = posbuf[0].abs.data * (10.0 ** -posbuf[0].abs.dec)
        y = posbuf[1].abs.data * (10.0 ** -posbuf[1].abs.dec)

        # Z from the laser macro variable.
        odbm = _ODBM()
        ret = self._fw.cnc_rdmacro(
            self._handle,
            c_short(self._macro_number),
            c_short(sizeof(odbm)),
            byref(odbm),
        )
        if ret != 0:
            return None
        z = odbm.mcr_val * (10.0 ** -odbm.dec_val)

        return CNCSample(x=float(x), y=float(y), z=float(z), timestamp=time.time())

    @property
    def description(self) -> str:
        return f"Fanuc FOCAS @ {self._ip}:{self._port}"
