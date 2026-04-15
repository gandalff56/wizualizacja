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
    create_string_buffer,
)
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CNCSample:
    """Single probe sample: machine X/Y + laser Z at a moment in time."""

    x: float
    y: float
    z: float
    timestamp: float


@dataclass
class CNCAlarm:
    """One active alarm on the controller."""

    alm_group: int          # FOCAS alarm category (SV, OT, PS, ...)
    alm_number: int         # numeric alarm id, e.g. 1026 for SV1026
    axis: int               # 0 if not axis-specific
    message: str            # raw message text from the control

    @property
    def code(self) -> str:
        """Short code like `SV1026` built from group and number."""
        group_names = {
            0: "PW", 1: "IO", 2: "PS", 3: "OT", 4: "OH", 5: "SV", 6: "SR",
            7: "MC", 8: "SP", 9: "DS", 10: "IE", 11: "BG", 12: "SN",
            13: "EX", 14: "PC", 15: "SW",
        }
        prefix = group_names.get(self.alm_group, f"G{self.alm_group}")
        return f"{prefix}{self.alm_number:04d}"


@dataclass
class CNCOperatorMessage:
    """One operator message currently displayed on the MSG screen."""

    number: int             # message id (e.g. 2001 for #3006 macro calls)
    text: str               # raw text as shown on screen


@dataclass
class CNCStatus:
    """Snapshot of what the controller is doing right now."""

    aut: int = -1           # MDI / MEM / EDIT / HND / JOG / TEACH ...
    run: int = -1           # STOP / HOLD / STRT / MSTR / ...
    motion: int = -1
    mstb: int = -1
    emergency: int = -1     # non-zero when EMG is pressed
    alarm: int = -1         # non-zero when any alarm is active
    edit: int = -1
    program_name: str = ""
    program_o_number: int = 0
    current_line: int = 0

    @property
    def status_text(self) -> str:
        if self.emergency:
            return "EMG"
        if self.alarm:
            return "ALARM"
        run_names = {0: "****", 1: "STOP", 2: "HOLD", 3: "STRT", 4: "MSTR"}
        return run_names.get(self.run, f"RUN?({self.run})")

    @property
    def mode_text(self) -> str:
        aut_names = {
            0: "MDI", 1: "MEM", 2: "****", 3: "EDIT", 4: "HND",
            5: "JOG", 6: "TCH/HND", 7: "TCH/JOG", 8: "INC", 9: "RMT",
            10: "TEST",
        }
        return aut_names.get(self.aut, f"M?({self.aut})")


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

    def read_alarms(self) -> List[CNCAlarm]:
        """Return the list of active alarms. Default: empty (no support)."""
        return []

    def read_operator_messages(self) -> List[CNCOperatorMessage]:
        """Return currently-displayed operator messages. Default: empty."""
        return []

    def read_status(self) -> CNCStatus:
        """Return a snapshot of controller status. Default: blank."""
        return CNCStatus()

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

    def read_alarms(self) -> List[CNCAlarm]:
        if not self._connected:
            return []
        # Fabricate an alarm every ~30 s so the ERR window has something
        # to display during development.
        if int(self._t) % 30 < 5:
            return [
                CNCAlarm(
                    alm_group=5, alm_number=1026, axis=3,
                    message="AXIS OVERLOAD (Z)",
                )
            ]
        return []

    def read_operator_messages(self) -> List[CNCOperatorMessage]:
        if not self._connected:
            return []
        # Rotate through a few canned messages for testing the window.
        bucket = int(self._t / 10) % 3
        if bucket == 0:
            return [CNCOperatorMessage(
                number=2001,
                text="B2149_PRD-246617_DEV2_F loaded",
            )]
        if bucket == 1:
            return [
                CNCOperatorMessage(
                    number=2005,
                    text="Waiting for data to be processed by the office",
                ),
                CNCOperatorMessage(
                    number=2010,
                    text="CH1: LASER SCANNER DIRTY",
                ),
            ]
        return [CNCOperatorMessage(number=2020, text="Program start")]

    def read_status(self) -> CNCStatus:
        if not self._connected:
            return CNCStatus()
        # Alternate between RUN and HOLD so the status text visibly changes.
        run = 3 if int(self._t) % 12 < 8 else 2
        return CNCStatus(
            aut=1, run=run, motion=0, mstb=0,
            emergency=0, alarm=0, edit=0,
            program_name="PROG01",
            program_o_number=1234,
            current_line=120 + int(self._t) % 40,
        )

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


class _ODBST(Structure):
    """Status info. See fwlib32.h `ODBST`."""

    _fields_ = [
        ("dummy", c_short * 2),
        ("aut", c_short),
        ("run", c_short),
        ("motion", c_short),
        ("mstb", c_short),
        ("emergency", c_short),
        ("alarm", c_short),
        ("edit", c_short),
    ]


class _ODBEXEPRG(Structure):
    """Currently executing program name. See fwlib32.h `ODBEXEPRG`."""

    _fields_ = [
        ("name", c_char * 36),
        ("o_num", c_long),
    ]


class _ODBALMMSG2(Structure):
    """Alarm message entry. See fwlib32.h `ODBALMMSG2`."""

    _fields_ = [
        ("dummy", c_long),
        ("alm_grp", c_short),
        ("alm_no", c_short),
        ("axis", c_short),
        ("dummy2", c_char),
        ("msg_len", c_char),
        ("alm_msg", c_char * 64),
    ]


class _OPMSG3(Structure):
    """Operator message entry (cnc_rdopmsg3). See fwlib32.h `OPMSG3`."""

    _fields_ = [
        ("datano", c_short),
        ("type", c_short),
        ("char_num", c_short),
        ("data", c_char * 256),
    ]


class _ODBOPMSG3(Structure):
    """Operator message readout (up to 4 messages at once)."""

    _fields_ = [
        ("datano_s", c_short),
        ("type", c_short),
        ("msg1", _OPMSG3),
        ("msg2", _OPMSG3),
        ("msg3", _OPMSG3),
        ("msg4", _OPMSG3),
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

    # -- status, alarms, operator messages -----------------------------------

    def read_status(self) -> CNCStatus:
        if not self._connected or self._fw is None:
            return CNCStatus()
        st = _ODBST()
        try:
            if self._fw.cnc_statinfo(self._handle, byref(st)) != 0:
                return CNCStatus()
        except Exception:
            return CNCStatus()

        prg = _ODBEXEPRG()
        prog_name = ""
        o_num = 0
        line = 0
        try:
            if self._fw.cnc_exeprgname(self._handle, byref(prg)) == 0:
                prog_name = prg.name.decode("ascii", errors="replace").strip("\x00")
                o_num = int(prg.o_num)
        except Exception:
            pass
        # cnc_rdexecprog returns the current block text; for now we just
        # surface the block number via cnc_rdactpt which is more portable.
        try:
            pt = c_long(0)
            if hasattr(self._fw, "cnc_rdactpt"):
                self._fw.cnc_rdactpt(self._handle, byref(pt))
                line = int(pt.value)
        except Exception:
            pass

        return CNCStatus(
            aut=int(st.aut), run=int(st.run), motion=int(st.motion),
            mstb=int(st.mstb), emergency=int(st.emergency),
            alarm=int(st.alarm), edit=int(st.edit),
            program_name=prog_name, program_o_number=o_num,
            current_line=line,
        )

    def read_alarms(self) -> List[CNCAlarm]:
        if not self._connected or self._fw is None:
            return []
        MAX = 10
        buf = (_ODBALMMSG2 * MAX)()
        count = c_short(MAX)
        try:
            ret = self._fw.cnc_rdalmmsg2(
                self._handle, c_short(-1),  # -1 = all alarm types
                byref(count), byref(buf),
            )
        except Exception:
            return []
        if ret != 0:
            return []
        out: List[CNCAlarm] = []
        for i in range(int(count.value)):
            entry = buf[i]
            try:
                msg_len = ord(entry.msg_len) if isinstance(entry.msg_len, bytes) else int(entry.msg_len)
            except Exception:
                msg_len = 64
            raw = bytes(entry.alm_msg)[:msg_len]
            text = raw.decode("ascii", errors="replace").strip()
            out.append(CNCAlarm(
                alm_group=int(entry.alm_grp),
                alm_number=int(entry.alm_no),
                axis=int(entry.axis),
                message=text,
            ))
        return out

    def read_operator_messages(self) -> List[CNCOperatorMessage]:
        if not self._connected or self._fw is None:
            return []
        buf = _ODBOPMSG3()
        try:
            ret = self._fw.cnc_rdopmsg3(
                self._handle, c_short(0),
                c_short(sizeof(buf)), byref(buf),
            )
        except Exception:
            return []
        if ret != 0:
            return []
        out: List[CNCOperatorMessage] = []
        for m in (buf.msg1, buf.msg2, buf.msg3, buf.msg4):
            if m.type == 0 and m.datano == 0 and m.char_num == 0:
                continue
            raw = bytes(m.data)[:max(0, int(m.char_num))]
            text = raw.decode("ascii", errors="replace").strip("\x00 ")
            if not text:
                continue
            out.append(CNCOperatorMessage(number=int(m.datano), text=text))
        return out

    @property
    def description(self) -> str:
        return f"Fanuc FOCAS @ {self._ip}:{self._port}"
