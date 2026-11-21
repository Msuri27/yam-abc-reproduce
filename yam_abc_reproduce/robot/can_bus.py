"""Bring the CAN buses up.

Run at go-live (first Start Teleop) on the hardware path and re-exposed as the GUI
"Reset CAN" action. Reuses i2rt's vendored reset_all_can.sh.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "third_party" / "i2rt" / "scripts" / "reset_all_can.sh"
)
_CHANNEL = re.compile(r"can[A-Za-z0-9_.-]{1,12}\Z")


def list_can_interfaces() -> list[str]:
    """Every CAN interface present on this machine, udev-named or raw (``can_left``,
    ``can0``, ...). Feeds the GUI's channel pickers."""
    net = Path("/sys/class/net")
    if not net.is_dir():
        return []
    return sorted(p.name for p in net.iterdir() if p.name.startswith("can"))


def check_can_up(channels: list[str], timeout_s: float = 5.0) -> list[str]:
    """Return the subset of ``channels`` that are NOT present-and-up."""
    down: list[str] = []
    for ch in channels:
        try:
            r = subprocess.run(
                ["ip", "link", "show", ch],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (subprocess.SubprocessError, OSError):
            down.append(ch)
            continue
        out = r.stdout or ""
        if r.returncode != 0 or ("state UP" not in out and "state UNKNOWN" not in out):
            down.append(ch)
    return down


def reset_can_buses(
    timeout_s: float = 30.0, channels: list[str] | None = None
) -> tuple[bool, str]:
    """Reset CAN interfaces at 1 Mbit/s, optionally limiting the operation to channels.

    With ``channels=None`` this preserves the historical all-interface reset. Supplying
    names is safer on mixed stations: unrelated buses (for example a mobile base on
    ``can0``) are not brought down or reconfigured.
    """
    if channels is not None:
        selected = list(dict.fromkeys(channels))
        invalid = [name for name in selected if not _CHANNEL.fullmatch(name)]
        if invalid:
            return False, f"invalid CAN interface name(s): {invalid}"
        if not selected:
            return False, "no CAN interfaces were selected"
        ip = shutil.which("ip") or "/usr/sbin/ip"
        sudo = [] if os.geteuid() == 0 else ["sudo", "-n"]
        failures = []
        for channel in selected:
            for state in (["down"], ["up", "type", "can", "bitrate", "1000000"]):
                try:
                    result = subprocess.run(
                        [*sudo, ip, "link", "set", channel, *state],
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.SubprocessError, OSError) as exc:
                    failures.append(f"{channel}: {exc}")
                    continue
                if result.returncode:
                    message = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
                    failures.append(f"{channel}: {message}")
        if failures:
            return False, "\n".join(failures)
        return True, f"Reset CAN interfaces: {', '.join(selected)}"

    if not _SCRIPT.exists():
        return False, f"CAN setup script not found: {_SCRIPT}"
    try:
        r = subprocess.run(
            ["bash", str(_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, str(exc)
    out = (r.stdout or "").strip()
    if r.returncode == 0:
        return True, out
    return False, (r.stderr or out or f"exit code {r.returncode}").strip()
