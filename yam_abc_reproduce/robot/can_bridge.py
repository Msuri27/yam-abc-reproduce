"""Pi-side TCP service that exclusively owns the station's CAN devices."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import socket
import threading
import time
from typing import Any

import numpy as np

from ..config import (
    StationConfig,
    build_station_config,
    controller_channel_for,
    is_passive_gello,
    robot_channel_for,
)
from ..runtime import ArmUnit, _arm_name, build_arm_units
from .can_bus import check_can_up, reset_can_buses
from .remote import robot_config_signature

_MAX_MESSAGE_BYTES = 64 * 1024
_WATCHDOG_SECONDS = 0.5
_LOG = logging.getLogger("yam_abc_reproduce.can_bridge")


def _needed_channels(cfg: StationConfig) -> list[str]:
    robots = cfg.robot.robots or [cfg.robot.active_robot()]
    channels = [robot_channel_for(robot) for robot in robots]
    channels.extend(controller_channel_for(cfg.robot.controller_for(robot)) for robot in robots)
    return channels


class CanHardwareBridge:
    """Single-client, request/response bridge for local leader and follower hardware."""

    def __init__(self, cfg: StationConfig, token: str, units: list[ArmUnit] | None = None):
        if not token:
            raise ValueError("a non-empty bridge token is required")
        self.cfg = cfg
        self._token = token
        self._units = units or []
        self._by_name = {unit.name: unit for unit in self._units}
        if len(self._by_name) != len(self._units):
            raise ValueError("station contains duplicate arm names")
        self._layout = [
            {"name": _arm_name(robot.type), "dofs": cfg.robot.num_arm_joints + 1}
            for robot in (cfg.robot.robots or [cfg.robot.active_robot()])
        ]
        if len({arm["name"] for arm in self._layout}) != len(self._layout):
            raise ValueError("station contains duplicate arm names")
        if self._units and self._contract() != [
            {"name": unit.name, "dofs": unit.robot.num_dofs()} for unit in self._units
        ]:
            raise ValueError("hardware unit layout does not match the station config")
        self._lock = threading.RLock()
        self._active = False
        self._last_request = 0.0
        self._timed_out = False
        self._stopping = threading.Event()
        self._listening = threading.Event()
        self._client_socket: socket.socket | None = None
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="can_bridge_watchdog", daemon=True
        )

    def _contract(self) -> list[dict[str, Any]]:
        return self._layout

    def _write(
        self, writer, seq: int, result: dict[str, Any] | None = None, error: str | None = None
    ):
        response = {"seq": seq, "ok": error is None}
        if error is None:
            response["result"] = result or {}
        else:
            response["error"] = error
        writer.write(json.dumps(response, allow_nan=False, separators=(",", ":")).encode() + b"\n")
        writer.flush()

    def _read(self, reader) -> dict[str, Any] | None:
        raw = reader.readline(_MAX_MESSAGE_BYTES + 1)
        if not raw:
            return None
        if len(raw) > _MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
            raise ValueError("request exceeds the message limit or is not newline-terminated")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        return request

    def _unit(self, name: Any) -> ArmUnit:
        if not isinstance(name, str) or name not in self._by_name:
            raise ValueError(f"unknown arm {name!r}")
        return self._by_name[name]

    def _snapshot(self) -> dict[str, Any]:
        arms: dict[str, Any] = {}
        for unit in self._units:
            if unit.agent is None:
                raise RuntimeError(f"{unit.name} has no leader in the Pi hardware stack")
            state_snapshot = getattr(unit.agent, "state_snapshot", None)
            if not callable(state_snapshot):
                raise RuntimeError(f"{unit.name} leader does not support bridge snapshots")
            leader = state_snapshot()
            follower = unit.robot.get_observations()
            arms[unit.name] = {
                "leader": leader,
                "follower": {
                    "joint_pos": np.asarray(follower["joint_pos"], dtype=np.float64)
                    .reshape(-1)
                    .tolist(),
                    "gripper_pos": np.asarray(follower["gripper_pos"], dtype=np.float64)
                    .reshape(-1)
                    .tolist(),
                },
            }
        return {"arms": arms}

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "snapshot":
            return self._snapshot()
        if op == "command":
            unit = self._unit(request.get("arm"))
            position = np.asarray(request.get("position"), dtype=np.float64).reshape(-1)
            if position.size != unit.robot.num_dofs() or not np.isfinite(position).all():
                raise ValueError(
                    f"{unit.name} command must have {unit.robot.num_dofs()} finite values"
                )
            if not 0.0 <= position[-1] <= 1.0:
                raise ValueError(f"{unit.name} gripper command must be in [0, 1]")
            unit.robot.command_joint_pos(position)
            return {}
        if op == "hold":
            self._unit(request.get("arm")).robot.stop()
            return {}
        if op == "relax":
            self._unit(request.get("arm")).robot.relax()
            return {}
        if op == "power_off":
            unit = self._unit(request.get("arm"))
            power_off = getattr(unit.robot, "power_off", None)
            if not callable(power_off):
                raise RuntimeError(f"{unit.name} robot driver does not support power off")
            return power_off()
        if op == "zero":
            side = request.get("side")
            if side not in ("left", "right"):
                raise ValueError("zero side must be 'left' or 'right'")
            unit = self._unit(side)
            if unit.agent is None:
                raise RuntimeError(f"{side} leader is not configured")
            zero = getattr(unit.agent, "hardware_zero", None)
            if not callable(zero):
                raise RuntimeError(f"{side} leader does not support hardware zeroing")
            return {"devices": zero()}
        if op == "reset":
            return {"message": self._rebuild_hardware()}
        raise ValueError(f"unknown operation {op!r}")

    def _handle_connection(self, conn: socket.socket, address) -> None:
        _LOG.info("Thor connected from %s", address)
        conn.settimeout(5.0)
        reader = conn.makefile("rb")
        writer = conn.makefile("wb")
        authenticated = False
        try:
            while not self._stopping.is_set():
                request = self._read(reader)
                if request is None:
                    return
                seq = request.get("seq")
                if not isinstance(seq, int) or isinstance(seq, bool):
                    raise ValueError("request sequence must be an integer")
                try:
                    if not authenticated:
                        if request.get("op") != "hello":
                            raise ValueError("first request must authenticate with hello")
                        supplied = request.get("token")
                        if not isinstance(supplied, str) or not hmac.compare_digest(
                            supplied, self._token
                        ):
                            raise PermissionError("bridge token rejected")
                        expected = request.get("arms")
                        actual = self._contract()
                        if expected != actual:
                            raise ValueError(
                                f"station arm layout mismatch: Thor={expected}, Pi={actual}"
                            )
                        supplied_config = request.get("robot_config")
                        if not isinstance(supplied_config, str) or not hmac.compare_digest(
                            supplied_config, robot_config_signature(self.cfg)
                        ):
                            raise ValueError(
                                "Thor/Pi robot settings differ; synchronize the robot section "
                                "of configs/station_yam.yaml"
                            )
                        initialize = request.get("initialize", True)
                        if not isinstance(initialize, bool):
                            raise ValueError("hello initialize must be a boolean")
                        if initialize and not self._units:
                            with self._lock:
                                self._units = _build_hardware(self.cfg)
                                self._by_name = {unit.name: unit for unit in self._units}
                        authenticated = True
                        conn.settimeout(None)
                        with self._lock:
                            self._active = True
                            self._last_request = time.monotonic()
                            self._timed_out = False
                        self._write(writer, seq, {"arms": actual})
                        continue

                    with self._lock:
                        result = self._dispatch(request)
                        self._last_request = time.monotonic()
                        self._timed_out = False
                    self._write(writer, seq, result)
                except Exception as exc:
                    _LOG.exception("bridge request %s failed", request.get("op"))
                    self._write(writer, seq, error=f"{type(exc).__name__}: {exc}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _LOG.info("bridge connection ended: %s", exc)
        finally:
            try:
                reader.close()
            except OSError:
                pass
            try:
                writer.close()
            except OSError:
                pass
            with self._lock:
                if authenticated:
                    self._hold_all()
                self._active = False
            _LOG.info("Thor disconnected: %s", address)

    def _hold_all(self) -> None:
        for unit in self._units:
            try:
                unit.robot.stop()
            except Exception:
                _LOG.exception("failed to hold %s follower", unit.name)

    def _watchdog_loop(self) -> None:
        while not self._stopping.wait(0.1):
            with self._lock:
                if (
                    self._active
                    and not self._timed_out
                    and time.monotonic() - self._last_request > _WATCHDOG_SECONDS
                ):
                    _LOG.error(
                        "Thor link stale for %.2f s; holding all followers", _WATCHDOG_SECONDS
                    )
                    self._hold_all()
                    self._timed_out = True
                    self._active = False
                    if self._client_socket is not None:
                        try:
                            self._client_socket.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass

    @staticmethod
    def _release_units(units: list[ArmUnit]) -> None:
        for unit in units:
            if unit.agent is not None:
                try:
                    unit.agent.stop()
                except Exception:
                    _LOG.exception("failed to release %s leader", unit.name)
            try:
                unit.robot.relax()
            except Exception:
                _LOG.exception("failed to relax %s follower", unit.name)

    def _rebuild_hardware(self) -> str:
        old_units, self._units = self._units, []
        self._by_name = {}
        self._release_units(old_units)
        del old_units
        ok, message = reset_can_buses(channels=_needed_channels(self.cfg))
        if not ok:
            raise RuntimeError(f"CAN reset failed: {message}")
        down = check_can_up(_needed_channels(self.cfg))
        if down:
            raise RuntimeError(f"CAN interfaces not up after reset: {down}")
        return message

    def serve_forever(self, host: str, port: int) -> None:
        self._watchdog.start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)
            listener.settimeout(0.5)
            self._listening.set()
            _LOG.info("listening on %s:%d for %d arm(s)", host, port, len(self._layout))
            while not self._stopping.is_set():
                try:
                    conn, address = listener.accept()
                except TimeoutError:
                    continue
                with conn:
                    with self._lock:
                        self._client_socket = conn
                    try:
                        self._handle_connection(conn, address)
                    finally:
                        with self._lock:
                            self._client_socket = None

    def close(self) -> None:
        self._stopping.set()
        with self._lock:
            self._hold_all()
            self._active = False
            conn = self._client_socket
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._watchdog.is_alive():
            self._watchdog.join(timeout=1.0)


def _build_hardware(cfg: StationConfig) -> list[ArmUnit]:
    robots = cfg.robot.robots or [cfg.robot.active_robot()]
    unsupported = [
        cfg.robot.controller_for(robot).type
        for robot in robots
        if not is_passive_gello(cfg.robot.controller_for(robot).type)
    ]
    if unsupported:
        raise ValueError(
            "remote hardware mode currently requires passive GELLO leaders; "
            f"unsupported controller(s): {unsupported}"
        )
    ok, message = reset_can_buses(channels=_needed_channels(cfg))
    if not ok:
        raise RuntimeError(f"CAN reset failed: {message}")
    needed = _needed_channels(cfg)
    down = check_can_up(needed)
    if down:
        raise RuntimeError(f"CAN interfaces not up after reset: {down}")
    return build_arm_units(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="yam-abc-hardware",
        description="Own YAM CAN devices on the Pi and serve snapshots/commands to the Thor.",
    )
    parser.add_argument("--station", default="configs/station_yam.yaml")
    parser.add_argument("--host", default="0.0.0.0", help="bind address; prefer the Pi's wired IP")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    token = os.environ.get("YAM_CAN_BRIDGE_TOKEN")
    if not token:
        parser.error("set YAM_CAN_BRIDGE_TOKEN to a shared secret before starting the bridge")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = build_station_config(args.station)
    bridge = CanHardwareBridge(cfg, token)
    try:
        bridge.serve_forever(args.host, args.port)
    except KeyboardInterrupt:
        _LOG.info("shutting down")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
