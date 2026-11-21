"""TCP client and proxy devices for a Pi-owned YAM hardware bridge."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import threading
from dataclasses import asdict
from typing import Any, BinaryIO

import numpy as np

from ..config import StationConfig, is_passive_gello
from ..runtime import ArmUnit, _arm_name
from .interface import RobotInterface
from .yam_adapter import YamTeleop

_MAX_MESSAGE_BYTES = 64 * 1024


class RemoteHardwareError(RuntimeError):
    """The remote hardware service rejected a request or became unreachable."""


def robot_config_signature(cfg: StationConfig) -> str:
    encoded = json.dumps(asdict(cfg.robot), sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _contract(cfg: StationConfig) -> list[dict[str, Any]]:
    robots = cfg.robot.robots or [cfg.robot.active_robot()]
    return [
        {"name": _arm_name(robot.type), "dofs": cfg.robot.num_arm_joints + 1} for robot in robots
    ]


class RemoteHardwareClient:
    """One synchronous TCP connection to the machine that owns the CAN buses.

    Requests are lock-step: the client never queues control messages behind stale ones.
    """

    def __init__(
        self,
        host: str,
        port: int,
        cfg: StationConfig,
        token: str | None = None,
        timeout: float = 2.0,
        initialize: bool = True,
    ) -> None:
        token = token if token is not None else os.environ.get("YAM_CAN_BRIDGE_TOKEN")
        if not token:
            raise ValueError("set YAM_CAN_BRIDGE_TOKEN on both the Pi and Thor")
        self._expected = {arm["name"]: arm["dofs"] for arm in _contract(cfg)}
        self._config_signature = robot_config_signature(cfg)
        self._lock = threading.RLock()
        self._sequence = 0
        self._states: dict[str, dict[str, Any]] = {}
        self._closed = False
        self._socket: socket.socket | None = None
        self._reader: BinaryIO | None = None
        try:
            self._socket = socket.create_connection((host, int(port)), timeout=timeout)
            self._socket.settimeout(timeout)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._reader = self._socket.makefile("rb")
            response = self._rpc(
                "hello",
                token=token,
                arms=[{"name": name, "dofs": dofs} for name, dofs in self._expected.items()],
                robot_config=self._config_signature,
                initialize=initialize,
            )
            actual = {
                arm["name"]: int(arm["dofs"])
                for arm in response.get("arms", [])
                if isinstance(arm, dict) and "name" in arm and "dofs" in arm
            }
            if actual != self._expected:
                raise RemoteHardwareError(
                    f"station arm layout does not match Pi: Thor={self._expected}, Pi={actual}"
                )
        except Exception:
            self.close()
            raise

    def _close_unlocked(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader is not None:
            try:
                self._reader.close()
            except OSError:
                pass
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    @property
    def connected(self) -> bool:
        return not self._closed

    def _rpc(self, op: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RemoteHardwareError("Pi hardware bridge connection is closed")
            self._sequence += 1
            request = {"seq": self._sequence, "op": op, **payload}
            try:
                if self._socket is None or self._reader is None:
                    raise RemoteHardwareError("Pi hardware bridge connection is unavailable")
                encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
                if len(encoded) + 1 > _MAX_MESSAGE_BYTES:
                    raise ValueError("request exceeds the bridge message limit")
                self._socket.sendall(encoded + b"\n")
                raw = self._reader.readline(_MAX_MESSAGE_BYTES + 1)
                if not raw or len(raw) > _MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
                    raise RemoteHardwareError("Pi closed the bridge or sent an invalid response")
                response = json.loads(raw)
            except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._close_unlocked()
                raise RemoteHardwareError(
                    f"Pi hardware bridge request {op!r} failed: {exc}"
                ) from exc
            if not isinstance(response, dict) or response.get("seq") != self._sequence:
                self._close_unlocked()
                raise RemoteHardwareError("Pi hardware bridge returned an out-of-order response")
            if not response.get("ok"):
                raise RemoteHardwareError(str(response.get("error", "Pi bridge request failed")))
            result = response.get("result", {})
            if not isinstance(result, dict):
                raise RemoteHardwareError("Pi hardware bridge returned an invalid result")
            return result

    def refresh(self) -> None:
        result = self._rpc("snapshot")
        arms = result.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(self._expected):
            raise RemoteHardwareError("Pi returned a snapshot with the wrong arm roster")
        checked: dict[str, dict[str, Any]] = {}
        for name, dofs in self._expected.items():
            state = arms[name]
            try:
                follower = state["follower"]
                leader = state["leader"]
                joints = np.asarray(follower["joint_pos"], dtype=np.float64).reshape(-1)
                grip = np.asarray(follower["gripper_pos"], dtype=np.float64).reshape(-1)
                leader_arm = np.asarray(leader["arm"], dtype=np.float64).reshape(-1)
                leader_grip = float(leader["gripper"])
                buttons = [bool(x) for x in leader["buttons"]]
                raw_value, cal_value = leader.get("raw"), leader.get("cal")
                raw = (
                    None
                    if raw_value is None
                    else np.asarray(raw_value, dtype=np.float64).reshape(-1)
                )
                cal = (
                    None
                    if cal_value is None
                    else np.asarray(cal_value, dtype=np.float64).reshape(-1)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RemoteHardwareError(f"Pi returned malformed state for {name}: {exc}") from exc
            if (
                joints.size != dofs - 1
                or grip.size != 1
                or leader_arm.size != dofs - 1
                or len(buttons) != 2
                or (raw is not None and raw.size != dofs)
                or (cal is not None and cal.size != dofs)
                or not all(np.isfinite(v).all() for v in (joints, grip, leader_arm))
                or not math.isfinite(leader_grip)
            ):
                raise RemoteHardwareError(f"Pi returned invalid joint state for {name}")
            checked[name] = {
                "follower": {"joint_pos": joints.copy(), "gripper_pos": grip.copy()},
                "leader": {
                    "arm": leader_arm.copy(),
                    "gripper": leader_grip,
                    "buttons": buttons,
                    "raw": None if raw is None else raw.copy(),
                    "cal": None if cal is None else cal.copy(),
                },
            }
        with self._lock:
            self._states = checked

    def state(self, arm: str) -> dict[str, Any]:
        with self._lock:
            if arm not in self._states:
                raise RemoteHardwareError("no fresh Pi snapshot yet; refresh the bridge first")
            return self._states[arm]

    def command(self, arm: str, position: np.ndarray) -> None:
        values = np.asarray(position, dtype=np.float64).reshape(-1)
        if values.size != self._expected[arm] or not np.isfinite(values).all():
            raise ValueError(f"{arm} command must contain {self._expected[arm]} finite values")
        self._rpc("command", arm=arm, position=values.tolist())

    def hold(self, arm: str) -> None:
        self._rpc("hold", arm=arm)

    def relax(self, arm: str) -> None:
        self._rpc("relax", arm=arm)

    def power_off(self, arm: str) -> dict[str, Any]:
        return self._rpc("power_off", arm=arm)

    def zero(self, side: str) -> dict[str, Any]:
        return self._rpc("zero", side=side)

    def reset_hardware(self) -> str:
        result = self._rpc("reset")
        return str(result.get("message", "Pi CAN buses reset"))


class _RemoteRobot(RobotInterface):
    def __init__(self, client: RemoteHardwareClient, name: str, num_arm_joints: int):
        self._client = client
        self._name = name
        self._n = num_arm_joints

    def num_dofs(self) -> int:
        return self._n + 1

    def get_joint_pos(self) -> np.ndarray:
        state = self._client.state(self._name)["follower"]
        return np.concatenate([state["joint_pos"], state["gripper_pos"]])

    def command_joint_pos(self, pos: np.ndarray) -> None:
        self._client.command(self._name, pos)

    def get_observations(self) -> dict[str, np.ndarray]:
        state = self._client.state(self._name)["follower"]
        return {
            "joint_pos": state["joint_pos"].copy(),
            "gripper_pos": state["gripper_pos"].copy(),
        }

    def stop(self) -> None:
        self._client.hold(self._name)

    def relax(self) -> None:
        self._client.relax(self._name)

    def power_off(self) -> dict[str, Any]:
        return self._client.power_off(self._name)


class _RemotePassiveLeader:
    def __init__(self, client: RemoteHardwareClient, name: str):
        self._client = client
        self._name = name

    def get_state(self) -> tuple[np.ndarray, float, list[bool]]:
        leader = self._client.state(self._name)["leader"]
        return leader["arm"].copy(), leader["gripper"], list(leader["buttons"])

    def leader_angles(self) -> tuple[np.ndarray, np.ndarray]:
        leader = self._client.state(self._name)["leader"]
        if leader["raw"] is None or leader["cal"] is None:
            raise RemoteHardwareError(f"Pi did not return raw GELLO angles for {self._name}")
        return leader["raw"].copy(), leader["cal"].copy()

    def command_arm(self, arm_joints: np.ndarray) -> None:
        raise RuntimeError("remote passive GELLO leaders do not support force feedback")

    def stop(self) -> None:
        return None


def build_remote_arm_units(cfg: StationConfig, client: RemoteHardwareClient) -> list[ArmUnit]:
    """Build the same teleop stack over cached Pi snapshots and remote follower commands."""
    robots = cfg.robot.robots or [cfg.robot.active_robot()]
    units: list[ArmUnit] = []
    for robot_cfg in robots:
        name = _arm_name(robot_cfg.type)
        controller = cfg.robot.controller_for(robot_cfg)
        if not is_passive_gello(controller.type):
            raise ValueError(
                f"remote hardware mode currently requires passive GELLO leaders; "
                f"{controller.type!r} is not passive"
            )
        robot = _RemoteRobot(client, name, cfg.robot.num_arm_joints)
        leader = _RemotePassiveLeader(client, name)
        agent = YamTeleop(leader=leader, follower=robot, bilateral_kp=0.0)
        units.append(ArmUnit(name, robot, agent))
    return units
