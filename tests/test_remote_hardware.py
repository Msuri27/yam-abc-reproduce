import socket
import threading
import time

import numpy as np
import pytest

import yam_abc_reproduce.robot.can_bridge as can_bridge
from yam_abc_reproduce.config import (
    ControllerConfig,
    RobotConfig,
    RobotUnitConfig,
    StationConfig,
)
from yam_abc_reproduce.robot.can_bridge import CanHardwareBridge
from yam_abc_reproduce.robot.remote import (
    RemoteHardwareClient,
    RemoteHardwareError,
    build_remote_arm_units,
)
from yam_abc_reproduce.runtime import ArmUnit


class FakeRobot:
    def __init__(self):
        self.position = np.zeros(7)
        self.stopped = 0

    def num_dofs(self):
        return 7

    def get_joint_pos(self):
        return self.position.copy()

    def get_observations(self):
        return {"joint_pos": self.position[:6].copy(), "gripper_pos": self.position[6:].copy()}

    def command_joint_pos(self, position):
        self.position = np.asarray(position, dtype=float).copy()

    def stop(self):
        self.stopped += 1

    def relax(self):
        pass


class FakeAgent:
    def state_snapshot(self):
        return {
            "arm": [0.1] * 6,
            "gripper": 0.75,
            "buttons": [False, True],
            "raw": [0.2] * 7,
            "cal": [0.1] * 7,
        }

    def hardware_zero(self):
        return list(range(7))

    def stop(self):
        pass


def _cfg():
    return StationConfig(
        robot=RobotConfig(
            robots=[RobotUnitConfig("yam_left", "flexible_4310")],
            controllers=[ControllerConfig("passive_gello_left", "yam_left")],
        )
    )


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_bridge_snapshot_commands_and_zero_round_trip():
    robot = FakeRobot()
    bridge = CanHardwareBridge(_cfg(), "secret", [ArmUnit("left", robot, FakeAgent())])
    port = _free_port()
    server = threading.Thread(target=bridge.serve_forever, args=("127.0.0.1", port), daemon=True)
    server.start()
    assert bridge._listening.wait(1)
    client = RemoteHardwareClient("127.0.0.1", port, _cfg(), token="secret")
    try:
        client.refresh()
        state = client.state("left")
        np.testing.assert_allclose(state["leader"]["arm"], [0.1] * 6)
        np.testing.assert_allclose(state["follower"]["joint_pos"], np.zeros(6))
        assert state["leader"]["buttons"] == [False, True]

        (unit,) = build_remote_arm_units(_cfg(), client)
        assert unit.agent.read_inputs() == ([False, True], 0.75)
        raw, calibrated = unit.agent.leader_raw()
        np.testing.assert_allclose(raw, [0.2] * 7)
        np.testing.assert_allclose(calibrated, [0.1] * 7)
        target = np.array([0.1] * 6 + [0.75])
        np.testing.assert_allclose(unit.agent.act(unit.robot.get_observations()), target)
        np.testing.assert_allclose(robot.position, target)
        assert client.zero("left")["devices"] == list(range(7))
    finally:
        client.close()
        bridge.close()
        server.join(timeout=2)
    assert not server.is_alive()
    assert robot.stopped >= 1


def test_bridge_supports_motorized_yam_leader_without_raw_encoder_angles():
    cfg = _cfg()
    cfg.robot.controllers = [ControllerConfig("yam_lead_left", "yam_left")]
    robot = FakeRobot()
    agent = FakeAgent()
    agent.state_snapshot = lambda: {
        "arm": [0.1] * 6,
        "gripper": 0.75,
        "buttons": [False, True],
        "raw": None,
        "cal": None,
    }
    bridge = CanHardwareBridge(cfg, "secret", [ArmUnit("left", robot, agent)])
    port = _free_port()
    server = threading.Thread(target=bridge.serve_forever, args=("127.0.0.1", port), daemon=True)
    server.start()
    assert bridge._listening.wait(1)
    client = RemoteHardwareClient("127.0.0.1", port, cfg, token="secret")
    try:
        client.refresh()
        (unit,) = build_remote_arm_units(cfg, client)
        assert unit.agent.read_inputs() == ([False, True], 0.75)
        assert unit.agent.leader_raw() is None
        target = np.array([0.1] * 6 + [0.75])
        np.testing.assert_allclose(unit.agent.act(unit.robot.get_observations()), target)
        np.testing.assert_allclose(robot.position, target)
    finally:
        client.close()
        bridge.close()
        server.join(timeout=2)
    assert not server.is_alive()


def test_bridge_rejects_station_layout_mismatch():
    robot = FakeRobot()
    bridge = CanHardwareBridge(_cfg(), "secret", [ArmUnit("left", robot, FakeAgent())])
    port = _free_port()
    server = threading.Thread(target=bridge.serve_forever, args=("127.0.0.1", port), daemon=True)
    server.start()
    assert bridge._listening.wait(1)
    mismatched = StationConfig(
        robot=RobotConfig(
            robots=[
                RobotUnitConfig("yam_left", "flexible_4310"),
                RobotUnitConfig("yam_right", "flexible_4310"),
            ],
            controllers=[
                ControllerConfig("passive_gello_left", "yam_left"),
                ControllerConfig("passive_gello_right", "yam_right"),
            ],
        )
    )
    try:
        with pytest.raises(RemoteHardwareError, match="station arm layout"):
            RemoteHardwareClient("127.0.0.1", port, mismatched, token="secret")
    finally:
        bridge.close()
        server.join(timeout=2)
    assert not server.is_alive()


def test_bridge_rejects_different_robot_calibration():
    robot = FakeRobot()
    bridge = CanHardwareBridge(_cfg(), "secret", [ArmUnit("left", robot, FakeAgent())])
    port = _free_port()
    server = threading.Thread(target=bridge.serve_forever, args=("127.0.0.1", port), daemon=True)
    server.start()
    assert bridge._listening.wait(1)
    different_cfg = _cfg()
    different_cfg.robot.leader_joint_signs = [1] * 7
    try:
        with pytest.raises(RemoteHardwareError, match="robot settings differ"):
            RemoteHardwareClient("127.0.0.1", port, different_cfg, token="secret")
    finally:
        bridge.close()
        server.join(timeout=2)
    assert not server.is_alive()


def test_bridge_watchdog_holds_and_closes_stale_client():
    robot = FakeRobot()
    bridge = CanHardwareBridge(_cfg(), "secret", [ArmUnit("left", robot, FakeAgent())])
    port = _free_port()
    server = threading.Thread(target=bridge.serve_forever, args=("127.0.0.1", port), daemon=True)
    server.start()
    assert bridge._listening.wait(1)
    client = RemoteHardwareClient("127.0.0.1", port, _cfg(), token="secret")
    try:
        client.refresh()
        time.sleep(0.7)
        with pytest.raises(RemoteHardwareError):
            client.refresh()
        assert robot.stopped >= 1
    finally:
        client.close()
        bridge.close()
        server.join(timeout=2)
    assert not server.is_alive()


def test_bridge_reset_releases_hardware_without_reopening_it(monkeypatch):
    robot = FakeRobot()
    bridge = CanHardwareBridge(_cfg(), "secret", [ArmUnit("left", robot, FakeAgent())])
    monkeypatch.setattr(can_bridge, "reset_can_buses", lambda channels=None: (True, "CAN reset"))
    monkeypatch.setattr(can_bridge, "check_can_up", lambda channels: [])
    port = _free_port()
    server = threading.Thread(target=bridge.serve_forever, args=("127.0.0.1", port), daemon=True)
    server.start()
    assert bridge._listening.wait(1)
    client = RemoteHardwareClient("127.0.0.1", port, _cfg(), token="secret")
    try:
        assert client.reset_hardware() == "CAN reset"
        assert bridge._units == []
    finally:
        client.close()
        bridge.close()
        server.join(timeout=2)
    assert not server.is_alive()


def test_bridge_opens_hardware_only_when_thor_connects(monkeypatch):
    robot = FakeRobot()
    opened = []
    monkeypatch.setattr(
        can_bridge,
        "_build_hardware",
        lambda cfg: opened.append(True) or [ArmUnit("left", robot, FakeAgent())],
    )
    bridge = CanHardwareBridge(_cfg(), "secret")
    port = _free_port()
    server = threading.Thread(target=bridge.serve_forever, args=("127.0.0.1", port), daemon=True)
    server.start()
    assert bridge._listening.wait(1)
    assert opened == []
    client = RemoteHardwareClient("127.0.0.1", port, _cfg(), token="secret")
    try:
        assert opened == [True]
        client.refresh()
    finally:
        client.close()
        bridge.close()
        server.join(timeout=2)
    assert not server.is_alive()
