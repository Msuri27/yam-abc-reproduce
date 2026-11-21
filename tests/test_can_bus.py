from types import SimpleNamespace

from yam_abc_reproduce.robot import can_bus


def test_selective_reset_only_touches_configured_channels(monkeypatch):
    calls = []
    monkeypatch.setattr(can_bus.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(can_bus.shutil, "which", lambda name: "/usr/sbin/ip")
    monkeypatch.setattr(
        can_bus.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    channels = ["can_left", "can_right", "can_lead_l", "can_lead_r"]

    ok, output = can_bus.reset_can_buses(channels=channels)

    assert ok
    assert output == "Reset CAN interfaces: can_left, can_right, can_lead_l, can_lead_r"
    assert len(calls) == 2 * len(channels)
    assert all("can0" not in command for command in calls)
    assert calls[0] == ["sudo", "-n", "/usr/sbin/ip", "link", "set", "can_left", "down"]
    assert calls[1] == [
        "sudo", "-n", "/usr/sbin/ip", "link", "set", "can_left", "up", "type", "can",
        "bitrate", "1000000",
    ]


def test_selective_reset_rejects_invalid_channel_names(monkeypatch):
    monkeypatch.setattr(
        can_bus.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    ok, message = can_bus.reset_can_buses(channels=["can_left; reboot"])

    assert not ok
    assert "invalid CAN interface" in message
