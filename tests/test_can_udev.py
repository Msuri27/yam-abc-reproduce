from scripts import setup_can_udev


def test_identify_ignores_the_separate_main_can_interface(monkeypatch):
    interfaces = {"can0", "can1", "can2", "can3", "can4"}
    excluded = {"can0"}
    removed = iter(("can1", "can2", "can3"))
    cached = {f"can{i}": f"serial-{i}" for i in range(1, 5)}
    monkeypatch.setattr(
        setup_can_udev,
        "can_interfaces",
        lambda ignored=None: interfaces - (ignored or set()),
    )
    monkeypatch.setattr(
        setup_can_udev,
        "wait_for_change",
        lambda before, want_removed, timeout=120, excluded=None: (
            next(removed) if want_removed else None
        ),
    )
    monkeypatch.setattr(setup_can_udev, "serial_of_cache", cached)
    monkeypatch.setattr(setup_can_udev, "serial_of", lambda iface: cached[iface])
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    mapping = setup_can_udev.identify(setup_can_udev.DEFAULT_ROLES, excluded=excluded)

    assert mapping == {
        "can_left": "serial-1",
        "can_right": "serial-2",
        "can_lead_l": "serial-3",
        "can_lead_r": "serial-4",
    }
