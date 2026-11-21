from scripts import setup_can_udev


def test_identify_ignores_the_separate_main_can_interface(monkeypatch):
    interfaces = {"can0", "can1", "can2", "can3", "can4"}
    excluded = {"can0"}
    unplug_order = iter(("can1", "can2", "can3"))
    unplugged = []
    cached = {f"can{i}": f"serial-{i}" for i in range(1, 5)}

    def fake_input(prompt):
        if "UNPLUG" in prompt:
            iface = next(unplug_order)
            interfaces.remove(iface)
            unplugged.append(iface)
        else:
            interfaces.add(unplugged.pop())

    def fake_wait(before, want_removed, timeout=120, excluded=None):
        current = interfaces - (excluded or set())
        diff = (before - current) if want_removed else (current - before)
        assert len(diff) == 1
        return diff.pop()

    monkeypatch.setattr(
        setup_can_udev,
        "can_interfaces",
        lambda ignored=None: interfaces - (ignored or set()),
    )
    monkeypatch.setattr(setup_can_udev, "wait_for_change", fake_wait)
    monkeypatch.setattr(setup_can_udev, "serial_of_cache", cached)
    monkeypatch.setattr(setup_can_udev, "serial_of", lambda iface: cached[iface])
    monkeypatch.setattr("builtins.input", fake_input)

    mapping = setup_can_udev.identify(setup_can_udev.DEFAULT_ROLES, excluded=excluded)

    assert mapping == {
        "can_left": "serial-1",
        "can_right": "serial-2",
        "can_lead_l": "serial-3",
        "can_lead_r": "serial-4",
    }
