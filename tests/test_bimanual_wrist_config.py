from pathlib import Path

from yam_abc_reproduce.config import build_station_config, controller_channel_for, robot_channel_for


def test_bimanual_wrist_station_uses_all_four_arm_buses_and_three_cameras():
    path = Path(__file__).parents[1] / "configs" / "station_yam_bimanual_wrist.yaml"
    cfg = build_station_config(path)

    assert [robot_channel_for(robot) for robot in cfg.robot.robots] == [
        "can_left",
        "can_right",
    ]
    assert [controller_channel_for(controller) for controller in cfg.robot.controllers] == [
        "can_lead_l",
        "can_lead_r",
    ]
    assert [controller.type for controller in cfg.robot.controllers] == [
        "yam_lead_left",
        "yam_lead_right",
    ]
    assert [(camera.name, camera.role) for camera in cfg.cameras] == [
        ("top", "top"),
        ("left", "left"),
        ("right", "right"),
    ]
