from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bimanual_vla.collection.teleop_bimanual import (
    DEFAULT_LEFT_MASTER,
    DEFAULT_LEFT_SLAVE,
    DEFAULT_RIGHT_MASTER,
    DEFAULT_RIGHT_SLAVE,
    _read_7d,
    _read_master_7d,
    _require_shared_bus_mapping,
)


def _message(**values):
    return SimpleNamespace(**values)


class FakeSharedBus:
    def GetArmJointMsgs(self):
        return _message(
            joint_state=_message(
                joint_1=1000,
                joint_2=2000,
                joint_3=3000,
                joint_4=4000,
                joint_5=5000,
                joint_6=6000,
            )
        )

    def GetArmGripperMsgs(self):
        return _message(gripper_state=_message(grippers_angle=35_000))

    def GetArmJointCtrl(self):
        return _message(
            joint_ctrl=_message(
                joint_1=7000,
                joint_2=8000,
                joint_3=9000,
                joint_4=10_000,
                joint_5=11_000,
                joint_6=12_000,
            )
        )

    def GetArmGripperCtrl(self):
        return _message(gripper_ctrl=_message(grippers_angle=70_000))


def _args(left_master="can0", left_slave="can0", right_master="can1", right_slave="can1"):
    return SimpleNamespace(
        left_master=left_master,
        left_slave=left_slave,
        right_master=right_master,
        right_slave=right_slave,
    )


def test_default_mapping_uses_one_shared_can_per_side():
    assert DEFAULT_LEFT_MASTER == DEFAULT_LEFT_SLAVE == "can0"
    assert DEFAULT_RIGHT_MASTER == DEFAULT_RIGHT_SLAVE == "can1"
    assert _require_shared_bus_mapping(_args()) == ("can0", "can1")


@pytest.mark.parametrize(
    "args, message",
    [
        (_args(left_slave="can2"), "left master and slave"),
        (_args(right_slave="can2"), "right master and slave"),
        (_args(right_master="can0", right_slave="can0"), "left and right"),
    ],
)
def test_non_shared_or_colliding_mapping_is_rejected(args, message):
    with pytest.raises(ValueError, match=message):
        _require_shared_bus_mapping(args)


def test_shared_bus_separates_slave_feedback_from_master_control_frames():
    bus = FakeSharedBus()

    slave_state = _read_7d(bus)
    master_action = _read_master_7d(bus)

    np.testing.assert_allclose(slave_state[:6], np.arange(1, 7) * 1000 / 57295.7795)
    np.testing.assert_allclose(master_action[:6], np.arange(7, 13) * 1000 / 57295.7795)
    assert slave_state[6] == pytest.approx(0.5)
    assert master_action[6] == pytest.approx(1.0)
