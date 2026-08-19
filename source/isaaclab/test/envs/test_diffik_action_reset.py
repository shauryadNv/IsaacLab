# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free tests for :class:`DifferentialInverseKinematicsAction` reset behavior."""

import pytest
import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

pytestmark = pytest.mark.unit


class _IKControllerStub:
    def __init__(self):
        self.reset_env_ids = []

    def reset(self, env_ids=None):
        self.reset_env_ids.append(env_ids)


def test_reset_forwards_env_ids_to_controller():
    action = object.__new__(DifferentialInverseKinematicsAction)
    action._raw_actions = torch.ones(3, 2)
    action._ik_controller = _IKControllerStub()
    env_ids = torch.tensor([0, 2])

    DifferentialInverseKinematicsAction.reset(action, env_ids)

    torch.testing.assert_close(action._raw_actions, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]))
    assert len(action._ik_controller.reset_env_ids) == 1
    assert action._ik_controller.reset_env_ids[0] is env_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
