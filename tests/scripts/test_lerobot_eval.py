#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.scripts.lerobot_eval import post_process_libero_gripper_action, rollout


class DummyPolicy(nn.Module):
    def __init__(self, action: torch.Tensor):
        super().__init__()
        self._action = action

    def reset(self):
        return None

    def select_action(self, batch):
        return self._action.clone()


class DummyVecEnv:
    def __init__(self):
        self.num_envs = 1
        self.envs = [SimpleNamespace(task_description="dummy task", task="dummy task")]
        self.metadata = {"render_fps": 30}
        self.unwrapped = self
        self.recorded_actions: list[np.ndarray] = []

    def reset(self, seed=None):
        observation = {"agent_pos": np.zeros((1, 3), dtype=np.float32)}
        return observation, {}

    def call(self, name):
        if name == "_max_episode_steps":
            return [1]
        if name == "task_description":
            return [self.envs[0].task_description]
        if name == "task":
            return [self.envs[0].task]
        raise AttributeError(name)

    def step(self, action: np.ndarray):
        self.recorded_actions.append(action.copy())
        observation = {"agent_pos": np.zeros((1, 3), dtype=np.float32)}
        reward = np.array([0.0], dtype=np.float32)
        terminated = np.array([True], dtype=bool)
        truncated = np.array([False], dtype=bool)
        info = {"final_info": {"is_success": np.array([True], dtype=bool)}}
        return observation, reward, terminated, truncated, info


def make_identity_pipeline():
    return PolicyProcessorPipeline[dict, dict](steps=[])


def make_identity_postprocessor():
    return PolicyProcessorPipeline[torch.Tensor, torch.Tensor](
        steps=[],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )


def test_post_process_libero_gripper_action_remaps_only_last_dimension():
    action = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)

    processed = post_process_libero_gripper_action(action)

    torch.testing.assert_close(action, torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32))
    torch.testing.assert_close(processed, torch.tensor([[0.1, -0.2, 0.4]], dtype=torch.float32))


def test_rollout_optionally_applies_libero_gripper_postprocess():
    base_action = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)

    for enabled, expected_gripper in ((False, 0.3), (True, 0.4)):
        env = DummyVecEnv()
        policy = DummyPolicy(base_action)

        rollout(
            env=env,
            policy=policy,
            env_preprocessor=make_identity_pipeline(),
            env_postprocessor=make_identity_pipeline(),
            preprocessor=make_identity_pipeline(),
            postprocessor=make_identity_postprocessor(),
            libero_gripper_postprocess=enabled,
        )

        np.testing.assert_allclose(
            env.recorded_actions[0],
            np.array([[0.1, -0.2, expected_gripper]], dtype=np.float32),
        )
