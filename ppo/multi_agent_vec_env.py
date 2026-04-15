from typing import List, Optional

import cv2
import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

import racecar_gym.envs.gym_api  # registers environments
from racecar_gym.envs.gym_api import MultiAgentRaceEnv


class MultiAgentSharedPolicyVecEnv(VecEnv):
    """Wraps MultiAgentRaceEnv as an SB3 VecEnv for shared-policy training.

    Each agent is treated as a separate "environment" so SB3's PPO can train
    a single policy across all agents. All agents share the same physics
    simulation and can collide with each other.

    Observation preprocessing:
        rgb_camera (H, W, 3) → grayscale (H, W, 1) → resize (img_size, img_size, 1)

    Action space:
        Box(-1, 1, (2,)) where [0]=motor, [1]=steering
    """

    def __init__(self, scenario_path: str, img_size: int = 84, render_mode: str = 'rgb_array_birds_eye'):
        self._env = MultiAgentRaceEnv(scenario=scenario_path, render_mode=render_mode)
        self._agent_ids: List[str] = list(self._env.scenario.agents.keys())
        self._n_agents = len(self._agent_ids)
        self._img_size = img_size
        self._render_mode = render_mode

        obs_space = gymnasium.spaces.Box(
            low=0, high=255,
            shape=(img_size, img_size, 1),
            dtype=np.uint8
        )
        action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        super().__init__(
            num_envs=self._n_agents,
            observation_space=obs_space,
            action_space=action_space,
        )

        self._actions: Optional[np.ndarray] = None
        self._last_obs: Optional[np.ndarray] = None

    def _preprocess_obs(self, rgb: np.ndarray) -> np.ndarray:
        """Convert (H, W, 3) RGB → (img_size, img_size, 1) uint8 grayscale."""
        # pybullet getCameraImage returns int32; cv2 requires uint8
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self._img_size, self._img_size), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]

    def _extract_obs(self, raw_obs: dict) -> np.ndarray:
        """Extract and preprocess observations for all agents → (n_agents, H, W, 1)."""
        return np.stack([
            self._preprocess_obs(raw_obs[aid]['rgb_camera'])
            for aid in self._agent_ids
        ], axis=0)

    def _build_action_dict(self, actions: np.ndarray) -> dict:
        """Convert (n_agents, 2) array to MultiAgentRaceEnv action dict."""
        return {
            aid: {
                'motor': np.array([actions[i, 0]], dtype=np.float32),
                'steering': np.array([actions[i, 1]], dtype=np.float32),
            }
            for i, aid in enumerate(self._agent_ids)
        }

    def reset(self) -> VecEnvObs:
        raw_obs, _ = self._env.reset(options={'mode': 'grid'})
        self._last_obs = self._extract_obs(raw_obs)
        return self._last_obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        action_dict = self._build_action_dict(self._actions)
        raw_obs, rewards, dones, _, states = self._env.step(action_dict)

        obs = self._extract_obs(raw_obs)
        self._last_obs = obs

        rew_array = np.array([rewards[aid] for aid in self._agent_ids], dtype=np.float32)
        done_array = np.array([dones[aid] for aid in self._agent_ids], dtype=bool)
        info_list = [states[aid] for aid in self._agent_ids]

        # Auto-reset: if any agent is done, reset the whole environment
        if done_array.any():
            raw_obs, _ = self._env.reset(options={'mode': 'grid'})
            obs = self._extract_obs(raw_obs)
            self._last_obs = obs

        return obs, rew_array, done_array, info_list

    def close(self) -> None:
        self._env.close()

    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        return self._env.render()

    def get_images(self):
        return [self._env.render()]

    # --- Required abstract methods (not used in training) ---

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        return [getattr(self._env, attr_name)] * self._n_agents

    def set_attr(self, attr_name: str, value, indices: VecEnvIndices = None) -> None:
        setattr(self._env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs):
        method = getattr(self._env, method_name)
        return [method(*method_args, **method_kwargs)] * self._n_agents

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None):
        return [False] * self._n_agents

    def seed(self, seed: Optional[int] = None):
        pass
