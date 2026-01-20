#!/usr/bin/env python3
"""
Benchmark Runner (Official-compatible)

Goal:
- Match LIBERO official example environment init/state/action behavior.
- Keep RemoteAgent + VideoLogger integration, but do NOT modify env state/action conventions.
"""

import os
import argparse
import random
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from agent import RemoteAgent
from misc.logger import VideoLogger


def configure_logging():
    """Suppress verbose library logs."""
    logging.getLogger('curobo').setLevel(logging.ERROR)
    logging.getLogger('robomimic').setLevel(logging.ERROR)
    logging.getLogger('robosuite').setLevel(logging.ERROR)


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def setup_benchmark_paths() -> Dict[str, str]:
    return {
        "benchmark_root": get_libero_path("benchmark_root"),
        "init_states": get_libero_path("init_states"),
        "datasets": get_libero_path("datasets"),
        "bddl_files": get_libero_path("bddl_files"),
    }


def get_benchmark_dict() -> Dict[str, Any]:
    return benchmark.get_benchmark_dict()


def get_task_bddl_path(task_suite, task_id: int) -> Tuple[str, str, str]:
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return task_bddl_file, task_name, task_description


def create_environment(bddl_file_path: str, seed: int = 0, cam_h: int = 256, cam_w: int = 256) -> OffScreenRenderEnv:
    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": cam_h,
        "camera_widths": cam_w,
        "camera_depths": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    env.reset()
    return env


def set_init_state(env: OffScreenRenderEnv, task_suite, task_id: int, init_state_id: int):
    init_states = task_suite.get_task_init_states(task_id)
    if init_state_id < 0 or init_state_id >= len(init_states):
        raise ValueError(f"init_state_id={init_state_id} out of range [0, {len(init_states)-1}]")
    env.set_init_state(init_states[init_state_id])


def stabilize_scene(env: OffScreenRenderEnv, steps: int = 10) -> Any:
    """
    Official behavior: execute dummy_action = [0.]*7 for a few steps.
    Do NOT depend on agent proprio or modify gripper convention.
    """
    dummy_action = [0.0] * 7
    obs = None
    for _ in range(steps):
        obs, _, _, _ = env.step(dummy_action)
    return obs


def step_agent(
    agent: RemoteAgent,
    obs: Any,
    debug: bool = False
) -> Tuple[np.ndarray, Optional[Any]]:
    """
    Thin adapter in case agent.step returns list; enforce np.ndarray float32 shape (7,).
    """
    out = agent.step(obs, debug=debug)

    # support both:
    #   action, bbox = agent.step(...)
    #   action = agent.step(...)
    if isinstance(out, tuple) and len(out) == 2:
        action, bbox = out
    else:
        action, bbox = out, None

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"Agent action must be 7-dim for LIBERO env.step; got shape={action.shape}")
    return action, bbox


def run_episode(
    env: OffScreenRenderEnv,
    agent: RemoteAgent,
    video_logger: VideoLogger,
    max_steps: int = 300,
    stabilize_steps: int = 10,
    debug: bool = False,
) -> bool:
    """
    Returns:
        success (bool): done=True within max_steps
    """
    # Stabilize by official dummy action steps (and obtain last obs)
    obs = stabilize_scene(env, steps=stabilize_steps)

    for _ in tqdm.tqdm(range(max_steps)):
        action, bbox = step_agent(agent, obs, debug=debug)
        obs, reward, done, info = env.step(action)

        # VideoLogger is user-defined; keep signature consistent with your existing code.
        video_logger.log_frame(obs, bbox)

        if done:
            video_logger.stop_recording(success=True)
            return

    video_logger.stop_recording(success=False)
    return