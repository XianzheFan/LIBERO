"""
Rollout pi0.5 on LIBERO and collect data in LeRobot format for DreamDojo fine-tuning.

Data layout produced:
  <output_dir>/
    meta/
      modality.json
      info.json
      episodes.jsonl
      tasks.jsonl
      stats.json
    data/
      chunk-000/
        episode_000000.parquet
        ...
    videos/
      chunk-000/
        observation.images.agentview/
          episode_000000.mp4
          ...
        observation.images.wrist/
          episode_000000.mp4
          ...

Action / state convention (matches DreamDojo's relative-action rebaselining):
  observation.state  [8D]: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)  -- absolute
  action             [7D]: eef_pos(3) + eef_axisangle(3) + gripper_qpos[0]  -- absolute EE state
    -> stored as absolute so WrappedLeRobotSingleDataset can compute chunk-relative deltas

In DreamDojo's 384-dim action vector the 7D arm action is placed in the reserved slot [169:176].
See DreamDojo/groot_dreams/data/dataset.py WrappedLeRobotSingleDataset.__getitem__.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Optional

import imageio
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ENV_FPS = 10          # LIBERO recorded video fps (matches imageio fps=10 in original main.py)
CHUNK_SIZE = 1000     # episodes per parquet/video chunk
STATE_DIM = 8         # eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
ACTION_DIM = 7        # eef_pos(3) + eef_axisangle(3) + gripper_qpos[0](1)

LIBERO_MODALITY = {
    "state": {
        "eef_pos": {
            "original_key": "observation.state",
            "start": 0, "end": 3,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
        "eef_rot": {
            "original_key": "observation.state",
            "start": 3, "end": 6,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
        "gripper": {
            "original_key": "observation.state",
            "start": 6, "end": 8,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
    },
    "action": {
        "arm": {
            "original_key": "action",
            "start": 0, "end": 7,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
    },
    "video": {
        "agentview": {
            "original_key": "observation.images.agentview",
        },
        "wrist": {
            "original_key": "observation.images.wrist",
        },
    },
    "annotation": {
        "language.task": {
            "original_key": "task_index",
        },
    },
}


@dataclasses.dataclass
class Args:
    # Model server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Data output
    output_dir: str = "data/libero_dreamdojo"
    save_failed: bool = False   # also save failed episodes

    seed: int = 7


class DreamDojoWriter:
    """Incrementally writes a LeRobot-format dataset compatible with DreamDojo."""

    def __init__(self, output_dir: pathlib.Path, env_fps: int = ENV_FPS):
        self.root = pathlib.Path(output_dir)
        self.env_fps = env_fps

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)

        self.episode_index: int = 0
        self.global_frame_index: int = 0

        # Registered tasks: description -> task_index
        self.task_registry: dict[str, int] = {}

        # Per-episode buffers (populated by record_step)
        self._ep_states: list[np.ndarray] = []
        self._ep_actions: list[np.ndarray] = []
        self._ep_agentview: list[np.ndarray] = []
        self._ep_wrist: list[np.ndarray] = []
        self._ep_task_index: Optional[int] = None

        # Accumulated stats for normalization (per column, list of arrays)
        self._stat_accum: dict[str, list[np.ndarray]] = {
            "observation.state": [],
            "action": [],
        }

        self._episodes_meta: list[dict] = []


    def begin_episode(self, task_description: str) -> None:
        if task_description not in self.task_registry:
            self.task_registry[task_description] = len(self.task_registry)
        self._ep_task_index = self.task_registry[task_description]
        self._ep_states = []
        self._ep_actions = []
        self._ep_agentview = []
        self._ep_wrist = []

    def record_step(
        self,
        agentview_img: np.ndarray,   # (H, W, 3) uint8
        wrist_img: np.ndarray,       # (H, W, 3) uint8
        eef_pos: np.ndarray,         # (3,)
        eef_quat: np.ndarray,        # (4,)  wxyz
        gripper_qpos: np.ndarray,    # (2,)
    ) -> None:
        axisangle = _quat2axisangle(eef_quat)                      # (3,)
        state = np.concatenate([eef_pos, axisangle, gripper_qpos]) # (8,)
        action = np.concatenate([eef_pos, axisangle, gripper_qpos[:1]])  # (7,)

        self._ep_states.append(state.astype(np.float64))
        self._ep_actions.append(action.astype(np.float64))
        self._ep_agentview.append(agentview_img)
        self._ep_wrist.append(wrist_img)

    def end_episode(self, success: bool) -> bool:
        """Flush episode to disk. Returns True if episode was written."""
        n = len(self._ep_states)
        if n == 0:
            return False

        ep_idx = self.episode_index
        chunk_idx = ep_idx // CHUNK_SIZE

        for cam_key, frames in [
            ("observation.images.agentview", self._ep_agentview),
            ("observation.images.wrist",     self._ep_wrist),
        ]:
            vid_dir = self.root / "videos" / f"chunk-{chunk_idx:03d}" / cam_key
            vid_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_dir / f"episode_{ep_idx:06d}.mp4"
            imageio.mimwrite(
                vid_path,
                [np.asarray(f) for f in frames],
                fps=self.env_fps,
                codec="libx264",
                quality=8,
            )

        data_dir = self.root / "data" / f"chunk-{chunk_idx:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for frame_idx, (state, action) in enumerate(
            zip(self._ep_states, self._ep_actions)
        ):
            rows.append({
                "observation.state": state,
                "action":            action,
                "timestamp":         frame_idx / self.env_fps,
                "frame_index":       frame_idx,
                "episode_index":     ep_idx,
                "index":             self.global_frame_index + frame_idx,
                "next.done":         (frame_idx == n - 1),
                "task_index":        self._ep_task_index,
                "success":           success,
            })

        df = pd.DataFrame(rows)
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        self._stat_accum["observation.state"].extend(self._ep_states)
        self._stat_accum["action"].extend(self._ep_actions)
        self._episodes_meta.append({
            "episode_index": ep_idx,
            "tasks":         [list(self.task_registry.keys())[self._ep_task_index]],
            "length":        n,
            "success":       bool(success),
        })

        self.global_frame_index += n
        self.episode_index += 1
        return True


    def finalize(self) -> None:
        """Write all meta files after all episodes are collected."""
        if self.episode_index == 0:
            logging.warning("No episodes were written; skipping finalize.")
            return

        total_ep = self.episode_index
        total_frames = self.global_frame_index

        info = {
            "codebase_version": "v2.0",
            "robot_type":       "single_arm",
            "fps":              self.env_fps,
            "data_path":        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path":       "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "chunks_size":      CHUNK_SIZE,
            "total_episodes":   total_ep,
            "total_frames":     total_frames,
            "total_tasks":      len(self.task_registry),
            "features": {
                "observation.state": {
                    "shape":  [STATE_DIM],
                    "names":  ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                               "eef_rot_x", "eef_rot_y", "eef_rot_z",
                               "gripper_0", "gripper_1"],
                    "dtype":  "float64",
                },
                "action": {
                    "shape": [ACTION_DIM],
                    "names": ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                              "eef_rot_x", "eef_rot_y", "eef_rot_z",
                              "gripper"],
                    "dtype": "float64",
                },
                "observation.images.agentview": {
                    "shape":  [LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3],
                    "names":  ["height", "width", "channel"],
                    "dtype":  "uint8",
                    "video_info": {"video.fps": float(self.env_fps)},
                },
                "observation.images.wrist": {
                    "shape":  [LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3],
                    "names":  ["height", "width", "channel"],
                    "dtype":  "uint8",
                    "video_info": {"video.fps": float(self.env_fps)},
                },
            },
        }
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        with open(self.root / "meta" / "modality.json", "w") as f:
            json.dump(LIBERO_MODALITY, f, indent=2)

        with open(self.root / "meta" / "tasks.jsonl", "w") as f:
            for desc, idx in self.task_registry.items():
                f.write(json.dumps({"task_index": idx, "task": desc}) + "\n")

        with open(self.root / "meta" / "episodes.jsonl", "w") as f:
            for meta in self._episodes_meta:
                f.write(json.dumps(meta) + "\n")

        stats = {}
        for col, arrays in self._stat_accum.items():
            if not arrays:
                continue
            data = np.stack(arrays, axis=0)  # (N, D)
            stats[col] = {
                "mean": data.mean(0).tolist(),
                "std":  data.std(0).tolist(),
                "min":  data.min(0).tolist(),
                "max":  data.max(0).tolist(),
                "q01":  np.quantile(data, 0.01, axis=0).tolist(),
                "q99":  np.quantile(data, 0.99, axis=0).tolist(),
            }
        with open(self.root / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logging.info(
            f"Dataset written to {self.root}: "
            f"{total_ep} episodes, {total_frames} frames, "
            f"{len(self.task_registry)} tasks."
        )


def collect_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}  ({num_tasks} tasks)")

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object":  280,
        "libero_goal":    300,
        "libero_10":      520,
        "libero_90":      400,
    }
    max_steps = max_steps_map.get(args.task_suite_name, 300)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    writer = DreamDojoWriter(pathlib.Path(args.output_dir), env_fps=ENV_FPS)

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm(range(num_tasks), desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        logging.info(f"\n=== Task {task_id}: {task_description} ===")

        for episode_idx in tqdm(range(args.num_trials_per_task), desc="episodes", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan: collections.deque = collections.deque()

            writer.begin_episode(task_description)

            t = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                try:
                    # Warm-up: let objects settle
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist     = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img_resized   = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(agentview, args.resize_size, args.resize_size)
                    )
                    wrist_resized = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size)
                    )

                    # record step (original 256×256 for video storage)
                    writer.record_step(
                        agentview_img=agentview,
                        wrist_img=wrist,
                        eef_pos=obs["robot0_eef_pos"].copy(),
                        eef_quat=obs["robot0_eef_quat"].copy(),
                        gripper_qpos=obs["robot0_gripper_qpos"].copy(),
                    )

                    if not action_plan:
                        element = {
                            "observation/image":       img_resized,
                            "observation/wrist_image": wrist_resized,
                            "observation/state": np.concatenate([
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            ]),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())

                    if done:
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Step exception: {e}")
                    break

            if done or args.save_failed:
                writer.end_episode(success=done)
                total_episodes += 1

            logging.info(
                f"Episode {episode_idx+1}: {'SUCCESS' if done else 'FAIL'} | "
                f"total {total_successes}/{total_episodes} "
                f"({100*total_successes/max(total_episodes,1):.1f}%)"
            )

    writer.finalize()

    logging.info(f"Final success rate: {total_successes}/{total_episodes} "
                 f"({100*total_successes/max(total_episodes,1):.1f}%)")


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name":  task_bddl_file,
        "camera_heights":  resolution,
        "camera_widths":   resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) → axis-angle (3,).
    Copied from robosuite transform_utils."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    collect_libero(args)
