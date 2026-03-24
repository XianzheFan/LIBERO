"""
VLAW-style synthetic trajectory generation via VLA + World Model autoregressive rollout.

Instead of executing actions in the real LIBERO simulator, this script:
1. Extracts the initial frame s_0 (and optionally wrist frame) from real collected trajectories.
2. Feeds s_0 + language instruction I to the pi05 VLA to predict an action chunk a_t.
3. Feeds (s_t, a_t) to DreamDojo (action-conditioned world model) to "imagine" the next frame s_{t+1}.
4. Loops autoregressively: s_{t+1} becomes the new input to the VLA -> predict a_{t+1} -> DreamDojo
   generates s_{t+2}, and so on until max time steps T.
5. Performs parallel multi-trajectory rollouts from the same initial frame to produce diverse
   synthetic trajectories (both successful and failed), thanks to stochastic action sampling
   (Flow-SDE) and diffusion-based video generation randomness.

The output is a LeRobot-format dataset compatible with DreamDojo fine-tuning, identical in schema
to `collect_dreamdojo_data.py`.

Reference: VLAW (Vision-Language-Action World model) — "Scaling Up VLAs with World Models"

Usage:
    # 1. Start DreamDojo server(s) — one per parallel rollout
    python examples/dreamdojo_server.py --checkpoint <ckpt> --experiment dreamdojo_2b_480_640_libero \
        --save-dir /tmp/dd_gen --port 8020

    # 2. Start pi05 VLA policy server (or use JAX model directly)
    python scripts/serve_sde_policy.py --config pi05_libero --checkpoint checkpoints/pi05_libero

    # 3. Run this script
    python third_party/libero/generate_vlaw_synthetic_data.py \
        --task_suite_name libero_10 \
        --num_rollouts_per_init 8 \
        --max_time_steps 520 \
        --output_dir data/libero/vlaw_synthetic
"""

import base64
import concurrent.futures
import dataclasses
import json
import logging
import math
import pathlib
import re
import tempfile
from typing import Optional

import cv2
import imageio
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import torch
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
from openpi_client import websocket_client_policy as _wcp
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ENV_FPS = 10
CHUNK_SIZE = 1000
STATE_DIM = 8       # eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
ACTION_DIM = 7      # eef_pos(3) + eef_axisangle(3) + gripper_qpos[0](1)
VLA_RESIZE = 224    # pi05 expects 224x224 input
DREAMDOJO_H, DREAMDOJO_W = 480, 640  # DreamDojo model resolution

# LIBERO OSC controller constants (matching low-level simulator dynamics)
OSC_ACTION_SCALE = 0.05       # ctrl_config["output_max"][0]
OSC_TRACKING_FACTOR = 0.35    # first-order physical tracking lag


def vla_delta_to_absolute(
    vla_actions: np.ndarray,
    current_state: np.ndarray,
    action_scale: float = OSC_ACTION_SCALE,
    tracking_factor: float = OSC_TRACKING_FACTOR,
) -> np.ndarray:
    """Convert VLA delta actions to absolute state format matching DreamDojo training data.

    VLA outputs: [delta_pos(3), delta_rot(3), gripper_cmd(1)]  range ~ [-1, 1]
    Training data: [abs_pos(3), abs_rot(3), gripper_qpos(1)]   absolute values

    The LIBERO OSC controller applies:
      1. clip(delta, -1, 1) * action_scale  → desired displacement
      2. tracking_factor lag: actual_move = (goal - current) * tracking_factor

    Args:
        vla_actions: (T, 7) raw VLA delta actions
        current_state: (8,) current robot state [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
    Returns:
        (T, 7) absolute actions in training-data convention
    """
    T = len(vla_actions)
    abs_actions = np.zeros((T, 7), dtype=np.float64)

    pos = current_state[:3].copy()
    rot = current_state[3:6].copy()
    gripper = current_state[6]  # first gripper qpos dim

    for i in range(T):
        # Position: clip + scale + tracking lag
        delta_pos = np.clip(vla_actions[i, :3], -1.0, 1.0) * action_scale
        goal_pos = pos + delta_pos
        pos = pos + (goal_pos - pos) * tracking_factor

        # Rotation: same treatment
        delta_rot = np.clip(vla_actions[i, 3:6], -1.0, 1.0) * action_scale
        goal_rot = rot + delta_rot
        rot = rot + (goal_rot - rot) * tracking_factor

        # Gripper: VLA outputs continuous [-1, 1] score;
        # training data stores gripper qpos (small, ~0-0.04).
        # Gripper > 0 → closing (qpos stays near current), < 0 → opening (qpos→0)
        # Use a simple mapping: hold current qpos for positive, decay toward 0 for negative
        if vla_actions[i, 6] > 0:
            gripper = gripper  # maintain (closing)
        else:
            gripper = gripper * 0.9  # slowly open

        abs_actions[i, :3] = pos
        abs_actions[i, 3:6] = rot
        abs_actions[i, 6] = gripper

    return abs_actions

LIBERO_MODALITY = {
    "state": {
        "eef_pos":  {"original_key": "observation.state", "start": 0, "end": 3,
                     "rotation_type": None, "absolute": True, "dtype": "float64", "range": None},
        "eef_rot":  {"original_key": "observation.state", "start": 3, "end": 6,
                     "rotation_type": None, "absolute": True, "dtype": "float64", "range": None},
        "gripper":  {"original_key": "observation.state", "start": 6, "end": 8,
                     "rotation_type": None, "absolute": True, "dtype": "float64", "range": None},
    },
    "action": {
        "arm": {"original_key": "action", "start": 0, "end": 7,
                "rotation_type": None, "absolute": True, "dtype": "float64", "range": None},
    },
    "video": {
        "agentview": {"original_key": "observation.images.agentview"},
        "wrist":     {"original_key": "observation.images.wrist"},
    },
    "annotation": {
        "language.task": {"original_key": "task_index"},
    },
}


@dataclasses.dataclass
class Args:
    # --- VLA policy server ---
    vla_host: str = "0.0.0.0"
    vla_port: int = 8000
    replan_steps: int = 5          # execute this many steps per VLA action chunk

    # --- DreamDojo world model server(s) ---
    dd_base_port: int = 8020       # parallel instances on ports 8020, 8021, ...
    dd_timeout: int = 600          # HTTP timeout for DreamDojo requests (seconds)
    dd_save_dir: str = "/tmp/vlaw_dd_gen"  # DreamDojo saves intermediate .mp4 here

    # --- LIBERO ---
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10       # simulator warm-up before extracting s_0

    # --- Rollout config ---
    max_time_steps: int = 520      # maximum autoregressive steps per trajectory
    num_rollouts_per_init: int = 8 # parallel rollouts from each initial state
    num_inits_per_task: int = 50   # how many initial states to sample per task

    # --- Output ---
    output_dir: str = "data/libero/vlaw_synthetic"

    seed: int = 42


def dreamdojo_generate(
    port: int,
    frame_np: np.ndarray,       # (H, W, 3) uint8 — will be resized to 480x640
    actions: np.ndarray,        # (T, 7) float32 — raw 7-DoF actions
    save_name: str,
    task_description: str = "",
    timeout: int = 600,
    seed: int = 0,
) -> Optional[str]:
    """Call DreamDojo server to generate a future video clip from (frame, actions).

    Returns the path to the saved .mp4 on the server, or None on failure.
    """
    # Resize to DreamDojo's expected resolution
    if frame_np.shape[:2] != (DREAMDOJO_H, DREAMDOJO_W):
        frame_np = cv2.resize(frame_np, (DREAMDOJO_W, DREAMDOJO_H))

    frame_bytes = base64.b64encode(frame_np.tobytes()).decode()
    payload = {
        "frame": frame_bytes,
        "frame_height": DREAMDOJO_H,
        "frame_width": DREAMDOJO_W,
        "actions": actions.tolist(),
        "save_name": save_name,
        "prompt": task_description,
        "seed": seed,
    }
    url = f"http://127.0.0.1:{port}/generate"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["save_path"]
    except Exception as e:
        logging.error(f"[DreamDojo port={port}] generation failed: {e}")
        return None


def decode_dreamdojo_video(video_path: str) -> list[np.ndarray]:
    """Read a DreamDojo-generated .mp4 and return list of (H, W, 3) uint8 frames."""
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = [np.asarray(f) for f in reader]
    reader.close()
    return frames


class SyntheticTrajectoryWriter:
    """Writes VLAW synthetic trajectories in LeRobot format (same schema as DreamDojoWriter)."""

    def __init__(self, output_dir: pathlib.Path, env_fps: int = ENV_FPS):
        self.root = pathlib.Path(output_dir)
        self.env_fps = env_fps

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)

        self.episode_index: int = 0
        self.global_frame_index: int = 0
        self.task_registry: dict[str, int] = {}

        self._ep_agentview: list[np.ndarray] = []
        self._ep_wrist: list[np.ndarray] = []
        self._ep_actions: list[np.ndarray] = []
        self._ep_states: list[np.ndarray] = []
        self._ep_task_index: Optional[int] = None

        self._stat_accum: dict[str, list[np.ndarray]] = {
            "observation.state": [],
            "action": [],
        }
        self._episodes_meta: list[dict] = []

    def begin_episode(self, task_description: str) -> None:
        if task_description not in self.task_registry:
            self.task_registry[task_description] = len(self.task_registry)
        self._ep_task_index = self.task_registry[task_description]
        self._ep_agentview = []
        self._ep_wrist = []
        self._ep_actions = []
        self._ep_states = []

    def record_step(
        self,
        agentview_img: np.ndarray,   # (H, W, 3) uint8
        wrist_img: np.ndarray,       # (H, W, 3) uint8
        action: np.ndarray,          # (7,) float64
        state: np.ndarray,           # (8,) float64
    ) -> None:
        self._ep_agentview.append(agentview_img)
        self._ep_wrist.append(wrist_img)
        self._ep_actions.append(action.astype(np.float64))
        self._ep_states.append(state.astype(np.float64))

    def end_episode(self, success: bool) -> bool:
        n = len(self._ep_states)
        if n == 0:
            return False

        ep_idx = self.episode_index
        chunk_idx = ep_idx // CHUNK_SIZE

        # Write videos
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

        # Write parquet
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
                "synthetic":         True,  # mark as VLAW-synthesized
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
            "synthetic":     True,
        })

        self.global_frame_index += n
        self.episode_index += 1
        return True

    def finalize(self) -> None:
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
            "generation_method": "VLAW_autoregressive_rollout",
            "features": {
                "observation.state": {
                    "shape": [STATE_DIM],
                    "names": ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                              "eef_rot_x", "eef_rot_y", "eef_rot_z",
                              "gripper_0", "gripper_1"],
                    "dtype": "float64",
                },
                "action": {
                    "shape": [ACTION_DIM],
                    "names": ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                              "eef_rot_x", "eef_rot_y", "eef_rot_z",
                              "gripper"],
                    "dtype": "float64",
                },
                "observation.images.agentview": {
                    "shape": [LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3],
                    "names": ["height", "width", "channel"],
                    "dtype": "uint8",
                    "video_info": {"video.fps": float(self.env_fps)},
                },
                "observation.images.wrist": {
                    "shape": [LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3],
                    "names": ["height", "width", "channel"],
                    "dtype": "uint8",
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
            data = np.stack(arrays, axis=0)
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


def extract_initial_state(env, initial_state, num_steps_wait: int) -> dict:
    """Reset the LIBERO env, wait for objects to settle, and extract s_0.

    Returns a dict with:
        agentview_img:  (256, 256, 3) uint8
        wrist_img:      (256, 256, 3) uint8
        eef_pos:        (3,)
        eef_quat:       (4,)
        gripper_qpos:   (2,)
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    # Let objects settle
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    return {
        "agentview_img": agentview,
        "wrist_img":     wrist,
        "eef_pos":       obs["robot0_eef_pos"].copy(),
        "eef_quat":      obs["robot0_eef_quat"].copy(),
        "gripper_qpos":  obs["robot0_gripper_qpos"].copy(),
    }


def vla_predict_action(
    client: _wcp.WebsocketClientPolicy,
    agentview_img: np.ndarray,    # (256, 256, 3) uint8
    wrist_img: np.ndarray,        # (256, 256, 3) uint8
    state: np.ndarray,            # (8,) robot state
    task_description: str,
    resize_size: int = VLA_RESIZE,
) -> np.ndarray:
    """Query the VLA (pi05) policy server to predict an action chunk.

    Returns action_chunk of shape (action_horizon, 7).
    """
    img_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agentview_img, resize_size, resize_size)
    )
    wrist_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )

    element = {
        "observation/image":       img_resized,
        "observation/wrist_image": wrist_resized,
        "observation/state":       state,
        "prompt":                  str(task_description),
    }
    result = client.infer(element)
    return result["actions"]  # (action_horizon, 7)


def autoregressive_rollout(
    rollout_id: int,
    init_state: dict,
    task_description: str,
    vla_client: _wcp.WebsocketClientPolicy,
    dd_port: int,
    max_steps: int,
    replan_steps: int,
    dd_timeout: int,
    dd_save_dir: str,
    seed: int,
    task_id: int = 0,
    init_idx: int = 0,
) -> dict:
    """Execute one full autoregressive rollout in imagination.

    Flow per step:
        1. VLA sees current frame s_t + instruction I -> predicts action chunk a_t
        2. DreamDojo sees (s_t, a_t) -> generates imagined next frames
        3. Take the last generated frame as s_{t+1}
        4. Repeat until max_steps

    Returns:
        {
            "agentview_frames": list of (H, W, 3) uint8,
            "wrist_frames":    list of (H, W, 3) uint8,
            "actions":         list of (7,) float64,
            "states":          list of (8,) float64,
            "num_steps":       int,
        }
    """
    # Initialize from real extracted state
    current_agentview = init_state["agentview_img"].copy()
    current_wrist = init_state["wrist_img"].copy()
    current_eef_pos = init_state["eef_pos"].copy()          # (3,)
    current_eef_quat = init_state["eef_quat"].copy()        # (4,)
    current_gripper_qpos = init_state["gripper_qpos"].copy() # (2,)

    current_state = np.concatenate([
        current_eef_pos,
        _quat2axisangle(current_eef_quat),
        current_gripper_qpos,
    ])  # (8,)

    agentview_frames = [current_agentview.copy()]
    wrist_frames = [current_wrist.copy()]
    actions_list = []
    states_list = [current_state.copy()]

    t = 0

    while t < max_steps:
        # VLA predicts action chunk from current (imagined) observation
        action_chunk = vla_predict_action(
            client=vla_client,
            agentview_img=current_agentview,
            wrist_img=current_wrist,
            state=current_state,
            task_description=task_description,
        )
        # Use replan_steps actions, capped by remaining budget
        n_use = min(replan_steps, len(action_chunk), max_steps - t)
        vla_chunk = np.array(action_chunk, dtype=np.float64)  # full VLA delta chunk

        # Convert VLA delta actions → absolute positions matching DreamDojo
        # training data convention (action == robot state)
        abs_chunk = vla_delta_to_absolute(vla_chunk, current_state)
        chunk_actions = abs_chunk[:n_use]  # (n_use, 7) absolute actions
        actions_for_dd = abs_chunk.astype(np.float32)  # full chunk for DreamDojo

        task_tag = re.sub(r"[^a-zA-Z0-9]+", "_", task_description).strip("_").lower()
        save_name = f"task{task_id}_{task_tag}_init{init_idx}_rollout{rollout_id}/step_{t:04d}"
        video_path = dreamdojo_generate(
            port=dd_port,
            frame_np=current_agentview,
            actions=actions_for_dd,
            save_name=save_name,
            task_description=task_description,
            timeout=dd_timeout,
            seed=seed + t,
        )

        if video_path is None:
            logging.warning(
                f"[Rollout {rollout_id}] DreamDojo failed at step {t}; truncating trajectory."
            )
            break

        # Decode generated video.
        # DreamDojo produces 13 frames for 12 action slots (zero-padded on server).
        # Frame 0 = input frame s_t; frame i = state after actions[0..i-1].
        # We sent n_use real actions, so frames 1..n_use are meaningful next states.
        generated_frames = decode_dreamdojo_video(video_path)
        if not generated_frames or len(generated_frames) < 2:
            logging.warning(
                f"[Rollout {rollout_id}] Empty/short video at step {t}; truncating."
            )
            break

        # Record each action step with its corresponding generated frame
        for i in range(n_use):
            action = chunk_actions[i]
            actions_list.append(np.asarray(action, dtype=np.float64))

            # Frame i+1 = observation after executing action i
            frame_idx = min(i + 1, len(generated_frames) - 1)
            next_frame = generated_frames[frame_idx]

            # Resize back to LIBERO resolution for storage and next VLA input
            if next_frame.shape[:2] != (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION):
                obs_frame = cv2.resize(
                    next_frame, (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)
                )
            else:
                obs_frame = next_frame

            # Update state from absolute action (already converted from VLA deltas)
            action_arr = np.asarray(action, dtype=np.float64)
            current_state = np.concatenate([
                action_arr[:3],                                   # eef_pos  (3,)
                action_arr[3:6],                                  # eef_axisangle (3,)
                np.array([action_arr[6], action_arr[6]]),         # gripper_qpos  (2,)
            ])

            agentview_frames.append(obs_frame.copy())
            wrist_frames.append(current_wrist.copy())
            states_list.append(current_state.copy())

        # Set current observation to the last meaningful generated frame
        # for the next VLA prediction
        current_agentview = agentview_frames[-1].copy()

        t += n_use

    logging.info(
        f"[Rollout {rollout_id}] Completed: {len(actions_list)} steps generated."
    )

    return {
        "agentview_frames": agentview_frames,
        "wrist_frames":     wrist_frames,
        "actions":          actions_list,
        "states":           states_list,
        "num_steps":        len(actions_list),
    }


def parallel_rollouts(
    init_state: dict,
    task_description: str,
    vla_host: str,
    vla_port: int,
    num_rollouts: int,
    dd_base_port: int,
    max_steps: int,
    replan_steps: int,
    dd_timeout: int,
    dd_save_dir: str,
    base_seed: int,
    task_id: int = 0,
    init_idx: int = 0,
) -> list[dict]:
    """Launch N parallel autoregressive rollouts from the same initial state.

    Each rollout uses a different DreamDojo server port and random seed to
    produce diverse trajectories. This is the "Parallel Multi-trajectory
    Rollouts" step from VLAW.

    Each rollout thread creates its own WebSocket client connection to avoid
    thread-safety issues with shared connections.

    Returns a list of rollout result dicts.
    """
    logging.info(
        f"Launching {num_rollouts} parallel rollouts for: '{task_description}'"
    )

    results = [None] * num_rollouts

    def _run_rollout(i: int) -> dict:
        # Each thread gets its own WebSocket client to avoid concurrent recv issues.
        thread_vla_client = _wcp.WebsocketClientPolicy(vla_host, vla_port)
        try:
            return autoregressive_rollout(
                rollout_id=i,
                init_state=init_state,
                task_description=task_description,
                vla_client=thread_vla_client,
                dd_port=dd_base_port + i,
                max_steps=max_steps,
                replan_steps=replan_steps,
                dd_timeout=dd_timeout,
                dd_save_dir=dd_save_dir,
                seed=base_seed + i * 1000,
                task_id=task_id,
                init_idx=init_idx,
            )
        finally:
            # Close the per-thread client connection.
            if hasattr(thread_vla_client, 'close'):
                thread_vla_client.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_rollouts) as executor:
        futures = {}
        for i in range(num_rollouts):
            fut = executor.submit(_run_rollout, i)
            futures[fut] = i

        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                logging.error(f"Rollout {idx} failed with exception: {e}")
                results[idx] = None

    valid = [r for r in results if r is not None and r["num_steps"] > 0]
    logging.info(
        f"Completed {len(valid)}/{num_rollouts} rollouts successfully."
    )
    return valid


def generate_vlaw_data(args: Args) -> None:
    np.random.seed(args.seed)

    # Initialize LIBERO (only for extracting initial frames s_0)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks} tasks)")

    max_steps = args.max_time_steps

    # Verify VLA policy server is reachable (each rollout thread creates its own connection)
    _test_client = _wcp.WebsocketClientPolicy(args.vla_host, args.vla_port)
    logging.info(f"Connected to VLA server at {args.vla_host}:{args.vla_port}")
    if hasattr(_test_client, 'close'):
        _test_client.close()
    del _test_client

    # Initialize dataset writer
    writer = SyntheticTrajectoryWriter(
        pathlib.Path(args.output_dir), env_fps=ENV_FPS
    )

    total_rollouts = 0
    total_written = 0

    for task_id in tqdm(range(num_tasks), desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed
        )
        logging.info(f"\n=== Task {task_id}: {task_description} ===")

        num_inits = min(args.num_inits_per_task, len(initial_states))

        for init_idx in tqdm(range(num_inits), desc="initial states", leave=False):
            logging.info(
                f"Extracting initial state {init_idx} for task '{task_description}'"
            )
            init_state = extract_initial_state(
                env, initial_states[init_idx], args.num_steps_wait
            )

            
            rollout_results = parallel_rollouts(
                init_state=init_state,
                task_description=task_description,
                vla_host=args.vla_host,
                vla_port=args.vla_port,
                num_rollouts=args.num_rollouts_per_init,
                dd_base_port=args.dd_base_port,
                max_steps=max_steps,
                replan_steps=args.replan_steps,
                dd_timeout=args.dd_timeout,
                dd_save_dir=args.dd_save_dir,
                base_seed=args.seed + task_id * 10000 + init_idx * 100,
                task_id=task_id,
                init_idx=init_idx,
            )

            
            for rollout in rollout_results:
                if rollout["num_steps"] == 0:
                    continue

                writer.begin_episode(task_description)

                for step_idx in range(rollout["num_steps"]):
                    action = rollout["actions"][step_idx]
                    state = rollout["states"][step_idx]
                    agentview = rollout["agentview_frames"][step_idx]
                    wrist = rollout["wrist_frames"][step_idx]

                    writer.record_step(
                        agentview_img=agentview,
                        wrist_img=wrist,
                        action=action,
                        state=state,
                    )

                # In pure imagination we don't have ground truth success signal;
                # heuristic: trajectories that reach max_steps are likely failures,
                # shorter ones may indicate natural completion.
                # A VLM judge (e.g. Gemini) could be added here for success filtering.
                is_full_length = (rollout["num_steps"] >= max_steps)
                writer.end_episode(success=not is_full_length)
                total_written += 1

            total_rollouts += len(rollout_results)

            logging.info(
                f"Init {init_idx}: wrote {len(rollout_results)} trajectories | "
                f"Total: {total_written} episodes, {total_rollouts} rollouts"
            )

        env.close()

    writer.finalize()

    logging.info(
        f"\nVLAW data generation complete.\n"
        f"  Total rollouts attempted: {total_rollouts}\n"
        f"  Total episodes written:   {total_written}\n"
        f"  Output directory:         {args.output_dir}"
    )


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> axis-angle (3,)."""
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = tyro.cli(Args)
    generate_vlaw_data(args)
