"""
Convert perturbed_trajs_libero_10 data (LeRobot v2.1 format) to DreamDojo-compatible
LeRobot v2.0 format.

Key transformations:
  - Column renames: state -> observation.state, actions -> action (unused; replaced)
  - Action semantics: perturbed stores raw control deltas; DreamDojo expects absolute
    EE state [eef_pos(3) + eef_axisangle(3) + gripper_qpos[0](1)] derived from state.
  - dtype: float32 -> float64 for state/action
  - Images: extracted from parquet PNG bytes -> mp4 video files
  - Added columns: next.done, success
  - Added meta files: modality.json, stats.json

Usage:
  python convert_perturbed_to_dreamdojo.py \
      --input_dir  /path/to/perturbed_trajs_libero_10 \
      --output_dir /path/to/output
"""

import io
import json
import logging
import pathlib
from typing import Literal

import imageio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tyro
import dataclasses


ENV_FPS = 10
CHUNK_SIZE = 1000
STATE_DIM = 8
ACTION_DIM = 7

LIBERO_ENV_RESOLUTION = 256

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
    input_dir: str = "/root/workspace/fxz/openpi/data/libero/perturbed_trajs_libero_10"
    output_dir: str = "/root/workspace/fxz/openpi/data/libero/perturbed_trajs_libero_10_dreamdojo"


def _decode_png_bytes(img_dict: dict) -> np.ndarray:
    """Decode a {'bytes': ..., 'path': ...} image dict to uint8 array."""
    buf = img_dict["bytes"]
    img = Image.open(io.BytesIO(buf))
    return np.array(img, dtype=np.uint8)


def convert_split(
    input_root: pathlib.Path,
    output_root: pathlib.Path,
    split_name: Literal["perturbed_success", "perturbed_failure"],
    success_flag: bool,
) -> None:
    src = input_root / split_name
    dst = output_root / split_name

    if not src.exists():
        logging.warning(f"Source {src} does not exist, skipping.")
        return

    # Read source tasks
    tasks_jsonl = src / "meta" / "tasks.jsonl"
    task_registry: dict[int, str] = {}
    with open(tasks_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            task_registry[rec["task_index"]] = rec["task"]

    # Read source episodes meta
    episodes_jsonl = src / "meta" / "episodes.jsonl"
    src_episodes = []
    with open(episodes_jsonl) as f:
        for line in f:
            src_episodes.append(json.loads(line))

    total_episodes = len(src_episodes)
    logging.info(f"Converting {split_name}: {total_episodes} episodes")

    # Prepare output dirs
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(parents=True, exist_ok=True)
    (dst / "videos").mkdir(parents=True, exist_ok=True)

    stat_states = []
    stat_actions = []
    out_episodes_meta = []
    global_frame_index = 0

    for ep_meta in tqdm(src_episodes, desc=split_name):
        ep_idx = ep_meta["episode_index"]
        chunk_idx = ep_idx // CHUNK_SIZE

        # Read source parquet
        src_parquet = src / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(src_parquet)
        n = len(df)

        # Extract images and write videos
        agentview_frames = []
        wrist_frames = []
        for _, row in df.iterrows():
            agentview_frames.append(_decode_png_bytes(row["image"]))
            wrist_frames.append(_decode_png_bytes(row["wrist_image"]))

        for cam_key, frames in [
            ("observation.images.agentview", agentview_frames),
            ("observation.images.wrist", wrist_frames),
        ]:
            vid_dir = dst / "videos" / f"chunk-{chunk_idx:03d}" / cam_key
            vid_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_dir / f"episode_{ep_idx:06d}.mp4"
            imageio.mimwrite(
                vid_path,
                frames,
                fps=ENV_FPS,
                codec="libx264",
                quality=8,
            )

        # Build new parquet rows
        rows = []
        for frame_idx in range(n):
            state = df["state"].iloc[frame_idx].astype(np.float64)  # (8,)
            # DreamDojo action = absolute EE state: eef_pos(3) + eef_axisangle(3) + gripper_qpos[0](1)
            action = np.concatenate([state[:6], state[6:7]]).astype(np.float64)  # (7,)

            stat_states.append(state)
            stat_actions.append(action)

            rows.append({
                "observation.state": state,
                "action": action,
                "timestamp": frame_idx / ENV_FPS,
                "frame_index": frame_idx,
                "episode_index": ep_idx,
                "index": global_frame_index + frame_idx,
                "next.done": (frame_idx == n - 1),
                "task_index": int(df["task_index"].iloc[frame_idx]),
                "success": success_flag,
            })

        out_df = pd.DataFrame(rows)
        data_dir = dst / "data" / f"chunk-{chunk_idx:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        out_episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": ep_meta["tasks"],
            "length": n,
            "success": success_flag,
        })

        global_frame_index += n

    info = {
        "codebase_version": "v2.0",
        "robot_type": "single_arm",
        "fps": ENV_FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": CHUNK_SIZE,
        "total_episodes": total_episodes,
        "total_frames": global_frame_index,
        "total_tasks": len(task_registry),
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
                "video_info": {"video.fps": float(ENV_FPS)},
            },
            "observation.images.wrist": {
                "shape": [LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3],
                "names": ["height", "width", "channel"],
                "dtype": "uint8",
                "video_info": {"video.fps": float(ENV_FPS)},
            },
        },
    }
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # modality.json
    with open(dst / "meta" / "modality.json", "w") as f:
        json.dump(LIBERO_MODALITY, f, indent=2)

    # tasks.jsonl
    with open(dst / "meta" / "tasks.jsonl", "w") as f:
        for idx in sorted(task_registry.keys()):
            f.write(json.dumps({"task_index": idx, "task": task_registry[idx]}) + "\n")

    # episodes.jsonl
    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for meta in out_episodes_meta:
            f.write(json.dumps(meta) + "\n")

    # stats.json
    if stat_states:
        states_arr = np.stack(stat_states, axis=0)
        actions_arr = np.stack(stat_actions, axis=0)
        stats = {}
        for col, data in [("observation.state", states_arr), ("action", actions_arr)]:
            stats[col] = {
                "mean": data.mean(0).tolist(),
                "std": data.std(0).tolist(),
                "min": data.min(0).tolist(),
                "max": data.max(0).tolist(),
                "q01": np.quantile(data, 0.01, axis=0).tolist(),
                "q99": np.quantile(data, 0.99, axis=0).tolist(),
            }
        with open(dst / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    logging.info(f"Done {split_name}: {total_episodes} episodes, {global_frame_index} frames")


def main(args: Args) -> None:
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    for split_name, success_flag in [
        ("perturbed_success", True),
        ("perturbed_failure", False),
        ("perturbed_success_resumed", True),
        ("perturbed_failure_resumed", False),
    ]:
        convert_split(input_dir, output_dir, split_name, success_flag)

    logging.info(f"All splits converted to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
