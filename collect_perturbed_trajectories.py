"""
Collect perturbed trajectories and save as LeRobot 2.0 datasets.

Based on multimodal_ambiguity_4traj.py — runs the same perturbation pipeline
but records full (obs, action, state) trajectories and saves them into two
separate LeRobot datasets:
  - <output_dir>/success/   — episodes where the task was completed
  - <output_dir>/failure/   — episodes where the task was not completed

Usage:
    # Start the SDE server:
    python scripts/serve_sde_policy.py --env libero --noise_level 1.0 --num_steps 3

    # Collect trajectories:
    python third_party/libero/collect_perturbed_trajectories.py \
        --task_suite_name libero_10 \
        --perturbations scene_swap obstacle occlusion object_swap_target \
        --output_dir data/libero/perturbed_trajs \
        --num_trials_per_task 20
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import json
import logging
import math
import pathlib
import shutil
from typing import Any

import cv2
import imageio
import numpy as np
import torch

_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wsc
import tqdm
import tyro

# Re-use perturbation classes from the evaluation script
from multimodal_ambiguity_4traj import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    ObstaclePerturbation,
    OcclusionPerturbation,
    ObjectSwapPerturbation,
    PerturbationType,
    SceneSwapPerturbation,
    LanguagePerturbationType,
    perturb_language,
    _quat2axisangle,
    _MAX_STEPS,
    draw_obstacles_on_image,
    draw_trajectory_on_image,
    TRAJ_COLORS,
)
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20
    replan_steps: int = 5

    perturbations: list[str] = dataclasses.field(
        default_factory=lambda: ["none"],
    )

    output_dir: str = "data/libero/perturbed_trajs"
    repo_name_success: str = "perturbed_success"
    repo_name_failure: str = "perturbed_failure"

    video_out_path: str = "data/libero/perturbed_videos"
    num_samples: int = 4  # number of trajectory samples for annotated video

    fps: int = 10
    seed: int = 7
    resume: bool = False


LEROBOT_FEATURES = {
    "image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],
    },
}


def _get_libero_env(task, resolution, seed):
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
    return env, task.language


def _get_or_create_dataset(
    output_dir: pathlib.Path, repo_name: str, fps: int, resume: bool = False,
) -> LeRobotDataset:
    """Create a fresh LeRobot dataset, or a new session dataset when *resume* is True.

    When resuming, the old dataset is kept untouched and a new dataset is created
    under ``<repo_name>_resumed`` so that collection can proceed without loading
    the (potentially large) existing dataset.  A final merge step combines them.
    """
    import lerobot.common.datasets.lerobot_dataset as _ld

    full_path = output_dir / repo_name
    original_home = _ld.HF_LEROBOT_HOME

    if resume and full_path.exists():
        # Create a fresh session dataset alongside the existing one.
        session_name = f"{repo_name}_resumed"
        session_path = output_dir / session_name
        if session_path.exists():
            shutil.rmtree(session_path)
        logging.info(
            "Resume mode: existing dataset at %s kept intact. "
            "New episodes -> %s (will be merged at the end).",
            full_path, session_path,
        )
        _ld.HF_LEROBOT_HOME = output_dir
        try:
            dataset = LeRobotDataset.create(
                repo_id=session_name,
                robot_type="panda",
                fps=fps,
                features=LEROBOT_FEATURES,
                image_writer_threads=4,
                image_writer_processes=2,
            )
        finally:
            _ld.HF_LEROBOT_HOME = original_home
        return dataset

    # Fresh start — remove stale data if any.
    if full_path.exists():
        shutil.rmtree(full_path)

    _ld.HF_LEROBOT_HOME = output_dir
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type="panda",
            fps=fps,
            features=LEROBOT_FEATURES,
            image_writer_threads=4,
            image_writer_processes=2,
        )
    finally:
        _ld.HF_LEROBOT_HOME = original_home

    return dataset


def _load_completed_episodes(meta_path: pathlib.Path) -> set[tuple[int, int]]:
    """Return set of (task_id, episode_idx) already collected, from metadata JSON."""
    if not meta_path.exists():
        return set()
    try:
        with open(meta_path) as f:
            data = json.load(f)
        return {
            (ep["task_id"], ep["episode_idx"])
            for ep in data.get("episodes", [])
        }
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning("Could not parse existing metadata (%s), starting fresh.", e)
        return set()


def _save_metadata(
    meta_path: pathlib.Path,
    args: Args,
    active_perturbations,
    metadata_log: list[dict],
    total_episodes: int,
    total_successes: int,
    total_failures: int,
) -> None:
    """Incrementally save collection metadata to JSON."""
    with open(meta_path, "w") as f:
        json.dump(
            {
                "task_suite": args.task_suite_name,
                "perturbations": [p.value for p in active_perturbations],
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "success_rate": total_successes / max(total_episodes, 1),
                "episodes": metadata_log,
            },
            f,
            indent=2,
            default=str,
        )


def collect(args: Args) -> None:
    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    # ---- Parse perturbations (same logic as multimodal_ambiguity_4traj) ----
    active_perturbations = set()
    for p in args.perturbations:
        try:
            active_perturbations.add(PerturbationType(p))
        except ValueError:
            raise ValueError(
                f"Unknown perturbation '{p}'. Choose from: "
                + ", ".join(pt.value for pt in PerturbationType)
            )
    if PerturbationType.NONE in active_perturbations:
        active_perturbations = set()

    logging.info("Active perturbations: %s", [p.value for p in active_perturbations])

    scene_swap = SceneSwapPerturbation(rng) if PerturbationType.SCENE_SWAP in active_perturbations else None
    obstacle = ObstaclePerturbation(rng) if PerturbationType.OBSTACLE in active_perturbations else None
    occlusion = OcclusionPerturbation(rng) if PerturbationType.OCCLUSION in active_perturbations else None
    object_swap_target = (
        ObjectSwapPerturbation(rng, involve_target=True)
        if PerturbationType.OBJECT_SWAP_TARGET in active_perturbations else None
    )
    object_swap_nontarget = (
        ObjectSwapPerturbation(rng, involve_target=False)
        if PerturbationType.OBJECT_SWAP_NONTARGET in active_perturbations else None
    )
    lang_perturbation_types = []
    if PerturbationType.MISIDENTIFY in active_perturbations:
        lang_perturbation_types.append(LanguagePerturbationType.MISIDENTIFY)
    if PerturbationType.AMBIGUOUS in active_perturbations:
        lang_perturbation_types.append(LanguagePerturbationType.AMBIGUOUS)
    if PerturbationType.UNFAMILIAR in active_perturbations:
        lang_perturbation_types.append(LanguagePerturbationType.UNFAMILIAR)

    # ---- Prepare LeRobot datasets ----
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_success = _get_or_create_dataset(output_dir, args.repo_name_success, args.fps, args.resume)
    ds_failure = _get_or_create_dataset(output_dir, args.repo_name_failure, args.fps, args.resume)

    # ---- LIBERO benchmark setup ----
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = _MAX_STEPS[args.task_suite_name]

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _wsc.WebsocketClientPolicy(args.host, args.port)

    # ---- Resume: load previously completed episodes ----
    meta_path = output_dir / "collection_metadata.json"
    completed_episodes: set[tuple[int, int]] = set()
    metadata_log: list[dict] = []
    total_episodes, total_successes, total_failures = 0, 0, 0

    if args.resume:
        completed_episodes = _load_completed_episodes(meta_path)
        if completed_episodes:
            # Restore metadata_log and counters from existing file
            try:
                with open(meta_path) as f:
                    prev = json.load(f)
                metadata_log = prev.get("episodes", [])
                total_episodes = prev.get("total_episodes", 0)
                total_successes = prev.get("total_successes", 0)
                total_failures = prev.get("total_failures", 0)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
            logging.info(
                "Resuming: %d episodes already done (%d success, %d failure). "
                "Skipping completed (task_id, episode_idx) pairs.",
                total_episodes, total_successes, total_failures,
            )

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc="Episodes", leave=False
        ):
            # Skip already-collected episodes when resuming
            if (task_id, episode_idx) in completed_episodes:
                # Advance RNG to keep deterministic state consistent
                rng.randint(1)
                continue

            episode_meta: dict[str, Any] = {
                "task_id": task_id,
                "episode_idx": episode_idx,
                "original_instruction": task_description,
                "perturbations_applied": [],
            }

            if obstacle is not None:
                obstacle.deactivate(env)

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Controller action scale
            mujoco_robot = env.env.robots[0]
            ctrl_config = mujoco_robot.controller_config
            if isinstance(ctrl_config, dict) and "output_max" in ctrl_config:
                action_scale = ctrl_config["output_max"][0]
            else:
                action_scale = 0.05

            obstacle_info: list[dict] = []

            # ---- Apply perturbations ----
            vision_perturbations = []
            if scene_swap is not None:
                vision_perturbations.append(("scene_swap", scene_swap))
            if obstacle is not None:
                vision_perturbations.append(("obstacle", obstacle))
            if occlusion is not None:
                vision_perturbations.append(("occlusion", occlusion))
            if object_swap_target is not None:
                vision_perturbations.append(("object_swap_target", object_swap_target))
            if object_swap_nontarget is not None:
                vision_perturbations.append(("object_swap_nontarget", object_swap_nontarget))

            if vision_perturbations:
                name, perturber = vision_perturbations[rng.randint(len(vision_perturbations))]
                if name == "obstacle":
                    meta = perturber.apply(env, num_obstacles=rng.randint(1, 3))
                    if meta["applied"]:
                        obstacle_info = meta["obstacles"]
                elif name in ("object_swap_target", "object_swap_nontarget"):
                    meta = perturber.apply(env, task_description=task_description)
                else:
                    meta = perturber.apply(env)
                episode_meta["perturbations_applied"].append(meta)

            # Re-render after perturbations
            sim = env.env.sim if hasattr(env, "env") else env.sim
            sim.forward()
            env._update_observables(force=True)
            obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()

            # Language perturbation
            prompt = task_description
            if lang_perturbation_types:
                lp_type = lang_perturbation_types[rng.randint(len(lang_perturbation_types))]
                prompt, lang_meta = perturb_language(task_description, lp_type, rng)
                episode_meta["perturbations_applied"].append(lang_meta)
            episode_meta["prompt_used"] = prompt

            # ---- Collect trajectory ----
            trajectory_frames: list[dict] = []  # buffered frames for LeRobot (clean)
            video_frames: list[np.ndarray] = []  # annotated frames for captioned video
            action_plan = collections.deque()
            t = 0
            done = False

            mujoco_sim = env.env.sim
            camera_name = "agentview"

            while t < max_steps + args.num_steps_wait:
                try:
                    # Camera matrices for trajectory projection
                    K = get_camera_intrinsic_matrix(
                        sim=mujoco_sim, camera_name=camera_name,
                        camera_height=LIBERO_ENV_RESOLUTION,
                        camera_width=LIBERO_ENV_RESOLUTION,
                    )
                    E = get_camera_extrinsic_matrix(
                        sim=mujoco_sim, camera_name=camera_name,
                    )

                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Raw images at env resolution (256×256) — for LeRobot dataset
                    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Resized images for the policy
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img_raw, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_raw, args.resize_size, args.resize_size)
                    )

                    # Draw obstacle overlays on the video image (not on LeRobot data)
                    img_viz = img.copy()
                    if obstacle_info:
                        img_viz = draw_obstacles_on_image(
                            img_viz, obstacle_info, K, E,
                            orig_res=LIBERO_ENV_RESOLUTION,
                            target_res=args.resize_size,
                        )
                    video_frames.append(img_viz.copy())

                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )).astype(np.float32)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": prompt,
                        }

                        # Sample multiple action chunks for annotated video
                        action_chunks = [
                            client.infer(element)["actions"]
                            for _ in range(args.num_samples)
                        ]

                        # Draw all trajectories on the video frame
                        img_multi = img_viz.copy()
                        for i, chunk in enumerate(action_chunks):
                            line_c, point_c = TRAJ_COLORS[i % len(TRAJ_COLORS)]
                            img_multi = draw_trajectory_on_image(
                                img=img_multi,
                                current_eef_pos=obs["robot0_eef_pos"],
                                action_chunk=chunk[:args.replan_steps],
                                K=K, E=E,
                                orig_res=LIBERO_ENV_RESOLUTION,
                                target_res=args.resize_size,
                                action_scale=action_scale,
                                line_color=line_c,
                                point_color=point_c,
                            )
                        video_frames[-1] = img_multi

                        # Execute the first sampled trajectory
                        action_plan.extend(action_chunks[0][:args.replan_steps])

                    action = action_plan.popleft()

                    # Block if colliding with obstacles
                    if obstacle_info:
                        eef_pos = obs["robot0_eef_pos"]
                        if ObstaclePerturbation.check_collision(eef_pos, obstacle_info):
                            action = np.array(LIBERO_DUMMY_ACTION)

                    # Save frame to buffer (raw images for LeRobot, no captions)
                    trajectory_frames.append({
                        "image": img_raw,
                        "wrist_image": wrist_raw,
                        "state": state,
                        "actions": np.array(action[:7], dtype=np.float32),
                        "task": prompt,
                    })

                    obs, reward, done, info = env.step(
                        action.tolist() if hasattr(action, "tolist") else action
                    )
                    if done:
                        break
                    t += 1

                except Exception as e:
                    logging.error("Exception at step %d: %s", t, e)
                    break

            # ---- Write episode to the appropriate LeRobot dataset (clean, no captions) ----
            success = bool(done)
            ds = ds_success if success else ds_failure

            for frame in trajectory_frames:
                ds.add_frame(frame)
            ds.save_episode()

            episode_meta["success"] = success
            episode_meta["steps"] = t
            episode_meta["num_frames"] = len(trajectory_frames)
            metadata_log.append(episode_meta)

            # ---- Save annotated video with captions (separate from LeRobot data) ----
            if video_frames:
                suffix = "success" if success else "failure"
                task_segment = task_description.replace(" ", "_")
                perturb_tag = "_".join(
                    m.get("perturbation", m.get("type", "unknown"))
                    for m in episode_meta["perturbations_applied"]
                ) or "baseline"

                # Build perturbation detail string for overlay
                perturb_details: list[str] = []
                for m in episode_meta["perturbations_applied"]:
                    ptype = m.get("perturbation", m.get("type", "unknown"))
                    if not m.get("applied", True):
                        perturb_details.append(f"{ptype} (not applied)")
                    elif ptype == "scene_swap":
                        perturb_details.append(f"scene_swap: preset={m.get('preset', '?')}")
                    elif ptype == "obstacle":
                        obs_names = [o["name"] for o in m.get("obstacles", [])]
                        perturb_details.append(f"obstacle: {obs_names}")
                    elif ptype == "occlusion":
                        perturb_details.append(f"occlusion: dir={m.get('direction', '?')}")
                    elif ptype in ("object_swap_target", "object_swap_nontarget"):
                        swapped = m.get("swapped", [])
                        label = "target" if m.get("involve_target") else "non-target"
                        perturb_details.append(
                            f"obj_swap({label}): {swapped[0]} <-> {swapped[1]}"
                            if len(swapped) == 2 else f"obj_swap({label})"
                        )
                    elif ptype in ("misidentify", "ambiguous", "unfamiliar"):
                        perturb_details.append(f'{ptype}: "{m.get("perturbed", "")}"')
                    else:
                        perturb_details.append(ptype)
                perturb_detail_str = " | ".join(perturb_details) if perturb_details else "baseline"

                video_size = 512
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.38
                line_height = 14
                margin_x = 4

                def _wrap_text(text, max_width):
                    words = text.split()
                    lines, current = [], ""
                    for word in words:
                        test = f"{current} {word}".strip()
                        if cv2.getTextSize(test, font, font_scale, 1)[0][0] > max_width and current:
                            lines.append(current)
                            current = word
                        else:
                            current = test
                    if current:
                        lines.append(current)
                    return lines

                annotated = []
                for frame in video_frames:
                    frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                    frame_bgr = cv2.resize(frame_bgr, (video_size, video_size),
                                           interpolation=cv2.INTER_LANCZOS4)

                    text_lines = []
                    for ln in _wrap_text(f"Task: {task_description}", video_size - 2 * margin_x):
                        text_lines.append((ln, (255, 255, 255)))
                    if prompt != task_description:
                        for ln in _wrap_text(f"Prompt: {prompt}", video_size - 2 * margin_x):
                            text_lines.append((ln, (180, 220, 255)))
                    for ln in _wrap_text(f"Perturb: {perturb_detail_str}", video_size - 2 * margin_x):
                        text_lines.append((ln, (200, 200, 200)))

                    bar_height = line_height * len(text_lines) + 4
                    overlay = frame_bgr.copy()
                    cv2.rectangle(overlay, (0, 0), (video_size, bar_height), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
                    for i, (text, color) in enumerate(text_lines):
                        cv2.putText(frame_bgr, text, (margin_x, line_height * (i + 1)),
                                    font, font_scale, color, 1, cv2.LINE_AA)
                    annotated.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"{task_segment}_ep{episode_idx}_{perturb_tag}_{suffix}.mp4",
                    annotated, fps=args.fps,
                )

            total_episodes += 1
            if success:
                total_successes += 1
            else:
                total_failures += 1

            logging.info(
                "Episode done. success=%s | Total: %d success, %d failure / %d episodes (%.1f%%)",
                success, total_successes, total_failures, total_episodes,
                total_successes / total_episodes * 100,
            )

            # Incrementally save metadata so we can resume from here
            _save_metadata(
                meta_path, args, active_perturbations,
                metadata_log, total_episodes, total_successes, total_failures,
            )

        env.close()

    # ---- Consolidate datasets ----
    # NOTE: consolidate() was removed in newer lerobot versions.
    # Episodes are already persisted via save_episode() during collection.

    # ---- Merge resumed session datasets back into the originals ----
    if args.resume:
        for repo_name in (args.repo_name_success, args.repo_name_failure):
            session_dir = output_dir / f"{repo_name}_resumed"
            original_dir = output_dir / repo_name
            if not session_dir.exists():
                continue
            # Copy parquet data chunks from session into original
            session_data = session_dir / "data"
            original_data = original_dir / "data"
            if session_data.exists():
                for chunk_dir in sorted(session_data.iterdir()):
                    if not chunk_dir.is_dir():
                        continue
                    # Find the next available chunk index in original
                    existing_chunks = sorted(original_data.iterdir()) if original_data.exists() else []
                    # Merge parquet files with offset episode numbering
                    existing_episodes = sorted(original_data.rglob("*.parquet")) if original_data.exists() else []
                    next_ep = len(existing_episodes)
                    for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                        # Determine target chunk (put all resumed in the last chunk)
                        target_chunk = existing_chunks[-1] if existing_chunks else original_data / "chunk-000"
                        target_chunk.mkdir(parents=True, exist_ok=True)
                        dest = target_chunk / f"episode_{next_ep:06d}.parquet"
                        shutil.copy2(parquet_file, dest)
                        next_ep += 1
            # Copy images
            session_images = session_dir / "images"
            original_images = original_dir / "images"
            if session_images.exists():
                for src_file in session_images.rglob("*"):
                    if src_file.is_file():
                        rel = src_file.relative_to(session_images)
                        dst = original_images / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst)
            # Clean up session directory
            shutil.rmtree(session_dir)
            logging.info("Merged resumed session into %s", original_dir)

    # Final metadata save
    _save_metadata(
        meta_path, args, active_perturbations,
        metadata_log, total_episodes, total_successes, total_failures,
    )

    logging.info(
        "Done! success=%d  failure=%d  rate=%.1f%%",
        total_successes, total_failures,
        total_successes / max(total_episodes, 1) * 100,
    )
    logging.info("Success dataset: %s", output_dir / args.repo_name_success)
    logging.info("Failure dataset: %s", output_dir / args.repo_name_failure)
    logging.info("Metadata: %s", meta_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    collect(args)
