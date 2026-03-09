"""
Runs the LIBERO evaluation loop (same as main_1traj_dataset.py) and additionally
queries Gemini every 4 seconds of rollout video for dense value estimation.
Results are written to gemini_values.txt inside each rollout directory.

Usage:
    python eval_with_gemini.py --task_suite_name libero_10
"""

import collections
import concurrent.futures
import dataclasses
import json
import logging
import math
import os
import pathlib
import tempfile
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import torch
import cv2
import numpy as np

_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from pydantic import BaseModel
from google import genai
from google.genai import types
import tqdm
import tyro

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ROLLOUT_FPS = 10

GEMINI_QUERY_INTERVAL_FRAMES = 40   # every 4s at 10 fps
GEMINI_HISTORY_FRAMES = 220          # send up to ~22s of context per query
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

class FrameEvaluation(BaseModel):
    reasoning: str
    score: float
    status: str


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _query_gemini_value(frames: list, task_description: str, step_idx: int) -> dict:
    """
    Write frames to a temp mp4, upload to Gemini, wait for processing,
    run inference, delete the remote file, and return a result dict.
    Called from a worker thread — must be thread-safe.
    """
    client = _get_gemini_client()
    tmp_path = None
    video_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        imageio.mimwrite(tmp_path, [np.asarray(x) for x in frames], fps=ROLLOUT_FPS)

        video_file = client.files.upload(file=tmp_path)

        file_info = client.files.get(name=video_file.name)
        while file_info.state.name == "PROCESSING":
            time.sleep(2)
            file_info = client.files.get(name=video_file.name)
        if file_info.state.name == "FAILED":
            return {"step": step_idx, "error": "Video processing failed"}

        prompt = (
            f'You are a top-tier robot action evaluation expert responsible for constructing a '
            f'Dense Value Function for an RL model. '
            f'The robot is performing the task: "{task_description}". '
            f'Based on the provided video sequence (including the past 17s of history), please '
            f'evaluate the robot\'s state **over the most recent 5s** and provide a **Value Score** '
            f'between **0.00** and **1.00**.\n'
            f'Rigorous Scoring Scale:\n'
            f'- 0.00 - 0.20 (Disengaged/Failure State): The robot is not in contact with the target '
            f'object, is moving in the wrong direction, or has just committed a serious destructive '
            f'error (e.g., knocking something over, dropping an item).\n'
            f'- 0.20 - 0.40 (Approach State): The robot\'s end-effector is moving correctly toward '
            f'the target object and preparing for contact, but stable interaction has not yet occurred.\n'
            f'- 0.40 - 0.60 (Initial Interaction State): Successful contact or grasping of the target '
            f'object has been achieved, but the core task logic has not yet begun '
            f'(e.g., has not yet started moving or placing the object).\n'
            f'- 0.60 - 0.80 (Critical Execution State): The core task is being executed smoothly and '
            f'is only one step away from the final goal state.\n'
            f'- 0.80 - 1.00 (Completion State): The task has been successfully accomplished.\n'
            f'Please output strictly in **JSON array format** (without any additional explanatory text). '
            f'Include reasoning (justification for the score based on the scale), score (final score, '
            f'rounded to two decimal places) and status. '
            f'**Example Format:** [{{"reasoning": "The end-effector is approaching the target but has '
            f'not yet made contact.", "score": 0.35, "status": "Approach State"}}]'
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, "\n[Current Video]:", video_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[FrameEvaluation],
                temperature=0.0,
            ),
        )
        return {"step": step_idx, "result": json.loads(response.text)}

    except Exception as e:
        return {"step": step_idx, "error": str(e)}

    finally:
        if video_file is not None:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _write_gemini_results(gemini_results: list, rollout_dir: pathlib.Path,
                          task_description: str, episode_idx: int, suffix: str):
    out_path = rollout_dir / "gemini_values.txt"
    with open(out_path, "w") as f:
        f.write(f"Task: {task_description}\n")
        f.write(f"Episode: {episode_idx}  Outcome: {suffix}\n")
        f.write("=" * 60 + "\n\n")
        for entry in gemini_results:
            step = entry.get("step", "?")
            ts = f"{step / ROLLOUT_FPS:.1f}s" if isinstance(step, int) else "?"
            f.write(f"[Frame {step} / ~{ts}]\n")
            if "error" in entry:
                f.write(f"  ERROR: {entry['error']}\n")
            else:
                for item in entry.get("result", []):
                    f.write(f"  Status : {item.get('status', '')}\n")
                    f.write(f"  Score  : {item.get('score', '')}\n")
                    f.write(f"  Reason : {item.get('reasoning', '')}\n")
            f.write("\n")
    logging.info(f"[Gemini] Results written to {out_path}")


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_object"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20
    video_out_path: str = "data/libero/output"
    seed: int = 7


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    policy_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            task_segment = task_description.replace(" ", "_")
            existing_dirs = list(
                pathlib.Path(args.video_out_path).glob(f"rollout_{task_segment}_ep{episode_idx}_*")
            )
            if existing_dirs:
                logging.info(f"Skip: {task_segment} (Episode {episode_idx})")
                if "success" in existing_dirs[0].name:
                    task_successes += 1
                    total_successes += 1
                task_episodes += 1
                total_episodes += 1
                continue

            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()

            mujoco_robot = env.env.robots[0]
            ctrl_config = mujoco_robot.controller_config
            if isinstance(ctrl_config, dict) and "output_max" in ctrl_config:
                action_scale = ctrl_config["output_max"][0]
            else:
                action_scale = 0.05

            obs = env.set_init_state(initial_states[episode_idx])

            camera_name = "agentview"
            img_height = LIBERO_ENV_RESOLUTION
            img_width = LIBERO_ENV_RESOLUTION
            mujoco_sim = env.env.sim

            t = 0
            done = False
            clean_images = []
            history_eef_pos = []
            history_K = []
            history_E = []
            history_actions = []

            # Gemini async state
            gemini_futures = []
            gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    K = get_camera_intrinsic_matrix(
                        sim=mujoco_sim,
                        camera_name=camera_name,
                        camera_height=img_height,
                        camera_width=img_width,
                    )
                    E = get_camera_extrinsic_matrix(sim=mujoco_sim, camera_name=camera_name)

                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    clean_images.append(img)
                    history_eef_pos.append(obs["robot0_eef_pos"].copy())
                    history_K.append(K)
                    history_E.append(E)

                    # ---- Gemini query every GEMINI_QUERY_INTERVAL_FRAMES ----
                    num_frames = len(clean_images)
                    if num_frames % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                        clip = list(clean_images[-GEMINI_HISTORY_FRAMES:])
                        future = gemini_executor.submit(
                            _query_gemini_value, clip, task_description, num_frames
                        )
                        gemini_futures.append(future)
                        logging.info(f"[Gemini] Submitted query at frame {num_frames}")

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = policy_client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"Policy only predicts {len(action_chunk)} steps, "
                            f"need at least {args.replan_steps}."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    history_actions.append(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # ---- Collect Gemini results (wait for all in-flight queries) ----
            gemini_executor.shutdown(wait=True)
            gemini_results = []
            for future in gemini_futures:
                try:
                    gemini_results.append(future.result())
                except Exception as e:
                    gemini_results.append({"error": str(e)})

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            rollout_folder_name = f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            rollout_dir = pathlib.Path(args.video_out_path) / rollout_folder_name
            rollout_dir.mkdir(parents=True, exist_ok=True)

            # ---- Write Gemini evaluations ----
            _write_gemini_results(gemini_results, rollout_dir, task_description, episode_idx, suffix)

            # ---- Save full video ----
            imageio.mimwrite(
                rollout_dir / "complete_video.mp4",
                [np.asarray(x) for x in clean_images],
                fps=ROLLOUT_FPS,
            )

            # ---- Trajectory visualisation (unchanged from main_1traj_dataset.py) ----
            num_chunks_to_plot = 4
            chunk_size = args.replan_steps * num_chunks_to_plot
            tracking_factor = 0.35

            for i in range(0, len(clean_images), chunk_size):
                clip_idx = i // chunk_size
                macro_traj_3d = [history_eef_pos[i]]

                for chunk_offset in range(num_chunks_to_plot):
                    start_step = i + chunk_offset * args.replan_steps
                    if start_step >= len(history_actions):
                        break
                    curr_pos = history_eef_pos[start_step].copy()
                    sub_actions = history_actions[start_step : start_step + args.replan_steps]
                    for step_action in sub_actions:
                        clipped_action = np.clip(step_action[:3], -1.0, 1.0)
                        delta_3d = clipped_action * action_scale
                        goal_pos = curr_pos + delta_3d
                        actual_movement = (goal_pos - curr_pos) * tracking_factor
                        curr_pos = curr_pos + actual_movement
                        macro_traj_3d.append(curr_pos)

                if len(macro_traj_3d) > 1:
                    img_with_traj = _draw_projected_trajectory(
                        img=clean_images[i],
                        traj_3d=macro_traj_3d,
                        K=history_K[i],
                        E=history_E[i],
                        orig_res=LIBERO_ENV_RESOLUTION,
                        target_res=args.resize_size,
                    )
                    imageio.imwrite(
                        rollout_dir / f"trajectory_frame_{clip_idx:03d}.png",
                        np.asarray(img_with_traj),
                    )

                clip_frames = clean_images[i : i + chunk_size]
                if clip_frames:
                    clip_frames_with_traj = []
                    for j, frame in enumerate(clip_frames):
                        frame_idx = i + j
                        drawn_frame = _draw_projected_trajectory(
                            img=frame,
                            traj_3d=macro_traj_3d,
                            K=history_K[frame_idx],
                            E=history_E[frame_idx],
                            orig_res=LIBERO_ENV_RESOLUTION,
                            target_res=args.resize_size,
                        )
                        clip_frames_with_traj.append(np.asarray(drawn_frame))
                    imageio.mimwrite(
                        rollout_dir / f"follow_up_clip_{clip_idx:03d}.mp4",
                        clip_frames_with_traj,
                        fps=ROLLOUT_FPS,
                    )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


# ---------------------------------------------------------------------------
# Helpers (copied from main_1traj_dataset.py to keep this file self-contained)
# ---------------------------------------------------------------------------

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _draw_projected_trajectory(img, traj_3d, K, E, orig_res=256, target_res=224):
    traj_3d = np.vstack(traj_3d)
    ones = np.ones((traj_3d.shape[0], 1))
    traj_3d_homo = np.hstack([traj_3d, ones])
    E_inv = np.linalg.inv(E)
    traj_cam_homo = (E_inv @ traj_3d_homo.T).T
    traj_cam = traj_cam_homo[:, :3]
    traj_2d_homo = (K @ traj_cam.T).T
    u = traj_2d_homo[:, 0] / traj_2d_homo[:, 2]
    v = traj_2d_homo[:, 1] / traj_2d_homo[:, 2]
    u = orig_res - 1 - u
    scale = target_res / orig_res
    u = u * scale
    v = v * scale
    img_drawn = img.copy()
    points_2d = np.vstack((u, v)).T.astype(np.int32)

    def in_bounds(pt):
        return -50 <= pt[0] <= target_res + 50 and -50 <= pt[1] <= target_res + 50

    for i in range(len(points_2d) - 1):
        pt1, pt2 = tuple(points_2d[i]), tuple(points_2d[i + 1])
        if in_bounds(pt1) and in_bounds(pt2):
            cv2.line(img_drawn, pt1, pt2, (235, 206, 135), 2)
            cv2.circle(img_drawn, pt1, 3, (0, 215, 255), -1)
    if in_bounds(tuple(points_2d[0])):
        cv2.circle(img_drawn, tuple(points_2d[0]), 5, (120, 200, 80), -1)
    if in_bounds(tuple(points_2d[-1])):
        cv2.circle(img_drawn, tuple(points_2d[-1]), 5, (255, 127, 80), -1)
    return img_drawn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
