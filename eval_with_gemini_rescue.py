"""
LIBERO evaluation with Gemini-based dense value monitoring and rescue logic.

Normal operation (same as eval_with_gemini.py):
  - Every 4s (40 frames at 10fps), asynchronously query Gemini for a value score.
  - Results are written to gemini_values.txt in each rollout directory.

Rescue logic (triggered at the next replanning step when):
  - Latest Gemini score < 0.20  (disengaged / failure state), OR
  - Latest Gemini score dropped >= 0.20 compared to the score 4s earlier
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
import threading
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import torch
import cv2
import numpy as np
import requests

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


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ROLLOUT_FPS = 10

GEMINI_QUERY_INTERVAL_FRAMES = 40   # 4s at 10fps
GEMINI_HISTORY_FRAMES = 200          # context window sent to Gemini (~20s)
GEMINI_VALUE_MODEL = "gemini-2.5-flash-preview-04-17"
GEMINI_SELECT_MODEL = "gemini-2.5-flash-preview-04-17"

RESCUE_SCORE_ABSOLUTE = 0.20   # trigger if latest score < this
RESCUE_SCORE_DROP = 0.20       # trigger if score dropped >= this vs 4s ago

WAN2_SHARED_PROMPT = """Please examine the robotic arm workspace in the input image. The drawn trajectory represents the future motion of the gripper's center: the green dot is the starting point, and the red dot is the destination.
Please imagine a video showing the complete process of the upper part of the robotic arm (including the gripper) moving from the start to the end (keeping the drawn trajectory stationary). Follow these physical rules:
1. In the final state, the center of the end-effector (gripper) must be perfectly aligned with the red dot. The base of the robotic arm must remain stationary.
2. Only the upper joints and the arm of the robotic arm move to perform tasks, while the base remains stationary.
3. Object Interaction Logic: Observe the relationship between the green dot (start position) and any objects in the scene.
- IF the green dot is currently positioned near an object in a way that implies a grasp or contact, or if the gripper is already holding it: Move that specific object along the trajectory so that it is still being held or manipulated by the gripper at the red dot.
- IF the green dot and the predicted path are in free space, not touching any objects: Execute the movement as pure free-space motion. No objects should be moved.
- Do not change the open/closed state of the gripper unless it is necessary for physical plausibility at the destination (e.g., placing an object).
4. Environment Stability: Keep all other background elements, lighting, camera angle, and non-interacted objects exactly identical to the original scene."""

TRAJ_COLORS = [
    ((235, 206, 135), (0, 215, 255)),
    ((144, 238, 144), (34, 139, 34)),
    ((255, 182, 193), (220, 20, 60)),
    ((173, 216, 230), (0, 0, 139)),
    ((221, 160, 221), (139, 0, 139)),
]


class ValueEvaluation(BaseModel):
    reasoning: str
    score: float
    status: str


class BestIndex(BaseModel):
    best_index: int


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _query_gemini_value(frames: list, task_description: str, step_idx: int,
                        score_history: list, lock: threading.Lock) -> dict:
    """
    Upload a video clip, query Gemini for a value score, append to score_history,
    and return the full result dict.  Called from a worker thread.
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
            f'object has been achieved, but the core task logic has not yet begun.\n'
            f'- 0.60 - 0.80 (Critical Execution State): The core task is being executed smoothly and '
            f'is only one step away from the final goal state.\n'
            f'- 0.80 - 1.00 (Completion State): The task has been successfully accomplished.\n'
            f'Please output strictly in **JSON array format** (without any additional explanatory text). '
            f'Include reasoning, score (rounded to two decimal places) and status. '
            f'**Example Format:** [{{"reasoning": "...", "score": 0.35, "status": "Approach State"}}]'
        )

        response = client.models.generate_content(
            model=GEMINI_VALUE_MODEL,
            contents=[prompt, "\n[Current Video]:", video_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[ValueEvaluation],
                temperature=0.0,
            ),
        )
        result = json.loads(response.text)

        # Record score in shared history (thread-safe)
        if result:
            score = result[0].get("score")
            if score is not None:
                with lock:
                    score_history.append((step_idx, float(score)))
                logging.info(f"[Gemini Value] frame={step_idx} score={score:.2f} status={result[0].get('status')}")

        return {"step": step_idx, "result": result}

    except Exception as e:
        logging.error(f"[Gemini Value] frame={step_idx} error: {e}")
        return {"step": step_idx, "error": str(e)}

    finally:
        if video_file is not None:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _check_rescue_needed(score_history: list, lock: threading.Lock) -> bool:
    """Return True if rescue should be triggered based on the score history."""
    with lock:
        if not score_history:
            return False
        sorted_scores = sorted(score_history, key=lambda x: x[0])

    latest_frame, latest_score = sorted_scores[-1]

    # Condition 1: absolute low score
    if latest_score < RESCUE_SCORE_ABSOLUTE:
        logging.info(f"[Rescue] Triggered: score {latest_score:.2f} < {RESCUE_SCORE_ABSOLUTE}")
        return True

    # Condition 2: dropped >= RESCUE_SCORE_DROP compared to the entry ~4s ago
    prev_score = None
    for frame_idx, score in reversed(sorted_scores[:-1]):
        if latest_frame - frame_idx >= GEMINI_QUERY_INTERVAL_FRAMES:
            prev_score = score
            break
    if prev_score is not None and (latest_score - prev_score) <= -RESCUE_SCORE_DROP:
        logging.info(
            f"[Rescue] Triggered: score dropped {prev_score:.2f} -> {latest_score:.2f} "
            f"(drop={prev_score - latest_score:.2f} >= {RESCUE_SCORE_DROP})"
        )
        return True

    return False


def _wan2_generate(port: int, image_path: str, prompt: str, save_path: str) -> bool:
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "image_path": image_path,
        "prompt": prompt,
        "save_path": save_path,
        "sampling_steps": 30,
    }
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"[Wan2 port={port}] generation failed: {e}")
        return False


def _gemini_select_best(current_video_path: str, candidate_paths: list,
                        task_description: str) -> int:
    """Ask Gemini to select the best candidate video. Returns index (0-based)."""
    client = _get_gemini_client()

    current_file = client.files.upload(file=current_video_path)
    cand_files = [client.files.upload(file=p) for p in candidate_paths]

    all_files = [current_file] + cand_files
    for f in all_files:
        info = client.files.get(name=f.name)
        while info.state.name == "PROCESSING":
            time.sleep(2)
            info = client.files.get(name=f.name)
        if info.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {f.name}")

    prompt = (
        f'You are an evaluation model in a robotic control system.\n'
        f'Based on the [Current Video] and the language command, select the most promising '
        f'candidate next video that best continues the task.\n\n'
        f'Language Command: "{task_description}"\n\n'
        f'Please evaluate the Current Video against the Candidate Next Videos '
        f'(Index 0 to {len(cand_files) - 1}).'
    )
    contents = [prompt, "\n[Current Video]:", current_file]
    for i, cf in enumerate(cand_files):
        contents += [f"\n[Candidate Next Video {i}]:", cf]

    response = client.models.generate_content(
        model=GEMINI_SELECT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BestIndex,
            temperature=0.2,
        ),
    )
    result = json.loads(response.text)

    client.files.delete(name=current_file.name)
    for cf in cand_files:
        client.files.delete(name=cf.name)

    return int(result["best_index"])


def _rescue_select_action(
    obs, img, wrist_img, K, E,
    replay_images_for_history: list,
    task_description: str,
    policy_client,
    replan_steps: int,
    step_save_dir: pathlib.Path,
    action_scale: float,
    resize_size: int,
    num_samples: int = 5,
    wan2_base_port: int = 8010,
) -> list:
    """
    Sample `num_samples` action chunks, generate candidate videos via Wan2.2,
    let Gemini pick the best, and return the chosen action chunk.
    Falls back to chunk[0] on any error.
    """
    step_save_dir.mkdir(parents=True, exist_ok=True)

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
    action_chunks = [policy_client.infer(element)["actions"] for _ in range(num_samples)]

    # Draw per-candidate trajectory images
    plan_image_paths = []
    for i, chunk in enumerate(action_chunks):
        line_c, point_c = TRAJ_COLORS[i % len(TRAJ_COLORS)]
        drawn = _draw_trajectory_on_image(
            img=img,
            current_eef_pos=obs["robot0_eef_pos"],
            action_chunk=chunk[:replan_steps],
            K=K, E=E,
            orig_res=LIBERO_ENV_RESOLUTION,
            target_res=resize_size,
            action_scale=action_scale,
            line_color=line_c,
            point_color=point_c,
        )
        img_path = str(step_save_dir / f"plan_{i}.png")
        cv2.imwrite(img_path, cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
        plan_image_paths.append(img_path)

    # Parallel Wan2.2 video generation
    tasks = [
        {
            "port": wan2_base_port + i,
            "img": plan_image_paths[i],
            "save": str(step_save_dir / f"output_{i}.mp4"),
        }
        for i in range(num_samples)
    ]
    logging.info("[Rescue] Launching parallel Wan2.2 video generation...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as ex:
        futures = [
            ex.submit(_wan2_generate, t["port"], t["img"], WAN2_SHARED_PROMPT, t["save"])
            for t in tasks
        ]
        concurrent.futures.wait(futures)

    candidate_paths = [t["save"] for t in tasks]
    # Filter to only paths that actually exist (in case some Wan2 workers failed)
    valid = [(i, p) for i, p in enumerate(candidate_paths) if os.path.exists(p)]
    if not valid:
        logging.warning("[Rescue] All Wan2.2 generations failed; using chunk 0.")
        return list(action_chunks[0][:replan_steps])

    # Save history video for Gemini context
    current_video_path = str(step_save_dir / "current_actual_video.mp4")
    imageio.mimwrite(
        current_video_path,
        [np.asarray(x) for x in replay_images_for_history],
        fps=ROLLOUT_FPS,
    )

    valid_indices, valid_paths = zip(*valid)
    try:
        raw_best = _gemini_select_best(current_video_path, list(valid_paths), task_description)
        # raw_best is an index into valid_paths; map back to original chunk index
        best_chunk_idx = valid_indices[min(raw_best, len(valid_indices) - 1)]
        logging.info(f"[Rescue] Gemini selected candidate {raw_best} -> chunk {best_chunk_idx}")
    except Exception as e:
        logging.error(f"[Rescue] Gemini selection failed: {e}. Using chunk 0.")
        best_chunk_idx = valid_indices[0]

    return list(action_chunks[best_chunk_idx][:replan_steps])


def _write_gemini_results(gemini_results: list, rollout_dir: pathlib.Path,
                          task_description: str, episode_idx: int, suffix: str):
    out_path = rollout_dir / "gemini_values.txt"
    with open(out_path, "w") as f:
        f.write(f"Task: {task_description}\n")
        f.write(f"Episode: {episode_idx}  Outcome: {suffix}\n")
        f.write("=" * 60 + "\n\n")
        for entry in sorted(gemini_results, key=lambda x: x.get("step", 0)):
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


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20
    video_out_path: str = "data/libero/output"
    seed: int = 7
    num_rescue_samples: int = 5     # action candidates when rescue fires
    wan2_base_port: int = 8010      # Wan2.2 server ports: base, base+1, ..., base+N-1


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
            clean_images = []         # raw frames for video saving
            history_eef_pos = []
            history_K = []
            history_E = []
            history_actions = []

            # Per-episode Gemini value state
            score_history: list = []        # [(frame_idx, score), ...]
            score_lock = threading.Lock()
            gemini_futures: list = []
            gemini_all_results: list = []   # for final txt output
            gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

            rescue_log: list = []  # record which frames triggered rescue

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

                    num_frames = len(clean_images)

                    # ---- Async Gemini value query every 4s ----
                    if num_frames % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                        clip = list(clean_images[-GEMINI_HISTORY_FRAMES:])
                        future = gemini_executor.submit(
                            _query_gemini_value,
                            clip, task_description, num_frames,
                            score_history, score_lock,
                        )
                        gemini_futures.append(future)
                        logging.info(f"[Gemini] Submitted value query at frame {num_frames}")

                    # ---- Replanning ----
                    if not action_plan:
                        rescue = _check_rescue_needed(score_history, score_lock)

                        if rescue:
                            logging.info(f"[Rescue] Activating at frame {num_frames}...")
                            rescue_log.append(num_frames)
                            step_save_dir = (
                                pathlib.Path(args.video_out_path)
                                / "rescue_steps"
                                / f"{task_segment}_ep{episode_idx}_frame{num_frames}"
                            )
                            best_actions = _rescue_select_action(
                                obs=obs,
                                img=img,
                                wrist_img=wrist_img,
                                K=K, E=E,
                                replay_images_for_history=clean_images,
                                task_description=task_description,
                                policy_client=policy_client,
                                replan_steps=args.replan_steps,
                                step_save_dir=step_save_dir,
                                action_scale=action_scale,
                                resize_size=args.resize_size,
                                num_samples=args.num_rescue_samples,
                                wan2_base_port=args.wan2_base_port,
                            )
                            action_plan.extend(best_actions)
                        else:
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
                                f"Policy predicts only {len(action_chunk)} steps, "
                                f"need >= {args.replan_steps}."
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

            # ---- Wait for all in-flight Gemini queries ----
            gemini_executor.shutdown(wait=True)
            for future in gemini_futures:
                try:
                    gemini_all_results.append(future.result())
                except Exception as e:
                    gemini_all_results.append({"error": str(e)})

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            rollout_folder_name = f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            rollout_dir = pathlib.Path(args.video_out_path) / rollout_folder_name
            rollout_dir.mkdir(parents=True, exist_ok=True)

            _write_gemini_results(
                gemini_all_results, rollout_dir, task_description, episode_idx, suffix
            )

            if rescue_log:
                with open(rollout_dir / "gemini_values.txt", "a") as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"Rescue activations ({len(rescue_log)}): frames {rescue_log}\n")

            imageio.mimwrite(
                rollout_dir / "complete_video.mp4",
                [np.asarray(x) for x in clean_images],
                fps=ROLLOUT_FPS,
            )

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
                    for step_action in history_actions[start_step : start_step + args.replan_steps]:
                        clipped = np.clip(step_action[:3], -1.0, 1.0)
                        curr_pos = curr_pos + (curr_pos + clipped * action_scale - curr_pos) * tracking_factor
                        macro_traj_3d.append(curr_pos.copy())

                if len(macro_traj_3d) > 1:
                    imageio.imwrite(
                        rollout_dir / f"trajectory_frame_{clip_idx:03d}.png",
                        np.asarray(_draw_projected_trajectory(
                            img=clean_images[i],
                            traj_3d=macro_traj_3d,
                            K=history_K[i],
                            E=history_E[i],
                            orig_res=LIBERO_ENV_RESOLUTION,
                            target_res=args.resize_size,
                        )),
                    )

                clip_frames = clean_images[i : i + chunk_size]
                if clip_frames:
                    clip_frames_with_traj = [
                        np.asarray(_draw_projected_trajectory(
                            img=frame,
                            traj_3d=macro_traj_3d,
                            K=history_K[i + j],
                            E=history_E[i + j],
                            orig_res=LIBERO_ENV_RESOLUTION,
                            target_res=args.resize_size,
                        ))
                        for j, frame in enumerate(clip_frames)
                    ]
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


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        **{"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    )
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


def _draw_trajectory_on_image(img, current_eef_pos, action_chunk, K, E,
                               orig_res=256, target_res=224, action_scale=0.05,
                               tracking_factor=0.35,
                               line_color=(235, 206, 135), point_color=(0, 215, 255)):
    traj_3d = [current_eef_pos]
    curr_pos = current_eef_pos.copy()
    for step_action in action_chunk:
        delta = np.clip(step_action[:3], -1.0, 1.0) * action_scale
        goal = curr_pos + delta
        curr_pos = curr_pos + (goal - curr_pos) * tracking_factor
        traj_3d.append(curr_pos.copy())
    return _project_and_draw(img, np.vstack(traj_3d), K, E, orig_res, target_res,
                             line_color, point_color)


def _draw_projected_trajectory(img, traj_3d, K, E, orig_res=256, target_res=224):
    return _project_and_draw(img, np.vstack(traj_3d), K, E, orig_res, target_res,
                             (235, 206, 135), (0, 215, 255))


def _project_and_draw(img, traj_3d, K, E, orig_res, target_res, line_color, point_color):
    ones = np.ones((traj_3d.shape[0], 1))
    traj_cam = (np.linalg.inv(E) @ np.hstack([traj_3d, ones]).T).T[:, :3]
    proj = (K @ traj_cam.T).T
    u = (orig_res - 1 - proj[:, 0] / proj[:, 2]) * (target_res / orig_res)
    v = (proj[:, 1] / proj[:, 2]) * (target_res / orig_res)

    img_drawn = img.copy()
    pts = np.vstack((u, v)).T.astype(np.int32)

    def ok(pt):
        return -50 <= pt[0] <= target_res + 50 and -50 <= pt[1] <= target_res + 50

    for i in range(len(pts) - 1):
        p1, p2 = tuple(pts[i]), tuple(pts[i + 1])
        if ok(p1) and ok(p2):
            cv2.line(img_drawn, p1, p2, line_color, 2)
            cv2.circle(img_drawn, p1, 3, point_color, -1)
    if ok(tuple(pts[0])):
        cv2.circle(img_drawn, tuple(pts[0]), 5, (120, 200, 80), -1)
    if ok(tuple(pts[-1])):
        cv2.circle(img_drawn, tuple(pts[-1]), 5, (255, 127, 80), -1)
    return img_drawn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
