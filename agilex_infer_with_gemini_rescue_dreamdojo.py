"""
Agilex robot inference with Gemini-based dense value evaluation and DreamDojo rescue.

Adapts eval_with_gemini_rescue_dreamdojo.py (LIBERO simulation) to the agilex real robot
infrastructure (ROS-based observation/action via agilex_infer.py).

When the Gemini value score drops below a threshold or drops sharply, the rescue mechanism:
1. Samples multiple action chunks from the policy
2. Sends each to a DreamDojo server to generate candidate future videos
3. Asks Gemini to pick the best candidate
4. Executes the chosen action chunk
"""

import argparse
import base64
import collections
import concurrent.futures
import json
import logging
import os
import pathlib
import random
import shutil
import signal
import sys
import tempfile
import termios
import threading
import time
import tty

import imageio
import numpy as np
import requests
import rospy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import check_keyboard_input, get_config, handle_interactive_mode, process_action
from rosoperator import RosOperator, get_ros_observation
from rotation import abs_6d_2_abs_euler, quat_2_euler

from google import genai
from google.genai import types
from pydantic import BaseModel


ROLLOUT_FPS = 10

GEMINI_QUERY_INTERVAL_FRAMES = 40
GEMINI_HISTORY_FRAMES = 200
GEMINI_VALUE_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_SELECT_MODEL = "gemini-3.1-flash-lite-preview"

RESCUE_SCORE_ABSOLUTE = 0.30
RESCUE_SCORE_DROP = 0.20

observation_window = None
observation_window_lock = threading.Lock()

shutdown_event = threading.Event()
inference_paused = threading.Event()
inference_paused.clear()

inference_stamp = 0


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


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def reset_observation_window():
    global observation_window
    with observation_window_lock:
        observation_window = None


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = collections.deque(maxlen=2)
            observation_window.append(
                {
                    "qpos": None,
                    "images": {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
                    "endpose": None,
                }
            )

    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, endpose_left, endpose_right = (
        get_ros_observation(args, ros_operator)
    )

    qpos = np.concatenate(
        (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)),
        axis=0,
    )

    left_pos = endpose_left.pose.position
    left_rpy = quat_2_euler(endpose_left.pose.orientation)
    left_gripper = puppet_arm_left.position[-1]
    endpose_left_arr = np.array([left_pos.x, left_pos.y, left_pos.z, left_rpy[0], left_rpy[1], left_rpy[2], left_gripper])

    right_pos = endpose_right.pose.position
    right_rpy = quat_2_euler(endpose_right.pose.orientation)
    right_gripper = puppet_arm_right.position[-1]
    endpose_right_arr = np.array(
        [right_pos.x, right_pos.y, right_pos.z, right_rpy[0], right_rpy[1], right_rpy[2], right_gripper]
    )

    endpose = np.concatenate((endpose_left_arr, endpose_right_arr), axis=0)

    with observation_window_lock:
        observation_window.append(
            {
                "qpos": qpos,
                "images": {
                    config["camera_names"][0]: img_front,
                    config["camera_names"][1]: img_right,
                    config["camera_names"][2]: img_left,
                },
                "endpose": endpose,
            }
        )


class StreamActionBuffer:

    def __init__(self, delay, exec_horizon, state_dim):
        self.delay = delay
        self.exec_horizon = exec_horizon
        self.lock = threading.Lock()

        self.cur_chunk = np.zeros((exec_horizon, state_dim))
        self.next_chunk = np.zeros((exec_horizon, state_dim))
        self.cur_len = 0
        self.next_len = 0

        self.cur_step = 0
        self.cur_stamp = 0

    def reset(self):
        with self.lock:
            self.cur_len = 0
            self.next_len = 0
            self.cur_step = 0
            self.cur_stamp = 0

    def should_launch_inference(self):
        if self.exec_horizon > self.delay:
            return self.cur_step == (self.exec_horizon - self.delay - 1)
        else:
            return self.cur_step == 0

    def integrate_first_chunk(self, actions_chunk: np.ndarray):
        with self.lock:
            assert self.cur_len == 0
            assert self.cur_stamp == 0
            assert actions_chunk.shape[0] == self.exec_horizon
            self.cur_chunk[: actions_chunk.shape[0]] = actions_chunk
            self.cur_len = self.exec_horizon

    def integrate_new_chunk(self, actions_chunk: np.ndarray):
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or empty when integrating new chunk")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            assert L == self.exec_horizon
            if self.cur_len == 0:
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                self.cur_chunk[:L] = actions_chunk
                self.cur_len = L
            else:
                self.next_chunk[:L] = actions_chunk
                self.next_len = L

    def integrate_new_chunk_streaming(self, actions_chunk: np.ndarray, stamp: int):
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or empty when integrating new chunk")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            if self.cur_len == 0:
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                safe_L = min(L, self.exec_horizon)
                self.cur_chunk[:safe_L] = actions_chunk[:safe_L]
                self.cur_len = safe_L
            else:
                if self.cur_stamp == stamp:
                    remaining_len = self.exec_horizon - self.cur_len
                    if remaining_len > 0:
                        safe_L = min(L, remaining_len)
                        self.cur_chunk[self.cur_len : self.cur_len + safe_L] = actions_chunk[:safe_L]
                        self.cur_len += safe_L
                else:
                    remaining_len = self.exec_horizon - self.next_len
                    if remaining_len > 0:
                        safe_L = min(L, remaining_len)
                        self.next_chunk[self.next_len : self.next_len + safe_L] = actions_chunk[:safe_L]
                        self.next_len += safe_L

    def integrate_rescue_chunk(self, actions: list):
        """Replace current action plan with rescue-selected actions."""
        with self.lock:
            L = min(len(actions), self.exec_horizon)
            arr = np.array(actions[:L])
            self.cur_chunk[:L] = arr
            self.cur_len = L
            self.cur_step = 0
            self.next_len = 0

    def get_next_action(self):
        with self.lock:
            if self.cur_step >= self.cur_len:
                return None

            action = self.cur_chunk[self.cur_step]
            self.cur_step += 1

            if self.cur_step == self.exec_horizon:
                self.cur_chunk, self.next_chunk = self.next_chunk, self.cur_chunk
                self.cur_len = self.next_len
                self.next_len = 0
                self.cur_step = 0
                self.cur_stamp += 1

            return action


def _dreamdojo_generate(port: int, frame_np: np.ndarray, actions: np.ndarray,
                        save_name: str, task_description: str = "",
                        seed: int = 0) -> str | None:
    url = f"http://127.0.0.1:{port}/generate"

    h, w = frame_np.shape[:2]
    frame_bytes = base64.b64encode(frame_np.tobytes()).decode()

    payload = {
        "frame": frame_bytes,
        "frame_height": h,
        "frame_width": w,
        "actions": actions.tolist(),
        "save_name": save_name,
        "prompt": task_description,
        "seed": seed,
    }
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["save_path"]
    except Exception as e:
        logging.error(f"[DreamDojo port={port}] generation failed: {e}")
        return None


def _query_gemini_value(frames: list, task_description: str, step_idx: int,
                        score_history: list, lock: threading.Lock) -> dict:
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
            f'Based on the provided video sequence (including the past history), please '
            f'evaluate the robot\'s state **over the most recent 2s** and provide a **Value Score** '
            f'between **0.00** and **1.00**.\n'
            f'IMPORTANT: Focus on the **final frames** of the video to judge the current state. '
            f'Do NOT give a high score just because the robot appeared to be on the right track earlier.\n'
            f'Rigorous Scoring Scale:\n'
            f'- 0.00 - 0.20 (Disengaged/Failure State): The robot is not in contact with the target '
            f'object, is moving in the wrong direction, has knocked the object away, or the object '
            f'has slipped out of the gripper.\n'
            f'- 0.20 - 0.40 (Approach State): The robot\'s end-effector is moving correctly toward '
            f'the target object, but has not yet made contact.\n'
            f'- 0.40 - 0.60 (Initial Interaction State): The gripper is touching or closing on the '
            f'object, but the object is NOT yet securely grasped or lifted.\n'
            f'- 0.60 - 0.80 (Critical Execution State): The object is securely grasped and being '
            f'lifted, but has not yet reached the goal height or position.\n'
            f'- 0.80 - 1.00 (Completion State): The task goal is fully achieved — for pick tasks, '
            f'the object is clearly lifted off the surface and stably held in the gripper.\n'
            f'Common failure patterns to watch for:\n'
            f'- Gripper closes but misses the object → score 0.10-0.20\n'
            f'- Object touched but not grasped (slides away) → score 0.20-0.30\n'
            f'- Object grasped but slips during lift → score 0.30-0.40\n'
            f'- Robot arm moving aimlessly or oscillating → score 0.05-0.15\n'
            f'Output strictly in **JSON array format**. Include reasoning, score (two decimal places) '
            f'and status. Example: [{{"reasoning": "...", "score": 0.35, "status": "Approach State"}}]'
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

        if result:
            score = result[0].get("score")
            if score is not None:
                with lock:
                    score_history.append((step_idx, float(score)))
                logging.info(
                    f"[Gemini Value] frame={step_idx} score={score:.2f} "
                    f"status={result[0].get('status')}"
                )

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
    with lock:
        if not score_history:
            return False
        sorted_scores = sorted(score_history, key=lambda x: x[0])

    latest_frame, latest_score = sorted_scores[-1]

    if latest_score <= RESCUE_SCORE_ABSOLUTE:
        logging.info(f"[Rescue] Triggered: score {latest_score:.2f} < {RESCUE_SCORE_ABSOLUTE}")
        return True

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


def _gemini_select_best(current_video_path: str, candidate_paths: list,
                        task_description: str) -> int:
    """Select the best candidate video using Gemini.

    To mitigate VLM position bias, the candidate order is randomized before
    querying and the selected index is mapped back to the original order.
    """
    client = _get_gemini_client()

    # Shuffle candidate order to counteract position bias
    num_cands = len(candidate_paths)
    shuffled_order = list(range(num_cands))
    random.shuffle(shuffled_order)
    shuffled_paths = [candidate_paths[i] for i in shuffled_order]
    logging.info(f"[Gemini Select] presentation order: {shuffled_order}")

    current_file = client.files.upload(file=current_video_path)
    cand_files = [client.files.upload(file=p) for p in shuffled_paths]

    for f in [current_file] + cand_files:
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
    for i, cf_file in enumerate(cand_files):
        contents += [f"\n[Candidate Next Video {i}]:", cf_file]

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
    for cf_file in cand_files:
        client.files.delete(name=cf_file.name)

    # Map shuffled index back to original index
    shuffled_best = int(result["best_index"])
    original_best = shuffled_order[min(shuffled_best, num_cands - 1)]
    logging.info(
        f"[Gemini Select] picked shuffled idx {shuffled_best} "
        f"-> original idx {original_best}"
    )
    return original_best


def _rescue_select_action(
    current_obs_snapshot: dict,
    replay_images: list,
    task_description: str,
    policy: "OpenpiClient",
    exec_horizon: int,
    step_save_dir: pathlib.Path,
    dd_base_port: int,
    num_samples: int = 5,
) -> tuple:
    """
    Sample `num_samples` action chunks from the policy, send parallel DreamDojo
    generation requests, let Gemini pick the best, return the chosen action chunk.

    Args:
        current_obs_snapshot: dict with keys "top", "right", "left", "instruction", "state".
        replay_images: list of recent frames (uint8 HWC RGB) for Gemini context video.
        task_description: language instruction.
        policy: OpenpiClient instance.
        exec_horizon: number of steps to execute from the chosen chunk.
        step_save_dir: directory to save rescue artifacts.
        dd_base_port: base port for DreamDojo servers.
        num_samples: number of candidate action chunks.

    Returns:
        (best_actions, selection_record)
    """
    step_save_dir.mkdir(parents=True, exist_ok=True)
    action_chunks = [policy.predict_action(current_obs_snapshot) for _ in range(num_samples)]

    # Use the front camera image for DreamDojo frame
    frame_img = current_obs_snapshot["top"]  # uint8 HWC RGB

    # Log action diversity for diagnostics
    for i in range(num_samples):
        arr = np.array(action_chunks[i][:exec_horizon], dtype=np.float32)
        logging.info(
            f"[Rescue] chunk_{i} actions mean={arr.mean():.6f} "
            f"std={arr.std():.6f} first={arr[0, :3]}"
        )

    save_prefix = step_save_dir.name
    base_seed = int(time.time() * 1e6) % (2**31)
    tasks = [
        {
            "port": dd_base_port + i,
            "actions": np.array(action_chunks[i][:exec_horizon], dtype=np.float32),
            "save_name": f"{save_prefix}/chunk_{i}",
            "seed": base_seed + i,
        }
        for i in range(num_samples)
    ]

    logging.info(f"[Rescue] Launching {num_samples} parallel DreamDojo generation requests...")

    def _submit(t):
        return _dreamdojo_generate(t["port"], frame_img, t["actions"], t["save_name"], task_description, seed=t["seed"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as ex:
        futures = {ex.submit(_submit, t): i for i, t in enumerate(tasks)}
        save_paths = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            save_paths[idx] = fut.result()

    valid = [(i, save_paths[i]) for i in range(num_samples)
             if save_paths.get(i) and os.path.exists(save_paths[i])]

    if not valid:
        logging.warning("[Rescue] All DreamDojo generations failed; using chunk 0.")
        return list(action_chunks[0][:exec_horizon]), {
            "num_candidates": 0, "candidate_paths": [], "raw_best": None,
            "best_chunk_idx": 0, "error": "All DreamDojo generations failed",
        }

    # Save current history video for Gemini context
    current_video_path = str(step_save_dir / "current_actual_video.mp4")
    imageio.mimwrite(
        current_video_path,
        [np.asarray(x) for x in replay_images],
        fps=ROLLOUT_FPS,
    )

    # Copy candidate videos into step_save_dir
    local_valid_paths = []
    for orig_i, orig_path in valid:
        dst = step_save_dir / f"output_{orig_i}.mp4"
        try:
            shutil.copy2(orig_path, dst)
            local_valid_paths.append((orig_i, str(dst)))
        except Exception as e:
            logging.warning(f"[Rescue] Could not copy {orig_path} -> {dst}: {e}")
            local_valid_paths.append((orig_i, orig_path))
    valid = local_valid_paths

    valid_indices, valid_paths = zip(*valid)
    selection_record = {
        "num_candidates": len(valid),
        "candidate_paths": list(valid_paths),
        "raw_best": None,
        "best_chunk_idx": None,
        "error": None,
    }
    try:
        raw_best = _gemini_select_best(current_video_path, list(valid_paths), task_description)
        best_chunk_idx = valid_indices[min(raw_best, len(valid_indices) - 1)]
        selection_record["raw_best"] = raw_best
        selection_record["best_chunk_idx"] = int(best_chunk_idx)
        logging.info(f"[Rescue] Gemini selected candidate {raw_best} -> chunk {best_chunk_idx}")
    except Exception as e:
        logging.error(f"[Rescue] Gemini selection failed: {e}. Using first valid chunk.")
        best_chunk_idx = valid_indices[0]
        selection_record["error"] = str(e)

    return list(action_chunks[best_chunk_idx][:exec_horizon]), selection_record


def _write_gemini_results(gemini_results: list, rollout_dir: pathlib.Path,
                          task_description: str, suffix: str,
                          rescue_log: list | None = None,
                          rescue_selections: list | None = None):
    out_path = rollout_dir / "gemini_values.txt"
    with open(out_path, "w") as f:
        f.write(f"Task: {task_description}\n")
        f.write(f"Outcome: {suffix}\n")
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

        if rescue_log:
            f.write("=" * 60 + "\n")
            f.write(f"Rescue activations ({len(rescue_log)}): frames {rescue_log}\n")
    logging.info(f"[Gemini] Results written to {out_path}")

    json_path = rollout_dir / "gemini_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": task_description,
            "outcome": suffix,
            "value_evaluations": sorted(gemini_results, key=lambda x: x.get("step", 0)),
            "rescue_activations": rescue_log or [],
            "rescue_selections": rescue_selections or [],
        }, f, indent=2)
    logging.info(f"[Gemini] JSON results written to {json_path}")


def inference_fn_sync(args, config, policy, ros_operator):
    global inference_stamp

    update_observation_window(args, config, ros_operator)

    start_time = time.perf_counter()

    with observation_window_lock:
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]

        if args.ctrl_type == "joint" or args.ctrl_type == "ee6d":
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            state = observation_window[-1]["endpose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

    payload = {
        "top": image_arrs[0],
        "right": image_arrs[1],
        "left": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state,
        "action_prefix": None,
        "delay": None,
    }

    if args.streaming:
        actions = policy.predict_action_streaming(payload)
    else:
        actions = policy.predict_action(payload)
    print(f"[Sync   {inference_stamp:2d}] Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms")
    inference_stamp += 1

    return actions


def inference_fn_async(args, config, policy, ros_operator, action_buffer):
    global inference_stamp

    while not rospy.is_shutdown():
        try:
            inference_paused.wait()

            print(f"[Async  {inference_stamp:2d}] Start inference")

            d = config["delay"]
            s = config["exec_horizon"]

            with action_buffer.lock:
                if action_buffer.cur_chunk is None or config["mode"] == "naive":
                    action_prefix = None
                    if config["mode"] == "rtc":
                        rospy.logwarn("RTC mode: action_prefix is None")
                else:
                    action_prefix = action_buffer.cur_chunk[(s - d) : s].copy()
                    assert action_prefix.shape[0] == d

            update_observation_window(args, config, ros_operator)

            start_time = time.perf_counter()

            with observation_window_lock:
                image_arrs = [
                    observation_window[-1]["images"][config["camera_names"][0]],
                    observation_window[-1]["images"][config["camera_names"][1]],
                    observation_window[-1]["images"][config["camera_names"][2]],
                ]

                if args.ctrl_type == "joint" or args.ctrl_type == "ee6d":
                    state = observation_window[-1]["qpos"]
                elif args.ctrl_type == "eef":
                    state = observation_window[-1]["endpose"]
                else:
                    raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

            payload = {
                "top": image_arrs[0],
                "right": image_arrs[1],
                "left": image_arrs[2],
                "instruction": config["language_instruction"],
                "state": state,
                "action_prefix": action_prefix,
                "delay": np.array(d),
            }

            from functools import partial
            on_actions_ready = partial(action_buffer.integrate_new_chunk_streaming, stamp=inference_stamp)

            if args.streaming:
                policy.predict_action_streaming(payload, on_actions_ready=on_actions_ready)
                print(
                    f"[Async  {inference_stamp:2d}] Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms"
                )
            else:
                actions = policy.predict_action(payload)
                print(
                    f"[Async  {inference_stamp:2d}] Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms"
                )
                if actions is not None and len(actions) > 0:
                    action_buffer.integrate_new_chunk(actions[d : s + d])
                else:
                    print("actions is None or len(actions) == 0")

            inference_stamp += 1
            inference_paused.clear()

        except Exception as e:
            rospy.logwarn(f"[inference_fn_async] {e}")
            time.sleep(0.1)
            continue


def start_inference_thread(args, config, policy, ros_operator, action_buffer):
    inference_thread = threading.Thread(
        target=inference_fn_async, args=(args, config, policy, ros_operator, action_buffer)
    )
    inference_thread.daemon = True
    inference_thread.start()


def model_inference(args, config, ros_operator):
    global inference_stamp

    policy = OpenpiClient(host=args.host, port=args.port)

    max_publish_step = config["episode_len"]
    left0 = config["left0"]
    right0 = config["right0"]
    task_description = config["language_instruction"]
    print(config)

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    print("Warmup the server...")
    policy.warmup(rtc=(args.mode == "rtc"), streaming=args.streaming)
    print("Server warmed up")

    output_dir = pathlib.Path(args.video_out_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_buffer = StreamActionBuffer(
        delay=config["delay"], exec_horizon=config["exec_horizon"], state_dim=config["state_dim"]
    )

    start_inference_thread(args, config, policy, ros_operator, action_buffer)

    try:
        episode_idx = 0
        while not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)

            # Reset for new episode
            reset_observation_window()
            action_buffer.reset()
            inference_paused.clear()
            inference_stamp = 0

            input("Press enter to start episode")
            task_time = time.time()
            ros_operator.puppet_arm_publish_continuous(left0, right0)

            # Gemini value tracking
            score_history: list = []
            score_lock = threading.Lock()
            gemini_futures: list = []
            gemini_all_results: list = []
            gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            rescue_log: list = []
            rescue_selections: list = []

            # Collected frames for Gemini video and final recording
            collected_frames: list = []
            frame_counter = 0

            task_segment = task_description.replace(" ", "_")
            rollout_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_running"
            rollout_dir.mkdir(parents=True, exist_ok=True)

            actions = inference_fn_sync(args, config, policy, ros_operator)
            action_buffer.integrate_first_chunk(actions[: config["exec_horizon"]])

            last_valid_act = None
            user_stopped = False

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                print(
                    f"[Step {t:4d}] cur_step={action_buffer.cur_step:3d} | "
                    f"cur_chunk={action_buffer.cur_len:3d} | next_chunk={action_buffer.next_len:3d} | "
                    f"cur_stamp={action_buffer.cur_stamp:3d}"
                )

                # Keyboard handling
                key = check_keyboard_input()
                if key == " ":
                    inference_paused.clear()
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        ros_operator.puppet_arm_publish_continuous(left0, right0)
                        user_stopped = True
                        break
                    elif result == "quit":
                        user_stopped = True
                        # Shutdown gemini executor before returning
                        gemini_executor.shutdown(wait=False)
                        return

                # Capture current frame for Gemini evaluation
                with observation_window_lock:
                    if observation_window is not None and len(observation_window) > 0:
                        latest_obs = observation_window[-1]
                        front_img = latest_obs["images"].get(config["camera_names"][0])
                        if front_img is not None:
                            collected_frames.append(np.asarray(front_img).copy())
                            frame_counter += 1

                # Periodic Gemini value query
                if frame_counter > 0 and frame_counter % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                    clip = list(collected_frames[-GEMINI_HISTORY_FRAMES:])
                    future = gemini_executor.submit(
                        _query_gemini_value,
                        clip, task_description, frame_counter,
                        score_history, score_lock,
                    )
                    gemini_futures.append(future)
                    logging.info(f"[Gemini] Submitted value query at frame {frame_counter}")

                # Check rescue at replan boundaries
                if action_buffer.cur_step == 0 and frame_counter > GEMINI_QUERY_INTERVAL_FRAMES:
                    rescue = _check_rescue_needed(score_history, score_lock)
                    if rescue:
                        logging.info(f"[Rescue] Activating at frame {frame_counter}...")
                        rescue_log.append(frame_counter)

                        # Pause normal async inference during rescue
                        inference_paused.clear()

                        # Build observation snapshot for rescue sampling
                        with observation_window_lock:
                            image_arrs = [
                                observation_window[-1]["images"][config["camera_names"][0]],
                                observation_window[-1]["images"][config["camera_names"][1]],
                                observation_window[-1]["images"][config["camera_names"][2]],
                            ]
                            if args.ctrl_type == "joint" or args.ctrl_type == "ee6d":
                                state = observation_window[-1]["qpos"]
                            elif args.ctrl_type == "eef":
                                state = observation_window[-1]["endpose"]
                            else:
                                raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

                        obs_snapshot = {
                            "top": image_arrs[0],
                            "right": image_arrs[1],
                            "left": image_arrs[2],
                            "instruction": task_description,
                            "state": state,
                            "action_prefix": None,
                            "delay": None,
                        }

                        step_save_dir = rollout_dir / "rescue_steps" / f"frame{frame_counter}"
                        best_actions, sel_record = _rescue_select_action(
                            current_obs_snapshot=obs_snapshot,
                            replay_images=collected_frames,
                            task_description=task_description,
                            policy=policy,
                            exec_horizon=config["exec_horizon"],
                            step_save_dir=step_save_dir,
                            dd_base_port=args.dd_base_port,
                            num_samples=args.num_rescue_samples,
                        )
                        sel_record["frame"] = frame_counter
                        rescue_selections.append(sel_record)

                        # Inject rescue actions into the buffer
                        action_buffer.integrate_rescue_chunk(best_actions)
                        logging.info(f"[Rescue] Injected {len(best_actions)} rescue actions")

                # Normal action execution
                if action_buffer.should_launch_inference() and not inference_paused.is_set():
                    inference_paused.set()
                    time.sleep(0.001)

                act = action_buffer.get_next_action()

                if act is None:
                    rospy.logwarn(f"[Step {t:4d}] act is None")
                    if last_valid_act is not None:
                        act = last_valid_act
                    else:
                        rate.sleep()
                        continue

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.puppet_arm_publish(left_action, right_action)
                elif args.ctrl_type == "ee6d":
                    act = abs_6d_2_abs_euler(act)
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.endpose_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.endpose_publish(left_action, right_action)

                if args.use_robot_base:
                    vel_action = act[14:16]
                    ros_operator.robot_base_publish(vel_action)

                t += 1
                last_valid_act = act
                rate.sleep()

            # Episode finished — collect Gemini results
            gemini_executor.shutdown(wait=True)
            for future in gemini_futures:
                try:
                    gemini_all_results.append(future.result())
                except Exception as e:
                    gemini_all_results.append({"error": str(e)})

            suffix = "stopped" if user_stopped else "done"
            final_rollout_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            rollout_dir.rename(final_rollout_dir)
            rollout_dir = final_rollout_dir

            _write_gemini_results(
                gemini_all_results, rollout_dir, task_description, suffix,
                rescue_log=rescue_log, rescue_selections=rescue_selections,
            )

            if collected_frames:
                imageio.mimwrite(
                    str(rollout_dir / "complete_video.mp4"),
                    [np.asarray(x) for x in collected_frames],
                    fps=ROLLOUT_FPS,
                )

            logging.info(f"Episode {episode_idx} finished: {suffix}")
            logging.info(f"Rescue activations: {len(rescue_log)}")

            episode_idx += 1

            # Reset to starting position
            ros_operator.puppet_arm_publish_continuous(left0, right0)

    finally:
        ros_operator.puppet_arm_publish_continuous(left0, right0)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Agilex inference with Gemini rescue + DreamDojo"
    )
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--img_front_depth_topic", type=str, default="/camera_f/depth/image_raw")
    parser.add_argument("--img_left_depth_topic", type=str, default="/camera_l/depth/image_raw")
    parser.add_argument("--img_right_depth_topic", type=str, default="/camera_r/depth/image_raw")
    parser.add_argument("--puppet_arm_left_cmd_topic", type=str, default="/master/joint_left")
    parser.add_argument("--puppet_arm_right_cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_left_topic", type=str, default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--endpose_left_cmd_topic", type=str, default="/puppet/pos_cmd_left")
    parser.add_argument("--endpose_right_cmd_topic", type=str, default="/puppet/pos_cmd_right")
    parser.add_argument("--endpose_left_topic", type=str, default="/puppet/end_pose_left")
    parser.add_argument("--endpose_right_topic", type=str, default="/puppet/end_pose_right")
    parser.add_argument("--robot_base_topic", type=str, default="/odom_raw")
    parser.add_argument("--robot_base_cmd_topic", type=str, default="/cmd_vel")
    parser.add_argument("--use_robot_base", action="store_true", default=False)
    # --- Inference params ---
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--arm_steps_length", type=float, nargs=7, default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2])
    parser.add_argument("--use_depth_image", action="store_true", default=False)
    parser.add_argument("--ctrl_type", type=str, choices=["joint", "eef", "ee6d"], default="joint")
    parser.add_argument("--host", type=str, default="10.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, required=True,
                        choices=["towel", "rubbish", "tissue", "beverage", "play_pp", "pick_pp", "react"])
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--exec_horizon", type=int, default=25)
    parser.add_argument("--mode", type=str, choices=["naive", "rtc"], default="rtc")
    parser.add_argument("--streaming", action="store_true")
    # --- Gemini rescue / DreamDojo params ---
    parser.add_argument("--video_out_path", type=str, default="data/agilex/output",
                        help="Directory for rollout videos and Gemini results")
    parser.add_argument("--num_rescue_samples", type=int, default=5,
                        help="Number of candidate action chunks to sample during rescue")
    parser.add_argument("--dd_base_port", type=int, default=8020,
                        help="Base port for DreamDojo servers (one per sample)")

    return parser.parse_args()


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)

    logging.basicConfig(level=logging.INFO)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
