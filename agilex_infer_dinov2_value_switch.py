"""
Agilex robot inference with DINOv2 Value Expert + Auxiliary Switch Head.

Replaces the original Gemini-based pipeline:

  Original                              →  New
  ─────────────────────────────────────────────────────────────────
  Gemini value score → rescue trigger   →  Auxiliary switch head (in pi05)
                                            predicts switch_prob; if > threshold
                                            → switch from ODE to SDE policy
  Gemini selects best DreamDojo video   →  DINOv2 value expert scores each
                                            generated video with sliding 4-frame
                                            windows; picks lowest-value candidate
                                            (closest to subtask completion)

Flow:
  1. Normal: ODE policy → action chunk → execute
  2. Switch head fires (switch_prob > threshold):
     a. Sample N action chunks from SDE policy
     b. Send each to DreamDojo → generate candidate future videos
     c. DINOv2 value expert scores each video (stride-1 sliding window of 4-frame clips,
        using current 3-view observation as context)
     d. Average values per candidate up to the first "dip" → pick lowest score
     e. Execute the winning action chunk

Usage:
  # 1. ODE policy server (with switch head)
  python scripts/serve_policy.py policy:checkpoint \\
      --policy.config pi05_libero_switch --policy.dir <ckpt> --port 8000

  # 2. SDE policy server
  python scripts/serve_policy.py policy:checkpoint \\
      --policy.config pi05_sde_libero --policy.dir <sde_ckpt> --port 8001

  # 3. DreamDojo servers (one per candidate, ports 8020..8024)

  # 4. Run inference
  python agilex_infer_dinov2_value_switch.py \\
      --task towel --host 10.0.0.1 --port 8000 \\
      --sde_host 10.0.0.1 --sde_port 8001 \\
      --value_expert_ckpt checkpoints/dinov2_value_expert/best_model.pt \\
      --switch_threshold 0.5 --num_sde_samples 5 --dd_base_port 8020
"""

import argparse
import base64
import collections
import concurrent.futures
import json
import logging
import os
import pathlib
import shutil
import signal
import sys
import termios
import threading
import time
import tty

import imageio
import numpy as np
import requests
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import check_keyboard_input, get_config, handle_interactive_mode, process_action
from rosoperator import RosOperator, get_ros_observation
from rotation import abs_6d_2_abs_euler, quat_2_euler
from train_dinov2_value_expert import DINOv2ValueExpert


ROLLOUT_FPS = 10

observation_window = None
observation_window_lock = threading.Lock()

shutdown_event = threading.Event()
inference_paused = threading.Event()
inference_paused.clear()
inference_stamp = 0


class ValueExpertScorer:
    """Wraps a trained DINOv2ValueExpert for sliding-window video scoring."""

    def __init__(self, checkpoint_path: str, num_clip_frames: int = 4,
                 dinov2_model: str = "dinov2_vitb14", attn_heads: int = 8,
                 attn_layers: int = 2, hidden_dim: int = 512,
                 device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_clip_frames = num_clip_frames

        self.model = DINOv2ValueExpert(
            num_clip_frames=num_clip_frames,
            num_obs_views=3,
            dinov2_model=dinov2_model,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            hidden_dim=hidden_dim,
            freeze_backbone=True,
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _img_to_tensor(self, img_hwc_uint8: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 → (3, H, W) float [0,1] on device."""
        return torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float().to(self.device) / 255.0

    @torch.no_grad()
    def score_video(
        self,
        video_frames: list[np.ndarray],
        obs_front: np.ndarray,
        obs_wrist_left: np.ndarray,
        obs_wrist_right: np.ndarray,
    ) -> np.ndarray:
        """
        Score a generated video using stride-1 sliding windows.

        Args:
            video_frames: list of L uint8 (H, W, 3) frames
            obs_front, obs_wrist_left, obs_wrist_right: current observation images (uint8 HWC)

        Returns:
            per_window_values: (num_windows,) array – lower = closer to subtask completion
        """
        L = len(video_frames)
        if L < self.num_clip_frames:
            # Pad by repeating last frame
            video_frames = list(video_frames) + [video_frames[-1]] * (self.num_clip_frames - L)
            L = len(video_frames)

        # Build tensors
        video_t = torch.stack([self._img_to_tensor(f) for f in video_frames])  # (L, 3, H, W)
        video_t = video_t.unsqueeze(0)  # (1, L, 3, H, W)

        obs_t = torch.stack([
            self._img_to_tensor(obs_front),
            self._img_to_tensor(obs_wrist_left),
            self._img_to_tensor(obs_wrist_right),
        ]).unsqueeze(0)  # (1, 3, 3, H, W)

        # Sliding window scoring
        values = self.model.score_video(video_t, obs_t,
                                        window_size=self.num_clip_frames, stride=1)
        return values.squeeze(0).cpu().numpy()  # (num_windows,)

    def aggregate_video_score(self, per_window_values: np.ndarray) -> float:
        """
        Aggregate per-window values into a single score for candidate ranking.

        Per DreamDojo: average values from the start up to the first "dip"
        (the point before value starts rising again). Lower = better.
        """
        if len(per_window_values) == 0:
            return float("inf")

        # Find the dip: the minimum before value starts consistently rising
        min_idx = 0
        min_val = per_window_values[0]
        for i in range(1, len(per_window_values)):
            if per_window_values[i] < min_val:
                min_val = per_window_values[i]
                min_idx = i
            elif per_window_values[i] > min_val + 0.05:
                # Value started rising — stop here
                break

        # Average from start up to (including) the dip
        return float(np.mean(per_window_values[: min_idx + 1]))


def _dreamdojo_generate(host: str, port: int, frame_np: np.ndarray, actions: np.ndarray,
                        save_name: str, task_description: str = "") -> str | None:
    """Send an action chunk to a DreamDojo server and get a generated video path."""
    url = f"http://{host}:{port}/generate"

    h, w = frame_np.shape[:2]
    frame_bytes = base64.b64encode(frame_np.tobytes()).decode()

    payload = {
        "frame": frame_bytes,
        "frame_height": h,
        "frame_width": w,
        "actions": actions.tolist(),
        "save_name": save_name,
        "prompt": task_description,
    }
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["save_path"]
    except Exception as e:
        logging.error(f"[DreamDojo port={port}] generation failed: {e}")
        return None


def _select_best_action_with_value_expert(
    obs_snapshot: dict,
    sde_policy: "OpenpiClient",
    value_scorer: ValueExpertScorer,
    exec_horizon: int,
    task_description: str,
    step_save_dir: pathlib.Path,
    dd_host: str,
    dd_base_port: int,
    num_samples: int = 5,
) -> tuple[list, dict]:
    """
    1. Sample N action chunks from SDE policy
    2. Generate N candidate future videos via DreamDojo (parallel)
    3. Score each video with DINOv2 value expert (sliding 4-frame windows)
    4. Pick the candidate with the lowest aggregated score (closest to subtask completion)

    Returns: (best_actions, selection_record)
    """
    step_save_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Sample N action chunks from SDE policy ---
    action_chunks = [sde_policy.predict_action(obs_snapshot) for _ in range(num_samples)]

    # --- Step 2: Parallel DreamDojo generation ---
    frame_img = obs_snapshot["top"]  # front camera for DreamDojo conditioning frame
    save_prefix = step_save_dir.name

    tasks = [
        {
            "host": dd_host,
            "port": dd_base_port + i,
            "actions": np.array(action_chunks[i][:exec_horizon], dtype=np.float32),
            "save_name": f"{save_prefix}/chunk_{i}",
        }
        for i in range(num_samples)
    ]

    logging.info(f"[Switch] Launching {num_samples} parallel DreamDojo generation requests...")

    def _submit(t):
        return _dreamdojo_generate(t["host"], t["port"], frame_img, t["actions"], t["save_name"], task_description)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as ex:
        futures = {ex.submit(_submit, t): i for i, t in enumerate(tasks)}
        save_paths = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            save_paths[idx] = fut.result()

    valid = [(i, save_paths[i]) for i in range(num_samples)
             if save_paths.get(i) and os.path.exists(save_paths[i])]

    if not valid:
        logging.warning("[Switch] All DreamDojo generations failed; using chunk 0.")
        return list(action_chunks[0][:exec_horizon]), {
            "num_candidates": 0, "scores": [], "best_idx": 0,
            "error": "All DreamDojo generations failed",
        }

    # Copy candidate videos into step_save_dir
    local_valid = []
    for orig_i, orig_path in valid:
        dst = step_save_dir / f"output_{orig_i}.mp4"
        try:
            shutil.copy2(orig_path, dst)
            local_valid.append((orig_i, str(dst)))
        except Exception as e:
            logging.warning(f"[Switch] Could not copy {orig_path} -> {dst}: {e}")
            local_valid.append((orig_i, orig_path))

    # --- Step 3: Score each video with DINOv2 value expert ---
    candidate_scores = {}
    per_window_details = {}

    for orig_i, video_path in local_valid:
        try:
            frames = imageio.mimread(video_path)
            frames = [np.asarray(f) for f in frames]
        except Exception as e:
            logging.error(f"[ValueExpert] Failed to read {video_path}: {e}")
            continue

        per_window = value_scorer.score_video(
            video_frames=frames,
            obs_front=obs_snapshot["top"],
            obs_wrist_left=obs_snapshot["left"],
            obs_wrist_right=obs_snapshot["right"],
        )
        agg_score = value_scorer.aggregate_video_score(per_window)
        candidate_scores[orig_i] = agg_score
        per_window_details[orig_i] = per_window.tolist()

        logging.info(
            f"[ValueExpert] Candidate {orig_i}: agg_score={agg_score:.4f} "
            f"(windows={len(per_window)}, min={per_window.min():.4f})"
        )

    if not candidate_scores:
        logging.warning("[Switch] All video scoring failed; using first valid chunk.")
        best_idx = local_valid[0][0]
    else:
        # --- Step 4: Pick lowest score (closest to subtask completion) ---
        best_idx = min(candidate_scores, key=candidate_scores.get)

    selection_record = {
        "num_candidates": len(local_valid),
        "candidate_scores": {str(k): v for k, v in candidate_scores.items()},
        "per_window_details": {str(k): v for k, v in per_window_details.items()},
        "best_idx": int(best_idx),
        "best_score": candidate_scores.get(best_idx, float("inf")),
    }

    logging.info(
        f"[Switch] Selected candidate {best_idx} "
        f"(score={candidate_scores.get(best_idx, 'N/A')})"
    )

    return list(action_chunks[best_idx][:exec_horizon]), selection_record


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        import rospy
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
            observation_window.append({
                "qpos": None,
                "images": {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                    config["camera_names"][2]: None,
                },
                "endpose": None,
            })

    import rospy
    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, endpose_left, endpose_right = (
        get_ros_observation(args, ros_operator)
    )

    qpos = np.concatenate(
        (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0,
    )

    left_pos = endpose_left.pose.position
    left_rpy = quat_2_euler(endpose_left.pose.orientation)
    left_gripper = puppet_arm_left.position[-1]
    endpose_left_arr = np.array([left_pos.x, left_pos.y, left_pos.z,
                                  left_rpy[0], left_rpy[1], left_rpy[2], left_gripper])

    right_pos = endpose_right.pose.position
    right_rpy = quat_2_euler(endpose_right.pose.orientation)
    right_gripper = puppet_arm_right.position[-1]
    endpose_right_arr = np.array([right_pos.x, right_pos.y, right_pos.z,
                                   right_rpy[0], right_rpy[1], right_rpy[2], right_gripper])

    endpose = np.concatenate((endpose_left_arr, endpose_right_arr), axis=0)

    with observation_window_lock:
        observation_window.append({
            "qpos": qpos,
            "images": {
                config["camera_names"][0]: img_front,
                config["camera_names"][1]: img_right,
                config["camera_names"][2]: img_left,
            },
            "endpose": endpose,
        })


class StreamActionBuffer:
    """Action double-buffer for async streaming inference (unchanged from original)."""

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
            assert self.cur_len == 0 and self.cur_stamp == 0
            assert actions_chunk.shape[0] == self.exec_horizon
            self.cur_chunk[: actions_chunk.shape[0]] = actions_chunk
            self.cur_len = self.exec_horizon

    def integrate_new_chunk(self, actions_chunk: np.ndarray):
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            import rospy
            rospy.logwarn("actions_chunk is None or empty")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            assert L == self.exec_horizon
            if self.cur_len == 0:
                import rospy
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                self.cur_chunk[:L] = actions_chunk
                self.cur_len = L
            else:
                self.next_chunk[:L] = actions_chunk
                self.next_len = L

    def integrate_new_chunk_streaming(self, actions_chunk: np.ndarray, stamp: int):
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            import rospy
            rospy.logwarn("actions_chunk is None or empty")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            if self.cur_len == 0:
                import rospy
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                safe_L = min(L, self.exec_horizon)
                self.cur_chunk[:safe_L] = actions_chunk[:safe_L]
                self.cur_len = safe_L
            else:
                if self.cur_stamp == stamp:
                    remaining = self.exec_horizon - self.cur_len
                    if remaining > 0:
                        safe_L = min(L, remaining)
                        self.cur_chunk[self.cur_len : self.cur_len + safe_L] = actions_chunk[:safe_L]
                        self.cur_len += safe_L
                else:
                    remaining = self.exec_horizon - self.next_len
                    if remaining > 0:
                        safe_L = min(L, remaining)
                        self.next_chunk[self.next_len : self.next_len + safe_L] = actions_chunk[:safe_L]
                        self.next_len += safe_L

    def integrate_rescue_chunk(self, actions: list):
        """Replace current plan with value-expert-selected actions."""
        with self.lock:
            L = min(len(actions), self.exec_horizon)
            self.cur_chunk[:L] = np.array(actions[:L])
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


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def inference_fn_sync(args, config, policy, ros_operator):
    """Synchronous single-step inference (used for first chunk)."""
    global inference_stamp
    import rospy

    update_observation_window(args, config, ros_operator)
    start_time = time.perf_counter()

    with observation_window_lock:
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]
        if args.ctrl_type in ("joint", "ee6d"):
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            state = observation_window[-1]["endpose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

    payload = {
        "top": image_arrs[0], "right": image_arrs[1], "left": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state, "action_prefix": None, "delay": None,
    }

    actions = policy.predict_action(payload) if not args.streaming else policy.predict_action_streaming(payload)

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Sync   {inference_stamp:2d}] Model inference: {elapsed:.1f}ms")
    inference_stamp += 1
    return actions


def inference_fn_async(args, config, policy, ros_operator, action_buffer):
    global inference_stamp
    import rospy

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
                if args.ctrl_type in ("joint", "ee6d"):
                    state = observation_window[-1]["qpos"]
                elif args.ctrl_type == "eef":
                    state = observation_window[-1]["endpose"]
                else:
                    raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

            payload = {
                "top": image_arrs[0], "right": image_arrs[1], "left": image_arrs[2],
                "instruction": config["language_instruction"],
                "state": state, "action_prefix": action_prefix, "delay": np.array(d),
            }

            from functools import partial
            on_actions_ready = partial(action_buffer.integrate_new_chunk_streaming, stamp=inference_stamp)

            if args.streaming:
                policy.predict_action_streaming(payload, on_actions_ready=on_actions_ready)
            else:
                actions = policy.predict_action(payload)
                if actions is not None and len(actions) > 0:
                    action_buffer.integrate_new_chunk(actions[d : s + d])
                else:
                    print("actions is None or len(actions) == 0")

            print(f"[Async  {inference_stamp:2d}] Inference: {(time.perf_counter() - start_time)*1000:.1f}ms")
            inference_stamp += 1
            inference_paused.clear()

        except Exception as e:
            import rospy
            rospy.logwarn(f"[inference_fn_async] {e}")
            time.sleep(0.1)


def start_inference_thread(args, config, policy, ros_operator, action_buffer):
    t = threading.Thread(target=inference_fn_async,
                         args=(args, config, policy, ros_operator, action_buffer))
    t.daemon = True
    t.start()


def _get_obs_snapshot(args, config):
    """Build observation snapshot dict from current observation window."""
    with observation_window_lock:
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]
        if args.ctrl_type in ("joint", "ee6d"):
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            state = observation_window[-1]["endpose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

    return {
        "top": image_arrs[0], "right": image_arrs[1], "left": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state, "action_prefix": None, "delay": None,
    }


def model_inference(args, config, ros_operator):
    global inference_stamp
    import rospy

    # ODE policy (with auxiliary switch head)
    policy = OpenpiClient(host=args.host, port=args.port)

    # SDE policy (queried when switch head triggers)
    sde_policy = None
    if args.sde_host:
        sde_policy = OpenpiClient(host=args.sde_host, port=args.sde_port)
        logging.info(f"SDE policy connected to {args.sde_host}:{args.sde_port}")

    # DINOv2 value expert
    value_scorer = None
    if args.value_expert_ckpt and os.path.exists(args.value_expert_ckpt):
        value_scorer = ValueExpertScorer(
            checkpoint_path=args.value_expert_ckpt,
            num_clip_frames=args.num_clip_frames,
            dinov2_model=args.dinov2_model,
            attn_heads=args.attn_heads,
            attn_layers=args.attn_layers,
            hidden_dim=args.value_hidden_dim,
        )
        logging.info(f"DINOv2 value expert loaded from {args.value_expert_ckpt}")
    elif args.value_expert_ckpt:
        logging.warning(f"Value expert checkpoint not found: {args.value_expert_ckpt}")

    can_switch = sde_policy is not None and value_scorer is not None

    max_publish_step = config["episode_len"]
    left0 = config["left0"]
    right0 = config["right0"]
    task_description = config["language_instruction"]
    print(config)

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    print("Warmup servers...")
    policy.warmup(rtc=(args.mode == "rtc"), streaming=args.streaming)
    if sde_policy:
        sde_policy.warmup(rtc=False, streaming=False)
    print("Servers warmed up")

    output_dir = pathlib.Path(args.video_out_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_buffer = StreamActionBuffer(
        delay=config["delay"], exec_horizon=config["exec_horizon"], state_dim=config["state_dim"],
    )
    start_inference_thread(args, config, policy, ros_operator, action_buffer)

    try:
        episode_idx = 0
        while not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()
            action_buffer.reset()
            inference_paused.clear()
            inference_stamp = 0

            input("Press enter to start episode")
            task_time = time.time()
            ros_operator.puppet_arm_publish_continuous(left0, right0)

            # Tracking
            switch_log: list = []
            value_selections: list = []
            collected_frames: list = []
            frame_counter = 0

            task_segment = task_description.replace(" ", "_")
            rollout_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_running"
            rollout_dir.mkdir(parents=True, exist_ok=True)

            # First synchronous inference
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
                        return

                # Capture current frame
                with observation_window_lock:
                    if observation_window is not None and len(observation_window) > 0:
                        front_img = observation_window[-1]["images"].get(config["camera_names"][0])
                        if front_img is not None:
                            collected_frames.append(np.asarray(front_img).copy())
                            frame_counter += 1

                # Auxiliary Switch Head check at replan boundaries
                if can_switch and action_buffer.cur_step == 0 and frame_counter > 0:
                    update_observation_window(args, config, ros_operator)
                    obs_snapshot = _get_obs_snapshot(args, config)

                    # Query ODE policy to get switch_prob
                    ode_result = policy.predict_action(obs_snapshot)
                    cur_switch_prob = None
                    if isinstance(ode_result, dict):
                        cur_switch_prob = ode_result.get("switch", None)

                    if cur_switch_prob is not None and cur_switch_prob > args.switch_threshold:
                        logging.info(
                            f"[Switch] Triggered at frame {frame_counter} "
                            f"(prob={cur_switch_prob:.3f} > {args.switch_threshold})"
                        )
                        switch_log.append({
                            "frame": frame_counter,
                            "switch_prob": float(cur_switch_prob),
                        })

                        # Pause normal async inference
                        inference_paused.clear()

                        # DreamDojo + DINOv2 value expert selection
                        step_save_dir = rollout_dir / "switch_steps" / f"frame{frame_counter}"
                        best_actions, sel_record = _select_best_action_with_value_expert(
                            obs_snapshot=obs_snapshot,
                            sde_policy=sde_policy,
                            value_scorer=value_scorer,
                            exec_horizon=config["exec_horizon"],
                            task_description=task_description,
                            step_save_dir=step_save_dir,
                            dd_host=args.dd_host,
                            dd_base_port=args.dd_base_port,
                            num_samples=args.num_sde_samples,
                        )
                        sel_record["frame"] = frame_counter
                        sel_record["switch_prob"] = float(cur_switch_prob)
                        value_selections.append(sel_record)

                        action_buffer.integrate_rescue_chunk(best_actions)
                        logging.info(f"[Switch] Injected {len(best_actions)} SDE actions")

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
                    ros_operator.robot_base_publish(act[14:16])

                t += 1
                last_valid_act = act
                rate.sleep()

            suffix = "stopped" if user_stopped else "done"
            final_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            rollout_dir.rename(final_dir)

            _write_results(final_dir, task_description, suffix, switch_log, value_selections)

            if collected_frames:
                imageio.mimwrite(
                    str(final_dir / "complete_video.mp4"),
                    [np.asarray(x) for x in collected_frames],
                    fps=ROLLOUT_FPS,
                )

            logging.info(f"Episode {episode_idx} finished: {suffix}")
            logging.info(f"Switch activations: {len(switch_log)}")
            episode_idx += 1
            ros_operator.puppet_arm_publish_continuous(left0, right0)

    finally:
        ros_operator.puppet_arm_publish_continuous(left0, right0)


def _write_results(rollout_dir: pathlib.Path, task_description: str, suffix: str,
                   switch_log: list, value_selections: list):
    json_path = rollout_dir / "switch_value_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": task_description,
            "outcome": suffix,
            "switch_activations": switch_log,
            "value_expert_selections": value_selections,
        }, f, indent=2)
    logging.info(f"Results written to {json_path}")

    txt_path = rollout_dir / "switch_value_results.txt"
    with open(txt_path, "w") as f:
        f.write(f"Task: {task_description}\nOutcome: {suffix}\n")
        f.write(f"Switch activations: {len(switch_log)}\n{'='*60}\n\n")
        for e in switch_log:
            f.write(f"  Frame {e['frame']}: switch_prob={e['switch_prob']:.3f}\n")
        f.write(f"\nValue expert selections: {len(value_selections)}\n{'='*60}\n\n")
        for s in value_selections:
            f.write(
                f"  Frame {s['frame']}: best_idx={s['best_idx']} "
                f"best_score={s.get('best_score', 'N/A')}\n"
                f"    scores={s.get('candidate_scores', {})}\n"
            )


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Agilex inference: Auxiliary Switch Head + DreamDojo + DINOv2 Value Expert"
    )
    # ---- ROS topics ----
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
    # ---- Inference ----
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--arm_steps_length", type=float, nargs=7,
                        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2])
    parser.add_argument("--use_depth_image", action="store_true", default=False)
    parser.add_argument("--ctrl_type", type=str, choices=["joint", "eef", "ee6d"], default="joint")
    parser.add_argument("--host", type=str, default="10.0.0.1",
                        help="ODE policy server (with switch head)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, required=True,
                        choices=["towel", "rubbish", "tissue", "beverage", "play_pp", "pick_pp", "react"])
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--exec_horizon", type=int, default=25)
    parser.add_argument("--mode", type=str, choices=["naive", "rtc"], default="rtc")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--video_out_path", type=str, default="data/agilex/output")
    # ---- SDE policy ----
    parser.add_argument("--sde_host", type=str, default=None,
                        help="SDE policy server host (required for switch mode)")
    parser.add_argument("--sde_port", type=int, default=8001)
    # ---- Auxiliary switch head ----
    parser.add_argument("--switch_threshold", type=float, default=0.5,
                        help="Prob threshold for switching ODE → SDE")
    # ---- DreamDojo ----
    parser.add_argument("--dd_host", type=str, default="127.0.0.1",
                        help="DreamDojo server host")
    parser.add_argument("--dd_base_port", type=int, default=8020,
                        help="Base port for DreamDojo servers (one per candidate)")
    parser.add_argument("--num_sde_samples", type=int, default=5,
                        help="Number of SDE action candidates")
    # ---- DINOv2 Value Expert ----
    parser.add_argument("--value_expert_ckpt", type=str, default=None,
                        help="Trained value expert checkpoint (.pt)")
    parser.add_argument("--num_clip_frames", type=int, default=4,
                        help="Frames per sliding window clip")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--attn_layers", type=int, default=2)
    parser.add_argument("--value_hidden_dim", type=int, default=512)

    return parser.parse_args()


def main():
    args = get_arguments()

    import rospy
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
