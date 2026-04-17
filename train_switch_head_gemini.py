"""
Two-phase pipeline for training a switch head to replace Gemini-based intervention detection.

Phase 1 – collect:
  Run LIBERO rollouts with the policy server, query Gemini for dense value scores
  every 4 seconds, and apply the rescue trigger logic to produce binary switch labels.
  Each replanning step is saved as an .npz file.

Phase 2 – train:
  Train a standalone DINOv2-based binary classifier on the collected data.
  Alternatively, the saved switch_label can be injected into the LeRobot dataset
  and used with the existing `pi05_libero_switch` training config.

Usage
-----
  # 1) Collect labeled data (needs policy server running + GOOGLE_API_KEY set)
  python train_switch_head_gemini.py collect \
      --host 10.0.0.1 --port 8000 \
      --output_dir data/switch_labels \
      --task_suite_name libero_10 \
      --num_trials_per_task 20

  # 2) Train standalone switch head
  python train_switch_head_gemini.py train \
      --data_dir data/switch_labels \
      --output_dir checkpoints/switch_head \
      --epochs 30

  # 3) (Optional) Export labels for pi05 integrated training
  python train_switch_head_gemini.py export \
      --data_dir data/switch_labels \
      --lerobot_repo physical-intelligence/libero \
      --output_repo data/libero_with_switch
"""

import argparse
import collections
import concurrent.futures
import glob
import json
import logging
import os
import pathlib
import sys
import tempfile
import threading
import time

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (match eval_with_gemini_rescue.py)
# ---------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ROLLOUT_FPS = 10

GEMINI_QUERY_INTERVAL_FRAMES = 40   # 4s at 10fps
GEMINI_HISTORY_FRAMES = 200         # ~20s context window
GEMINI_VALUE_MODEL = "gemini-2.5-flash-preview-04-17"

RESCUE_SCORE_ABSOLUTE = 0.20
RESCUE_SCORE_DROP = 0.20

# ============================================================================
#  Phase 1: Data collection with Gemini labeling
# ============================================================================

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _query_gemini_value(frames: list, task_description: str, step_idx: int,
                        score_history: list, lock: threading.Lock) -> dict:
    """Query Gemini for a value score on a video clip. Thread-safe."""
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class ValueEvaluation(BaseModel):
        reasoning: str
        score: float
        status: str

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
    """Return True if rescue should be triggered based on the Gemini score history."""
    with lock:
        if not score_history:
            return False
        sorted_scores = sorted(score_history, key=lambda x: x[0])

    latest_frame, latest_score = sorted_scores[-1]

    # Condition 1: absolute low score
    if latest_score < RESCUE_SCORE_ABSOLUTE:
        logging.info(f"[Rescue] Triggered: score {latest_score:.2f} < {RESCUE_SCORE_ABSOLUTE}")
        return True

    # Condition 2: score drop >= RESCUE_SCORE_DROP compared to ~4s ago
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


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle (matches eval_with_gemini_rescue.py)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(quat).as_rotvec()


def _get_libero_env(task, resolution, seed):
    """Create LIBERO env for a given task."""
    from libero.libero.envs import OffScreenRenderEnv

    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def collect(args):
    """Phase 1: Run LIBERO rollouts with Gemini scoring and save labeled data."""
    from libero.libero import benchmark, get_libero_path
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _wcp

    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    policy_client = _wcp.WebsocketClientPolicy(args.host, args.port)

    # Resume from existing samples (same as agilex version)
    existing = glob.glob(str(output_dir / "sample_*.npz"))
    global_sample_idx = len(existing)
    if global_sample_idx > 0:
        logging.info(f"Resuming from sample index {global_sample_idx}")

    stats = {"total_episodes": 0, "total_successes": 0,
             "total_rescue_steps": 0, "total_normal_steps": 0}

    # Per-episode metadata for later analysis
    all_episode_meta = []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        logging.info(f"\n{'='*60}")
        logging.info(f"Task {task_id+1}/{num_tasks}: {task_description}")
        logging.info(f"{'='*60}")

        for episode_idx in range(args.num_trials_per_task):
            # Skip already-completed episodes (for resume)
            task_segment = task_description.replace(" ", "_")
            existing_dirs = list(
                output_dir.glob(f"rollout_{task_segment}_ep{episode_idx}_*")
            )
            if existing_dirs:
                logging.info(f"  Skip: task {task_id} ep {episode_idx} (already collected)")
                continue

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan = collections.deque()

            t = 0
            done = False
            clean_images = []       # front camera history (for Gemini + clip)
            all_wrist_images = []   # wrist camera history (for clip)
            replan_records = []

            # Gemini scoring state
            score_history = []
            score_lock = threading.Lock()
            gemini_futures = []
            gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

            logging.info(f"  Episode {episode_idx+1}/{args.num_trials_per_task}...")

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            wrist_img, args.resize_size, args.resize_size
                        )
                    )

                    clean_images.append(img)
                    all_wrist_images.append(wrist_img)
                    num_frames = len(clean_images)

                    # ---- Async Gemini value query every GEMINI_QUERY_INTERVAL_FRAMES ----
                    if num_frames % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                        clip = list(clean_images[-GEMINI_HISTORY_FRAMES:])
                        future = gemini_executor.submit(
                            _query_gemini_value,
                            clip, task_description, num_frames,
                            score_history, score_lock,
                        )
                        gemini_futures.append(future)

                    # ---- Replanning: record observation + check rescue ----
                    if not action_plan:
                        rescue = _check_rescue_needed(score_history, score_lock)

                        state_vec = np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ))

                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state_vec,
                            "prompt": str(task_description),
                        }
                        action_chunk = policy_client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps

                        # Extract video clips of recent clip_len frames
                        clip_len = args.clip_len
                        img_clip = list(clean_images[-clip_len:])
                        wrist_clip = list(all_wrist_images[-clip_len:])
                        if len(img_clip) < clip_len:
                            pad_n = clip_len - len(img_clip)
                            img_clip = [img_clip[0]] * pad_n + img_clip
                            wrist_clip = [wrist_clip[0]] * pad_n + wrist_clip

                        # Record this replanning step (single-frame + video clip)
                        replan_records.append({
                            "frame_idx": num_frames,
                            "image": img.copy(),
                            "wrist_image": wrist_img.copy(),
                            "image_clip": np.stack(img_clip),       # (T, H, W, 3)
                            "wrist_clip": np.stack(wrist_clip),     # (T, H, W, 3)
                            "state": state_vec.copy(),
                            "actions": np.array(action_chunk[:args.replan_steps],
                                                dtype=np.float32),
                            "rescue": rescue,
                        })

                        action_plan.extend(action_chunk[:args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        stats["total_successes"] += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # Wait for all Gemini queries to finish
            gemini_executor.shutdown(wait=True)

            # ---- Post-process: re-label with final Gemini scores ----
            # Some Gemini queries might have returned after the replan step that
            # checked them. Re-apply rescue logic with the complete score history.
            sorted_scores = sorted(score_history, key=lambda x: x[0])

            for rec in replan_records:
                frame = rec["frame_idx"]
                # Rebuild score_history up to this frame
                scores_up_to_frame = [
                    (f, s) for f, s in sorted_scores if f <= frame
                ]
                if not scores_up_to_frame:
                    rec["switch_label"] = 0.0
                    continue

                latest_f, latest_s = scores_up_to_frame[-1]

                should_rescue = False
                # Condition 1: absolute low
                if latest_s < RESCUE_SCORE_ABSOLUTE:
                    should_rescue = True
                # Condition 2: score drop
                if not should_rescue:
                    prev_s = None
                    for f, s in reversed(scores_up_to_frame[:-1]):
                        if latest_f - f >= GEMINI_QUERY_INTERVAL_FRAMES:
                            prev_s = s
                            break
                    if prev_s is not None and (latest_s - prev_s) <= -RESCUE_SCORE_DROP:
                        should_rescue = True

                rec["switch_label"] = 1.0 if should_rescue else 0.0

            # ---- Save per-step .npz files ----
            n_rescue = sum(1 for r in replan_records if r["switch_label"] > 0.5)
            n_normal = len(replan_records) - n_rescue

            for rec in replan_records:
                save_path = output_dir / f"sample_{global_sample_idx:07d}.npz"
                np.savez_compressed(
                    save_path,
                    # Single-frame (for image-input model)
                    image=rec["image"],                          # (H, W, 3) uint8
                    wrist_image=rec["wrist_image"],              # (H, W, 3) uint8
                    # Video clips (for video-input model)
                    image_clip=rec["image_clip"],                # (T, H, W, 3) uint8
                    wrist_clip=rec["wrist_clip"],                # (T, H, W, 3) uint8
                    state=rec["state"].astype(np.float32),       # (7,)
                    actions=rec["actions"],                       # (replan_steps, 7) float32
                    switch_label=np.float32(rec["switch_label"]),  # 0.0 or 1.0
                    clip_len=np.int32(args.clip_len),
                    prompt=np.array(task_description),            # string
                    task_id=np.int32(task_id),
                    episode_idx=np.int32(episode_idx),
                    frame_idx=np.int32(rec["frame_idx"]),
                )
                global_sample_idx += 1

            stats["total_episodes"] += 1
            stats["total_rescue_steps"] += n_rescue
            stats["total_normal_steps"] += n_normal

            episode_meta = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "success": bool(done),
                "num_steps": len(replan_records),
                "num_rescue": n_rescue,
                "gemini_scores": sorted_scores,
            }
            all_episode_meta.append(episode_meta)

            # Create rollout dir marker for resume skip
            suffix = "success" if done else "failure"
            rollout_marker = output_dir / f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            rollout_marker.mkdir(parents=True, exist_ok=True)

            logging.info(
                f"  -> {suffix.upper()} | "
                f"steps={len(replan_records)} rescue={n_rescue} normal={n_normal}"
            )

        env.close()

    # Save collection metadata
    meta_path = output_dir / "collection_meta.json"
    serializable_meta = []
    for m in all_episode_meta:
        sm = dict(m)
        sm["gemini_scores"] = [(int(f), float(s)) for f, s in sm["gemini_scores"]]
        serializable_meta.append(sm)

    with open(meta_path, "w") as f:
        json.dump({"stats": stats, "episodes": serializable_meta}, f, indent=2)

    logging.info(f"\nCollection complete.")
    logging.info(f"  Total samples: {global_sample_idx}")
    logging.info(f"  Rescue steps : {stats['total_rescue_steps']}")
    logging.info(f"  Normal steps : {stats['total_normal_steps']}")
    logging.info(f"  Saved to     : {output_dir}")


# ============================================================================
#  Phase 2: Training
# ============================================================================

class DINOv2SwitchHead(nn.Module):
    """
    Standalone DINOv2-based binary classifier for switch/intervention prediction.

    Supports two data formats:
      - LIBERO (2 cameras): base_image + wrist_image
      - Agilex (3 cameras): top + right + left

    Encodes each image with a frozen DINOv2 backbone, concatenates features
    with the robot state, and outputs a switch probability.
    """

    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        hidden_dim: int = 256,
        state_dim: int = 7,
        num_cameras: int = 2,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_cameras = num_cameras

        self.backbone = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.feature_dim = self.backbone.embed_dim  # 768 for ViT-B/14
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        input_dim = num_cameras * self.feature_dim + state_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float [0,1] → (B, feature_dim)"""
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self.img_mean) / self.img_std
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(img)
        return self.backbone(img)

    def forward(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: list of (B, 3, H, W) float [0,1] tensors, length = num_cameras
            state:  (B, state_dim)

        Returns:
            logit: (B,) — raw logit (apply sigmoid for probability)
        """
        feats = [self._encode_image(img) for img in images]  # list of (B, D)
        combined = torch.cat(feats + [state], dim=-1)
        return self.classifier(combined).squeeze(-1)

    def predict_switch_prob(
        self, images: list[torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        """Returns switch probability in [0, 1]."""
        return torch.sigmoid(self.forward(images, state))


class SwitchLabelDataset(Dataset):
    """
    Loads .npz files produced by the 'collect' phase.

    Auto-detects format:
      - LIBERO: keys "image", "wrist_image"  → 2 cameras
      - Agilex: keys "top", "right", "left"  → 3 cameras
    """

    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz files found in {data_dir}")
        logging.info(f"Found {len(self.files)} samples in {data_dir}")

        # Detect format from first file
        first = np.load(self.files[0], allow_pickle=True)
        if "top" in first:
            self.format = "agilex"
            self.num_cameras = 3
            self.image_keys = ["top", "right", "left"]
        else:
            self.format = "libero"
            self.num_cameras = 2
            self.image_keys = ["image", "wrist_image"]
        self.state_dim = first["state"].shape[0]
        logging.info(
            f"  Detected format: {self.format} ({self.num_cameras} cameras, "
            f"state_dim={self.state_dim})"
        )

        # Count class balance
        labels = []
        for f in self.files:
            d = np.load(f)
            labels.append(float(d["switch_label"]))
        n_pos = sum(1 for l in labels if l > 0.5)
        n_neg = len(labels) - n_pos
        logging.info(f"  Class balance: {n_pos} rescue (pos) / {n_neg} normal (neg)")
        self._labels = labels

    def __len__(self):
        return len(self.files)

    @property
    def pos_weight(self) -> float:
        """Compute pos_weight for BCE loss to handle class imbalance."""
        n_pos = sum(1 for l in self._labels if l > 0.5)
        n_neg = len(self._labels) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        # (H, W, 3) uint8 → (3, H, W) float [0,1]
        images = []
        for key in self.image_keys:
            img = torch.from_numpy(data[key]).permute(2, 0, 1).float() / 255.0
            images.append(img)

        state = torch.from_numpy(data["state"].astype(np.float32))
        label = torch.tensor(float(data["switch_label"]), dtype=torch.float32)

        return images, state, label


def _collate_switch(batch):
    """Custom collate for variable-length image lists."""
    images_list, states, labels = zip(*batch)
    num_cameras = len(images_list[0])
    batched_images = [torch.stack([img[c] for img in images_list]) for c in range(num_cameras)]
    return batched_images, torch.stack(states), torch.stack(labels)


def train(args):
    """Phase 2: Train the standalone switch head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_dataset = SwitchLabelDataset(args.data_dir)
    val_dataset = SwitchLabelDataset(args.val_dir) if args.val_dir else None

    num_cameras = train_dataset.num_cameras
    state_dim = train_dataset.state_dim
    logging.info(
        f"Training with {num_cameras}-camera data ({train_dataset.format} format), "
        f"state_dim={state_dim}"
    )

    model = DINOv2SwitchHead(
        dinov2_model=args.dinov2_model,
        hidden_dim=args.hidden_dim,
        state_dim=state_dim,
        num_cameras=num_cameras,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_switch,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=_collate_switch,
        )
        if val_dataset
        else None
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Use pos_weight to handle class imbalance (rescue steps are typically rare)
    pw = torch.tensor([train_dataset.pos_weight], device=device)
    logging.info(f"BCE pos_weight: {pw.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_metric = -float("inf")

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()

        total_loss, num_batches = 0.0, 0
        train_correct, train_total = 0, 0

        for images, state, label in train_loader:
            images = [img.to(device) for img in images]
            state = state.to(device)
            label = label.to(device)

            logit = model(images, state)
            loss = criterion(logit, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pred = (torch.sigmoid(logit) > 0.5).float()
            train_correct += (pred == label).sum().item()
            train_total += label.numel()

        avg_train_loss = total_loss / max(num_batches, 1)
        train_acc = train_correct / max(train_total, 1)
        scheduler.step()

        # ---- Val ----
        val_str = "N/A"
        if val_loader is not None:
            model.eval()
            vl, vn = 0.0, 0
            val_correct, val_total = 0, 0
            val_tp, val_fp, val_fn = 0, 0, 0

            with torch.no_grad():
                for images, state, label in val_loader:
                    images = [img.to(device) for img in images]
                    state = state.to(device)
                    label = label.to(device)

                    logit = model(images, state)
                    vl += criterion(logit, label).item()
                    vn += 1

                    pred = (torch.sigmoid(logit) > 0.5).float()
                    val_correct += (pred == label).sum().item()
                    val_total += label.numel()
                    val_tp += ((pred == 1) & (label == 1)).sum().item()
                    val_fp += ((pred == 1) & (label == 0)).sum().item()
                    val_fn += ((pred == 0) & (label == 1)).sum().item()

            avg_val_loss = vl / max(vn, 1)
            val_acc = val_correct / max(val_total, 1)
            val_precision = val_tp / max(val_tp + val_fp, 1)
            val_recall = val_tp / max(val_tp + val_fn, 1)
            val_f1 = (
                2 * val_precision * val_recall / max(val_precision + val_recall, 1e-8)
            )

            val_str = (
                f"loss={avg_val_loss:.4f} acc={val_acc:.3f} "
                f"P={val_precision:.3f} R={val_recall:.3f} F1={val_f1:.3f}"
            )

            # Save best model by F1 (more meaningful than accuracy for imbalanced data)
            if val_f1 > best_val_metric:
                best_val_metric = val_f1
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logging.info(f"  -> Saved best model (F1={val_f1:.4f})")

        logging.info(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val=[{val_str}]  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), output_dir / "model_final.pt")
    logging.info(f"Training complete. Models saved to {output_dir}")


# ============================================================================
#  Phase 3 (optional): Export for pi05 integrated training
# ============================================================================

def export_for_pi05(args):
    """
    Inject switch_label into an existing LeRobot LIBERO dataset.

    Loads the existing HF dataset, maps each sample to the closest collected
    switch_label based on (task_id, episode_idx, frame_idx), and saves a new
    dataset with the switch_label column added.
    """
    try:
        import datasets
    except ImportError:
        logging.error("Install `datasets` library: pip install datasets")
        return

    data_dir = pathlib.Path(args.data_dir)

    # Load all collected labels into a lookup table
    label_files = sorted(glob.glob(str(data_dir / "sample_*.npz")))
    if not label_files:
        logging.error(f"No sample files found in {data_dir}")
        return

    logging.info(f"Loading {len(label_files)} label files...")
    label_lookup = {}  # (task_id, episode_idx) -> [(frame_idx, switch_label), ...]
    for f in label_files:
        d = np.load(f, allow_pickle=True)
        key = (int(d["task_id"]), int(d["episode_idx"]))
        if key not in label_lookup:
            label_lookup[key] = []
        label_lookup[key].append((int(d["frame_idx"]), float(d["switch_label"])))

    for key in label_lookup:
        label_lookup[key].sort()

    logging.info(f"Loaded labels for {len(label_lookup)} episodes")

    # Load HF dataset
    logging.info(f"Loading dataset from {args.lerobot_repo}...")
    ds = datasets.load_dataset(args.lerobot_repo, split="train")

    def add_switch_label(example, idx):
        """Map function: add switch_label based on closest collected label."""
        task_id = example.get("task_id", 0)
        episode_idx = example.get("episode_index", 0)
        frame_idx = example.get("frame_index", idx)

        key = (task_id, episode_idx)
        if key in label_lookup:
            entries = label_lookup[key]
            # Find closest frame
            closest = min(entries, key=lambda x: abs(x[0] - frame_idx))
            example["switch_label"] = closest[1]
        else:
            # No label collected for this episode → default to 0.0 (no switch)
            example["switch_label"] = 0.0

        return example

    logging.info("Adding switch_label column...")
    ds = ds.map(add_switch_label, with_indices=True)

    output_path = args.output_repo
    logging.info(f"Saving to {output_path}...")
    ds.save_to_disk(output_path)
    logging.info(f"Done. Dataset with switch_label saved to {output_path}")
    logging.info(
        "To train pi05 with integrated switch head, update the training config to "
        "load from this local dataset."
    )


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train switch head with Gemini labels (collect → train)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- collect ----
    p_collect = subparsers.add_parser(
        "collect", help="Run LIBERO rollouts with Gemini scoring and save labeled data"
    )
    p_collect.add_argument("--host", type=str, default="0.0.0.0")
    p_collect.add_argument("--port", type=int, default=8000)
    p_collect.add_argument("--resize_size", type=int, default=224)
    p_collect.add_argument("--replan_steps", type=int, default=5)
    p_collect.add_argument(
        "--task_suite_name", type=str, default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal",
                 "libero_10", "libero_90"],
    )
    p_collect.add_argument("--num_steps_wait", type=int, default=10)
    p_collect.add_argument("--num_trials_per_task", type=int, default=20)
    p_collect.add_argument(
        "--output_dir", type=str, default="data/switch_labels",
        help="Directory to save .npz files with switch labels",
    )
    p_collect.add_argument("--seed", type=int, default=7)
    p_collect.add_argument("--clip_len", type=int, default=20,
                           help="Number of recent frames per camera for video clips (default 20 = 2s)")

    # ---- train ----
    p_train = subparsers.add_parser(
        "train", help="Train standalone DINOv2-based switch head"
    )
    p_train.add_argument("--data_dir", type=str, required=True)
    p_train.add_argument("--val_dir", type=str, default=None)
    p_train.add_argument(
        "--output_dir", type=str, default="checkpoints/switch_head"
    )
    p_train.add_argument(
        "--dinov2_model", type=str, default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    )
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--freeze_backbone", action="store_true", default=True)
    p_train.add_argument(
        "--no_freeze_backbone", dest="freeze_backbone", action="store_false"
    )
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--save_every", type=int, default=5)

    # ---- export ----
    p_export = subparsers.add_parser(
        "export",
        help="Inject switch_label into LeRobot dataset for pi05 integrated training",
    )
    p_export.add_argument("--data_dir", type=str, required=True,
                          help="Directory of collected .npz files")
    p_export.add_argument("--lerobot_repo", type=str,
                          default="physical-intelligence/libero",
                          help="HuggingFace repo ID of the LIBERO dataset")
    p_export.add_argument("--output_repo", type=str,
                          default="data/libero_with_switch",
                          help="Local path to save the augmented dataset")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "collect":
        collect(args)
    elif args.command == "train":
        train(args)
    elif args.command == "export":
        export_for_pi05(args)


if __name__ == "__main__":
    main()
