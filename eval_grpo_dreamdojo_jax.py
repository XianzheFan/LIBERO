"""
Test-Time RL via GRPO (Group Relative Policy Optimization) with DreamDojo — JAX version.

Uses the native JAX/Flax NNX Pi0SDE model (pi05 by default) for both Flow-SDE
action sampling and gradient-based GRPO updates. No PyTorch dependency for the model.

Algorithm:
1. At each "Pause" (rescue trigger or periodic), sample n action chunks
   from the VLA via Flow-SDE (stochastic → diverse candidates).
2. For each candidate action, generate a future video via DreamDojo.
3. The VLM (Gemini) scores every generated future video with a reward.
4. Compute GRPO advantage: A_i = (r_i - mean(r)) / std(r)
5. Compute advantage-weighted flow-matching loss and update the VLA with
   nnx.value_and_grad + optax.
6. Execute the best-scoring action on the robot.
"""

import base64
import collections
import concurrent.futures
import dataclasses
import functools
import json
import logging
import math
import os
import pathlib
import shutil
import tempfile
import threading
import time as time_module

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import cv2
import numpy as np
import requests

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
from flax import traverse_util

import torch  # only for the patched torch.load workaround
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from openpi_client import image_tools
from pydantic import BaseModel
from google import genai
from google.genai import types
import tqdm
import tyro

import openpi.models.model as _model
import openpi.models.pi0_sde as _pi0_sde
import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.models.pi0_sde import Pi0SDEConfig
from openpi.shared import nnx_utils


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ROLLOUT_FPS = 10

GEMINI_QUERY_INTERVAL_FRAMES = 40
GEMINI_HISTORY_FRAMES = 200
GEMINI_VALUE_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_SCORE_MODEL = "gemini-3.1-flash-lite-preview"

RESCUE_SCORE_ABSOLUTE = 0.30
RESCUE_SCORE_DROP = 0.20

SDE_NOISE_LEVEL = 0.5
SDE_NUM_STEPS = 3

GRPO_NUM_SAMPLES = 4
GRPO_LR = 5e-5                # Larger LR since we only get ~5 updates per episode
GRPO_MAX_GRAD_NORM = 1.0
GRPO_NUM_FLOW_TIMES = 4       # Random timesteps to average flow loss over
GRPO_MIN_STD = 1e-4


class CandidateScore(BaseModel):
    reasoning: str
    score: float


class ValueEvaluation(BaseModel):
    reasoning: str
    score: float
    status: str


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _dreamdojo_generate(port: int, frame_np: np.ndarray, actions: np.ndarray,
                        save_name: str, task_description: str = "") -> str | None:
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
            time_module.sleep(2)
            file_info = client.files.get(name=video_file.name)
        if file_info.state.name == "FAILED":
            return {"step": step_idx, "error": "Video processing failed"}

        prompt = (
            f'You are a top-tier robot action evaluation expert responsible for constructing a '
            f'Dense Value Function for an RL model. '
            f'The robot is performing the task: "{task_description}". '
            f'Based on the provided video sequence, please evaluate the robot\'s state '
            f'**over the most recent 4s** and provide a **Value Score** between **0.00** and **1.00**.\n'
            f'Scoring: 0.00-0.20 Disengaged/Failure, 0.20-0.40 Approach, '
            f'0.40-0.60 Initial Interaction, 0.60-0.80 Critical Execution, 0.80-1.00 Completion.\n'
            f'Output strictly in **JSON array format** with reasoning, score, status.'
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
                logging.info(f"[Gemini Value] frame={step_idx} score={score:.2f}")
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
        logging.info(f"[Rescue] Triggered: drop {prev_score:.2f}->{latest_score:.2f}")
        return True
    return False


def _gemini_score_candidate(video_path: str, history_video_path: str,
                            task_description: str, candidate_idx: int) -> float:
    client = _get_gemini_client()
    history_file = None
    cand_file = None
    try:
        history_file = client.files.upload(file=history_video_path)
        cand_file = client.files.upload(file=video_path)
        for f in [history_file, cand_file]:
            info = client.files.get(name=f.name)
            while info.state.name == "PROCESSING":
                time_module.sleep(2)
                info = client.files.get(name=f.name)
            if info.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {f.name}")

        prompt = (
            f'You are a robot action evaluation expert. '
            f'The robot is performing: "{task_description}".\n'
            f'Evaluate how well the candidate future video progresses toward the goal.\n'
            f'Score 0.00-1.00. Output JSON with "reasoning" and "score".'
        )
        response = client.models.generate_content(
            model=GEMINI_SCORE_MODEL,
            contents=[
                prompt,
                "\n[History Video]:", history_file,
                "\n[Candidate Future Video]:", cand_file,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateScore,
                temperature=0.0,
            ),
        )
        result = json.loads(response.text)
        score = float(result.get("score", 0.0))
        logging.info(f"[GRPO Score] candidate={candidate_idx} score={score:.3f}")
        return score
    except Exception as e:
        logging.error(f"[GRPO Score] candidate={candidate_idx} error: {e}")
        return 0.0
    finally:
        for f in [history_file, cand_file]:
            if f is not None:
                try:
                    client.files.delete(name=f.name)
                except Exception:
                    pass


def _score_candidates_parallel(candidate_paths: list, history_video_path: str,
                               task_description: str) -> list[float]:
    scores = [0.0] * len(candidate_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidate_paths)) as ex:
        futures = {}
        for i, path in enumerate(candidate_paths):
            fut = ex.submit(_gemini_score_candidate, path, history_video_path, task_description, i)
            futures[fut] = i
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                scores[idx] = fut.result()
            except Exception:
                scores[idx] = 0.0
    return scores


def compute_grpo_advantages(rewards: list[float]) -> np.ndarray:
    r = np.array(rewards, dtype=np.float64)
    return ((r - r.mean()) / max(r.std(), GRPO_MIN_STD)).astype(np.float32)


def grpo_update_jax(
    model_def: nnx.GraphDef,
    params: nnx.State,
    opt_state: optax.OptState,
    tx: optax.GradientTransformation,
    trainable_filter: nnx.filterlib.Filter,
    observation: _model.Observation,
    action_chunks_jax: jnp.ndarray,       # (n, horizon, action_dim) — normalized
    advantages: jnp.ndarray,              # (n,)
    rng: jax.Array,
    num_flow_times: int = GRPO_NUM_FLOW_TIMES,
) -> tuple[nnx.State, optax.OptState, dict]:
    """
    Single GRPO gradient step on the JAX Pi0SDE model.

    For each candidate action a_i with advantage A_i, compute the flow-matching
    loss and weight by advantage:
        L = mean_i [ A_i * mean_t [ flow_loss(a_i, t) ] ]

    Uses nnx.value_and_grad with DiffState to only differentiate trainable params.
    """
    n = action_chunks_jax.shape[0]

    def loss_fn(model: _model.BaseModel, rng: jax.Array):
        total = jnp.float32(0.0)
        for t_idx in range(num_flow_times):
            step_rng = jax.random.fold_in(rng, t_idx)
            # compute_loss returns (n, action_horizon) — MSE per timestep
            per_step_loss = model.compute_loss(step_rng, observation, action_chunks_jax, train=False)
            # Mean over action horizon → (n,)
            per_candidate = jnp.mean(per_step_loss, axis=-1)
            # Advantage-weighted loss
            total = total + jnp.mean(advantages * per_candidate)
        return total / num_flow_times

    model = nnx.merge(model_def, params)
    model.train()

    diff_state = nnx.DiffState(0, trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rng)

    # Only update trainable params via optax
    trainable_params = params.filter(trainable_filter)
    updates, new_opt_state = tx.update(grads, opt_state, trainable_params)
    new_trainable = optax.apply_updates(trainable_params, updates)

    # Write updated trainable params back into the full model, then extract full params
    nnx.update(model, new_trainable)
    new_params = nnx.state(model)

    grad_norm = optax.global_norm(grads)

    stats = {
        "grpo_loss": float(loss),
        "grad_norm": float(grad_norm),
        "advantages": np.asarray(advantages).tolist(),
    }
    return new_params, new_opt_state, stats


@dataclasses.dataclass
class GRPOState:
    """Mutable container for the JAX model + optimizer state used during GRPO."""
    model_def: nnx.GraphDef
    params: nnx.State
    opt_state: optax.OptState
    tx: optax.GradientTransformation
    trainable_filter: nnx.filterlib.Filter
    rng: jax.Array
    policy: _policy.Policy  # wraps the same model for inference (transforms + SDE sampling)

    def update_policy_state(self):
        """Sync the Policy wrapper's internal model state after a GRPO update."""
        model = nnx.merge(self.model_def, self.params)
        model.eval()
        self.policy._model = model
        self.policy._sample_actions = nnx_utils.module_jit(model.sample_actions)


def _build_trainable_filter(is_pi05: bool) -> nnx.filterlib.Filter:
    """Build the trainable parameter filter based on model variant.

    Pi05 layers: action_in_proj, action_out_proj, time_mlp_in, time_mlp_out
    Pi0  layers: action_in_proj, action_out_proj, action_time_mlp_in, action_time_mlp_out, state_proj
    """
    if is_pi05:
        regex = r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
    else:
        regex = (
            r".*(action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out|state_proj).*"
        )
    return nnx.All(nnx.Param, nnx_utils.PathRegex(regex))


def _load_grpo_state(
    config_name: str,
    checkpoint_dir: str,
    sde_noise_level: float = SDE_NOISE_LEVEL,
    sde_num_steps: int = SDE_NUM_STEPS,
    lr: float = GRPO_LR,
    max_grad_norm: float = GRPO_MAX_GRAD_NORM,
    seed: int = 42,
) -> GRPOState:
    """Load JAX Pi0SDE model and set up optax optimizer for GRPO."""
    train_config = _config.get_config(config_name)

    # Override model config to Pi0SDEConfig (preserves pi05 flag from base config)
    original_model_config = train_config.model
    model_kwargs = dataclasses.asdict(original_model_config)
    sde_model_config = Pi0SDEConfig(
        **model_kwargs,
        noise_method="flow_sde",
        noise_level=sde_noise_level,
        num_steps=sde_num_steps,
    )
    train_config = dataclasses.replace(train_config, model=sde_model_config)

    # Load via create_trained_policy (JAX path)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

    model = policy._model  # Pi0SDE (Flax NNX module)
    model_def, params = nnx.split(model)

    # Optimizer: AdamW with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10),
    )

    # Select trainable params based on model variant
    is_pi05 = getattr(sde_model_config, "pi05", False)
    trainable_filter = _build_trainable_filter(is_pi05)
    opt_state = tx.init(params.filter(trainable_filter))

    n_params = sum(p.size for p in jax.tree.leaves(params))
    trainable_count = sum(p.size for p in jax.tree.leaves(params.filter(trainable_filter)))
    frozen_count = n_params - trainable_count
    logging.info(f"[GRPO] Loaded JAX Pi0SDE model (pi05={is_pi05}). Total params: {n_params:,}")
    logging.info(f"[GRPO] Trainable: {trainable_count:,} | Frozen: {frozen_count:,}")

    rng = jax.random.key(seed)

    return GRPOState(
        model_def=model_def,
        params=params,
        opt_state=opt_state,
        tx=tx,
        trainable_filter=trainable_filter,
        rng=rng,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# Core GRPO step: sample → generate videos → score → update → select best
# ---------------------------------------------------------------------------

def _grpo_rescue_and_update(
    obs, img, wrist_img,
    replay_images_for_history: list,
    task_description: str,
    grpo_state: GRPOState,
    replan_steps: int,
    step_save_dir: pathlib.Path,
    dd_base_port: int,
    num_samples: int = GRPO_NUM_SAMPLES,
    do_update: bool = True,
) -> tuple[list, dict]:
    step_save_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare normalized observation (shared for all samples) ---
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
        "prompt": str(task_description),
    }

    # Apply input transform once to get normalized observation
    inputs = jax.tree.map(lambda x: x, element)
    inputs = grpo_state.policy._input_transform(inputs)
    inputs_batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    obs_jax = _model.Observation.from_dict(inputs_batched)

    # --- Step 1: Sample n action chunks via Flow-SDE (single sampling) ---
    # Sample from model.sample_actions directly → get normalized actions
    # Then apply output_transform to get executable (denormalized) actions
    normalized_chunks = []  # for GRPO loss (model space)
    action_chunks = []      # for env execution (denormalized)

    for _ in range(num_samples):
        grpo_state.rng, sample_rng = jax.random.split(grpo_state.rng)
        model = nnx.merge(grpo_state.model_def, grpo_state.params)
        model.eval()
        raw_actions = model.sample_actions(sample_rng, obs_jax)
        # raw_actions: (1, action_horizon, action_dim) in normalized space
        normalized_chunks.append(raw_actions[0])  # remove batch dim → (horizon, action_dim)

        # Denormalize via output_transform to get env-executable actions
        output_dict = {
            "state": np.asarray(inputs_batched["state"][0]),
            "actions": np.asarray(raw_actions[0]),
        }
        output_dict = grpo_state.policy._output_transform(output_dict)
        action_chunks.append(output_dict["actions"])

    logging.info(f"[GRPO] Sampled {num_samples} action chunks via Flow-SDE (single pass)")

    # --- Step 2: Generate future videos via DreamDojo ---
    save_prefix = step_save_dir.name
    tasks = [
        {
            "port": dd_base_port + i,
            "actions": np.array(action_chunks[i][:replan_steps], dtype=np.float32),
            "save_name": f"{save_prefix}/chunk_{i}",
        }
        for i in range(num_samples)
    ]

    logging.info(f"[GRPO] Launching {num_samples} parallel DreamDojo requests...")

    def _submit(t):
        return _dreamdojo_generate(t["port"], img, t["actions"], t["save_name"], task_description)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as ex:
        futures = {ex.submit(_submit, t): i for i, t in enumerate(tasks)}
        save_paths = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            save_paths[idx] = fut.result()

    valid = [(i, save_paths[i]) for i in range(num_samples)
             if save_paths.get(i) and os.path.exists(save_paths[i])]

    if not valid:
        logging.warning("[GRPO] All DreamDojo generations failed; using chunk 0.")
        return list(action_chunks[0][:replan_steps]), {
            "num_candidates": 0, "error": "All DreamDojo generations failed",
            "grpo_update": None, "best_chunk_idx": 0,
        }

    history_video_path = str(step_save_dir / "history_video.mp4")
    imageio.mimwrite(
        history_video_path,
        [np.asarray(x) for x in replay_images_for_history[-GEMINI_HISTORY_FRAMES:]],
        fps=ROLLOUT_FPS,
    )

    local_paths = {}
    for orig_i, orig_path in valid:
        dst = step_save_dir / f"candidate_{orig_i}.mp4"
        try:
            shutil.copy2(orig_path, dst)
            local_paths[orig_i] = str(dst)
        except Exception as e:
            logging.warning(f"[GRPO] Could not copy {orig_path}->{dst}: {e}")
            local_paths[orig_i] = orig_path

    valid_indices = [i for i, _ in valid]
    valid_paths = [local_paths[i] for i in valid_indices]

    # --- Step 3: Score each candidate with Gemini ---
    logging.info(f"[GRPO] Scoring {len(valid_paths)} candidates with Gemini...")
    all_rewards = [0.0] * num_samples
    valid_rewards = _score_candidates_parallel(valid_paths, history_video_path, task_description)
    for vi, score in zip(valid_indices, valid_rewards):
        all_rewards[vi] = score

    logging.info(f"[GRPO] Rewards: {[f'{r:.3f}' for r in all_rewards]}")

    # --- Step 4: Compute GRPO advantages ---
    valid_rewards_arr = np.array(valid_rewards, dtype=np.float32)
    valid_advantages = compute_grpo_advantages(valid_rewards_arr.tolist())
    logging.info(f"[GRPO] Advantages: {[f'{a:.3f}' for a in valid_advantages]}")

    # --- Step 5: GRPO weight update ---
    grpo_stats = None
    if not do_update:
        logging.info("[GRPO] Skipping weight update (not yet at update interval).")
    elif len(valid_indices) >= 2:
        try:
            # Build batch of normalized actions for valid candidates: (n_valid, horizon, action_dim)
            valid_actions = jnp.stack([normalized_chunks[i] for i in valid_indices])

            # Replicate observation to match n_valid
            n_valid = len(valid_indices)
            obs_replicated = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (n_valid,) + x.shape[1:]),
                obs_jax,
            )

            advantages_jax = jnp.array(valid_advantages)

            grpo_state.rng, update_rng = jax.random.split(grpo_state.rng)

            new_params, new_opt_state, grpo_stats = grpo_update_jax(
                model_def=grpo_state.model_def,
                params=grpo_state.params,
                opt_state=grpo_state.opt_state,
                tx=grpo_state.tx,
                trainable_filter=grpo_state.trainable_filter,
                observation=obs_replicated,
                action_chunks_jax=valid_actions,
                advantages=advantages_jax,
                rng=update_rng,
            )
            grpo_state.params = new_params
            grpo_state.opt_state = new_opt_state

            # Sync policy wrapper with updated weights
            grpo_state.update_policy_state()

            logging.info(
                f"[GRPO] Update done: loss={grpo_stats['grpo_loss']:.4f} "
                f"grad_norm={grpo_stats['grad_norm']:.4f}"
            )
        except Exception as e:
            logging.error(f"[GRPO] Weight update failed: {e}", exc_info=True)
            grpo_stats = {"error": str(e)}
    else:
        logging.info("[GRPO] Skipping update: fewer than 2 valid candidates.")

    # --- Step 6: Select best action ---
    best_valid_idx = int(np.argmax(valid_rewards_arr))
    best_chunk_idx = valid_indices[best_valid_idx]
    logging.info(f"[GRPO] Best candidate: {best_chunk_idx} (reward={all_rewards[best_chunk_idx]:.3f})")

    record = {
        "num_candidates": len(valid),
        "valid_indices": valid_indices,
        "candidate_paths": valid_paths,
        "rewards": all_rewards,
        "advantages": valid_advantages.tolist(),
        "best_chunk_idx": int(best_chunk_idx),
        "grpo_update": grpo_stats,
    }
    return list(action_chunks[best_chunk_idx][:replan_steps]), record


def _write_gemini_results(gemini_results: list, rollout_dir: pathlib.Path,
                          task_description: str, episode_idx: int, suffix: str,
                          rescue_log: list | None = None,
                          grpo_records: list | None = None):
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

    json_path = rollout_dir / "gemini_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": task_description,
            "episode_index": episode_idx,
            "outcome": suffix,
            "value_evaluations": sorted(gemini_results, key=lambda x: x.get("step", 0)),
            "rescue_activations": rescue_log or [],
            "grpo_records": grpo_records or [],
        }, f, indent=2)


def _save_jax_checkpoint(params: nnx.State, path: pathlib.Path):
    """Save NNX params to an orbax checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(path / "params", {"params": params.to_pure_dict()})
    logging.info(f"[GRPO] Saved JAX checkpoint to {path}")


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_libero"
    checkpoint_dir: str = "checkpoints/pi05_libero"

    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20
    video_out_path: str = "data/libero/output_grpo_jax"
    seed: int = 7

    # --- DreamDojo ---
    dd_base_port: int = 8020

    # --- Flow-SDE ---
    sde_noise_level: float = SDE_NOISE_LEVEL
    sde_num_steps: int = SDE_NUM_STEPS

    grpo_num_samples: int = GRPO_NUM_SAMPLES
    grpo_lr: float = GRPO_LR
    grpo_max_grad_norm: float = GRPO_MAX_GRAD_NORM
    grpo_save_interval: int = 5              # Save checkpoint every N GRPO updates
    grpo_update_interval: int = 10           # Update policy weights every N rescues


def eval_libero_grpo(args: Args) -> None:
    np.random.seed(args.seed)

    logging.info("[GRPO] Loading JAX Pi0SDE model...")
    grpo_state = _load_grpo_state(
        args.config_name, args.checkpoint_dir,
        sde_noise_level=args.sde_noise_level,
        sde_num_steps=args.sde_num_steps,
        lr=args.grpo_lr,
        max_grad_norm=args.grpo_max_grad_norm,
        seed=args.seed,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps_map = {
        "libero_spatial": 220, "libero_object": 280,
        "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }
    max_steps = max_steps_map.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    total_episodes, total_successes = 0, 0
    total_grpo_updates = 0
    total_rescues = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            task_segment = task_description.replace(" ", "_")

            # Skip completed episodes
            existing_dirs = [
                d for d in pathlib.Path(args.video_out_path).glob(
                    f"rollout_{task_segment}_ep{episode_idx}_*"
                )
                if d.name.endswith("_success") or d.name.endswith("_failure")
            ]
            if existing_dirs:
                logging.info(f"Skip: {task_segment} (Episode {episode_idx})")
                if "success" in existing_dirs[0].name:
                    task_successes += 1
                    total_successes += 1
                task_episodes += 1
                total_episodes += 1
                continue

            rollout_dir = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_ep{episode_idx}_running"
            )
            rollout_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            clean_images = []

            score_history: list = []
            score_lock = threading.Lock()
            gemini_futures: list = []
            gemini_all_results: list = []
            gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            rescue_log: list = []
            grpo_records: list = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
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
                    num_frames = len(clean_images)

                    # Background Gemini value monitoring
                    if num_frames % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                        clip = list(clean_images[-GEMINI_HISTORY_FRAMES:])
                        future = gemini_executor.submit(
                            _query_gemini_value, clip, task_description,
                            num_frames, score_history, score_lock,
                        )
                        gemini_futures.append(future)

                    if not action_plan:
                        # Only intervene on rescue triggers, capped per episode
                        rescue = _check_rescue_needed(score_history, score_lock)

                        if rescue:
                            rescue_log.append(num_frames)
                            logging.info(
                                f"[GRPO] Rescue at frame {num_frames}"
                            )

                            total_rescues += 1
                            do_update = (total_rescues % args.grpo_update_interval == 0)

                            step_save_dir = rollout_dir / "grpo_steps" / f"frame{num_frames}"
                            best_actions, grpo_record = _grpo_rescue_and_update(
                                obs=obs,
                                img=img,
                                wrist_img=wrist_img,
                                replay_images_for_history=clean_images,
                                task_description=task_description,
                                grpo_state=grpo_state,
                                replan_steps=args.replan_steps,
                                step_save_dir=step_save_dir,
                                dd_base_port=args.dd_base_port,
                                num_samples=args.grpo_num_samples,
                                do_update=do_update,
                            )
                            grpo_record["frame"] = num_frames
                            grpo_records.append(grpo_record)
                            action_plan.extend(best_actions)
                            total_grpo_updates += 1

                            if (args.grpo_save_interval > 0
                                    and total_grpo_updates % args.grpo_save_interval == 0):
                                ckpt_path = (
                                    pathlib.Path(args.video_out_path)
                                    / f"grpo_checkpoint_step{total_grpo_updates}"
                                )
                                _save_jax_checkpoint(grpo_state.params, ckpt_path)
                        else:
                            # Normal VLA inference via Flow-SDE (uses GRPO-updated weights)
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate((
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )),
                                "prompt": str(task_description),
                            }
                            action_chunk = grpo_state.policy.infer(element)["actions"]
                            assert len(action_chunk) >= args.replan_steps
                            action_plan.extend(action_chunk[:args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(
                        action.tolist() if hasattr(action, 'tolist') else list(action)
                    )
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}", exc_info=True)
                    break

            gemini_executor.shutdown(wait=True)
            for future in gemini_futures:
                try:
                    gemini_all_results.append(future.result())
                except Exception as e:
                    gemini_all_results.append({"error": str(e)})

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            final_rollout_dir = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            )
            rollout_dir.rename(final_rollout_dir)
            rollout_dir = final_rollout_dir

            _write_gemini_results(
                gemini_all_results, rollout_dir, task_description, episode_idx, suffix,
                rescue_log=rescue_log, grpo_records=grpo_records,
            )

            imageio.mimwrite(
                rollout_dir / "complete_video.mp4",
                [np.asarray(x) for x in clean_images],
                fps=ROLLOUT_FPS,
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes: {total_episodes}, successes: {total_successes} "
                         f"({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"Total GRPO updates: {total_grpo_updates}")

        logging.info(f"Task success rate: {task_successes / task_episodes:.3f}")
        logging.info(f"Total success rate: {total_successes / total_episodes:.3f}")

    # Save final model
    final_ckpt = pathlib.Path(args.video_out_path) / "grpo_final_checkpoint"
    _save_jax_checkpoint(grpo_state.params, final_ckpt)

    logging.info(f"Total success rate: {total_successes / total_episodes:.3f}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total GRPO updates: {total_grpo_updates}")


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero_grpo(tyro.cli(Args))
