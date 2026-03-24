"""
Multimodal Ambiguity Evaluation + Test-Time Evolution (TTE).

Extends multimodal_ambiguity_4traj.py: instead of sampling 4 random action
chunks and blindly executing the first one, we run CEM evolution over
a population of candidates using MuJoCo forward simulation + fitness scoring.
The top-4 elites are drawn as the 4 trajectory visualisations, and the
overall best is executed.

This combats position-memorisation overfitting: even when the policy's mode
goes to the wrong location, CEM steers selection toward the trajectory
that actually approaches the correct target object.

Usage:
    # 1. Start the SDE server (high noise_level for diversity):
    python scripts/serve_sde_policy.py --env libero --noise_level 1.0 --num_steps 3

    # 2. Run with TTE + perturbations:
    python third_party/libero/multimodal_ambiguity_4traj_tte.py \
        --task_suite_name libero_10 \
        --perturbations object_swap_target \
        --population_size 16 --generations 3 --elite_ratio 0.25 \
        --fitness_mode oracle

    # 3. Run baseline (no perturbation, still uses TTE):
    python third_party/libero/multimodal_ambiguity_4traj_tte.py \
        --perturbations none --population_size 8 --generations 2

    # 4. Compare against vanilla (disable TTE by setting generations=0):
    python third_party/libero/multimodal_ambiguity_4traj_tte.py \
        --perturbations object_swap_target --generations 0
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Any

import cv2
import imageio
import numpy as np
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
)
import torch
import tqdm
import tyro

# Patch torch.load for compatibility
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# ---- Re-use everything from the original 4traj script ----
from multimodal_ambiguity_4traj import (
    SceneSwapPerturbation,
    ObstaclePerturbation,
    ObjectSwapPerturbation,
    OcclusionPerturbation,
    PerturbationType,
    LanguagePerturbationType,
    perturb_language,
    draw_trajectory_on_image,
    draw_obstacles_on_image,
    TRAJ_COLORS,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _MAX_STEPS,
    _get_libero_env,
    _quat2axisangle,
)

logger = logging.getLogger(__name__)


# =========================================================================
# TTE: Test-Time Evolution (CEM in action space)
# =========================================================================

# --- Target-body detection for oracle fitness ---

_TARGET_KEYWORDS = [
    "moka_pot", "moka", "frypan", "pan", "bowl", "mug", "plate",
    "butter", "cream_cheese", "tomato_sauce", "chocolate_pudding",
    "alphabet_soup", "book", "basket", "caddy", "ketchup",
]


def _detect_target_body(env, instruction: str) -> str | None:
    """Find the MuJoCo body whose name best matches the instruction (first target)."""
    bodies = _detect_target_bodies(env, instruction)
    return bodies[0] if bodies else None


def _detect_target_bodies(env, instruction: str) -> list[str]:
    """Find all MuJoCo bodies whose names match keywords in the instruction.

    Returns a list of (body_name) sorted by keyword length descending,
    deduplicated by keyword so that e.g. 'alphabet_soup' doesn't also
    produce a match for 'soup' on the same body.
    """
    inner = env.env if hasattr(env, "env") else env
    model = inner.sim.model
    instr = instruction.lower().replace(" ", "_")

    # Collect all (keyword_length, body_name, keyword) matches
    matches: list[tuple[int, str, str]] = []
    for i in range(model.nbody):
        name = model.body_id2name(i)
        if not name:
            continue
        nl = name.lower()
        for kw in _TARGET_KEYWORDS:
            if kw in nl and kw in instr:
                matches.append((len(kw), name, kw))

    # Sort by keyword length descending, deduplicate by body name
    matches.sort(key=lambda x: -x[0])
    seen_bodies: set[str] = set()
    result: list[str] = []
    for _, body_name, _ in matches:
        if body_name not in seen_bodies:
            seen_bodies.add(body_name)
            result.append(body_name)
    return result


def _check_target_grasped(
    env, target_body: str, obs: dict,
    grasp_threshold: float = 0.08,
) -> bool:
    """Check if the EEF is near the target with gripper closed."""
    eef_pos = obs["robot0_eef_pos"]
    gripper_qpos = obs["robot0_gripper_qpos"]
    gripper_closed = np.mean(gripper_qpos) < 0.04
    pos = _body_pos(env, target_body)
    dist = np.linalg.norm(eef_pos - pos)
    return dist < grasp_threshold and gripper_closed


def _check_near_with_gripper_open(
    env, target_body: str, obs: dict,
    near_threshold: float = 0.12,
) -> bool:
    """Check if the EEF is near the target with gripper open (released)."""
    eef_pos = obs["robot0_eef_pos"]
    gripper_qpos = obs["robot0_gripper_qpos"]
    gripper_open = np.mean(gripper_qpos) > 0.04
    pos = _body_pos(env, target_body)
    dist = np.linalg.norm(eef_pos - pos)
    return dist < near_threshold and gripper_open


# Destination keywords — these are where objects get placed, not picked up
_DESTINATION_KEYWORDS = ["basket", "caddy", "plate", "bowl"]


def _classify_targets(
    target_bodies: list[str], instruction: str,
) -> tuple[list[str], str | None]:
    """Split detected bodies into pick targets and a destination.

    Returns:
        pick_targets: bodies to pick up (ordered)
        destination:  body to place objects into, or None
    """
    instr = instruction.lower().replace(" ", "_")
    pick_targets: list[str] = []
    destination: str | None = None

    for body in target_bodies:
        bl = body.lower()
        is_dest = any(kw in bl for kw in _DESTINATION_KEYWORDS)
        # Also check if the instruction uses "in the <dest>" pattern
        if is_dest:
            destination = body
        else:
            pick_targets.append(body)

    return pick_targets, destination


def _body_pos(env, body_name: str) -> np.ndarray:
    inner = env.env if hasattr(env, "env") else env
    bid = inner.sim.model.body_name2id(body_name)
    return inner.sim.data.body_xpos[bid].copy()


def compute_oracle_fitness(
    env,
    obs_seq: list[dict],
    actions: np.ndarray,
    target_body: str,
    w_distance: float = 1.0,
    w_grasp: float = 2.0,
    w_smooth: float = 0.1,
    return_breakdown: bool = False,
    phase: str = "pick",
) -> float | tuple[float, dict]:
    """Privileged fitness: closer to target = higher score.

    *phase* controls the grasp/release bonus:
      - "pick": rewards closing gripper near the target.
      - "place": rewards opening gripper near the destination.
    """
    if not obs_seq or not target_body:
        if return_breakdown:
            return 0.0, {}
        return 0.0

    target_pos = _body_pos(env, target_body)
    eef_start = obs_seq[0]["robot0_eef_pos"]
    eef_end = obs_seq[-1]["robot0_eef_pos"]

    d_start = np.linalg.norm(eef_start - target_pos)
    d_end = np.linalg.norm(eef_end - target_pos)

    # Getting closer
    f_dist = (d_start - d_end) * w_distance
    # Continuous proximity shaping: stronger reward as EEF gets closer
    f_prox = 1.0 / (d_end + 0.01) - 1.0 / (d_start + 0.01)
    # Grasp / release bonus
    f_grasp = 0.0
    if phase == "pick":
        # Reward closing gripper near target
        if d_end < 0.08 and len(actions) > 0 and actions[-1, -1] > 0:
            f_grasp = w_grasp
    else:
        # Place phase: reward opening gripper near destination
        if d_end < 0.12 and len(actions) > 0 and actions[-1, -1] < 0:
            f_grasp = w_grasp
    # Smoothness penalty
    f_smooth = 0.0
    if len(actions) > 1:
        jerk = np.mean(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1))
        f_smooth = -jerk * w_smooth

    fit = f_dist + f_prox + f_grasp + f_smooth

    if return_breakdown:
        return fit, {
            "d_start": float(d_start),
            "d_end": float(d_end),
            "f_dist": float(f_dist),
            "f_prox": float(f_prox),
            "f_grasp": float(f_grasp),
            "f_smooth": float(f_smooth),
            "phase": phase,
        }
    return fit


def compute_vlm_fitness(
    vlm_model,
    obs_seq: list[dict],
    instruction: str,
) -> float:
    """VLM-based fitness (requires external vlm_model)."""
    if vlm_model is None or not obs_seq:
        return 0.0
    img = obs_seq[-1].get("agentview_image")
    if img is None:
        return 0.0
    return vlm_model.predict_progress_value(img, instruction)


# --- Forward simulation ---

def _get_inner_env(env):
    """Return the underlying robosuite env that owns timestep/done."""
    return env.env if hasattr(env, "env") else env


def _forward_simulate(
    env, actions: np.ndarray, saved_state: np.ndarray,
    record_images: bool = False,
) -> tuple[list[dict], list[np.ndarray]]:
    """Rollout actions from a saved state.

    Saves and restores the environment's internal timestep and done flag
    so that forward simulation does not corrupt the main episode counter.

    Returns:
        obs_list: list of observations (len = steps+1).
        frames:   list of RGB images if *record_images* is True, else [].
    """
    inner = _get_inner_env(env)
    saved_timestep = inner.timestep
    saved_done = inner.done

    obs = env.set_init_state(saved_state)
    # Reset counter for clean simulation
    inner.timestep = 0
    inner.done = False

    obs_list = [obs]
    frames: list[np.ndarray] = []
    if record_images:
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
    for a in actions:
        obs, _, done, _ = env.step(a.tolist())
        obs_list.append(obs)
        if record_images:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        if done:
            break

    # Restore original counters
    inner.timestep = saved_timestep
    inner.done = saved_done
    return obs_list, frames


# --- CEM evolver ---

@dataclasses.dataclass
class TTEConfig:
    population_size: int = 16
    generations: int = 3
    elite_ratio: float = 0.25
    mutation_std: float = 0.35
    mutation_decay: float = 0.85
    fresh_ratio: float = 0.25
    crossover_prob: float = 0.3
    sim_horizon: int = 10
    fitness_mode: str = "oracle"        # "oracle" | "vlm" | "composite"
    record_evolution: bool = False       # save per-generation rollout frames
    early_stop_best: float = 5.0        # stop CEM when best fitness exceeds this (meaningful progress)
    # Oracle weights
    w_distance: float = 1.0
    w_grasp: float = 2.0
    w_smooth: float = 0.1
    w_vlm: float = 1.0


class TestTimeEvolver:
    def __init__(
        self,
        client: _websocket_client_policy.WebsocketClientPolicy,
        cfg: TTEConfig,
        vlm_model: Any = None,
    ):
        self.client = client
        self.cfg = cfg
        self.vlm_model = vlm_model
        # Warm-start: carry elites from previous replan step
        self._prev_elites: list[np.ndarray] = []
        self._prev_replan_steps: int = 0  # how many steps were executed
        # Pick-and-place state machine
        self._pick_targets: list[str] = []   # objects to pick (ordered)
        self._destination: str | None = None  # where to place them
        self._current_pick_idx: int = 0       # which object we're on
        self._phase: str = "pick"             # "pick" or "place"

    def reset_warm_start(self):
        """Clear warm-start and target tracking state (call at the start of each episode)."""
        self._prev_elites = []
        self._prev_replan_steps = 0
        self._pick_targets = []
        self._destination = None
        self._current_pick_idx = 0
        self._phase = "pick"

    # -- helpers --

    @staticmethod
    def _build_element(obs: dict, prompt: str, resize: int = 224) -> dict:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
        return {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": np.concatenate([
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            ]),
            "prompt": prompt,
        }

    def _sample_n(self, element: dict, n: int) -> list[np.ndarray]:
        return [np.array(self.client.infer(element)["actions"]) for _ in range(n)]

    def _fitness(
        self, env, candidate: np.ndarray, saved_state: np.ndarray,
        instruction: str, target_body: str | None,
        return_breakdown: bool = False,
    ) -> float | tuple[float, dict]:
        c = self.cfg
        acts = candidate[: c.sim_horizon]
        obs_seq, _ = _forward_simulate(env, acts, saved_state)

        score = 0.0
        breakdown = {}
        if c.fitness_mode in ("oracle", "composite") and target_body:
            if return_breakdown:
                s, breakdown = compute_oracle_fitness(
                    env, obs_seq, acts, target_body,
                    c.w_distance, c.w_grasp, c.w_smooth,
                    return_breakdown=True,
                    phase=self._phase,
                )
                score += s
            else:
                score += compute_oracle_fitness(
                    env, obs_seq, acts, target_body,
                    c.w_distance, c.w_grasp, c.w_smooth,
                    phase=self._phase,
                )
        if c.fitness_mode in ("vlm", "composite"):
            score += c.w_vlm * compute_vlm_fitness(
                self.vlm_model, obs_seq, instruction,
            )
        if return_breakdown:
            return score, breakdown
        return score

    @staticmethod
    def _mutate(chunk: np.ndarray, std: float) -> np.ndarray:
        m = chunk + np.random.randn(*chunk.shape) * std
        m[:, :6] = np.clip(m[:, :6], -1.0, 1.0)
        m[:, 6] = np.clip(m[:, 6], -1.0, 1.0)
        return m

    @staticmethod
    def _crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = np.random.rand(a.shape[0]) < 0.5
        c = a.copy()
        c[mask] = b[mask]
        return c

    # -- main entry --

    def evolve(
        self,
        env,
        obs: dict,
        prompt: str,
        replan_steps: int = 5,
        resize: int = 224,
        replan_step_idx: int = -1,
    ) -> tuple[np.ndarray, list[np.ndarray], dict]:
        """Run CEM evolution.

        Returns:
            best_actions:  (action_horizon, action_dim) — the winning trajectory.
            top4_actions:  list of 4 np.ndarray — the top-4 elites (for visualisation).
            meta:          dict with per-generation stats.
        """
        c = self.cfg

        # Place phase: reduce CEM intensity — forward sim with grasped
        # objects is unreliable, so use only 1 generation for light filtering
        effective_generations = c.generations
        if self._phase == "place" and c.generations > 1:
            effective_generations = 1
            logger.info("  TTE place phase: reducing to 1 CEM generation")

        # If generations==0, fall back to vanilla (sample one, return it)
        if effective_generations <= 0:
            element = self._build_element(obs, prompt, resize)
            chunks = self._sample_n(element, 4)
            return chunks[0], chunks, {"generations": [], "best_fitness": None}

        saved_state = env.get_sim_state()
        inner = _get_inner_env(env)
        saved_timestep = inner.timestep
        saved_done = inner.done

        # Detect target bodies for oracle fitness (multi-target support)
        target_body = None
        target_bodies: list[str] = []
        if c.fitness_mode in ("oracle", "composite"):
            target_bodies = _detect_target_bodies(env, prompt)

            # Initialize pick/place state on first call
            if not self._pick_targets and not self._destination:
                self._pick_targets, self._destination = _classify_targets(target_bodies, prompt)
                logger.info(
                    f"  TTE pick targets: {self._pick_targets}, "
                    f"destination: {self._destination}"
                )

            # State machine transitions
            prev_phase = self._phase
            if self._phase == "pick" and self._current_pick_idx < len(self._pick_targets):
                current_obj = self._pick_targets[self._current_pick_idx]
                if _check_target_grasped(env, current_obj, obs):
                    if self._destination:
                        self._phase = "place"
                        logger.info(
                            f"  TTE '{current_obj}' grasped → switching to place phase, "
                            f"target: {self._destination}"
                        )
                    else:
                        # No destination, just move to next object
                        self._current_pick_idx += 1
                        logger.info(f"  TTE '{current_obj}' grasped → next object")
            elif self._phase == "place" and self._destination:
                if _check_near_with_gripper_open(env, self._destination, obs):
                    placed_obj = self._pick_targets[self._current_pick_idx]
                    self._current_pick_idx += 1
                    self._phase = "pick"
                    logger.info(
                        f"  TTE '{placed_obj}' placed at '{self._destination}' → "
                        f"pick phase, obj_idx={self._current_pick_idx}"
                    )

            # Clear warm-start on phase transition (old elites are harmful)
            if self._phase != prev_phase:
                self._prev_elites = []
                logger.info(f"  TTE phase changed {prev_phase} → {self._phase}, cleared warm-start")

            # Determine current target body
            if self._phase == "pick" and self._current_pick_idx < len(self._pick_targets):
                target_body = self._pick_targets[self._current_pick_idx]
            elif self._phase == "place" and self._destination:
                target_body = self._destination
            elif self._destination:
                target_body = self._destination  # all picked, head to destination
            elif target_bodies:
                target_body = target_bodies[-1]

        if target_body:
            logger.info(
                f"TTE replan_step={replan_step_idx} phase={self._phase} "
                f"target body: {target_body} (picks: {self._pick_targets}, dest: {self._destination})"
            )

        element = self._build_element(obs, prompt, resize)

        # Warm-start: shift previous elites by removing executed steps
        warm_candidates: list[np.ndarray] = []
        if self._prev_elites and self._prev_replan_steps > 0:
            for elite in self._prev_elites:
                shifted = elite[self._prev_replan_steps:]
                if len(shifted) > 0:
                    # Pad with zeros at the end to maintain action horizon
                    pad = np.zeros((self._prev_replan_steps, elite.shape[1]))
                    warm_candidates.append(np.concatenate([shifted, pad], axis=0))
            if warm_candidates:
                logger.info(f"  TTE warm-start: injecting {len(warm_candidates)} shifted elites from previous step")

        # Initial population: warm candidates + fresh policy samples
        n_fresh_init = c.population_size - len(warm_candidates)
        population = warm_candidates + self._sample_n(element, max(1, n_fresh_init))

        n_elite = max(1, int(c.population_size * c.elite_ratio))
        mut_std = c.mutation_std

        meta: dict[str, Any] = {"generations": [], "target_body": target_body, "all_targets": target_bodies, "replan_step": replan_step_idx}
        # Per-generation evolution frames: list of (gen_idx, candidate_rank, frames)
        evolution_records: list[dict] = []
        best_overall = population[0]
        best_overall_fit = -float("inf")

        for gen in range(effective_generations):
            # Evaluate
            fits = np.array([
                self._fitness(env, cand, saved_state, prompt, target_body)
                for cand in population
            ])
            order = np.argsort(fits)[::-1]

            gen_best = fits[order[0]]
            if gen_best > best_overall_fit:
                best_overall_fit = gen_best
                best_overall = population[order[0]].copy()

            # Get breakdown for the best candidate
            _, best_breakdown = self._fitness(
                env, population[order[0]], saved_state, prompt, target_body,
                return_breakdown=True,
            )

            gen_std = float(fits.std())
            gen_mean = float(fits.mean())
            meta["generations"].append({
                "best": float(gen_best),
                "mean": gen_mean,
                "std": gen_std,
                "best_breakdown": best_breakdown,
            })
            bd = best_breakdown
            logger.info(
                f"  TTE replan_step={replan_step_idx} gen {gen}: best={gen_best:.4f}  "
                f"mean={gen_mean:.4f}  std={gen_std:.4f}  mut_std={mut_std:.4f}  "
                f"[d={bd.get('d_end', 0):.4f} dist={bd.get('f_dist', 0):.4f} "
                f"prox={bd.get('f_prox', 0):.4f} grasp={bd.get('f_grasp', 0):.1f}]"
            )

            # Early stopping: best fitness already good enough
            if gen > 0 and gen_best > c.early_stop_best:
                logger.info(
                    f"  TTE early stop at gen {gen}: best={gen_best:.4f} > {c.early_stop_best}"
                )
                break

            # Record forward-sim frames for top candidates
            if c.record_evolution:
                n_record = min(4, len(population))
                for rank in range(n_record):
                    idx = order[rank]
                    acts = population[idx][:c.sim_horizon]
                    _, frames = _forward_simulate(
                        env, acts, saved_state, record_images=True,
                    )
                    evolution_records.append({
                        "gen": gen,
                        "rank": rank,
                        "fitness": float(fits[idx]),
                        "frames": frames,
                    })
                env.set_init_state(saved_state)

            # Elites
            elites = [population[i].copy() for i in order[:n_elite]]

            # Adaptive mutation: scale up when fitness is poor
            # With proximity shaping, best < 1.0 means barely approaching
            adaptive_boost = max(1.0, min(3.0, 1.0 / (abs(gen_best) + 0.1))) if gen_best < 1.0 else 1.0
            effective_mut_std = mut_std * adaptive_boost

            # Adaptive fresh ratio: inject more fresh samples when fitness is poor
            if gen_best < 0.5:
                n_fresh = max(1, int(c.population_size * 0.5))
            else:
                n_fresh = max(1, int(c.population_size * c.fresh_ratio))

            # Build next generation
            nxt: list[np.ndarray] = list(elites)  # keep elites

            n_mutants = c.population_size - n_elite - n_fresh
            for _ in range(n_mutants):
                p = elites[np.random.randint(len(elites))]
                if c.crossover_prob > 0 and len(elites) > 1 and np.random.rand() < c.crossover_prob:
                    q = elites[np.random.randint(len(elites))]
                    child = self._crossover(p, q)
                else:
                    child = p.copy()
                nxt.append(self._mutate(child, effective_mut_std))

            nxt.extend(self._sample_n(element, n_fresh))
            population = nxt[: c.population_size]
            mut_std *= c.mutation_decay

        # Record forward-sim video of the best candidate for this replan step
        best_sim_acts = best_overall[:c.sim_horizon]
        _, best_frames = _forward_simulate(
            env, best_sim_acts, saved_state, record_images=True,
        )
        meta["best_frames"] = best_frames

        # Restore env to the state before evolution (physics + counters)
        env.set_init_state(saved_state)

        # Collect top-4 for visualisation (re-evaluate final population)
        fits_final = np.array([
            self._fitness(env, cand, saved_state, prompt, target_body)
            for cand in population
        ])
        env.set_init_state(saved_state)
        # Restore counters to what they were before evolve() was called
        inner.timestep = saved_timestep
        inner.done = saved_done

        order_final = np.argsort(fits_final)[::-1]
        top4 = [population[order_final[i]].copy() for i in range(min(4, len(population)))]

        # Best overall might be from an earlier generation
        if best_overall_fit > fits_final[order_final[0]]:
            top4[0] = best_overall.copy()

        meta["best_fitness"] = float(best_overall_fit)
        meta["evolution_records"] = evolution_records

        # Save elites for warm-starting next replan step
        n_save = max(1, int(c.population_size * c.elite_ratio))
        self._prev_elites = [population[order_final[i]].copy() for i in range(min(n_save, len(population)))]
        self._prev_replan_steps = replan_steps

        return best_overall, top4, meta


# =========================================================================
# CLI args (extends original 4traj Args with TTE params)
# =========================================================================

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20

    perturbations: list[str] = dataclasses.field(default_factory=lambda: ["none"])

    num_samples: int = 4  # kept for compatibility (top-4 visualised)

    # --- TTE parameters ---
    population_size: int = 16
    generations: int = 3
    elite_ratio: float = 0.25
    mutation_std: float = 0.35
    mutation_decay: float = 0.85
    fresh_ratio: float = 0.25
    crossover_prob: float = 0.3
    sim_horizon: int = 10
    fitness_mode: str = "oracle"
    record_evolution: bool = False  # save per-generation CEM rollout videos
    early_stop_best: float = 5.0     # stop CEM early when best fitness exceeds this
    w_distance: float = 1.0
    w_grasp: float = 2.0
    w_smooth: float = 0.1
    w_vlm: float = 1.0

    video_out_path: str = "data/libero/tte_ambiguity_videos"
    results_out_path: str = "data/libero/tte_ambiguity_results.json"

    seed: int = 7


# =========================================================================
# Main evaluation loop
# =========================================================================

def eval_with_tte(args: Args) -> None:
    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    # Parse perturbations
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

    logging.info(f"Active perturbations: {[p.value for p in active_perturbations]}")

    # Perturbation objects
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

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = _MAX_STEPS[args.task_suite_name]

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # TTE evolver
    tte_cfg = TTEConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_ratio=args.elite_ratio,
        mutation_std=args.mutation_std,
        mutation_decay=args.mutation_decay,
        fresh_ratio=args.fresh_ratio,
        crossover_prob=args.crossover_prob,
        sim_horizon=args.sim_horizon,
        fitness_mode=args.fitness_mode,
        record_evolution=args.record_evolution,
        early_stop_best=args.early_stop_best,
        w_distance=args.w_distance,
        w_grasp=args.w_grasp,
        w_smooth=args.w_smooth,
        w_vlm=args.w_vlm,
    )
    evolver = TestTimeEvolver(client, tte_cfg)

    all_results: list[dict] = []
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes", leave=False):
            episode_meta: dict[str, Any] = {
                "task_id": task_id,
                "episode_idx": episode_idx,
                "original_instruction": task_description,
                "perturbations_applied": [],
                "tte_meta": [],
            }

            if obstacle is not None:
                obstacle.deactivate(env)

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Controller action scale (for trajectory drawing)
            mujoco_robot = env.env.robots[0]
            ctrl_config = mujoco_robot.controller_config
            if isinstance(ctrl_config, dict) and "output_max" in ctrl_config:
                action_scale = ctrl_config["output_max"][0]
            else:
                action_scale = 0.05

            obstacle_info = []
            # Apply vision perturbations
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

            action_plan = collections.deque()
            t = 0
            replan_step_counter = 0
            evolver.reset_warm_start()
            replay_images_single = []
            replay_images_multi = []
            done = False

            camera_name = "agentview"
            mujoco_sim = env.env.sim

            logging.info(f"\nTask: {task_description} | Prompt: {prompt}")

            while t < max_steps + args.num_steps_wait:
                try:
                    K = get_camera_intrinsic_matrix(
                        sim=mujoco_sim, camera_name=camera_name,
                        camera_height=LIBERO_ENV_RESOLUTION, camera_width=LIBERO_ENV_RESOLUTION,
                    )
                    E = get_camera_extrinsic_matrix(sim=mujoco_sim, camera_name=camera_name)

                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Preprocess images
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if obstacle_info:
                        img = draw_obstacles_on_image(
                            img, obstacle_info, K, E,
                            orig_res=LIBERO_ENV_RESOLUTION, target_res=args.resize_size,
                        )

                    replay_images_single.append(img.copy())
                    replay_images_multi.append(img.copy())

                    if not action_plan:
                        # ============ TTE replaces vanilla sampling ============
                        best_actions, top4_actions, tte_meta = evolver.evolve(
                            env, obs, prompt,
                            replan_steps=args.replan_steps,
                            resize=args.resize_size,
                            replan_step_idx=replan_step_counter,
                        )
                        episode_meta["tte_meta"].append(tte_meta)

                        # Save per-replan-step video of the best candidate's forward sim
                        best_frames = tte_meta.get("best_frames", [])
                        if best_frames:
                            step_video_path = (
                                pathlib.Path(args.video_out_path)
                                / f"{task_description.replace(' ', '_')}_ep{episode_idx}_replan{replan_step_counter}.mp4"
                            )
                            imageio.mimwrite(str(step_video_path), best_frames, fps=10)

                        replan_step_counter += 1

                        # --- Draw top-4 evolved trajectories (same viz as original) ---
                        img_multi = img.copy()
                        for i, chunk in enumerate(top4_actions[:4]):
                            line_c, point_c = TRAJ_COLORS[i % len(TRAJ_COLORS)]

                            img_single_traj = draw_trajectory_on_image(
                                img=img, current_eef_pos=obs["robot0_eef_pos"],
                                action_chunk=chunk[:args.replan_steps],
                                K=K, E=E,
                                orig_res=LIBERO_ENV_RESOLUTION, target_res=args.resize_size,
                                action_scale=action_scale,
                                line_color=line_c, point_color=point_c,
                            )
                            img_multi = draw_trajectory_on_image(
                                img=img_multi, current_eef_pos=obs["robot0_eef_pos"],
                                action_chunk=chunk[:args.replan_steps],
                                K=K, E=E,
                                orig_res=LIBERO_ENV_RESOLUTION, target_res=args.resize_size,
                                action_scale=action_scale,
                                line_color=line_c, point_color=point_c,
                            )
                            # First trajectory (best elite) for single-traj video
                            if i == 0:
                                replay_images_single[-1] = img_single_traj

                        replay_images_multi[-1] = img_multi

                        # Execute the best evolved trajectory
                        action_plan.extend(best_actions[:args.replan_steps])
                        # =======================================================

                    action = action_plan.popleft()

                    # Obstacle collision check
                    if obstacle_info:
                        eef_pos = obs["robot0_eef_pos"]
                        if ObstaclePerturbation.check_collision(eef_pos, obstacle_info):
                            episode_meta.setdefault("obstacle_collisions", 0)
                            episode_meta["obstacle_collisions"] += 1
                            action = np.array(LIBERO_DUMMY_ACTION)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            episode_meta["success"] = bool(done)
            episode_meta["steps"] = t
            all_results.append(episode_meta)

            # ---- Save replay videos ----
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            perturb_tag = "_".join(
                m.get("perturbation", m.get("type", "unknown"))
                for m in episode_meta["perturbations_applied"]
            ) or "baseline"

            # Build detailed perturbation description (match 4traj style)
            perturb_details = []
            for m in episode_meta["perturbations_applied"]:
                ptype = m.get("perturbation", m.get("type", "unknown"))
                if ptype == "scene_swap":
                    preset = m.get("preset", "?")
                    perturb_details.append(
                        f"scene_swap: preset={preset}"
                    )
                elif ptype == "obstacle":
                    obs_names = [o["name"] for o in m.get("obstacles", [])]
                    n_geoms = m.get("geoms_activated", 0)
                    perturb_details.append(
                        f"obstacle: {obs_names} ({n_geoms} geoms)"
                    )
                elif ptype == "occlusion":
                    direction = m.get("direction", "?")
                    orig_fov = m.get("original_fov", "?")
                    perturb_details.append(
                        f"occlusion: dir={direction}, fov {orig_fov}->{max(30.0, float(orig_fov) - 10.0) if isinstance(orig_fov, (int, float)) else '?'}"
                    )
                elif ptype in ("object_swap_target", "object_swap_nontarget"):
                    swapped = m.get("swapped", [])
                    involve = m.get("involve_target", None)
                    label = "target" if involve else "non-target"
                    perturb_details.append(
                        f"obj_swap({label}): {swapped[0]} <-> {swapped[1]}"
                        if len(swapped) == 2 else f"obj_swap({label})"
                    )
                elif ptype == "misidentify":
                    replaced = m.get("replaced", m.get("swapped_words", {}))
                    perturbed = m.get("perturbed", "")
                    perturb_details.append(
                        f"misidentify: {replaced} => \"{perturbed}\""
                    )
                elif ptype == "ambiguous":
                    perturbed = m.get("perturbed", "")
                    perturb_details.append(
                        f"ambiguous: \"{perturbed}\""
                    )
                elif ptype == "unfamiliar":
                    perturbed = m.get("perturbed", "")
                    perturb_details.append(
                        f"unfamiliar: \"{perturbed}\""
                    )
                else:
                    perturb_details.append(ptype)
            perturb_detail_str = " | ".join(perturb_details) if perturb_details else "baseline"

            video_size = 512
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            line_height = 14
            margin_x = 4

            def _wrap_text(text, _font, _font_scale, max_width):
                words = text.split()
                lines, current = [], ""
                for word in words:
                    test = f"{current} {word}".strip()
                    tw = cv2.getTextSize(test, _font, _font_scale, 1)[0][0]
                    if tw > max_width and current:
                        lines.append(current)
                        current = word
                    else:
                        current = test
                if current:
                    lines.append(current)
                return lines

            def _annotate_frames(frames):
                annotated = []
                for frame in frames:
                    f_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                    f_bgr = cv2.resize(f_bgr, (video_size, video_size),
                                       interpolation=cv2.INTER_LANCZOS4)
                    text_lines = []
                    for ln in _wrap_text(f"Task: {task_description}", font, font_scale, video_size - 2 * margin_x):
                        text_lines.append((ln, (255, 255, 255)))
                    if prompt != task_description:
                        for ln in _wrap_text(f"Prompt: {prompt}", font, font_scale, video_size - 2 * margin_x):
                            text_lines.append((ln, (180, 220, 255)))
                    for ln in _wrap_text(f"Perturb: {perturb_detail_str} | TTE gen={args.generations} pop={args.population_size}",
                                         font, font_scale, video_size - 2 * margin_x):
                        text_lines.append((ln, (200, 200, 200)))

                    bar_height = line_height * len(text_lines) + 4
                    overlay = f_bgr.copy()
                    cv2.rectangle(overlay, (0, 0), (video_size, bar_height), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, f_bgr, 0.5, 0, f_bgr)
                    for i, (txt, clr) in enumerate(text_lines):
                        cv2.putText(f_bgr, txt, (margin_x, line_height * (i + 1)),
                                    font, font_scale, clr, 1, cv2.LINE_AA)
                    annotated.append(cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB))
                return annotated

            if replay_images_single:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"{task_segment}_ep{episode_idx}_{perturb_tag}_{suffix}_tte.mp4",
                    _annotate_frames(replay_images_single), fps=10,
                )
            if replay_images_multi:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"{task_segment}_ep{episode_idx}_{perturb_tag}_{suffix}_tte_multi.mp4",
                    _annotate_frames(replay_images_multi), fps=10,
                )

            # ---- Save CEM evolution process video ----
            # Collects all evolution_records across replan steps in this episode
            all_evo_records = []
            for m in episode_meta.get("tte_meta", []):
                all_evo_records.extend(m.get("evolution_records", []))

            if all_evo_records:
                evo_frames = []
                for rec in all_evo_records:
                    gen_idx, rank, fit = rec["gen"], rec["rank"], rec["fitness"]
                    for fr in rec["frames"]:
                        fr_resized = cv2.resize(
                            np.asarray(fr), (video_size, video_size),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                        fr_bgr = cv2.cvtColor(fr_resized, cv2.COLOR_RGB2BGR)
                        label = f"Gen {gen_idx} | Rank {rank} | Fit {fit:.3f}"
                        cv2.putText(fr_bgr, label, (margin_x, video_size - 10),
                                    font, font_scale, (0, 255, 255), 1, cv2.LINE_AA)
                        evo_frames.append(cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB))
                if evo_frames:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path)
                        / f"{task_segment}_ep{episode_idx}_{perturb_tag}_{suffix}_tte_evolution.mp4",
                        evo_frames, fps=10,
                    )

            logging.info(
                f"Episode done. Success={done} | "
                f"Total: {total_successes}/{total_episodes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(
            f"Task {task_id}: {task_successes}/{task_episodes} "
            f"({task_successes / max(task_episodes, 1) * 100:.1f}%)"
        )
        env.close()

    # Save results
    results_path = pathlib.Path(args.results_out_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "task_suite": args.task_suite_name,
        "perturbations": [p.value for p in active_perturbations],
        "tte_config": dataclasses.asdict(tte_cfg),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / max(total_episodes, 1),
        "episodes": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logging.info(f"Results saved to {results_path}")
    logging.info(
        f"Final: {total_successes}/{total_episodes} "
        f"({total_successes / max(total_episodes, 1) * 100:.1f}%)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_with_tte(tyro.cli(Args))
