"""
Test-Time Evolution (TTE) for SDE Policies.

Instead of blindly executing the first sampled trajectory, TTE maintains a
*population* of candidate action chunks, evaluates each by forward-simulating
inside a cloned MuJoCo state, scores the outcome with a configurable fitness
function, and evolves the population over several generations using CEM
(Cross-Entropy Method) before committing to the best trajectory.

This combats position-memorisation overfitting: even if the policy's mode
points at the wrong location, the stochastic SDE samples span a range of
targets, and the fitness function steers selection toward the semantically
correct one.

Usage (standalone):
    python third_party/libero/test_time_evolution.py \
        --task_suite_name libero_10 \
        --population_size 16 --generations 3 --elite_ratio 0.25 \
        --fitness_mode oracle

Integration:
    from test_time_evolution import TestTimeEvolver, FitnessConfig
    evolver = TestTimeEvolver(client, FitnessConfig(mode="oracle"))
    best_actions = evolver.evolve(env, obs, prompt, replan_steps=5)
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import logging
import math
import pathlib
import collections
from typing import Any, Callable, Protocol

import cv2
import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp
import tqdm
import tyro

logger = logging.getLogger(__name__)


class FitnessMode(enum.Enum):
    """Which fitness signal to use for trajectory evaluation."""
    ORACLE = "oracle"        # Privileged: use sim object positions
    VLM = "vlm"              # Use VLM reward model
    CLIP = "clip"            # Use CLIP image-text similarity
    COMPOSITE = "composite"  # Weighted combination


@dataclasses.dataclass
class FitnessConfig:
    mode: str = "oracle"
    target_body_name: str | None = None
    vlm_model: Any = None  # Externally provided VLM scorer
    clip_model: Any = None
    clip_preprocess: Any = None

    w_distance: float = 1.0      # Reward for getting closer to target
    w_grasp: float = 2.0         # Reward for gripper closing near target
    w_smoothness: float = 0.1    # Penalty for jerky actions
    w_vlm: float = 1.0           # VLM progress reward weight
    sim_horizon: int = 10        # How many steps to forward-simulate


def _extract_target_body(env, instruction: str) -> str | None:
    """Heuristic: find the MuJoCo body whose name best matches the
    instruction's target object."""
    inner = env.env if hasattr(env, "env") else env
    sim = inner.sim
    model = sim.model

    keywords = [
        "moka_pot", "moka", "frypan", "pan", "bowl", "mug", "plate",
        "butter", "cream_cheese", "tomato_sauce", "chocolate_pudding",
        "alphabet_soup", "book", "basket", "caddy",
    ]

    instruction_lower = instruction.lower().replace(" ", "_")

    best_match = None
    best_score = 0
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if not body_name:
            continue
        name_lower = body_name.lower()
        # Score = longest keyword substring match with instruction
        for kw in keywords:
            if kw in name_lower and kw in instruction_lower:
                score = len(kw)
                if score > best_score:
                    best_score = score
                    best_match = body_name
    return best_match


def _get_body_pos(env, body_name: str) -> np.ndarray:
    """Get 3D position of a named body in the simulation."""
    inner = env.env if hasattr(env, "env") else env
    sim = inner.sim
    body_id = sim.model.body_name2id(body_name)
    return sim.data.body_xpos[body_id].copy()


def _get_eef_pos(obs: dict) -> np.ndarray:
    return obs["robot0_eef_pos"].copy()


def _compute_oracle_fitness(
    env,
    obs_sequence: list[dict],
    actions: np.ndarray,
    target_body: str,
    config: FitnessConfig,
) -> float:
    """Privileged fitness using ground-truth object positions from sim."""
    if not obs_sequence:
        return -float("inf")

    target_pos = _get_body_pos(env, target_body)
    eef_start = _get_eef_pos(obs_sequence[0])
    eef_end = _get_eef_pos(obs_sequence[-1])

    dist_start = np.linalg.norm(eef_start - target_pos)
    dist_end = np.linalg.norm(eef_end - target_pos)

    # Reward for getting closer
    distance_reward = (dist_start - dist_end) * config.w_distance

    # Reward for being close at the end
    proximity_bonus = max(0, 0.05 - dist_end) * 10.0

    # Gripper reward: if close to target and gripper is closing
    grasp_reward = 0.0
    if dist_end < 0.08:
        last_action = actions[-1] if len(actions) > 0 else None
        if last_action is not None and last_action[-1] > 0:  # gripper closing
            grasp_reward = config.w_grasp

    # Smoothness penalty
    if len(actions) > 1:
        diffs = np.diff(actions[:, :3], axis=0)
        jerk = np.mean(np.linalg.norm(diffs, axis=-1))
        smoothness_penalty = -jerk * config.w_smoothness
    else:
        smoothness_penalty = 0.0

    return distance_reward + proximity_bonus + grasp_reward + smoothness_penalty


def _compute_vlm_fitness(
    env,
    obs_sequence: list[dict],
    instruction: str,
    config: FitnessConfig,
) -> float:
    """Use VLM model to score trajectory outcome."""
    if config.vlm_model is None:
        logger.warning("VLM model not provided, returning 0")
        return 0.0
    if not obs_sequence:
        return -float("inf")

    # Score the final observation image
    final_img = obs_sequence[-1].get("agentview_image")
    if final_img is None:
        return 0.0

    return config.vlm_model.predict_progress_value(final_img, instruction)


def _compute_clip_fitness(
    obs_sequence: list[dict],
    instruction: str,
    config: FitnessConfig,
) -> float:
    """Use CLIP to measure alignment between final frame and instruction."""
    if config.clip_model is None or config.clip_preprocess is None:
        logger.warning("CLIP model not provided, returning 0")
        return 0.0
    if not obs_sequence:
        return -float("inf")

    import torch

    final_img = obs_sequence[-1].get("agentview_image")
    if final_img is None:
        return 0.0

    device = next(config.clip_model.parameters()).device
    image_input = config.clip_preprocess(
        image_tools.convert_to_uint8(final_img)
    ).unsqueeze(0).to(device)

    import clip as clip_module
    text_input = clip_module.tokenize([instruction]).to(device)

    with torch.no_grad():
        img_feat = config.clip_model.encode_image(image_input)
        txt_feat = config.clip_model.encode_text(text_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        similarity = (img_feat @ txt_feat.T).item()

    return similarity


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _forward_simulate(
    env,
    actions: np.ndarray,
    saved_state: np.ndarray,
) -> list[dict]:
    """Forward-simulate a sequence of actions from a saved MuJoCo state.

    Returns a list of observation dicts (one per step + initial).
    The environment is restored to `saved_state` before simulation begins.
    """
    # Restore simulator state
    obs = env.set_init_state(saved_state)
    obs_list = [obs]

    for action in actions:
        obs, reward, done, info = env.step(action.tolist())
        obs_list.append(obs)
        if done:
            break

    return obs_list


def _restore_state(env, saved_state: np.ndarray) -> dict:
    """Restore env to a saved state and return the observation."""
    return env.set_init_state(saved_state)


# ---------------------------------------------------------------------------
# Test-Time Evolver (CEM-based)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EvolverConfig:
    population_size: int = 16       # Number of candidate trajectories
    generations: int = 3            # Evolution rounds
    elite_ratio: float = 0.25       # Top fraction to keep as elites
    mutation_std: float = 0.15      # Std of Gaussian mutation on actions
    mutation_decay: float = 0.7     # Decay mutation_std each generation
    fresh_ratio: float = 0.25       # Fraction of population re-sampled fresh
    crossover_prob: float = 0.3     # Probability of crossover between elites
    replan_steps: int = 5           # Action horizon to execute


class TestTimeEvolver:
    """CEM-based test-time evolution over action trajectories.

    The evolver:
      1. Samples an initial population from the SDE policy (diverse via noise).
      2. Forward-simulates each candidate in a cloned MuJoCo state.
      3. Scores trajectories with the configured fitness function.
      4. Selects elites, mutates, and optionally crosses over.
      5. Repeats for `generations` rounds.
      6. Returns the best action chunk to execute.
    """

    def __init__(
        self,
        client: _wcp.WebsocketClientPolicy,
        fitness_config: FitnessConfig | None = None,
        evolver_config: EvolverConfig | None = None,
    ):
        self.client = client
        self.fc = fitness_config or FitnessConfig()
        self.ec = evolver_config or EvolverConfig()

    def _build_policy_input(
        self, obs: dict, prompt: str, resize_size: int = 224
    ) -> dict:
        """Build the observation dict expected by the websocket policy."""
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, resize_size, resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
        )

        quat = obs["robot0_eef_quat"].copy()
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if np.isclose(den, 0.0):
            axis_angle = np.zeros(3)
        else:
            axis_angle = (quat[:3] * 2.0 * np.arccos(quat[3])) / den

        state = np.concatenate([
            obs["robot0_eef_pos"],
            axis_angle,
            obs["robot0_gripper_qpos"],
        ])

        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": prompt,
        }

    def _sample_from_policy(self, element: dict, n: int) -> list[np.ndarray]:
        """Sample n action chunks from the SDE policy."""
        return [
            np.array(self.client.infer(element)["actions"])
            for _ in range(n)
        ]

    def _evaluate_fitness(
        self,
        env,
        candidate: np.ndarray,
        saved_state: np.ndarray,
        instruction: str,
        target_body: str | None,
    ) -> float:
        """Evaluate one candidate action chunk by forward simulation."""
        actions_to_sim = candidate[: self.fc.sim_horizon]
        obs_seq = _forward_simulate(env, actions_to_sim, saved_state)

        mode = FitnessMode(self.fc.mode)
        fitness = 0.0

        if mode == FitnessMode.ORACLE:
            if target_body is None:
                return 0.0
            fitness = _compute_oracle_fitness(
                env, obs_seq, actions_to_sim, target_body, self.fc
            )
        elif mode == FitnessMode.VLM:
            fitness = _compute_vlm_fitness(env, obs_seq, instruction, self.fc)
        elif mode == FitnessMode.CLIP:
            fitness = _compute_clip_fitness(obs_seq, instruction, self.fc)
        elif mode == FitnessMode.COMPOSITE:
            if target_body:
                fitness += _compute_oracle_fitness(
                    env, obs_seq, actions_to_sim, target_body, self.fc
                )
            fitness += self.fc.w_vlm * _compute_vlm_fitness(
                env, obs_seq, instruction, self.fc
            )

        return fitness

    def _mutate(self, action_chunk: np.ndarray, std: float) -> np.ndarray:
        """Gaussian mutation on an action chunk."""
        noise = np.random.randn(*action_chunk.shape) * std
        mutant = action_chunk + noise
        # Clip to [-1, 1] for the position/rotation dims, keep gripper
        mutant[:, :6] = np.clip(mutant[:, :6], -1.0, 1.0)
        mutant[:, 6] = np.clip(mutant[:, 6], -1.0, 1.0)
        return mutant

    def _crossover(
        self, parent_a: np.ndarray, parent_b: np.ndarray
    ) -> np.ndarray:
        """Uniform crossover between two action chunks."""
        mask = np.random.rand(parent_a.shape[0]) < 0.5
        child = parent_a.copy()
        child[mask] = parent_b[mask]
        return child

    def evolve(
        self,
        env,
        obs: dict,
        prompt: str,
        *,
        resize_size: int = 224,
        verbose: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Run CEM evolution and return (best_action_chunk, metadata).

        Args:
            env: LIBERO OffScreenRenderEnv (will be used for forward sim).
            obs: Current observation dict from the environment.
            prompt: Language instruction.
            resize_size: Image resize for policy input.
            verbose: Log per-generation stats.

        Returns:
            best_actions: np.ndarray of shape (action_horizon, action_dim)
            meta: dict with evolution statistics
        """
        ec = self.ec
        fc = self.fc

        # Save current simulator state for rollback
        saved_state = env.get_sim_state()

        # Auto-detect target object
        target_body = fc.target_body_name
        if target_body is None and FitnessMode(fc.mode) in (
            FitnessMode.ORACLE, FitnessMode.COMPOSITE
        ):
            target_body = _extract_target_body(env, prompt)
            if target_body:
                logger.info(f"Auto-detected target body: {target_body}")
            else:
                logger.warning(
                    "Could not auto-detect target body from instruction. "
                    "Oracle fitness will return 0."
                )

        # Build policy input
        element = self._build_policy_input(obs, prompt, resize_size)

        # --- Generation 0: initial population from SDE policy ---
        population = self._sample_from_policy(element, ec.population_size)

        num_elites = max(1, int(ec.population_size * ec.elite_ratio))
        num_fresh = max(1, int(ec.population_size * ec.fresh_ratio))
        mutation_std = ec.mutation_std

        meta = {
            "generations": [],
            "target_body": target_body,
            "population_size": ec.population_size,
        }

        best_overall = None
        best_overall_fitness = -float("inf")

        for gen in range(ec.generations):
            # Evaluate fitness for each candidate
            fitnesses = []
            for cand in population:
                f = self._evaluate_fitness(
                    env, cand, saved_state, prompt, target_body
                )
                fitnesses.append(f)

            fitnesses = np.array(fitnesses)
            sorted_idx = np.argsort(fitnesses)[::-1]  # descending

            gen_best_fitness = fitnesses[sorted_idx[0]]
            gen_mean_fitness = fitnesses.mean()
            gen_std_fitness = fitnesses.std()

            if verbose:
                logger.info(
                    f"  Gen {gen}: best={gen_best_fitness:.4f} "
                    f"mean={gen_mean_fitness:.4f} std={gen_std_fitness:.4f}"
                )

            meta["generations"].append({
                "best": float(gen_best_fitness),
                "mean": float(gen_mean_fitness),
                "std": float(gen_std_fitness),
            })

            # Track overall best
            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall = population[sorted_idx[0]].copy()

            # Select elites
            elite_idx = sorted_idx[:num_elites]
            elites = [population[i].copy() for i in elite_idx]

            # Build next generation
            next_pop = []

            # 1. Keep elites unchanged
            for e in elites:
                next_pop.append(e)

            # 2. Mutants from elites
            num_mutants = ec.population_size - num_elites - num_fresh
            for _ in range(num_mutants):
                parent = elites[np.random.randint(len(elites))]
                if (
                    ec.crossover_prob > 0
                    and len(elites) > 1
                    and np.random.rand() < ec.crossover_prob
                ):
                    other = elites[np.random.randint(len(elites))]
                    child = self._crossover(parent, other)
                else:
                    child = parent.copy()
                child = self._mutate(child, mutation_std)
                next_pop.append(child)

            # 3. Fresh samples from SDE policy (maintains exploration)
            fresh = self._sample_from_policy(element, num_fresh)
            next_pop.extend(fresh)

            population = next_pop[: ec.population_size]
            mutation_std *= ec.mutation_decay

        # Final evaluation if we haven't found best yet
        if best_overall is None:
            best_overall = population[0]

        # Restore env to original state
        _restore_state(env, saved_state)

        meta["best_fitness"] = float(best_overall_fitness)
        return best_overall, meta


LIBERO_ENV_RESOLUTION = 256

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20
    population_size: int = 16
    generations: int = 3
    elite_ratio: float = 0.25
    mutation_std: float = 0.15
    mutation_decay: float = 0.7
    fresh_ratio: float = 0.25
    crossover_prob: float = 0.3
    fitness_mode: str = "oracle"
    sim_horizon: int = 10

    video_out_path: str = "data/libero/tte_videos"
    results_out_path: str = "data/libero/tte_results.json"
    seed: int = 7


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_with_tte(args: Args) -> None:
    """Evaluation loop using Test-Time Evolution."""
    import json
    from libero.libero import benchmark

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = _MAX_STEPS[args.task_suite_name]

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _wcp.WebsocketClientPolicy(args.host, args.port)

    fitness_config = FitnessConfig(
        mode=args.fitness_mode,
        sim_horizon=args.sim_horizon,
    )
    evolver_config = EvolverConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_ratio=args.elite_ratio,
        mutation_std=args.mutation_std,
        mutation_decay=args.mutation_decay,
        fresh_ratio=args.fresh_ratio,
        crossover_prob=args.crossover_prob,
        replan_steps=args.replan_steps,
    )
    evolver = TestTimeEvolver(client, fitness_config, evolver_config)

    all_results: list[dict] = []
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed
        )

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc="Episodes", leave=False
        ):
            episode_meta: dict[str, Any] = {
                "task_id": task_id,
                "episode_idx": episode_idx,
                "instruction": task_description,
                "tte_meta": [],
            }

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            action_plan = collections.deque()
            t = 0
            replay_images = []
            done = False

            logger.info(f"\nTask: {task_description}")

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(
                        obs["agentview_image"][::-1, ::-1]
                    )
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            img, args.resize_size, args.resize_size
                        )
                    )
                    replay_images.append(img)

                    if not action_plan:
                        # === Test-Time Evolution ===
                        best_actions, tte_meta = evolver.evolve(
                            env, obs, task_description,
                            resize_size=args.resize_size,
                            verbose=True,
                        )
                        episode_meta["tte_meta"].append(tte_meta)
                        action_plan.extend(
                            best_actions[: args.replan_steps]
                        )

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logger.error(f"Exception: {e}", exc_info=True)
                    break

            task_episodes += 1
            total_episodes += 1
            episode_meta["success"] = bool(done)
            episode_meta["steps"] = t
            all_results.append(episode_meta)

            # Save video
            if replay_images:
                suffix = "success" if done else "failure"
                tag = task_description.replace(" ", "_")
                video_size = 512
                font = cv2.FONT_HERSHEY_SIMPLEX

                annotated = []
                for frame in replay_images:
                    f_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                    f_bgr = cv2.resize(
                        f_bgr, (video_size, video_size),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    cv2.putText(
                        f_bgr, f"TTE | {task_description}",
                        (4, 14), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    annotated.append(
                        cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
                    )

                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"{tag}_ep{episode_idx}_tte_{suffix}.mp4",
                    annotated, fps=10,
                )

            logger.info(
                f"Episode done. Success={done} | "
                f"Total: {total_successes}/{total_episodes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logger.info(
            f"Task {task_id}: {task_successes}/{task_episodes} "
            f"({task_successes / max(task_episodes, 1) * 100:.1f}%)"
        )
        env.close()

    results_path = pathlib.Path(args.results_out_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "task_suite": args.task_suite_name,
        "tte_config": {
            "population_size": args.population_size,
            "generations": args.generations,
            "elite_ratio": args.elite_ratio,
            "fitness_mode": args.fitness_mode,
        },
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / max(total_episodes, 1),
        "episodes": all_results,
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")
    logger.info(
        f"Final: {total_successes}/{total_episodes} "
        f"({total_successes / max(total_episodes, 1) * 100:.1f}%)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_with_tte(args)
