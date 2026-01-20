import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tqdm
import tyro
import time
import zmq
from collections import deque
import torch
from typing import Dict, Tuple, Optional, Any
import numpy.typing as npt

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class RemoteAgent:
    PROPRIO_HISTORY_SIZE = 4

    def __init__(self, port: int) -> None:
        self._setup_zmq_connection(port)
        self._initialize_state()

    def _setup_zmq_connection(self, port: int) -> None:
        try:
            self.zmq_context = zmq.Context()
            self.socket = self.zmq_context.socket(zmq.REQ)
            self.socket.connect(f"tcp://127.0.0.1:{port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 40000)  # 40s timeout
        except Exception as e:
            raise ConnectionError(f"Failed to establish ZMQ connection: {e}")

    def _initialize_state(self) -> None:
        self.proprio_history = deque(maxlen=self.PROPRIO_HISTORY_SIZE)
        self.pred_actions = deque()

    def get_current_proprio(self, obs: Dict[str, Any]) -> npt.NDArray[np.float64]:
        current_proprio = np.concatenate(
            [obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"][-1:]]
        )
        return current_proprio

    def _process_proprio(self, obs: Dict[str, Any]) -> None:
        current_proprio = self.get_current_proprio(obs)
        self.proprio_history.append(current_proprio)
        while len(self.proprio_history) < self.proprio_history.maxlen:
            self.proprio_history.append(self.proprio_history[-1])

    # -----------------------------
    # Action path
    # -----------------------------

    def step(
        self, obs: Dict[str, Any], instruction: str, debug: bool = False
    ) -> Tuple[npt.NDArray[np.float64], Optional[Any]]:
        """
        Forward action from server to env.step() as-is.
        """
        self._process_proprio(obs)

        if len(self.pred_actions) == 0:
            self._post_and_get(obs, instruction, debug=debug)

        action, bbox = self.pred_actions.popleft()

        # Ensure correct shape/type only (NOT a semantic transformation)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Agent action must be 7-dim; got shape={action.shape}")

        action[6] = -action[6]  # 1 for open, -1 for close
        return action, bbox

    def _post_and_get(self, obs: Dict[str, Any], instruction: str, debug: bool = False) -> None:
        data = {
            "image_array": [obs["agentview_image"][::-1, ::-1]],
            "image_wrist_array": [obs["robot0_eye_in_hand_image"][::-1, ::-1]],
            "depth_array": [obs["agentview_depth"][::-1, ::-1]],
            "depth_wrist_array": [obs["robot0_eye_in_hand_depth"][::-1, ::-1]],
            "proprio_array": [np.copy(proprio) for proprio in self.proprio_history],
            "env_id": 1,
            "text": instruction,
        }

        self.socket.send_pyobj(data)
        response = self.socket.recv_pyobj()

        bbox = response.get("debug", {}).get("bbox", None)

        # Enqueue each action as-is
        for a in response["result"]:
            self.pred_actions.append([a, bbox])

        if debug:
            print("-" * 40)
            p0 = data["proprio_array"][-4] if len(data["proprio_array"]) >= 4 else data["proprio_array"][0]
            p1 = data["proprio_array"][-1]
            print("proprio transition", [round(v, 4) for v in p0[:3]], [round(v, 4) for v in p1[:3]])
            print("proprio gripper", float(p0[-1]), float(p1[-1]))
            acts = response["result"]
            print("raw z actions", [round(float(x[2]), 4) for x in acts[:10]])
            print("raw gripper actions", [float(x[6]) for x in acts[:10]])


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 6688

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    out_path: str = "data/"  # Path to save videos
    save_name: str = ""
    save_videos: bool = False  # Whether to save videos of the evaluation

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    assert args.save_name != "", "Please provide a save name for the evaluation run."

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.save_videos:
        video_out_path = pathlib.Path(args.out_path) / "videos" / args.save_name / args.task_suite_name
        video_out_path.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = RemoteAgent(args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    # wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    action, bbox = client.step(obs, task_description, debug=False)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            if args.save_videos:
                task_segment = task_description.replace(" ", "_")
                time_suffix = str(time.time())[-4:]
                imageio.mimwrite(
                    video_out_path / f"rollout_{task_segment}_{time_suffix}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Save results to a txt file
    (pathlib.Path(args.out_path) / "results").mkdir(parents=True, exist_ok=True)
    results_path = pathlib.Path(args.out_path) / "results" / f"{args.save_name}_results.txt"
    with open(results_path, "a") as f:
        f.write(f"Task suite: {args.task_suite_name}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Total successes: {total_successes}\n")
        f.write(f"Success rate: {float(total_successes) / float(total_episodes)}\n")
        f.write("\n")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
