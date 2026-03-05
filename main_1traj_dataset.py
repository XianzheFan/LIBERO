import collections
import dataclasses
import logging
import math
import pathlib

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
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"
        # "libero_spatial"
        # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 20  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/output"

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

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

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

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
            action_plan = collections.deque()
            
            mujoco_robot = env.env.robots[0]
            ctrl_config = mujoco_robot.controller_config
            # Maximum constraints of the first three dimensions (X, Y, Z translation)
            if isinstance(ctrl_config, dict) and "output_max" in ctrl_config:
                action_scale = ctrl_config["output_max"][0] 
            else:
                action_scale = 0.05

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            
            camera_name = "agentview"
            img_height = LIBERO_ENV_RESOLUTION  # 256
            img_width = LIBERO_ENV_RESOLUTION  # 256
            mujoco_sim = env.env.sim

            # Setup
            t = 0
            replay_images = []
            clean_images = []
            
            expected_eef_pos = None

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    K = get_camera_intrinsic_matrix(
                        sim=mujoco_sim,
                        camera_name=camera_name,
                        camera_height=img_height,
                        camera_width=img_width
                    )
                    # World to Camera
                    E = get_camera_extrinsic_matrix(
                        sim=mujoco_sim,
                        camera_name=camera_name
                    )
                    
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)
                    clean_images.append(img)

                    if not action_plan:
                        # if expected_eef_pos is not None:
                        #     actual_eef_pos = obs["robot0_eef_pos"]
                        #     # Calculate the Euclidean distance error in 3D space (m)
                        #     error_dist = np.linalg.norm(expected_eef_pos - actual_eef_pos)
                        #     logging.info(f"Step {t:03d} | Tracking Error: {error_dist:.4f}m | "
                        #                  f"Expected: {np.round(expected_eef_pos, 3)} | Actual: {np.round(actual_eef_pos, 3)}")
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
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

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        
                        curr_pos_sim = obs["robot0_eef_pos"].copy()
                        tracking_factor = 0.35
                        for step_action in action_chunk[:args.replan_steps]:
                            clipped_action = np.clip(step_action[:3], -1.0, 1.0)
                            delta_3d = clipped_action * action_scale
                            goal_pos = curr_pos_sim + delta_3d
                            actual_movement = (goal_pos - curr_pos_sim) * tracking_factor
                            curr_pos_sim = curr_pos_sim + actual_movement
                        expected_eef_pos = curr_pos_sim
                        
                        img_with_traj = draw_trajectory_on_image(
                            img=img, 
                            current_eef_pos=obs["robot0_eef_pos"], 
                            action_chunk=action_chunk[:args.replan_steps], 
                            K=K, 
                            E=E,
                            orig_res=LIBERO_ENV_RESOLUTION,
                            target_res=args.resize_size,
                            action_scale=action_scale
                        )
                        # Replace the trackless images originally saved in the playback list
                        replay_images[-1] = img_with_traj
                        
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
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
            task_segment = task_description.replace(" ", "_")
            
            rollout_folder_name = f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            rollout_dir = pathlib.Path(args.video_out_path) / rollout_folder_name
            rollout_dir.mkdir(parents=True, exist_ok=True)
            
            imageio.mimwrite(
                rollout_dir / "complete_video.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            # The index i corresponds to time t; replay_images[i] represents the frame containing the trajectory
            # clean_images[i+1 : i + replan_steps] is the sequence of unannotated frames following that image
            for i in range(0, len(replay_images), args.replan_steps):
                # If an early stop occurs and the sequence length is insufficient, slicing will automatically handle the boundaries
                clip_idx = i // args.replan_steps
                imageio.imwrite(
                    rollout_dir / f"trajectory_frame_{clip_idx:03d}.png",
                    np.asarray(replay_images[i])
                )
                clip_frames = clean_images[i + 1 : i + args.replan_steps]
                
                # If the segment is empty (for example, if the robot finishes exactly at the replan_step frame), no video will be saved
                if len(clip_frames) > 0:
                    imageio.mimwrite(
                        rollout_dir / f"follow_up_clip_{clip_idx:03d}.mp4",
                        [np.asarray(x) for x in clip_frames],
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


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
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


def draw_trajectory_on_image(img, current_eef_pos, action_chunk, K, E, orig_res=256, target_res=224, action_scale=0.05, pos_limit=None, tracking_factor=0.35):
    """
    img: Preprocessed image
    tracking_factor: Simulation of the physical controller's lag rate (0.0 to 1.0). Based on log measurements, 0.35 closely approximates physical reality.
    """
    traj_3d = [current_eef_pos]
    curr_pos = current_eef_pos.copy()
    
    for step_action in action_chunk:
        # Extract positional action and apply clipping, consistent with low-level simulation logic
        delta_action = step_action[:3]
        clipped_action = np.clip(delta_action, -1.0, 1.0)
        delta_3d = clipped_action * action_scale
        
        # Calculate the absolute target point (Goal) set by the low-level controller
        goal_pos = curr_pos + delta_3d
        if pos_limit is not None:
            goal_pos = np.clip(goal_pos, pos_limit[0], pos_limit[1])
            
        # Simulate first-order physical tracking lag
        # The robotic arm cannot reach goal_pos instantaneously; 
        # the actual displacement is only tracking_factor times the desired increment.
        actual_movement = (goal_pos - curr_pos) * tracking_factor
        next_pos = curr_pos + actual_movement
        
        traj_3d.append(next_pos)
        curr_pos = next_pos
        
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
    
    for i in range(len(points_2d) - 1):
        pt1 = tuple(points_2d[i])
        pt2 = tuple(points_2d[i+1])
        cv2.line(img_drawn, pt1, pt2, (235, 206, 135), 2)  
        cv2.circle(img_drawn, pt1, 3, (0, 215, 255), -1)   
        
    cv2.circle(img_drawn, tuple(points_2d[0]), 5, (120, 200, 80), -1)  
    cv2.circle(img_drawn, tuple(points_2d[-1]), 5, (255, 127, 80), -1) 
    
    return img_drawn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)