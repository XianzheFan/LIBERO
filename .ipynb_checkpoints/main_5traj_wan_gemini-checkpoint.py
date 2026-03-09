import collections
import dataclasses
import logging
import math
import pathlib
import requests
import concurrent.futures
import time
import json

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import torch
import cv2
import numpy as np

from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

WAN2_SHARED_PROMPT = """Please examine the robotic arm workspace in the input image. The drawn trajectory represents the future motion of the gripper's center: the green dot is the starting point, and the red dot is the destination. 
Please imagine a video showing the complete process of the upper part of the robotic arm (including the gripper) moving from the start to the end (keeping the drawn trajectory stationary). Follow these physical rules: 
1. In the final state, the center of the end-effector (gripper) must be perfectly aligned with the red dot. The base of the robotic arm must remain stationary. 
2. Only the upper joints and the arm of the robotic arm move to perform tasks, while the base remains stationary. 
3. Object Interaction Logic: Observe the relationship between the green dot (start position) and any objects in the scene. 
- IF the green dot is currently positioned near an object in a way that implies a grasp or contact, or if the gripper is already holding it: Move that specific object along the trajectory so that it is still being held or manipulated by the gripper at the red dot. 
- IF the green dot and the predicted path are in free space, not touching any objects: Execute the movement as pure free-space motion. No objects should be moved. 
- Do not change the open/closed state of the gripper unless it is necessary for physical plausibility at the destination (e.g., placing an object). 
4. Environment Stability: Keep all other background elements, lighting, camera angle, and non-interacted objects exactly identical to the original scene."""


class FrameEvaluation(BaseModel):
    best_index: int


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
    # num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def request_generation(port: int, image_path: str, prompt: str, save_path: str):
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "image_path": image_path,
        "prompt": prompt,
        "save_path": save_path,
        "sampling_steps": 30
    }
    
    logging.info(f"[Port {port}] Submitting task: {image_path}...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        end_time = time.time()
        logging.info(f"[Port {port}] Task completed! Time elapsed: {end_time - start_time:.2f}s. Saved to: {save_path}")
        return True
    except Exception as e:
        logging.error(f"[Port {port}] Task failed: {e}")
        return False


def evaluate_videos_with_gemini(current_video_path: str, candidate_paths: list, language_command: str):
    load_dotenv()
    client = genai.Client(http_options={'api_version': 'v1alpha'})
    
    logging.info("Uploading videos to Google GenAI...")
    current_video_file = client.files.upload(file=current_video_path)
    
    candidate_files = []
    for path in candidate_paths:
        candidate_files.append(client.files.upload(file=path))

    def wait_for_files_active(*files):
        logging.info("Waiting for video processing...")
        for f in files:
            file_info = client.files.get(name=f.name)
            while file_info.state.name == "PROCESSING":
                time.sleep(2)
                file_info = client.files.get(name=f.name)
            if file_info.state.name == "FAILED":
                raise ValueError(f"Video processing failed for file: {f.name}")
        logging.info("All videos are ready!")

    wait_for_files_active(current_video_file, *candidate_files)

    prompt_instruction = f"""
You are an evaluation model in a robotic control system. 
Based on the [Current Video] and the [Language Command], your task is to select the most accurate video from the provided [Candidate Next Videos] that best follows the intended trajectory.

Language Command: "{language_command}"

Please evaluate the Current Video against the Candidate Next Videos (Index 0 to {len(candidate_files)-1}).
"""
    contents = [prompt_instruction, "\n[Current Video]:", current_video_file]
    for i, cand_file in enumerate(candidate_files):
        contents.extend([f"\n[Candidate Next Video {i}]:", cand_file])

    logging.info("Requesting Gemini for evaluation...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FrameEvaluation,
            temperature=0.2
        )
    )

    result_dict = json.loads(response.text)
    
    logging.info("Cleaning up uploaded files...")
    client.files.delete(name=current_video_file.name)
    for cand_file in candidate_files:
        client.files.delete(name=cand_file.name)
        
    return result_dict['best_index']


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
            replay_images_single = []
            replay_images_multi = []

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

                    # Placeholder: This will be replaced by the actual adopted or rendered image
                    replay_images_single.append(img.copy())
                    replay_images_multi.append(img.copy())

                    if not action_plan:
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
                        num_samples = 5
                        action_chunks = [client.infer(element)["actions"] for _ in range(num_samples)]
                        
                        task_segment = task_description.replace(" ", "_")
                        step_save_dir = pathlib.Path(args.video_out_path) / "trajectories" / f"{task_segment}_ep{episode_idx}_step{t}"
                        step_save_dir.mkdir(parents=True, exist_ok=True)
                        
                        colors = [
                            ((235, 206, 135), (0, 215, 255)),
                            ((144, 238, 144), (34, 139, 34)),
                            ((255, 182, 193), (220, 20, 60)),
                            ((173, 216, 230), (0, 0, 139)),
                            ((221, 160, 221), (139, 0, 139))
                        ]
                        
                        img_multi = img.copy()
                        img_single_trajs = []
                        plan_image_paths = []
                        
                        for i, chunk in enumerate(action_chunks):
                            line_c, point_c = colors[i]
                            
                            img_single_traj = draw_trajectory_on_image(
                                img=img, current_eef_pos=obs["robot0_eef_pos"], 
                                action_chunk=chunk[:args.replan_steps], K=K, E=E,
                                orig_res=LIBERO_ENV_RESOLUTION, target_res=args.resize_size, action_scale=action_scale,
                                line_color=line_c, point_color=point_c
                            )
                            img_single_trajs.append(img_single_traj)
                            
                            img_filename = f"plan_{i}.png"
                            cv2.imwrite(
                                str(step_save_dir / img_filename),
                                cv2.cvtColor(img_single_traj, cv2.COLOR_RGB2BGR)
                            )
                            plan_image_paths.append(str(step_save_dir / img_filename))
                            
                            img_multi = draw_trajectory_on_image(
                                img=img_multi, current_eef_pos=obs["robot0_eef_pos"], 
                                action_chunk=chunk[:args.replan_steps], K=K, E=E,
                                orig_res=LIBERO_ENV_RESOLUTION, target_res=args.resize_size, action_scale=action_scale,
                                line_color=line_c, point_color=point_c
                            )
                        
                        replay_images_multi[-1] = img_multi
                        
                        # Submit parallel video generation tasks to Wan2.2
                        tasks = []
                        for i in range(num_samples):
                            tasks.append({
                                "port": 8010 + i,
                                "img": plan_image_paths[i],
                                "save": str(step_save_dir / f"output_{i}.mp4")
                            })
                            
                        logging.info("Initializing parallel Wan2.2 video generation...")
                        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                            futures = []
                            for task_info in tasks:
                                futures.append(
                                    executor.submit(
                                        request_generation, 
                                        task_info["port"], 
                                        task_info["img"], 
                                        WAN2_SHARED_PROMPT, 
                                        task_info["save"]
                                    )
                                )
                            concurrent.futures.wait(futures)
                            
                        generated_candidate_paths = [t["save"] for t in tasks]
                        
                        # Generate current history video (from start to current t, including all visualization elements)
                        current_video_path = str(step_save_dir / "current_actual_video.mp4")
                        imageio.mimwrite(
                            current_video_path,
                            [np.asarray(x) for x in replay_images_multi],
                            fps=10
                        )
                        
                        try:
                            best_index = evaluate_videos_with_gemini(
                                current_video_path=current_video_path,
                                candidate_paths=generated_candidate_paths,
                                language_command=task_description
                            )
                            logging.info(f"==> Gemini selected Plan {best_index} as the best action.")
                        except Exception as e:
                            logging.error(f"Gemini evaluation failed: {e}. Defaulting to index 0.")
                            best_index = 0

                        # Record the selected optimal trajectory to output a single-trajectory video at the end
                        replay_images_single[-1] = img_single_trajs[best_index]
                        # Feed only the best action into the environment planning queue
                        action_plan.extend(action_chunks[best_index][: args.replan_steps])

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
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images_single],
                fps=10,
            )
            
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx}_{suffix}_multi.mp4",
                [np.asarray(x) for x in replay_images_multi],
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


def draw_trajectory_on_image(img, current_eef_pos, action_chunk, K, E, orig_res=256, target_res=224, action_scale=0.05, pos_limit=None, tracking_factor=0.35, line_color=(235, 206, 135), point_color=(0, 215, 255)):
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
    
    # World to Camera Transform (MUST use inverse of E)
    E_inv = np.linalg.inv(E)
    traj_cam_homo = (E_inv @ traj_3d_homo.T).T
    traj_cam = traj_cam_homo[:, :3]
    
    # Project to 2D pixel plane
    traj_2d_homo = (K @ traj_cam.T).T
    
    # Divide by depth Z to get 2D pixel coordinates (u, v)
    u = traj_2d_homo[:, 0] / traj_2d_homo[:, 2]
    v = traj_2d_homo[:, 1] / traj_2d_homo[:, 2]
    
    # Compensate for 180-degree image rotation and resizing
    u = orig_res - 1 - u
    scale = target_res / orig_res
    u = u * scale
    v = v * scale
    
    img_drawn = img.copy()
    points_2d = np.vstack((u, v)).T.astype(np.int32)
    
    # Draw the trajectory line and points
    for i in range(len(points_2d) - 1):
        pt1 = tuple(points_2d[i])
        pt2 = tuple(points_2d[i+1])
        cv2.line(img_drawn, pt1, pt2, line_color, 2)  # Gold line
        cv2.circle(img_drawn, pt1, 3, point_color, -1)   # Blue point
        
    cv2.circle(img_drawn, tuple(points_2d[0]), 5, (120, 200, 80), -1)  # BGR Emerald Green start
    cv2.circle(img_drawn, tuple(points_2d[-1]), 5, (255, 127, 80), -1) # BGR Coral Red end
    
    return img_drawn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)