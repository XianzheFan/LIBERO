import argparse
import os
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque
from functools import partial

import numpy as np
import rospy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import check_keyboard_input, get_config, handle_interactive_mode, process_action
from rosoperator import RosOperator, get_ros_observation
from rotation import abs_6d_2_abs_euler, quat_2_euler

observation_window = None
observation_window_lock = threading.Lock()

shutdown_event = threading.Event()
# When clear: inference thread blocks. When set: inference thread runs.
inference_paused = threading.Event()
inference_paused.clear()  # paused by default

inference_stamp = 0  # the stamp of the current inference


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
    """Reset observation window for a new episode."""
    global observation_window
    with observation_window_lock:
        observation_window = None


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = deque(maxlen=2)

            # Append the first dummy image
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
    endpose_left = np.array([left_pos.x, left_pos.y, left_pos.z, left_rpy[0], left_rpy[1], left_rpy[2], left_gripper])

    right_pos = endpose_right.pose.position
    right_rpy = quat_2_euler(endpose_right.pose.orientation)
    right_gripper = puppet_arm_right.position[-1]
    endpose_right = np.array(
        [right_pos.x, right_pos.y, right_pos.z, right_rpy[0], right_rpy[1], right_rpy[2], right_gripper]
    )

    endpose = np.concatenate((endpose_left, endpose_right), axis=0)

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

        self.cur_step = 0  # current step in cur_chunk
        self.cur_stamp = 0  # indicate the origin inference stamp of the current chunk

    def reset(self):
        """Reset buffer state for a new episode."""
        with self.lock:
            self.cur_len = 0
            self.next_len = 0
            self.cur_step = 0
            self.cur_stamp = 0

    def should_launch_inference(self):
        if self.exec_horizon > self.delay:
            return self.cur_step == (self.exec_horizon - self.delay - 1)
        else:
            # continuous inference mode: launch inference when the current chunk is starting
            return self.cur_step == 0

    def integrate_first_chunk(self, actions_chunk: np.ndarray):
        # only for the first chunk inferenced by sync inference, already selected [0:s)
        with self.lock:
            assert self.cur_len == 0, "cur_len should be 0 when starting"
            assert self.cur_stamp == 0, "cur_stamp should be 0 when starting"
            assert actions_chunk.shape[0] == self.exec_horizon, f"{actions_chunk.shape[0]} != {self.exec_horizon}"
            self.cur_chunk[: actions_chunk.shape[0]] = actions_chunk
            self.cur_len = self.exec_horizon

    def integrate_new_chunk(self, actions_chunk: np.ndarray):
        # only for the new chunk inferenced by async inference, already selected [d:s+d)
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or len(actions_chunk) == 0 when integrating new chunk")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            assert L == self.exec_horizon, f"{L} != {self.exec_horizon}"
            if self.cur_len == 0:
                # warning: this should not happen
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                self.cur_chunk[:L] = actions_chunk
                self.cur_len = L
            else:
                self.next_chunk[:L] = actions_chunk
                self.next_len = L

    def integrate_new_chunk_streaming(self, actions_chunk: np.ndarray, stamp: int):
        # only for the new chunk inferenced by async streaming inference, already removed [0:d), but can be different lengths
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or len(actions_chunk) == 0 when integrating new chunk")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            if self.cur_len == 0:
                # warning: this should not happen
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                safe_L = min(L, self.exec_horizon)
                self.cur_chunk[:safe_L] = actions_chunk[:safe_L]
                self.cur_len = safe_L
                print(f"cur_chunk extend to {self.cur_len} at stamp {stamp}")
            else:
                if self.cur_stamp == stamp:
                    # current chunk is already executing, extend it
                    remaining_len = self.exec_horizon - self.cur_len
                    if remaining_len > 0:
                        safe_L = min(L, remaining_len)
                        self.cur_chunk[self.cur_len : self.cur_len + safe_L] = actions_chunk[:safe_L]
                        self.cur_len += safe_L
                        print(f"cur_chunk extend to {self.cur_len} with {safe_L} new actions at stamp {stamp}")
                    else:
                        print(f"cur_chunk is already enough at stamp {stamp}")
                else:
                    remaining_len = self.exec_horizon - self.next_len
                    if remaining_len > 0:
                        safe_L = min(L, remaining_len)
                        self.next_chunk[self.next_len : self.next_len + safe_L] = actions_chunk[:safe_L]
                        self.next_len += safe_L
                        print(f"next_chunk extend to {self.next_len} with {safe_L}/{L} new actions at stamp {stamp}")
                    else:
                        print(f"next_chunk is already enough at stamp {stamp}")

    def get_next_action(self):
        with self.lock:
            if self.cur_step >= self.cur_len:
                return None

            action = self.cur_chunk[self.cur_step]
            self.cur_step += 1

            # should only execute [0:s) of the current chunk, switch to next chunk
            if self.cur_step == self.exec_horizon:
                self.cur_chunk, self.next_chunk = self.next_chunk, self.cur_chunk
                self.cur_len = self.next_len
                self.next_len = 0
                self.cur_step = 0
                self.cur_stamp += 1

            return action


def inference_fn_sync(args, config, policy, ros_operator):
    global inference_stamp

    update_observation_window(args, config, ros_operator)

    start_time = time.perf_counter()

    with observation_window_lock:
        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]

        if args.ctrl_type == "joint" or args.ctrl_type == "ee6d":
            # state: Abs Joint 14dim, ee6d also use abs joint state input (FK in model)
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            # state: Abs EEF 14dim
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

            # Use action_buffer's internal lock to safely read cur_chunk
            with action_buffer.lock:
                if action_buffer.cur_chunk is None or config["mode"] == "naive":
                    action_prefix = None
                    if config["mode"] == "rtc":
                        rospy.logwarn("RTC mode: action_prefix is None")
                else:
                    action_prefix = action_buffer.cur_chunk[(s - d) : s].copy()  # last d actions of the current chunk
                    assert action_prefix.shape[0] == d, f"{action_prefix.shape[0]} != {d}"

            update_observation_window(args, config, ros_operator)

            start_time = time.perf_counter()

            with observation_window_lock:
                # fetch images in sequence [front, right, left]
                image_arrs = [
                    observation_window[-1]["images"][config["camera_names"][0]],
                    observation_window[-1]["images"][config["camera_names"][1]],
                    observation_window[-1]["images"][config["camera_names"][2]],
                ]

                if args.ctrl_type == "joint" or args.ctrl_type == "ee6d":
                    # state: Abs Joint 14dim, ee6d also use abs joint state input (FK in model)
                    state = observation_window[-1]["qpos"]
                elif args.ctrl_type == "eef":
                    # state: Abs EEF 14dim
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


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    global inference_stamp

    policy = OpenpiClient(host=args.host, port=args.port)

    max_publish_step = config["episode_len"]

    left0 = config["left0"]
    right0 = config["right0"]

    print(config)

    ros_operator.puppet_arm_publish_continuous(left0, right0)

    print("Warmup the server...")
    policy.warmup(rtc=(args.mode == "rtc"), streaming=args.streaming)
    print("Server warmed up")

    input("Press enter to continue")
    task_time = time.time()
    ros_operator.puppet_arm_publish_continuous(left0, right0)

    # Create action buffer once (outside the loop)
    action_buffer = StreamActionBuffer(
        delay=config["delay"], exec_horizon=config["exec_horizon"], state_dim=config["state_dim"]
    )

    # Start inference thread once (outside the loop)
    start_inference_thread(args, config, policy, ros_operator, action_buffer)

    try:
        # Inference loop
        while not rospy.is_shutdown():
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)

            # Reset observation window and action buffer for new episode
            reset_observation_window()
            action_buffer.reset()

            inference_paused.clear()
            inference_stamp = 0

            # At beginning, launch sync inference
            actions = inference_fn_sync(args, config, policy, ros_operator)
            action_buffer.integrate_first_chunk(actions[: config["exec_horizon"]])

            last_valid_act = None

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                print(
                    f"[Step {t:4d}] cur_step={action_buffer.cur_step:3d} | cur_chunk={action_buffer.cur_len:3d} | next_chunk={action_buffer.next_len:3d} | cur_stamp={action_buffer.cur_stamp:3d}"
                )
                # Check for keyboard input (space to enter interactive mode)
                key = check_keyboard_input()
                if key == " ":
                    inference_paused.clear()
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        # Reset to starting position
                        ros_operator.puppet_arm_publish_continuous(left0, right0)
                        input("Press enter to continue")
                        task_time = time.time()
                        break  # Break inner loop to restart
                    elif result == "quit":
                        return  # Exit the function entirely
                    # 'continue' just resumes the loop

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
    finally:
        ros_operator.puppet_arm_publish_continuous(left0, right0)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_left_cmd_topic",
        default="/master/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_right_cmd_topic",
        default="/master/joint_right",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_topic",
        action="store",
        type=str,
        help="puppet_arm_left_topic",
        default="/puppet/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_topic",
        action="store",
        type=str,
        help="puppet_arm_right_topic",
        default="/puppet/joint_right",
        required=False,
    )
    parser.add_argument(
        "--endpose_left_cmd_topic",
        action="store",
        type=str,
        help="endpose_left_cmd_topic",
        default="/puppet/pos_cmd_left",
        required=False,
    )
    parser.add_argument(
        "--endpose_right_cmd_topic",
        action="store",
        type=str,
        help="endpose_right_cmd_topic",
        default="/puppet/pos_cmd_right",
        required=False,
    )
    parser.add_argument(
        "--endpose_left_topic",
        action="store",
        type=str,
        default="/puppet/end_pose_left",
        required=False,
    )
    parser.add_argument(
        "--endpose_right_topic",
        action="store",
        type=str,
        default="/puppet/end_pose_right",
        required=False,
    )
    parser.add_argument(
        "--robot_base_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/odom_raw",
        required=False,
    )
    parser.add_argument(
        "--robot_base_cmd_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/cmd_vel",
        required=False,
    )
    parser.add_argument(
        "--use_robot_base",
        action="store_true",
        help="Whether to use the robot base to move around",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        nargs=7,
        help="The maximum change allowed for each joint per timestep (7 values)",
        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2],
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["joint", "eef", "ee6d"],
        help="Control type for the robot arm",
        default="joint",
    )
    parser.add_argument(
        "--host",
        action="store",
        type=str,
        help="Websocket server host",
        default="10.0.0.1",
        required=False,
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        help="Websocket server port",
        default=8000,
        required=False,
    )
    parser.add_argument(
        "--task",
        action="store",
        type=str,
        help="Task name",
        required=True,
        choices=["towel", "rubbish", "tissue", "beverage", "play_pp", "pick_pp", "react"],
    )
    parser.add_argument(
        "--delay",
        type=int,
        help="Delay in steps",
        default=4,
        required=False,
    )
    parser.add_argument(
        "--exec_horizon",
        type=int,
        help="Execution horizon in steps",
        default=25,
        required=False,
    )
    parser.add_argument(
        "--mode",
        action="store",
        type=str,
        choices=["naive", "rtc"],
        help="Mode of the inference",
        default="rtc",
        required=False,
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Whether to use streaming inference",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)

    # Set terminal to raw mode for non-blocking keyboard input
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