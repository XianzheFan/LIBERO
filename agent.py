#!/usr/bin/env python3
import numpy as np
import math
import zmq
from collections import deque
from typing import Dict, Tuple, Optional, Any
import numpy.typing as npt


class RemoteAgent:
    PROPRIO_HISTORY_SIZE = 4

    def __init__(self, instruction: str, port: int) -> None:
        self._validate_inputs(instruction, port)
        self._setup_zmq_connection(port)
        self._initialize_state(instruction)

    def _validate_inputs(self, instruction: str, port: int) -> None:
        if not instruction.strip():
            raise ValueError("Instruction cannot be empty")
        if not (1 <= port <= 65535):
            raise ValueError(f"Port must be between 1-65535, got {port}")

    def _setup_zmq_connection(self, port: int) -> None:
        try:
            self.zmq_context = zmq.Context()
            self.socket = self.zmq_context.socket(zmq.REQ)
            self.socket.connect(f"tcp://127.0.0.1:{port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 40000)  # 40s timeout
        except Exception as e:
            raise ConnectionError(f"Failed to establish ZMQ connection: {e}")

    def _initialize_state(self, instruction: str) -> None:
        self.proprio_history = deque(maxlen=self.PROPRIO_HISTORY_SIZE)
        self.instruction = instruction
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
        self, obs: Dict[str, Any], debug: bool = False
    ) -> Tuple[npt.NDArray[np.float64], Optional[Any]]:
        """
        Forward action from server to env.step() as-is.
        """
        self._process_proprio(obs)

        if len(self.pred_actions) == 0:
            self._post_and_get(obs, debug=debug)

        action, bbox = self.pred_actions.popleft()

        # Ensure correct shape/type only (NOT a semantic transformation)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Agent action must be 7-dim; got shape={action.shape}")

        action[6] = -action[6]  # 1 for open, -1 for close
        return action, bbox

    def _post_and_get(self, obs: Dict[str, Any], debug: bool = False) -> None:
        data = {
            "image_array": [obs["agentview_image"][::-1, ::-1]],
            "image_wrist_array": [obs["robot0_eye_in_hand_image"][::-1, ::-1]],
            "depth_array": [obs["agentview_depth"][::-1, ::-1]],
            "depth_wrist_array": [obs["robot0_eye_in_hand_depth"][::-1, ::-1]],
            "proprio_array": [np.copy(proprio) for proprio in self.proprio_history],
            "env_id": 1,
            "text": self.instruction,
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