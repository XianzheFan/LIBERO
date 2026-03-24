"""
Multimodal Ambiguity Evaluation for LIBERO.

This module introduces systematic perturbations to LIBERO tasks to evaluate
policy robustness under multi-solution scenarios. Perturbations span two modalities:

Vision perturbations:
  1. Scene swap: alter camera viewpoints / lighting to simulate
     unfamiliar visual conditions.
  2. Obstacle insertion: place obstacles in the workspace so the robot
     must find alternative paths.
  3. Viewpoint occlusion: shift or partially block camera views so
     the target object is not immediately visible.
  4. Object swap: exchange the positions of two movable objects so
     the policy must cope with unexpected spatial layouts.

Language perturbations:
  1. Object misidentification: replace the correct object name
     in the instruction with a plausible but incorrect name.
  2. Ambiguous instructions: rewrite instructions to be vague or
     underspecified.
  3. Unfamiliar instructions: paraphrase instructions with unusual
     vocabulary or indirect phrasing.

Usage:
    # Run evaluation with all perturbations enabled:
    python examples/libero/multimodal_ambiguity.py \
        --task_suite_name libero_10 \
        --perturbations scene_swap obstacle occlusion object_swap misidentify ambiguous unfamiliar

    # Run with only vision perturbations:
    python examples/libero/multimodal_ambiguity.py \
        --perturbations scene_swap obstacle occlusion object_swap

    # Run baseline (no perturbation):
    python examples/libero/multimodal_ambiguity.py --perturbations none
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import logging
import math
import pathlib
import random
import re
from typing import Any

import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


class SceneSwapPerturbation:
    """Modify camera parameters at the MuJoCo level to simulate
    different viewpoints / lighting conditions, creating visual distribution
    shift while keeping the physical task unchanged."""

    CAMERA_PRESETS = [
        # (pos_delta, quat_euler_delta_deg) applied to the agentview camera
        {"name": "left_shifted", "pos_delta": np.array([0.0, -0.10, 0.04]),
         "angle_delta": np.array([4.0, 0.0, -8.0])},
        {"name": "right_shifted", "pos_delta": np.array([0.0, 0.10, 0.04]),
         "angle_delta": np.array([4.0, 0.0, 8.0])},
        {"name": "top_down", "pos_delta": np.array([0.0, 0.0, 0.15]),
         "angle_delta": np.array([15.0, 0.0, 0.0])},
        {"name": "zoomed_out", "pos_delta": np.array([-0.12, 0.0, 0.08]),
         "angle_delta": np.array([5.0, 0.0, 0.0])},
    ]

    def __init__(self, rng: np.random.RandomState | None = None):
        self.rng = rng or np.random.RandomState()

    def apply(self, env: OffScreenRenderEnv, preset_idx: int | None = None) -> dict:
        """Apply a camera perturbation to the environment's MuJoCo model.
        Should be called **after** env.reset() and set_init_state().
        Returns:
            dict with metadata about the applied perturbation.
        """
        if preset_idx is None:
            preset_idx = self.rng.randint(len(self.CAMERA_PRESETS))
        preset = self.CAMERA_PRESETS[preset_idx]

        sim = env.env.sim if hasattr(env, "env") else env.sim
        model = sim.model

        cam_id = model.camera_name2id("agentview")
        if cam_id < 0:
            logging.warning("agentview camera not found; skipping scene swap.")
            return {"perturbation": "scene_swap", "applied": False}

        # Perturb camera position
        original_pos = model.cam_pos[cam_id].copy()
        model.cam_pos[cam_id] += preset["pos_delta"]

        # Perturb camera orientation (small euler perturbation)
        angle_rad = np.deg2rad(preset["angle_delta"])
        # Build incremental rotation matrices
        cx, cy, cz = np.cos(angle_rad)
        sx, sy, sz = np.sin(angle_rad)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R_delta = Rz @ Ry @ Rx

        # Current camera rotation matrix from quat
        original_mat = model.cam_mat0[cam_id].reshape(3, 3).copy()
        new_mat = (R_delta @ original_mat).flatten()
        model.cam_mat0[cam_id] = new_mat

        sim.forward()

        return {
            "perturbation": "scene_swap",
            "applied": True,
            "preset": preset["name"],
            "original_pos": original_pos.tolist(),
        }


class ObstaclePerturbation:
    """Insert box-shaped obstacles into the MuJoCo scene so the robot
    must find alternative trajectories to reach the target object. Different
    obstacle placements encourage different bypass strategies.

    Implementation uses the **pre-reserved geom slots** approach: the task XML
    must contain hidden geoms named ``obstacle_0``, ``obstacle_1``, … with
    initial ``rgba="0 0 0 0"`` (transparent), ``size="0.001 0.001 0.001"``
    (negligible), ``contype="0"`` and ``conaffinity="0"`` (no collision).
    At runtime we "activate" them by writing real size / pos / rgba / collision
    flags and calling ``sim.forward()``.

    Example XML snippet to add inside the ``<worldbody>``::

        <body name="obstacle_slot_0" pos="0 0 0">
          <geom name="obstacle_0" type="box" size="0.001 0.001 0.001"
                rgba="0 0 0 0" contype="0" conaffinity="0"/>
        </body>
        <body name="obstacle_slot_1" pos="0 0 0">
          <geom name="obstacle_1" type="box" size="0.001 0.001 0.001"
                rgba="0 0 0 0" contype="0" conaffinity="0"/>
        </body>
        <!-- add more slots as needed -->
    """

    # Maximum number of reserved geom slots to look for (obstacle_0 … obstacle_N-1)
    MAX_SLOTS = 8

    # Obstacle configurations: (half-size, relative position offset from table center)
    # Table is ~80cm x 80cm; obstacles must be large enough to be clearly
    # visible in the camera and to physically block the robot arm.
    OBSTACLE_CONFIGS = [
        {"name": "front_wall", "size": [0.02, 0.15, 0.08],
         "pos_offset": [0.08, 0.0, 0.08], "rgba": [0.8, 0.2, 0.2, 1.0]},
        {"name": "left_barrier", "size": [0.12, 0.02, 0.06],
         "pos_offset": [0.0, -0.12, 0.06], "rgba": [0.2, 0.8, 0.2, 1.0]},
        {"name": "right_barrier", "size": [0.12, 0.02, 0.06],
         "pos_offset": [0.0, 0.12, 0.06], "rgba": [0.2, 0.2, 0.8, 1.0]},
        {"name": "overhead_bar", "size": [0.15, 0.15, 0.01],
         "pos_offset": [0.0, 0.0, 0.18], "rgba": [0.7, 0.7, 0.1, 0.8]},
    ]

    def __init__(self, rng: np.random.RandomState | None = None):
        self.rng = rng or np.random.RandomState()
        # Track which geom slots we activated so we can deactivate later.
        self._active_geom_ids: list[int] = []

    @staticmethod
    def _find_reserved_geom_ids(model, max_slots: int = 8) -> list[int]:
        """Return MuJoCo geom ids for reserved ``obstacle_<i>`` geoms."""
        ids: list[int] = []
        for i in range(max_slots):
            name = f"obstacle_{i}"
            try:
                gid = model.geom_name2id(name)
            except Exception:
                # mujoco-py raises ValueError when name not found
                gid = -1
            if gid >= 0:
                ids.append(gid)
        return ids

    @staticmethod
    def _activate_geom(model, geom_id: int, pos, size, rgba,
                       contype: int = 1, conaffinity: int = 1) -> None:
        """Write physical properties into a reserved geom slot to make it
        visible and collidable."""
        model.geom_size[geom_id] = size
        model.geom_pos[geom_id] = pos
        model.geom_rgba[geom_id] = rgba
        model.geom_contype[geom_id] = contype
        model.geom_conaffinity[geom_id] = conaffinity

    @staticmethod
    def _deactivate_geom(model, geom_id: int) -> None:
        """Hide a geom slot: transparent, tiny, no collision."""
        model.geom_size[geom_id] = [0.001, 0.001, 0.001]
        model.geom_rgba[geom_id] = [0.0, 0.0, 0.0, 0.0]
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    
    def apply(
        self,
        env: OffScreenRenderEnv,
        num_obstacles: int = 1,
        config_indices: list[int] | None = None,
    ) -> dict:
        """Activate pre-reserved obstacle geoms in the MuJoCo simulation.
        Should be called **after** ``env.reset()`` and ``set_init_state()``.
        Returns:
            dict with metadata about the placed obstacles.
        """
        sim = env.env.sim if hasattr(env, "env") else env.sim
        model = sim.model

        # Discover available reserved geom slots
        available_ids = self._find_reserved_geom_ids(model, self.MAX_SLOTS)
        if not available_ids:
            logging.warning(
                "No reserved obstacle geom slots (obstacle_0, obstacle_1, …) "
                "found in the MuJoCo model. Add <geom name='obstacle_0' …/> "
                "etc. to the task XML. Falling back to software-only collision "
                "checking."
            )

        if config_indices is None:
            config_indices = self.rng.choice(
                len(self.OBSTACLE_CONFIGS),
                size=min(num_obstacles, len(self.OBSTACLE_CONFIGS)),
                replace=False,
            ).tolist()

        placed: list[dict] = []
        self._active_geom_ids = []

        # Get approximate table position (usually near world origin for LIBERO)
        table_pos = np.array([0.0, 0.0, 0.8])  # default LIBERO table height

        for slot_idx, cfg_idx in enumerate(config_indices):
            cfg = self.OBSTACLE_CONFIGS[cfg_idx]
            obstacle_pos = table_pos + np.array(cfg["pos_offset"])

            # If a reserved geom slot is available, activate it (physics + visual)
            if slot_idx < len(available_ids):
                geom_id = available_ids[slot_idx]
                self._activate_geom(
                    model,
                    geom_id,
                    pos=obstacle_pos,
                    size=cfg["size"],
                    rgba=cfg["rgba"],
                )
                self._active_geom_ids.append(geom_id)
                logging.info(
                    "Activated obstacle geom id=%d ('%s') at pos=%s size=%s",
                    geom_id, cfg["name"], obstacle_pos.tolist(), cfg["size"],
                )
            else:
                logging.warning(
                    "Not enough reserved geom slots for obstacle '%s' "
                    "(need slot %d, have %d). Using software collision only.",
                    cfg["name"], slot_idx, len(available_ids),
                )

            placed.append({
                "name": cfg["name"],
                "pos": obstacle_pos.tolist(),
                "half_size": cfg["size"],
                "rgba": cfg["rgba"],
                "geom_activated": slot_idx < len(available_ids),
            })

        # Propagate model changes into the simulation
        sim.forward()

        return {
            "perturbation": "obstacle",
            "applied": len(placed) > 0,
            "obstacles": placed,
            "geoms_activated": len(self._active_geom_ids),
        }

    def deactivate(self, env: OffScreenRenderEnv) -> None:
        """Reset all previously activated obstacle geoms back to hidden state.

        Useful when calling ``env.reset()`` is not enough (e.g. when reusing
        the same model across episodes).
        """
        if not self._active_geom_ids:
            return
        sim = env.env.sim if hasattr(env, "env") else env.sim
        model = sim.model
        for geom_id in self._active_geom_ids:
            self._deactivate_geom(model, geom_id)
        sim.forward()
        self._active_geom_ids = []

    @staticmethod
    def check_collision(eef_pos: np.ndarray, obstacles: list[dict], margin: float = 0.02) -> bool:
        """Check if the end-effector is inside any obstacle bounding box.

        This is a software-level fallback. When geom slots are properly
        activated, MuJoCo handles physics collision natively.
        """
        for obs in obstacles:
            center = np.array(obs["pos"])
            half = np.array(obs["half_size"]) + margin
            if np.all(np.abs(eef_pos - center) < half):
                return True
        return False


class ObjectSwapPerturbation:
    """Swap the positions of two movable objects in the scene.

    This tests whether the policy can still identify and manipulate the
    correct target when objects appear in unexpected locations.  The swap
    is performed by exchanging the 3-D positions stored in the free-joint
    qpos of two randomly chosen objects; orientations are kept unchanged.
    """

    def __init__(self, rng: np.random.RandomState | None = None):
        self.rng = rng or np.random.RandomState()

    def apply(self, env: OffScreenRenderEnv) -> dict:
        inner_env = env.env if hasattr(env, "env") else env
        sim = inner_env.sim
        model = sim.model

        # Discover movable objects via objects_dict (set by BDDLBaseDomain)
        objects_dict = getattr(inner_env, "objects_dict", None)
        if objects_dict is None or len(objects_dict) < 2:
            logging.warning("Cannot find enough objects for swap perturbation.")
            return {"perturbation": "object_swap", "applied": False}

        # Filter to objects with free joints
        swappable = []
        for obj_name, obj in objects_dict.items():
            if not hasattr(obj, "joints") or not obj.joints:
                continue
            joint_name = obj.joints[-1]
            try:
                joint_id = model.joint_name2id(joint_name)
            except Exception:
                continue
            if model.jnt_type[joint_id] == 0:  # mjJNT_FREE
                swappable.append({
                    "name": obj_name,
                    "joint_name": joint_name,
                    "qpos_addr": int(model.jnt_qposadr[joint_id]),
                })

        if len(swappable) < 2:
            logging.warning("Fewer than 2 swappable objects found.")
            return {"perturbation": "object_swap", "applied": False}

        # Pick two random objects and swap their positions
        idx = self.rng.choice(len(swappable), size=2, replace=False)
        obj_a = swappable[idx[0]]
        obj_b = swappable[idx[1]]

        addr_a = obj_a["qpos_addr"]
        addr_b = obj_b["qpos_addr"]

        # Swap position (3D) only; keep each object's original orientation
        pos_a = sim.data.qpos[addr_a:addr_a + 3].copy()
        pos_b = sim.data.qpos[addr_b:addr_b + 3].copy()

        sim.data.qpos[addr_a:addr_a + 3] = pos_b
        sim.data.qpos[addr_b:addr_b + 3] = pos_a

        sim.forward()

        logging.info(
            "Swapped objects '%s' (pos=%s) and '%s' (pos=%s)",
            obj_a["name"], pos_a.tolist(), obj_b["name"], pos_b.tolist(),
        )

        return {
            "perturbation": "object_swap",
            "applied": True,
            "swapped": [obj_a["name"], obj_b["name"]],
            "pos_a_original": pos_a.tolist(),
            "pos_b_original": pos_b.tolist(),
        }


class OcclusionPerturbation:
    """Shift the camera or add visual occluders so the target object
    is partially hidden, requiring the policy to *search* or rely on wrist
    camera / memory to locate the target."""

    def __init__(self, rng: np.random.RandomState | None = None):
        self.rng = rng or np.random.RandomState()

    def apply(self, env: OffScreenRenderEnv) -> dict:
        """Apply occlusion by shifting the agentview camera so that the
        workspace is partially out of frame, forcing reliance on the wrist
        camera for object localisation.

        Returns:
            dict with metadata.
        """
        sim = env.env.sim if hasattr(env, "env") else env.sim
        model = sim.model

        cam_id = model.camera_name2id("agentview")
        if cam_id < 0:
            return {"perturbation": "occlusion", "applied": False}

        # Shift camera to create partial occlusion
        direction = self.rng.choice(["left", "right", "up"])
        if direction == "left":
            model.cam_pos[cam_id][1] -= 0.13
        elif direction == "right":
            model.cam_pos[cam_id][1] += 0.13
        else:  # up
            model.cam_pos[cam_id][2] += 0.13
            # Also tilt down less so objects go out of view
            angle_rad = np.deg2rad([-10.0, 0.0, 0.0])
            cx, sx = np.cos(angle_rad[0]), np.sin(angle_rad[0])
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            original_mat = model.cam_mat0[cam_id].reshape(3, 3).copy()
            model.cam_mat0[cam_id] = (Rx @ original_mat).flatten()

        # Reduce field of view to make occlusion stronger
        original_fov = model.cam_fovy[cam_id]
        model.cam_fovy[cam_id] = max(30.0, original_fov - 10.0)

        sim.forward()

        return {
            "perturbation": "occlusion",
            "applied": True,
            "direction": direction,
            "original_fov": float(original_fov),
        }


# Mapping of LIBERO object names to plausible wrong names
_OBJECT_CONFUSIONS: dict[str, list[str]] = {
    "moka pot": ["coffee maker", "kettle", "teapot"],
    "frypan": ["saucepan", "wok", "skillet"],
    "black bowl": ["dark cup", "charcoal plate", "black mug"],
    "white mug": ["cream cup", "white bowl", "porcelain glass"],
    "yellow and white mug": ["striped cup", "gold mug", "beige mug"],
    "alphabet soup": ["letter cereal", "alphabet pasta", "character soup"],
    "cream cheese box": ["butter box", "cheese block", "cream container"],
    "tomato sauce": ["ketchup", "red sauce", "marinara"],
    "butter": ["margarine", "cream cheese", "spread"],
    "chocolate pudding": ["brown dessert", "cocoa cream", "chocolate mousse"],
    "book": ["notebook", "manual", "textbook"],
    "basket": ["container", "bin", "box"],
    "plate": ["dish", "tray", "saucer"],
    "stove": ["burner", "cooktop", "hotplate"],
    "microwave": ["oven", "heater", "toaster"],
    "cabinet": ["cupboard", "shelf", "closet"],
    "drawer": ["compartment", "slot", "tray"],
    "caddy": ["holder", "organizer", "rack"],
}

# Templates for making instructions ambiguous
_AMBIGUITY_TEMPLATES = [
    "do something with {obj}",
    "handle the thing over there",
    "move that object to the place",
    "put it somewhere appropriate",
    "deal with {obj} somehow",
    "take care of the item on the table",
    "do the task with the stuff",
    "arrange {obj} properly",
]

# Templates for unfamiliar / indirect instructions
_UNFAMILIAR_TEMPLATES = [
    "Effectuate the translocation of {obj} to the designated receptacle",
    "Kindly proceed to manipulate {obj} in the prescribed manner",
    "The {obj} requires repositioning — execute accordingly",
    "Initiate the grasping protocol for {obj} and complete placement",
    "Engage the end-effector with {obj}, then fulfill the objective",
    "Perform the requisite operation involving {obj}",
    "Undertake the necessary spatial rearrangement of {obj}",
    "Accomplish the transfer maneuver with respect to {obj}",
]


class LanguagePerturbationType(enum.Enum):
    MISIDENTIFY = "misidentify"
    AMBIGUOUS = "ambiguous"
    UNFAMILIAR = "unfamiliar"


def perturb_language(
    instruction: str,
    perturbation_type: LanguagePerturbationType,
    rng: np.random.RandomState | None = None,
) -> tuple[str, dict]:
    """Apply a language perturbation to a task instruction.

    Args:
        instruction: Original LIBERO task instruction.
        perturbation_type: Which type of language perturbation to apply.
        rng: Random state for reproducibility.

    Returns:
        (perturbed_instruction, metadata_dict)
    """
    rng = rng or np.random.RandomState()
    meta = {"original": instruction, "type": perturbation_type.value}

    if perturbation_type == LanguagePerturbationType.MISIDENTIFY:
        return _perturb_misidentify(instruction, rng, meta)
    elif perturbation_type == LanguagePerturbationType.AMBIGUOUS:
        return _perturb_ambiguous(instruction, rng, meta)
    elif perturbation_type == LanguagePerturbationType.UNFAMILIAR:
        return _perturb_unfamiliar(instruction, rng, meta)
    else:
        return instruction, meta


def _perturb_misidentify(
    instruction: str, rng: np.random.RandomState, meta: dict
) -> tuple[str, dict]:
    """Replace a known object name with a wrong but plausible one."""
    new_instruction = instruction
    replaced = False
    for obj_name, confusions in _OBJECT_CONFUSIONS.items():
        if obj_name in instruction:
            wrong_name = confusions[rng.randint(len(confusions))]
            new_instruction = instruction.replace(obj_name, wrong_name, 1)
            meta["replaced"] = {obj_name: wrong_name}
            replaced = True
            break
    if not replaced:
        # Fallback: randomly swap two nouns if no known object found
        words = instruction.split()
        nouns_idx = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
        if len(nouns_idx) >= 2:
            i, j = rng.choice(nouns_idx, 2, replace=False)
            words[i], words[j] = words[j], words[i]
            new_instruction = " ".join(words)
            meta["swapped_words"] = (words[j], words[i])
    meta["perturbed"] = new_instruction
    return new_instruction, meta


def _perturb_ambiguous(
    instruction: str, rng: np.random.RandomState, meta: dict
) -> tuple[str, dict]:
    """Make the instruction vague / underspecified."""
    # Extract a likely object name for partial context
    obj_mention = "the object"
    for obj_name in _OBJECT_CONFUSIONS:
        if obj_name in instruction:
            obj_mention = obj_name
            break

    template = _AMBIGUITY_TEMPLATES[rng.randint(len(_AMBIGUITY_TEMPLATES))]
    new_instruction = template.format(obj=obj_mention)
    meta["perturbed"] = new_instruction
    return new_instruction, meta


def _perturb_unfamiliar(
    instruction: str, rng: np.random.RandomState, meta: dict
) -> tuple[str, dict]:
    """Rephrase with unusual vocabulary."""
    obj_mention = "the object"
    for obj_name in _OBJECT_CONFUSIONS:
        if obj_name in instruction:
            obj_mention = obj_name
            break

    template = _UNFAMILIAR_TEMPLATES[rng.randint(len(_UNFAMILIAR_TEMPLATES))]
    new_instruction = template.format(obj=obj_mention)
    meta["perturbed"] = new_instruction
    return new_instruction, meta


def draw_obstacles_on_image(
    img, obstacles, K, E, orig_res=256, target_res=224, alpha=1.0,
):
    """Draw 3D obstacle boxes projected onto the 2D image as semi-transparent overlays."""
    if not obstacles:
        return img
    img_drawn = img.copy()
    E_inv = np.linalg.inv(E)
    scale = target_res / orig_res

    for obs_info in obstacles:
        center = np.array(obs_info["pos"])
        half = np.array(obs_info["half_size"])
        rgba = obs_info.get("rgba", [0.8, 0.2, 0.2, 1.0])

        # 8 corners of the 3D box
        corners = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    corners.append(center + np.array([dx, dy, dz]) * half)
        corners = np.array(corners)

        # Project to 2D
        ones = np.ones((corners.shape[0], 1))
        corners_homo = np.hstack([corners, ones])
        cam_homo = (E_inv @ corners_homo.T).T
        cam = cam_homo[:, :3]
        proj = (K @ cam.T).T
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]

        # Compensate for 180-degree rotation and resize
        u = (orig_res - 1 - u) * scale
        v = v * scale

        # Axis-aligned bounding rectangle from projected corners
        u_min, u_max = int(np.clip(u.min(), 0, target_res - 1)), int(np.clip(u.max(), 0, target_res - 1))
        v_min, v_max = int(np.clip(v.min(), 0, target_res - 1)), int(np.clip(v.max(), 0, target_res - 1))

        color_rgb = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        overlay = img_drawn.copy()
        cv2.rectangle(overlay, (u_min, v_min), (u_max, v_max), color_rgb, -1)
        cv2.addWeighted(overlay, alpha, img_drawn, 1 - alpha, 0, img_drawn)
        cv2.rectangle(img_drawn, (u_min, v_min), (u_max, v_max), color_rgb, 2)

    return img_drawn


class PerturbationType(enum.Enum):
    """All available perturbation types."""
    NONE = "none"
    # Vision
    SCENE_SWAP = "scene_swap"
    OBSTACLE = "obstacle"
    OCCLUSION = "occlusion"
    OBJECT_SWAP = "object_swap"
    # Language
    MISIDENTIFY = "misidentify"
    AMBIGUOUS = "ambiguous"
    UNFAMILIAR = "unfamiliar"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20

    perturbations: list[str] = dataclasses.field(
        default_factory=lambda: ["none"],
    )

    video_out_path: str = "data/libero/ambiguity_videos"
    results_out_path: str = "data/libero/ambiguity_results.json"

    seed: int = 7


_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
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


def eval_with_perturbations(args: Args) -> None:
    """Main evaluation loop with multimodal perturbations."""
    import json

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    # Parse requested perturbations
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

    # Initialize perturbation objects
    scene_swap = SceneSwapPerturbation(rng) if PerturbationType.SCENE_SWAP in active_perturbations else None
    obstacle = ObstaclePerturbation(rng) if PerturbationType.OBSTACLE in active_perturbations else None
    occlusion = OcclusionPerturbation(rng) if PerturbationType.OCCLUSION in active_perturbations else None
    object_swap = ObjectSwapPerturbation(rng) if PerturbationType.OBJECT_SWAP in active_perturbations else None

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

    # Results tracking
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
            }

            # Deactivate any obstacle geoms from the previous episode before resetting, so the model starts clean.
            if obstacle is not None:
                obstacle.deactivate(env)

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            obstacle_info = []
            # Collect available vision perturbations and randomly pick one per episode
            vision_perturbations = []
            if scene_swap is not None:
                vision_perturbations.append(("scene_swap", scene_swap))
            if obstacle is not None:
                vision_perturbations.append(("obstacle", obstacle))
            if occlusion is not None:
                vision_perturbations.append(("occlusion", occlusion))
            if object_swap is not None:
                vision_perturbations.append(("object_swap", object_swap))

            if vision_perturbations:
                name, perturber = vision_perturbations[rng.randint(len(vision_perturbations))]
                if name == "obstacle":
                    meta = perturber.apply(env, num_obstacles=rng.randint(1, 3))
                    if meta["applied"]:
                        obstacle_info = meta["obstacles"]
                else:
                    meta = perturber.apply(env)
                episode_meta["perturbations_applied"].append(meta)

            # Re-render observations after vision perturbations
            sim = env.env.sim if hasattr(env, "env") else env.sim
            sim.forward()
            env._update_observables(force=True)
            obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()

            prompt = task_description
            if lang_perturbation_types:
                lp_type = lang_perturbation_types[rng.randint(len(lang_perturbation_types))]
                prompt, lang_meta = perturb_language(task_description, lp_type, rng)
                episode_meta["perturbations_applied"].append(lang_meta)
            episode_meta["prompt_used"] = prompt

            action_plan = collections.deque()
            t = 0
            replay_images = []
            done = False

            camera_name = "agentview"
            mujoco_sim = env.env.sim

            logging.info(f"\nTask: {task_description} | Prompt: {prompt}")

            while t < max_steps + args.num_steps_wait:
                try:
                    # Get camera matrices for obstacle projection
                    K = get_camera_intrinsic_matrix(
                        sim=mujoco_sim, camera_name=camera_name,
                        camera_height=LIBERO_ENV_RESOLUTION, camera_width=LIBERO_ENV_RESOLUTION,
                    )
                    E = get_camera_extrinsic_matrix(
                        sim=mujoco_sim, camera_name=camera_name,
                    )

                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Resize for policy input
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Draw obstacle overlays on the image for visualization
                    if obstacle_info:
                        img = draw_obstacles_on_image(
                            img, obstacle_info, K, E,
                            orig_res=LIBERO_ENV_RESOLUTION,
                            target_res=args.resize_size,
                        )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt": prompt,
                        }
                        action_chunk = client.infer(element)["actions"]
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Block action if eef is inside or would enter an obstacle
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

            # Save video with task instruction overlay
            if replay_images:
                suffix = "success" if done else "failure"
                tag = task_description.replace(" ", "_")
                perturb_tag = "_".join(
                    m.get("perturbation", m.get("type", "unknown"))
                    for m in episode_meta["perturbations_applied"]
                ) or "baseline"

                # Annotate each frame with the task instruction
                video_size = 512
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.38
                line_height = 14
                margin_x = 4

                def _wrap_text(text, font, font_scale, max_width):
                    """Split text into lines that fit within max_width."""
                    words = text.split()
                    lines, current = [], ""
                    for word in words:
                        test = f"{current} {word}".strip()
                        tw = cv2.getTextSize(test, font, font_scale, 1)[0][0]
                        if tw > max_width and current:
                            lines.append(current)
                            current = word
                        else:
                            current = test
                    if current:
                        lines.append(current)
                    return lines

                annotated_frames = []
                for frame in replay_images:
                    frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                    # Upscale for clearer text
                    frame_bgr = cv2.resize(frame_bgr, (video_size, video_size),
                                           interpolation=cv2.INTER_LANCZOS4)

                    # Collect all text lines
                    text_lines = []  # (text, color)
                    for line in _wrap_text(f"Task: {task_description}", font, font_scale, video_size - 2 * margin_x):
                        text_lines.append((line, (255, 255, 255)))
                    if prompt != task_description:
                        for line in _wrap_text(f"Prompt: {prompt}", font, font_scale, video_size - 2 * margin_x):
                            text_lines.append((line, (180, 220, 255)))
                    for line in _wrap_text(f"Perturb: {perturb_tag}", font, font_scale, video_size - 2 * margin_x):
                        text_lines.append((line, (200, 200, 200)))

                    # Draw semi-transparent background bar
                    bar_height = line_height * len(text_lines) + 4
                    overlay = frame_bgr.copy()
                    cv2.rectangle(overlay, (0, 0), (video_size, bar_height), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

                    # Draw text lines
                    for i, (text, color) in enumerate(text_lines):
                        y = line_height * (i + 1)
                        cv2.putText(frame_bgr, text, (margin_x, y),
                                    font, font_scale, color, 1, cv2.LINE_AA)

                    annotated_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"{tag}_ep{episode_idx}_{perturb_tag}_{suffix}.mp4",
                    annotated_frames,
                    fps=10,
                )

            logging.info(
                f"Episode done. Success={done} | "
                f"Total: {total_successes}/{total_episodes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(
            f"Task {task_id} success rate: "
            f"{task_successes}/{task_episodes} "
            f"({task_successes / max(task_episodes, 1) * 100:.1f}%)"
        )
        env.close()

    # Save detailed results
    results_path = pathlib.Path(args.results_out_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "task_suite": args.task_suite_name,
        "perturbations": [p.value for p in active_perturbations],
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / max(total_episodes, 1),
        "episodes": all_results,
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logging.info(f"Results saved to {results_path}")
    logging.info(
        f"Final success rate: {total_successes}/{total_episodes} "
        f"({total_successes / max(total_episodes, 1) * 100:.1f}%)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_with_perturbations(args)
