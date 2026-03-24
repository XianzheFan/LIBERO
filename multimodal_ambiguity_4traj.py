"""
Multimodal Ambiguity Evaluation for LIBERO — with 4-trajectory visualization.

Based on multimodal_ambiguity.py, this variant samples 4 action chunks at each
replan step and draws them on the image (both individually and overlaid), similar
to main_5traj.py but with 4 trajectories and multimodal perturbations.

Usage:
    # Start the SDE server with desired noise_level (higher → more diverse trajectories):
    python scripts/serve_sde_policy.py --env libero --noise_level 1.0 --num_steps 3

    # Then run evaluation:
    python examples/libero/multimodal_ambiguity_4traj.py \
        --task_suite_name libero_10 \
        --perturbations scene_swap obstacle occlusion object_swap_target object_swap_nontarget misidentify ambiguous unfamiliar

    # Run baseline (no perturbation):
    python examples/libero/multimodal_ambiguity_4traj.py --perturbations none
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import json
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
import torch
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
LIBERO_ENV_RESOLUTION = 256


class SceneSwapPerturbation:
    """Modify camera parameters to simulate different viewpoints / lighting."""

    CAMERA_PRESETS = [
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
        if preset_idx is None:
            preset_idx = self.rng.randint(len(self.CAMERA_PRESETS))
        preset = self.CAMERA_PRESETS[preset_idx]

        sim = env.env.sim if hasattr(env, "env") else env.sim
        model = sim.model

        cam_id = model.camera_name2id("agentview")
        if cam_id < 0:
            logging.warning("agentview camera not found; skipping scene swap.")
            return {"perturbation": "scene_swap", "applied": False}

        original_pos = model.cam_pos[cam_id].copy()
        model.cam_pos[cam_id] += preset["pos_delta"]

        angle_rad = np.deg2rad(preset["angle_delta"])
        cx, cy, cz = np.cos(angle_rad)
        sx, sy, sz = np.sin(angle_rad)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R_delta = Rz @ Ry @ Rx

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
    """Insert box-shaped obstacles into the MuJoCo scene."""

    MAX_SLOTS = 8

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
        self._active_geom_ids: list[int] = []

    @staticmethod
    def _find_reserved_geom_ids(model, max_slots: int = 8) -> list[int]:
        ids: list[int] = []
        for i in range(max_slots):
            name = f"obstacle_{i}"
            try:
                gid = model.geom_name2id(name)
            except Exception:
                gid = -1
            if gid >= 0:
                ids.append(gid)
        return ids

    @staticmethod
    def _activate_geom(model, geom_id: int, pos, size, rgba,
                       contype: int = 1, conaffinity: int = 1) -> None:
        model.geom_size[geom_id] = size
        model.geom_pos[geom_id] = pos
        model.geom_rgba[geom_id] = rgba
        model.geom_contype[geom_id] = contype
        model.geom_conaffinity[geom_id] = conaffinity

    @staticmethod
    def _deactivate_geom(model, geom_id: int) -> None:
        model.geom_size[geom_id] = [0.001, 0.001, 0.001]
        model.geom_rgba[geom_id] = [0.0, 0.0, 0.0, 0.0]
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    def apply(self, env: OffScreenRenderEnv, num_obstacles: int = 1,
              config_indices: list[int] | None = None) -> dict:
        sim = env.env.sim if hasattr(env, "env") else env.sim
        model = sim.model

        available_ids = self._find_reserved_geom_ids(model, self.MAX_SLOTS)
        if not available_ids:
            logging.warning(
                "No reserved obstacle geom slots found in the MuJoCo model.")

        if config_indices is None:
            config_indices = self.rng.choice(
                len(self.OBSTACLE_CONFIGS),
                size=min(num_obstacles, len(self.OBSTACLE_CONFIGS)),
                replace=False,
            ).tolist()

        placed: list[dict] = []
        self._active_geom_ids = []
        table_pos = np.array([0.0, 0.0, 0.8])

        for slot_idx, cfg_idx in enumerate(config_indices):
            cfg = self.OBSTACLE_CONFIGS[cfg_idx]
            obstacle_pos = table_pos + np.array(cfg["pos_offset"])

            if slot_idx < len(available_ids):
                geom_id = available_ids[slot_idx]
                self._activate_geom(model, geom_id, pos=obstacle_pos,
                                    size=cfg["size"], rgba=cfg["rgba"])
                self._active_geom_ids.append(geom_id)

            placed.append({
                "name": cfg["name"],
                "pos": obstacle_pos.tolist(),
                "half_size": cfg["size"],
                "rgba": cfg["rgba"],
                "geom_activated": slot_idx < len(available_ids),
            })

        sim.forward()
        return {
            "perturbation": "obstacle",
            "applied": len(placed) > 0,
            "obstacles": placed,
            "geoms_activated": len(self._active_geom_ids),
        }

    def deactivate(self, env: OffScreenRenderEnv) -> None:
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

    Two modes are supported via *involve_target*:
      - True  ("object_swap_target"):    one of the swapped objects is the
        target mentioned in the task instruction.
      - False ("object_swap_nontarget"): neither swapped object is the target;
        the scene layout changes but the target stays in place.
    """

    def __init__(self, rng: np.random.RandomState | None = None,
                 involve_target: bool = True):
        self.rng = rng or np.random.RandomState()
        self.involve_target = involve_target

    @staticmethod
    def _is_target_object(obj_name: str, task_description: str) -> bool:
        """Check whether *obj_name* is mentioned in the task description.

        LIBERO object names typically look like ``akita_black_bowl_1`` while
        the task description says ``black bowl``.  We strip a leading vendor /
        brand token and a trailing numeric suffix, then check whether any
        contiguous sub-sequence of the remaining tokens appears in the
        description.  For example ``akita_black_bowl_1`` produces the
        candidate ``black bowl`` which matches ``pick up the black bowl``.
        """
        desc_lower = task_description.lower()
        # Split object name into tokens
        tokens = obj_name.lower().replace("_", " ").split()

        # Strip trailing numeric suffix (e.g. "1", "02")
        if tokens and tokens[-1].isdigit():
            tokens = tokens[:-1]
        if not tokens:
            return False

        # Try all contiguous sub-sequences from longest to shortest
        for length in range(len(tokens), 0, -1):
            for start in range(len(tokens) - length + 1):
                candidate = " ".join(tokens[start:start + length])
                if len(candidate) >= 3 and candidate in desc_lower:
                    return True
        return False

    def apply(self, env: OffScreenRenderEnv,
              task_description: str = "") -> dict:
        perturbation_label = (
            "object_swap_target" if self.involve_target
            else "object_swap_nontarget"
        )

        inner_env = env.env if hasattr(env, "env") else env
        sim = inner_env.sim
        model = sim.model

        # Discover movable objects via objects_dict (set by BDDLBaseDomain)
        objects_dict = getattr(inner_env, "objects_dict", None)
        if objects_dict is None or len(objects_dict) < 2:
            logging.warning("Cannot find enough objects for swap perturbation.")
            return {"perturbation": perturbation_label, "applied": False}

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
                    "is_target": self._is_target_object(obj_name, task_description),
                })

        if len(swappable) < 2:
            logging.warning("Fewer than 2 swappable objects found.")
            return {"perturbation": perturbation_label, "applied": False}

        logging.info(
            "Swappable objects: %s | task: '%s'",
            [(o["name"], "TARGET" if o["is_target"] else "non-target") for o in swappable],
            task_description,
        )

        # Split into target / non-target pools
        target_objs = [o for o in swappable if o["is_target"]]
        nontarget_objs = [o for o in swappable if not o["is_target"]]

        if self.involve_target:
            # One object must be a target, the other can be anything else
            if not target_objs or len(swappable) < 2:
                logging.warning("No target object found for object_swap_target.")
                return {"perturbation": perturbation_label, "applied": False}
            obj_a = target_objs[self.rng.randint(len(target_objs))]
            others = [o for o in swappable if o["name"] != obj_a["name"]]
            if not others:
                return {"perturbation": perturbation_label, "applied": False}
            obj_b = others[self.rng.randint(len(others))]
        else:
            # Neither object should be the target
            if len(nontarget_objs) < 2:
                logging.warning(
                    "Fewer than 2 non-target swappable objects for "
                    "object_swap_nontarget."
                )
                return {"perturbation": perturbation_label, "applied": False}
            idx = self.rng.choice(len(nontarget_objs), size=2, replace=False)
            obj_a = nontarget_objs[idx[0]]
            obj_b = nontarget_objs[idx[1]]

        addr_a = obj_a["qpos_addr"]
        addr_b = obj_b["qpos_addr"]

        # Swap position (3D) only; keep each object's original orientation
        pos_a = sim.data.qpos[addr_a:addr_a + 3].copy()
        pos_b = sim.data.qpos[addr_b:addr_b + 3].copy()

        sim.data.qpos[addr_a:addr_a + 3] = pos_b
        sim.data.qpos[addr_b:addr_b + 3] = pos_a

        sim.forward()

        logging.info(
            "Swapped objects '%s' (pos=%s) and '%s' (pos=%s) [involve_target=%s]",
            obj_a["name"], pos_a.tolist(), obj_b["name"], pos_b.tolist(),
            self.involve_target,
        )

        return {
            "perturbation": perturbation_label,
            "applied": True,
            "involve_target": self.involve_target,
            "swapped": [obj_a["name"], obj_b["name"]],
            "pos_a_original": pos_a.tolist(),
            "pos_b_original": pos_b.tolist(),
        }


class OcclusionPerturbation:
    """Shift the camera to partially hide the target object."""

    def __init__(self, rng: np.random.RandomState | None = None):
        self.rng = rng or np.random.RandomState()

    def apply(self, env: OffScreenRenderEnv) -> dict:
        sim = env.env.sim if hasattr(env, "env") else env.sim
        model = sim.model

        cam_id = model.camera_name2id("agentview")
        if cam_id < 0:
            return {"perturbation": "occlusion", "applied": False}

        direction = self.rng.choice(["left", "right", "up"])
        if direction == "left":
            model.cam_pos[cam_id][1] -= 0.13
        elif direction == "right":
            model.cam_pos[cam_id][1] += 0.13
        else:
            model.cam_pos[cam_id][2] += 0.13
            angle_rad = np.deg2rad([-10.0, 0.0, 0.0])
            cx, sx = np.cos(angle_rad[0]), np.sin(angle_rad[0])
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            original_mat = model.cam_mat0[cam_id].reshape(3, 3).copy()
            model.cam_mat0[cam_id] = (Rx @ original_mat).flatten()

        original_fov = model.cam_fovy[cam_id]
        model.cam_fovy[cam_id] = max(30.0, original_fov - 10.0)

        sim.forward()

        return {
            "perturbation": "occlusion",
            "applied": True,
            "direction": direction,
            "original_fov": float(original_fov),
        }


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


def _perturb_misidentify(instruction, rng, meta):
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
        words = instruction.split()
        nouns_idx = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
        if len(nouns_idx) >= 2:
            i, j = rng.choice(nouns_idx, 2, replace=False)
            words[i], words[j] = words[j], words[i]
            new_instruction = " ".join(words)
            meta["swapped_words"] = (words[j], words[i])
    meta["perturbed"] = new_instruction
    return new_instruction, meta


def _perturb_ambiguous(instruction, rng, meta):
    obj_mention = "the object"
    for obj_name in _OBJECT_CONFUSIONS:
        if obj_name in instruction:
            obj_mention = obj_name
            break
    template = _AMBIGUITY_TEMPLATES[rng.randint(len(_AMBIGUITY_TEMPLATES))]
    new_instruction = template.format(obj=obj_mention)
    meta["perturbed"] = new_instruction
    return new_instruction, meta


def _perturb_unfamiliar(instruction, rng, meta):
    obj_mention = "the object"
    for obj_name in _OBJECT_CONFUSIONS:
        if obj_name in instruction:
            obj_mention = obj_name
            break
    template = _UNFAMILIAR_TEMPLATES[rng.randint(len(_UNFAMILIAR_TEMPLATES))]
    new_instruction = template.format(obj=obj_mention)
    meta["perturbed"] = new_instruction
    return new_instruction, meta


class PerturbationType(enum.Enum):
    NONE = "none"
    SCENE_SWAP = "scene_swap"
    OBSTACLE = "obstacle"
    OCCLUSION = "occlusion"
    OBJECT_SWAP_TARGET = "object_swap_target"
    OBJECT_SWAP_NONTARGET = "object_swap_nontarget"
    MISIDENTIFY = "misidentify"
    AMBIGUOUS = "ambiguous"
    UNFAMILIAR = "unfamiliar"


def draw_trajectory_on_image(
    img, current_eef_pos, action_chunk, K, E,
    orig_res=256, target_res=224, action_scale=0.05,
    pos_limit=None, tracking_factor=0.35,
    line_color=(235, 206, 135), point_color=(0, 215, 255),
):
    """Project a 3D action chunk trajectory onto a 2D image and draw it."""
    traj_3d = [current_eef_pos]
    curr_pos = current_eef_pos.copy()

    for step_action in action_chunk:
        delta_action = step_action[:3]
        clipped_action = np.clip(delta_action, -1.0, 1.0)
        delta_3d = clipped_action * action_scale

        goal_pos = curr_pos + delta_3d
        if pos_limit is not None:
            goal_pos = np.clip(goal_pos, pos_limit[0], pos_limit[1])

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

    # Compensate for 180-degree image rotation and resizing
    u = orig_res - 1 - u
    scale = target_res / orig_res
    u = u * scale
    v = v * scale

    img_drawn = img.copy()
    points_2d = np.vstack((u, v)).T.astype(np.int32)

    for i in range(len(points_2d) - 1):
        pt1 = tuple(points_2d[i])
        pt2 = tuple(points_2d[i + 1])
        cv2.line(img_drawn, pt1, pt2, line_color, 1)
        cv2.circle(img_drawn, pt1, 2, point_color, -1)

    cv2.circle(img_drawn, tuple(points_2d[0]), 2, (120, 200, 80), -1)   # Green start
    cv2.circle(img_drawn, tuple(points_2d[-1]), 2, (255, 127, 80), -1)  # Coral end

    return img_drawn


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

    num_samples: int = 4  # number of trajectory samples to visualize

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


# 4 distinct colour pairs: (line_color_BGR, point_color_BGR)
TRAJ_COLORS = [
    ((235, 206, 135), (0, 215, 255)),    # gold / orange
    ((144, 238, 144), (34, 139, 34)),     # light green / dark green
    ((255, 182, 193), (220, 20, 60)),     # pink / crimson
    ((173, 216, 230), (0, 0, 139)),       # light blue / dark blue
]


def eval_with_perturbations(args: Args) -> None:
    """Main evaluation loop with multimodal perturbations and 4-trajectory viz."""
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

            # Deactivate any obstacle geoms from the previous episode
            if obstacle is not None:
                obstacle.deactivate(env)

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # --- Get controller action scale ---
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

            # Re-render observations after vision perturbations
            sim = env.env.sim if hasattr(env, "env") else env.sim
            sim.forward()
            env._update_observables(force=True)
            obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()

            # Apply language perturbation
            prompt = task_description
            if lang_perturbation_types:
                lp_type = lang_perturbation_types[rng.randint(len(lang_perturbation_types))]
                prompt, lang_meta = perturb_language(task_description, lp_type, rng)
                episode_meta["perturbations_applied"].append(lang_meta)
            episode_meta["prompt_used"] = prompt

            action_plan = collections.deque()
            t = 0
            replay_images_single = []
            replay_images_multi = []
            done = False

            camera_name = "agentview"
            mujoco_sim = env.env.sim

            logging.info(f"\nTask: {task_description} | Prompt: {prompt}")

            while t < max_steps + args.num_steps_wait:
                try:
                    # Get camera matrices for trajectory projection
                    K = get_camera_intrinsic_matrix(
                        sim=mujoco_sim,
                        camera_name=camera_name,
                        camera_height=LIBERO_ENV_RESOLUTION,
                        camera_width=LIBERO_ENV_RESOLUTION,
                    )
                    E = get_camera_extrinsic_matrix(
                        sim=mujoco_sim,
                        camera_name=camera_name,
                    )

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

                    # Draw obstacle overlays on the image for visualization
                    if obstacle_info:
                        img = draw_obstacles_on_image(
                            img, obstacle_info, K, E,
                            orig_res=LIBERO_ENV_RESOLUTION,
                            target_res=args.resize_size,
                        )

                    replay_images_single.append(img.copy())
                    replay_images_multi.append(img.copy())

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

                        # ---- Sample 4 action chunks ----
                        action_chunks = [client.infer(element)["actions"] for _ in range(args.num_samples)]

                        img_multi = img.copy()

                        for i, chunk in enumerate(action_chunks):
                            line_c, point_c = TRAJ_COLORS[i]

                            # Draw single trajectory image
                            img_single_traj = draw_trajectory_on_image(
                                img=img,
                                current_eef_pos=obs["robot0_eef_pos"],
                                action_chunk=chunk[:args.replan_steps],
                                K=K, E=E,
                                orig_res=LIBERO_ENV_RESOLUTION,
                                target_res=args.resize_size,
                                action_scale=action_scale,
                                line_color=line_c,
                                point_color=point_c,
                            )

                            # Draw all trajectories overlaid on the same image
                            img_multi = draw_trajectory_on_image(
                                img=img_multi,
                                current_eef_pos=obs["robot0_eef_pos"],
                                action_chunk=chunk[:args.replan_steps],
                                K=K, E=E,
                                orig_res=LIBERO_ENV_RESOLUTION,
                                target_res=args.resize_size,
                                action_scale=action_scale,
                                line_color=line_c,
                                point_color=point_c,
                            )

                            # Use the first trajectory for the single-traj replay
                            if i == 0:
                                replay_images_single[-1] = img_single_traj

                        replay_images_multi[-1] = img_multi

                        # Execute the first sampled trajectory
                        action_plan.extend(action_chunks[0][:args.replan_steps])

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

            # Save replay videos with upscaling and text annotation
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            perturb_tag = "_".join(
                m.get("perturbation", m.get("type", "unknown"))
                for m in episode_meta["perturbations_applied"]
            ) or "baseline"

            # Build detailed perturbation description for video overlay
            perturb_details: list[str] = []
            for m in episode_meta["perturbations_applied"]:
                ptype = m.get("perturbation", m.get("type", "unknown"))
                if not m.get("applied", True):
                    perturb_details.append(f"{ptype} (not applied)")
                    continue

                if ptype == "scene_swap":
                    preset = m.get("preset", "?")
                    pos_delta = m.get("original_pos", [])
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

            def _annotate_frames(frames):
                """Upscale frames to video_size and overlay text annotations."""
                annotated = []
                for frame in frames:
                    frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                    frame_bgr = cv2.resize(frame_bgr, (video_size, video_size),
                                           interpolation=cv2.INTER_LANCZOS4)

                    text_lines = []
                    for line in _wrap_text(f"Task: {task_description}", font, font_scale, video_size - 2 * margin_x):
                        text_lines.append((line, (255, 255, 255)))
                    if prompt != task_description:
                        for line in _wrap_text(f"Prompt: {prompt}", font, font_scale, video_size - 2 * margin_x):
                            text_lines.append((line, (180, 220, 255)))
                    for line in _wrap_text(f"Perturb: {perturb_detail_str}", font, font_scale, video_size - 2 * margin_x):
                        text_lines.append((line, (200, 200, 200)))

                    bar_height = line_height * len(text_lines) + 4
                    overlay = frame_bgr.copy()
                    cv2.rectangle(overlay, (0, 0), (video_size, bar_height), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

                    for i, (text, color) in enumerate(text_lines):
                        y = line_height * (i + 1)
                        cv2.putText(frame_bgr, text, (margin_x, y),
                                    font, font_scale, color, 1, cv2.LINE_AA)

                    annotated.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                return annotated

            if replay_images_single:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"{task_segment}_ep{episode_idx}_{perturb_tag}_{suffix}.mp4",
                    _annotate_frames(replay_images_single),
                    fps=10,
                )

            if replay_images_multi:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"{task_segment}_ep{episode_idx}_{perturb_tag}_{suffix}_multi.mp4",
                    _annotate_frames(replay_images_multi),
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
