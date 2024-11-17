# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import io
from typing import List, Optional

import carb
import numpy as np
import omni
import omni.kit.commands
import torch
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.sensor import ContactSensor

from pxr import Gf


class LegRobotFlatTerrainPolicy:
    """The Leg Humanoid running Flat Terrain Policy Locomotion Policy"""

    def __init__(
        self,
        prim_path: str,
        name: str = "leg_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize leg robot and import flat terrain policy.

        Args:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            usd_path {str} -- robot usd filepath in the directory
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot

        """
        self._stage = get_current_stage()
        self._prim_path = prim_path
        prim = get_prim_at_path(self._prim_path)
        assets_root_path = get_assets_root_path()
        if not prim.IsValid():
            prim = define_prim(self._prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")

                asset_path = "/home/ryz2/DanielWorkspace/RL/ROS2_IsaacSim_Validation/usd/leg00/leg00.usd"

                prim.GetReferences().AddReference(asset_path)

        self.robot = Articulation(prim_path=self._prim_path + "/base", name=name, position=position, orientation=orientation)
        
        from omni.isaac.sensor import _sensor
        self.contact_sensor_interface = _sensor.acquire_contact_sensor_interface()

        self._dof_control_modes: List[int] = list()

        # Policy
        policy_root_path = "/home/ryz2/DanielWorkspace/RL/ROS2_IsaacSim_Validation/isaac_sim_extension/custom_humanoid_2_python/exported_model"
        file_content = omni.client.read_file(policy_root_path + "/policy.pt")[2]
        file = io.BytesIO(memoryview(file_content).tobytes())

        self._policy = torch.jit.load(file)
        self._base_vel_lin_scale = 1
        self._base_vel_ang_scale = 1
        self._action_scale = 1.0
        self._default_joint_pos = [
            0.0,
            0.0,
            0.0,
            0.0,
            -0.2,
            -0.2,
            0.25,
            0.25,
            0.0,
            0.0
        ]
        self._previous_action = np.zeros(10)
        self._policy_counter = 0

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy.

        Argument:
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(45)
        # Base lin vel
        obs[:3] = self._base_vel_lin_scale * lin_vel_b
        # Base ang vel
        obs[3:6] = self._base_vel_ang_scale * ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9] = self._base_vel_lin_scale * command[0]
        obs[10] = self._base_vel_lin_scale * command[1]
        obs[11] = self._base_vel_ang_scale * command[2]
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        # obs[12:31] = current_joint_pos - self._default_joint_pos
        obs[12:22] = current_joint_pos
        obs[22:32] = current_joint_vel
        # Previous Action
        obs[32:42] = self._previous_action
        # Contact States
        left_contact_state = self.contact_sensor_interface.get_sensor_reading("/World/leg_robot/L_toe/Contact_Sensor", use_latest_data = True).in_contact
        right_contact_state = self.contact_sensor_interface.get_sensor_reading("/World/leg_robot/R_toe/Contact_Sensor", use_latest_data = True).in_contact
        obs[42] = float(left_contact_state)
        obs[43] = float(right_contact_state)
        # Height
        obs[44] = pos_IB[2]

        return obs

    def advance(self, dt, command):
        """
        Compute the desired articulation action and apply them to the robot articulation.

        Argument:
        dt {float} -- Timestep update in the world.
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % 4 == 0:
            obs = self._compute_observation(command)
            with torch.no_grad():
                obs = torch.from_numpy(obs).view(1, -1).float()
                self.action = self._policy(obs).detach().view(-1).numpy()
            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self._default_joint_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the articulation interface, set up robot drive mode,
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes("force")
        self.robot.get_articulation_controller().switch_control_mode("position")
        # initialize robot parameter, set joint properties based on the values from env param

        # H1 joint order
        # ['left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint',
        #  'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint',
        #  'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint',
        # 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_joint', 'right_ankle_joint', 'left_elbow_joint', 'right_elbow_joint']
        # stiffness = np.array([150, 150, 200, 150, 150, 40, 40, 200, 200, 40, 40, 200, 200, 40, 40, 20, 20, 40, 40])
        stiffness = np.array([30] * 10)
        # damping = np.array([5, 5, 5, 5, 5, 10, 10, 5, 5, 10, 10, 5, 5, 10, 10, 4, 4, 10, 10])
        damping = np.array([5] * 10)
        max_effort = np.array(
            [300, 300, 300, 300, 300, 300, 300, 300, 30, 30]
        )
        max_vel = np.zeros(10) + 100.0
        self.robot._articulation_view.set_gains(kps=stiffness, kds=damping)
        self.robot._articulation_view.set_max_efforts(max_effort)
        self.robot._articulation_view.set_max_joint_velocities(max_vel)

    def post_reset(self) -> None:
        """
        Post Reset robot articulation
        """
        self.robot.post_reset()
