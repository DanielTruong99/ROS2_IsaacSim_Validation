# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import carb
import numpy as np
import omni
import omni.appwindow  # Contains handle to keyboard
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.examples.humanoid.h1 import H1FlatTerrainPolicy


class HumanoidExample(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 200.0
        self._world_settings["rendering_dt"] = 8.0 / 200.0
        self._enter_toggled = 0
        self._base_command = [0.0, 0.0, 0.0]
        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [0.75, 0.0, 0.0],
            "UP": [0.75, 0.0, 0.0],
            # yaw command (positive)
            "NUMPAD_4": [0.0, 0.0, 0.75],
            "LEFT": [0.0, 0.0, 0.75],
            # yaw command (negative)
            "NUMPAD_6": [0.0, 0.0, -0.75],
            "RIGHT": [0.0, 0.0, -0.75],
        }

    def setup_scene(self) -> None:
        world = self.get_world()
        world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )

        self.h1 = H1FlatTerrainPolicy(
            prim_path="/World/H1",
            name="H1",
            position=np.array([0, 0, 1.05]),
        )

    async def setup_post_load(self) -> None:
        world = self.get_world()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._physics_ready = False
        world.add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._timeline_timer_callback_fn
        )
        await world.play_async()
        self.h1.initialize()

    async def setup_post_reset(self) -> None:
        self.h1.post_reset()
        world = self.get_world()
        self._physics_ready = False
        await world.play_async()
        self.h1.initialize()

    def on_physics_step(self, step_size) -> None:
        if self._physics_ready:
            self.h1.advance(step_size, self._base_command)
        else:
            self._physics_ready = True

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Subscriber callback to when kit is updated."""
        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.h1:
            self.h1.post_reset()

    def world_cleanup(self):
        world = self.get_world()
        self._event_timer_callback = None
        if world.physics_callback_exists("physics_step"):
            world.remove_physics_callback("physics_step")
