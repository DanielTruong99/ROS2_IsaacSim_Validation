import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float32, Int8MultiArray
import torch  # type: ignore

# import numpy as np # type: ignore

from robot_controller_rl.States import HighLevelState, LowLevelState


class WalkingPolicyControlNode(Node):
    def __init__(self):
        super().__init__("walking_policy_control_node")

        # Load the PyTorch model
        self.model_path = "/home/ryz2/DanielWorkspace/RL/ROS2_IsaacSim_Validation/src/robot_controller_rl/robot_controller_rl/policy.pt"  # Update this with your model path
        try:
            self.model = torch.load(self.model_path)
            self.model.eval()  # Set the model to evaluation mode
            self.get_logger().info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        # Subscribe to required topics
        self.body_twist_subscription = self.create_subscription(
            Twist, "/base_twist", self.body_twist_callback, 10
        )
        self.imu_subscription = self.create_subscription(
            Imu, "/imu", self.imu_callback, 10
        )
        self.base_gravity_subscription = self.create_subscription(
            Float32MultiArray, "/base_gravity", self.base_gravity_callback, 10
        )
        self.contact_states_subscription = self.create_subscription(
            Int8MultiArray, "/contact_states", self.contact_states_callback, 10
        )
        self.height_data_subscription = self.create_subscription(
            Float32, "/height_data", self.height_data_callback, 10
        )
        self.joint_state_subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

        # Publisher for joint commands
        self.joint_command_publisher = self.create_publisher(
            JointState, "/joint_commands", 10
        )

        # Setup buffer for calculations
        self.setup_buffer()

        # Timer to calculate walking policy at 100Hz (10 ms interval)
        self.control_frequency = 100  # Hz
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_loop
        )

        #! successfully initialized
        self.get_logger().info("Walking Policy Control Node initialized.")

    def setup_buffer(self):
        # Store latest sensor data
        self.latest_body_twist = None
        self.latest_imu_data = None
        self.latest_base_gravity = None
        self.latest_contact_states = None
        self.latest_height_data = None
        self.latest_joint_state_data = None

        # Instantiate high-level and low-level state classes
        self.high_level_state = HighLevelState()
        self.low_level_state = LowLevelState()
        self.previous_output = torch.zeros(
            10, dtype=torch.float32
        )  # Adjust size to model output
        self.commands = torch.zeros(
            3, dtype=torch.float32
        )  # Initialize commands to zeros

        self.commands[0] = 0.65  # Set the forward velocity command

        self.joint_names = [
            "L_hip_joint",
            "R_hip_joint",
            "L_hip2_joint",
            "R_hip2_joint",
            "L_thigh_joint",
            "R_thigh_joint",
            "L_calf_joint",
            "R_calf_joint",
            "L_toe_joint",
            "R_toe_joint",
        ]
        self.default_joint_pos = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, -0.2, -0.2, 0.25, 0.25, 0.0, 0.0]],
            dtype=torch.float32,
        )

    # Callbacks for each subscribed topic
    def body_twist_callback(self, msg):
        self.latest_body_twist = [msg.linear.x, msg.linear.y, msg.linear.z]

    def imu_callback(self, msg):
        self.latest_imu_data = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ]

    def base_gravity_callback(self, msg):
        self.latest_base_gravity = list(msg.data)

    def contact_states_callback(self, msg):
        self.latest_contact_states = list(msg.data)

    def height_data_callback(self, msg):
        self.latest_height_data = [msg.data / 0.65]  # Normalize height data

    def joint_state_callback(self, msg):
        self.latest_joint_state_data = {
            "positions": list(msg.position),
            "velocities": list(msg.velocity),
        }

    def preprocess_input(self):
        # Ensure all necessary data is available
        if (
            self.latest_body_twist is None
            or self.latest_imu_data is None
            or self.latest_base_gravity is None
            or self.latest_contact_states is None
            or self.latest_height_data is None
            or self.latest_joint_state_data is None
        ):
            return None

        # Update high-level and low-level states in place
        self.high_level_state.update(
            base_linear_velocity=self.latest_body_twist,
            imu_data=self.latest_imu_data,
            projected_gravity=self.latest_base_gravity,
            foot_contact_state=self.latest_contact_states,
            height_data=self.latest_height_data,
        )
        self.low_level_state.update(
            joint_positions=self.latest_joint_state_data["positions"],
            joint_velocities=self.latest_joint_state_data["velocities"],
        )

        self.get_logger().warn("received sensor data...")

        # Construct input tensor by concatenating all components directly
        input_tensor = torch.cat(
            (
                self.high_level_state.base_linear_velocity,
                self.high_level_state.imu_data,
                self.high_level_state.projected_gravity,
                self.commands,
                self.low_level_state.joint_positions,
                self.low_level_state.joint_velocities,
                self.previous_output,
                self.high_level_state.foot_contact_state,
                self.high_level_state.height_data,
            ),
            dim=0,
        ).unsqueeze(
            0
        )  # Add batch dimension

        return input_tensor

    def postprocess_output(self, output):
        self.previous_output = self.previous_output if torch.isnan(output).any() else output # Store the output for the next iteration

        processed_actions = 1.0 * self.previous_output + self.default_joint_pos
        return processed_actions.squeeze().tolist()

    def control_loop(self):
        """
        Executes the control loop for the robot.
        ! This method is called at a fixed frequency to calculate the walking policy.
        ! This is main logic.

        This method performs the following steps:
        1. Prepares the input vector from the latest sensor data.
        2. Checks if the input data is available; if not, logs a warning and skips the iteration.
        3. Runs inference using the pre-trained model to calculate the walking policy.
        4. Processes the output from the model to generate joint commands.
        5. Converts the processed actions to a list and publishes it as joint commands.
        6. Logs the published joint commands.

        Note:
            - This method assumes that the model, preprocess_input, postprocess_output,
              joint_command_publisher, and get_logger methods are defined elsewhere in the class.
            - The method uses PyTorch for inference and ROS2 for publishing joint commands.

        Returns:
            None
        """
        # Prepare the input vector from the latest sensor data
        input_data = self.preprocess_input()

        if input_data is None:
            self.get_logger().warn("Waiting for sensor data...")
            return  # Skip this iteration if data is incomplete

        # Run inference to calculate the walking policy
        with torch.no_grad():  # Disable gradient calculations for inference
            output = self.model(input_data).squeeze()

        processed_actions = self.postprocess_output(output)

        # Convert the output tensor to a list and publish it as joint commands
        joint_command_msg = JointState()
        joint_command_msg.name = self.joint_names
        joint_command_msg.position = processed_actions

        # Publish the joint commands
        self.joint_command_publisher.publish(joint_command_msg)
        self.get_logger().info(
            f"Published joint commands: {joint_command_msg.position}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = WalkingPolicyControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
