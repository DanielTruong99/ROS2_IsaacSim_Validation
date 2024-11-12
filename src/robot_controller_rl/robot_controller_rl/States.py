import torch

class HighLevelState:
    def __init__(self):
        # Initialize tensors for high-level state data
        self.base_linear_velocity = torch.zeros(3, dtype=torch.float32)
        self.imu_data = torch.zeros(3, dtype=torch.float32)
        self.projected_gravity = torch.zeros(3, dtype=torch.float32)
        self.foot_contact_state = torch.zeros(2, dtype=torch.float32)  # Example size
        self.height_data = torch.zeros(1, dtype=torch.float32)  # Example size

    def update(self, base_linear_velocity, imu_data, projected_gravity, foot_contact_state, height_data):
        # Update each tensor in place for memory efficiency
        self.base_linear_velocity.copy_(torch.tensor(base_linear_velocity, dtype=torch.float32))
        self.imu_data.copy_(torch.tensor(imu_data, dtype=torch.float32))
        self.projected_gravity.copy_(torch.tensor(projected_gravity, dtype=torch.float32))
        self.foot_contact_state.copy_(torch.tensor(foot_contact_state, dtype=torch.float32))
        self.height_data.copy_(torch.tensor(height_data, dtype=torch.float32))


class LowLevelState:
    def __init__(self):
        # Initialize tensors for low-level state data
        self.joint_positions = torch.zeros(10, dtype=torch.float32)  # Adjust size based on joint count
        self.joint_velocities = torch.zeros(10, dtype=torch.float32)

    def update(self, joint_positions, joint_velocities):
        # Update each tensor in place for memory efficiency
        self.joint_positions.copy_(torch.tensor(joint_positions, dtype=torch.float32))
        self.joint_velocities.copy_(torch.tensor(joint_velocities, dtype=torch.float32))

