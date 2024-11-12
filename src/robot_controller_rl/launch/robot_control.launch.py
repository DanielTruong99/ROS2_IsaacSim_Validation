from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_controller_rl',
            executable='walking_policy_node',
            name='walking_policy_control_node',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
            # parameters=[
            #     # Add any parameters if needed, like the model path
            #     {'model_path': 'path/to/your/model.pt'}  # Update the model path if using a parameter
            # ],
            # remappings=[
            #     # Add any topic remappings if needed
            #     ('/imu', '/imu/data'),  # Example: Remapping default /imu to a specific topic like /imu/data
            #     ('/joint_states', '/robot/joint_states'),
            #     ('/joint_commands', '/robot/joint_commands')
            # ]
        )
    ])
