from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_controller_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),  # Install launch files
    ],
    install_requires=['setuptools', 'rclpy', 'torch'],
    zip_safe=True,
    maintainer='danieltruong',
    maintainer_email='datthanh0123734449@gmail.com',
    description='ROS 2 package for walking policy control using a neural network',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'walking_policy_node = robot_controller_rl.robot_control_node:main',
        ],
    },
)
