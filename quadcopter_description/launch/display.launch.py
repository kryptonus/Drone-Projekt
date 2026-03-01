import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('quadcopter_description')
    xacro_file = os.path.join(pkg_path, 'urdf', 'quadcopter.urdf.xacro')
    robot_description_raw = xacro.process_file(xacro_file).toxml()
    world_file = os.path.join(pkg_path, 'launch', 'quadcopter.world')

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file,
                 '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description_raw,
                'use_sim_time': True
            }]
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description',
                       '-entity', 'stealth_quad',
                       '-x', '0.0', '-y', '0.0', '-z', '0.5'],
            output='screen'
        ),
    ])
