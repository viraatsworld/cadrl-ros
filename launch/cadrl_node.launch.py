import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Declare Launch Argument (equivalent to <arg> in ROS 1)
    jackal_speed_arg = DeclareLaunchArgument(
        'jackal_speed',
        default_value='1.2',
        description='The default speed for the Jackal.'
    )

    jackal_speed = LaunchConfiguration('jackal_speed')

    # 2. Node Definition (equivalent to <node> in ROS 1)
    cadrl_node = Node(
        package='cadrl_ros',
        executable='cadrl_node.py',
        name='cadrl_node',
        output='screen',
        # Namespace is set here
        namespace='/JA01',

        # 3. Remappings (equivalent to <remap> in ROS 1)
        # The structure is [('from', 'to')]
        remappings=[
            # Publications (Note: '~' prefix for relative topic names is automatically handled
            # for unmapped topics, but here we explicitly define the remappings)
            ('~/other_vels', 'other_vels'),
            ('~/nn_cmd_vel', 'nn_cmd_vel'),
            ('~/pose_marker', 'pose_marker'),
            ('~/path_marker', 'path_marker'),
            ('~/goal_path_marker', 'goal_path_marker'),
            ('~/agent_marker', 'other_agents_marker'),
            ('~/agent_markers', 'other_agents_markers'),

            # Subscriptions
            ('~/pose', 'pose'),
            ('~/velocity', 'velocity'),
            ('~/safe_actions', 'local_path_finder/safe_actions'),
            ('~/planner_mode', 'planner_fsm/mode'),
            ('~/goal', 'move_base_simple/goal'),
            ('~/clusters', 'cluster/output/clusters'),
            ('~/peds', 'ped_manager/ped_recent'),
        ],

        # 4. Parameters (equivalent to <param> in ROS 1)
        # Parameters should be set relative to the node.
        # The '~' prefix is not needed in the key as parameters are local by default.
        parameters=[
            {'jackal_speed': jackal_speed}
        ]
    )

    # 5. Return the LaunchDescription
    return LaunchDescription([
        jackal_speed_arg,
        cadrl_node,
    ])

#