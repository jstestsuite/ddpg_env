"""Gazebo DQN stage world + TurtleBot3 burger + goal marker, set up for fast training.

    ros2 launch ddpg_env stage.launch.py [stage:=1] [gui:=true] [rtf:=0]
                                         [physics_step:=0.004] [solver_iters:=50]

rtf is the target real-time factor (sim seconds per wall second); 0 = as fast as possible.
physics_step / solver_iters trade physics precision for speed (the stock world uses 0.001 / 150).
The agent itself is started separately with `ros2 run ddpg_env train`.
"""

import os
import re
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLD_NAME = 'dqn'


def make_world(stage, rtf, physics_step, solver_iters):
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    with open(os.path.join(tb3_gazebo, 'worlds', f'turtlebot3_dqn_stage{stage}.world')) as f:
        world = f.read()
    world = re.sub(r'<max_step_size>[^<]*<', f'<max_step_size>{physics_step}<', world)
    world = re.sub(r'<iters>[^<]*<', f'<iters>{solver_iters}<', world)
    world = re.sub(r'<real_time_factor>[^<]*<', f'<real_time_factor>{rtf}<', world)
    world = re.sub(r'<real_time_update_rate>[^<]*<',
                   f'<real_time_update_rate>{rtf / physics_step if rtf > 0 else 0}<', world)
    if stage == '2':
        # The pillar model is non-static (stage 3's plugin rotates it); as free bodies
        # resting on the floor they cut sim speed ~20x. Stage 2 never moves them.
        world = world.replace('<uri>model://turtlebot3_dqn_world/obstacles</uri>',
                              '<uri>model://turtlebot3_dqn_world/obstacles</uri>'
                              '<static>true</static>')
    path = os.path.join(tempfile.gettempdir(),
                        f'ddpg_env_stage{stage}_rtf{rtf}_dt{physics_step}_it{solver_iters}.world')
    with open(path, 'w') as f:
        f.write(world)
    return path


def launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    stage = arg('stage')
    gui = arg('gui').lower() in ('true', '1')
    world = make_world(stage, float(arg('rtf')), float(arg('physics_step')), int(arg('solver_iters')))

    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    tb3_launch = os.path.join(tb3_gazebo, 'launch')

    def gz(args):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': args}.items())

    actions = [gz(f'-r -s -v2 {world}')]
    if gui:
        actions.append(gz('-g -v2'))
    actions += [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_launch, 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_launch, 'spawn_turtlebot3.launch.py')),
            launch_arguments={'x_pose': '0.0', 'y_pose': '0.0'}.items()),
        # Goal marker (static, visual only); the env moves it with set_pose
        Node(package='ros_gz_sim', executable='create', output='screen', arguments=[
            '-world', WORLD_NAME, '-name', 'goal_box', '-x', '0.5', '-y', '0.0', '-z', '0.0',
            '-file', os.path.join(tb3_gazebo, 'models', 'turtlebot3_dqn_world', 'goal_box',
                                  'model.sdf')]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             arguments=[f'/world/{WORLD_NAME}/set_pose@ros_gz_interfaces/srv/SetEntityPose']),
    ]
    return actions


def generate_launch_description():
    # The turtlebot3 launch files read this at load time
    os.environ.setdefault('TURTLEBOT3_MODEL', 'burger')
    return LaunchDescription([
        DeclareLaunchArgument('stage', default_value='1', description='DQN stage world 1-4'),
        DeclareLaunchArgument('gui', default_value='true', description='open the Gazebo window'),
        DeclareLaunchArgument('rtf', default_value='0',
                              description='target real-time factor (0 = as fast as possible)'),
        DeclareLaunchArgument('physics_step', default_value='0.004',
                              description='physics step in seconds (stock world: 0.001)'),
        DeclareLaunchArgument('solver_iters', default_value='50',
                              description='ODE solver iterations (stock world: 150)'),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models')),
        OpaqueFunction(function=launch_setup),
    ])
