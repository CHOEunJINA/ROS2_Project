import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration

print(os.path.realpath(__file__))

ld = LaunchDescription()


def generate_launch_description():
    # configuration
    world = LaunchConfiguration('world')
    print('world =', world)
    world_file_name = 'gazebo_launch.world'
    world = os.path.join(get_package_share_directory('ros2_term_project'),
                         'worlds', world_file_name)
    print('world file name = %s' % world)

    # ld = LaunchDescription()
    declare_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    gazebo_run = ExecuteProcess(
        cmd=['gazebo', '-s', 'libgazebo_ros_factory.so', world],
        output='screen')

    spawn_car = ExecuteProcess(
        cmd=['python3', os.path.join(os.getcwd(), 'src', 'ros2_term_project', 'test', 'spawn_car.py')],
        output='screen'
    )

    return LaunchDescription([
        declare_argument,
        gazebo_run,
        spawn_car,
    ])