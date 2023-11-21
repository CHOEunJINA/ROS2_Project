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
    world_file_name = 'car_track.world'
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

	# spawn prius_hybrid
    spawn_car_action= ExecuteProcess(
        cmd=['python3', os.path.join(os.getcwd(), 'src', 'ros2_term_project', 'test', 'spawn_car.py')],
        output='screen'
    )
    
    py_follower_node = Node(
        package='ros2_term_project',
        namespace='/',
        executable='follower',
        name='follower',
        remappings=[
            ('/start_car', 'demo/cmd_demo'),
        ]
    )


    return LaunchDescription([
        declare_argument,
        gazebo_run,
        spawn_car_action
    ])
