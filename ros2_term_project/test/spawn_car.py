import os
import sys
import rclpy

def main(args=None):
    rclpy.init(args=args)

    os.system("ros2 run gazebo_ros spawn_entity.py -file /home/ros2/Ros2Projects/oom_ws/src/ros2_term_project/models/prius_hybrid_PR001.sdf -entity PR001")
    os.system("ros2 run gazebo_ros spawn_entity.py -file /home/ros2/Ros2Projects/oom_ws/src/ros2_term_project/models/prius_hybrid_PR002.sdf -entity PR002" )

    rclpy.shutdown()

if __name__ == '__main__':
    main()