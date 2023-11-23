import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from line_tracker import LineTracker
from std_msgs.msg import String
import cv_bridge
import datetime as dt
from enum import Enum
import threading
import time
import numpy

import cv2
import numpy as np


class LineFollower(Node):
    def __init__(self, line_tracker: LineTracker):
        super().__init__('line_follower')
        self.line_tracker = line_tracker
        self.bridge = cv_bridge.CvBridge()

        self.subscription_p = self.create_subscription(String, '/start_car', self.listener_callback, 10)

        # PR002
        self._subscription = self.create_subscription(Image, '/camera2/image_raw', self.image_callback, 10)
        self._subscription2 = self.create_subscription(Image, '/camera1/image_raw', self.stop_line_callback, 10)
        self._subscription3 = self.create_subscription(Image, '/camera1/image_raw', self.actor_collusion_callback, 10)

        # PR001
        self._subscription4 = self.create_subscription(Image, '/camera4/image_raw', self.image_callback, 10)
        self._subscription5 = self.create_subscription(Image, '/camera3/image_raw', self.stop_line_callback, 10)
        self._subscription6 = self.create_subscription(Image, '/camera3/image_raw', self.actor_collusion_callback, 10)

        # self.lidar_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # self._publisher = self.create_publisher(Twist, 'start_car', 1)

        self._publisher = self.create_publisher(Twist, '/PR002/cmd_demo', 10)
        self._publisher2 = self.create_publisher(Twist, '/PR001/cmd_demo', 10)

        self.twist = Twist()
        self.twist.linear.x = 3.0
        self.img = None

        # obstacle
        self.obstacle_found = False

        # stop_line
        self.count = 0
        self.sensorFlag = False
        self.timer = None  # 타이머 변수 추가
        self.set_timer(39, self.first_decrease_speed)
        self.start_time = time.time()
        self.car_choice = ''

    def listener_callback(self, msg):
        print(f"Received message: {msg.data}")
        self.car_choice = msg.data

    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.car_choice == 'PR001':
            self.line_tracker.process2(img)

        elif self.car_choice == 'PR002':
            self.line_tracker.process(img)

        self.twist.angular.z = (-1) * self.line_tracker._delta / 300
        self.get_logger().info('angular.z = %f' % self.twist.angular.z)

        if self.car_choice == 'PR001':
            self._publisher2.publish(self.twist)

        elif self.car_choice == 'PR002':
            self._publisher.publish(self.twist)

    def actor_collusion_callback(self, msg: Image):

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_actor = np.array([110, 50, 50])
        upper_actor = np.array([130, 255, 255])
        actor_mask = cv2.inRange(hsv, lower_actor, upper_actor)

        contours, _ = cv2.findContours(actor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("actor_mask", actor_mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)

                self.stop()
                time.sleep(3)
                self.twist.linear.x = 2.7
                self.get_logger().info('linear.x = %f' % self.twist.linear.x)
                self.get_logger().info('count = %f' % self.count)

                if self.car_choice == 'PR001':
                    self._publisher2.publish(self.twist)

                elif self.car_choice == 'PR002':
                    self._publisher.publish(self.twist)

    def stop_line_callback(self, msg: Image):
        if time.time() - self.start_time < 10:  # 타임 시간 동안은 기능을 사용하지 않음
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        stop_line_mask = cv2.inRange(hsv, lower_white, upper_white)

        h, w, d = img.shape
        search_top = int(h / 2 - 50)
        search_bot = int(h)
        stop_line_mask[0:search_top, 0:w] = 0
        stop_line_mask[search_bot:h, 0:w] = 0
        M = cv2.moments(stop_line_mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1)
            err = cy - h / 2
            self._delta = err

        cv2.imshow("stop_line_mask", stop_line_mask)
        cv2.waitKey(3)

        if np.any(stop_line_mask >= 230):

            if not self.sensorFlag:
                self.count += 1

            self.sensorFlag = True

            if self.count == 1:
                self.stop()
                time.sleep(3)
                self.twist.linear.x = 2.7
                self.get_logger().info('linear.x = %f' % self.twist.linear.x)
                self.get_logger().info('count = %f' % self.count)

                if self.car_choice == 'PR001':
                    self._publisher2.publish(self.twist)

                elif self.car_choice == 'PR002':
                    self._publisher.publish(self.twist)

                self.set_timer(25, self.increase_speed)  # 20초 후에 increase_speed 호출

            elif self.count == 2:
                self.stop()
                time.sleep(4)
                self.twist.linear.x = 3.2
                self.get_logger().info('linear.x = %f' % self.twist.linear.x)
                self.get_logger().info('count = %f' % self.count)

                if self.car_choice == 'PR001':
                    self._publisher2.publish(self.twist)

                elif self.car_choice == 'PR002':
                    self._publisher.publish(self.twist)

                self.set_timer(14, self.second_decrease_speed)

            elif self.count == 3:
                self.get_logger().info('count = %f' % self.count)
                self.stop()

        else:
            self.sensorFlag = False

    def stop(self):
        self.get_logger().info('stop() called... ')
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0

        if self.car_choice == 'PR001':
            self._publisher2.publish(self.twist)

        elif self.car_choice == 'PR002':
            self._publisher.publish(self.twist)

    def increase_speed(self):
        self.twist.linear.x = 3.55
        self.get_logger().info('linear.x = %f' % self.twist.linear.x)

        if self.car_choice == 'PR001':
            self._publisher2.publish(self.twist)

        elif self.car_choice == 'PR002':
            self._publisher.publish(self.twist)

    def first_decrease_speed(self):
        self.twist.linear.x = 2.75
        self.get_logger().info('linear.x = %f' % self.twist.linear.x)

        if self.car_choice == 'PR001':
            self._publisher2.publish(self.twist)

        elif self.car_choice == 'PR002':
            self._publisher.publish(self.twist)

    def second_decrease_speed(self):
        self.twist.linear.x = 2.6
        self.get_logger().info('linear.x = %f' % self.twist.linear.x)

        if self.car_choice == 'PR001':
            self._publisher2.publish(self.twist)

        elif self.car_choice == 'PR002':
            self._publisher.publish(self.twist)

    def set_timer(self, duration, callback):
        if self.timer is not None:
            self.timer.cancel()
        self.timer = threading.Timer(duration, callback)
        self.timer.start()

    @property
    def publisher(self):
        if self.car_choice == 'PR001':
            return self._publisher2

        elif self.car_choice == 'PR002':
            return self._publisher

def main():
    rclpy.init()
    tracker = LineTracker()
    follower = LineFollower(tracker)
    rclpy.spin(follower)
    follower.destroy_node()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        follower.stop()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
