import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from .line_tracker import LineTracker
import cv_bridge
import time
import numpy
import cv2
import numpy as np

class LineFollower(Node):
    def __init__(self, line_tracker: LineTracker):
        super().__init__('line_follower')
        self.line_tracker = line_tracker
        self.bridge = cv_bridge.CvBridge()
        self._subscription = self.create_subscription(Image, '/camera2/image_raw', self.image_callback, 10)
        self._subscription2 = self.create_subscription(Image, '/camera1/image_raw',self.stop_line_callback, 10)
        self._publisher = self.create_publisher(Twist, 'cmd_vel', 1)
        self.twist = Twist()
        self.twist.linear.x = 2.8
        self.img = None
        self.is_stopped = False 
        self.count =0
        self.start_time = time.time()  # 시작 시간 저장
        
    def stop_line_callback(self, msg: Image):
        if time.time() - self.start_time < 10:  # 타임 시간 동안은 기능을 사용하지 않음
            return
            
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200]) 
        upper_white = np.array([180, 30, 255])
        stop_line_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        h, w, d = img.shape
        search_top = int( h / 2 - 50 )
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
        cv2.imshow("window1", img)
        cv2.imshow("stop_line_mask", stop_line_mask)
        cv2.waitKey(3)   
        if np.any(stop_line_mask == 255):  
            self.count += 1
            if self.count < 3:
                self.stop()  
                time.sleep(3)  
                self.twist.linear.x = 2.8
                self.get_logger().info('linear.x = %f' % self.twist.linear.x)            
                self._publisher.publish(self.twist) 
                self.is_stopped = True     
            if self.count == 3:
                self.stop()  
                time.sleep(100)  
                                 
                         
                    
    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.line_tracker.process(img)
        self.twist.angular.z = (-1) * self.line_tracker._delta / 220
        self._publisher.publish(self.twist)
 
    def stop(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self._publisher.publish(self.twist)

    @property
    def publisher(self):
        return self._publisher


def main():
        rclpy.init()
        tracker = LineTracker()
        follower = LineFollower(tracker)
        try:
            rclpy.spin(follower)
        except KeyboardInterrupt:
            follower.stop()
            follower.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    
    
