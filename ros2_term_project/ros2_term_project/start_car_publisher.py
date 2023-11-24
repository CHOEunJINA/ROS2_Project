import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QComboBox

class CarSelector(Node):
    def __init__(self):
        super().__init__('car_selector')
        self.publisher = None

    def select_car(self, car_id):
        topic_name = f"/car{car_id}/cmd_vel"
        self.publisher = self.create_publisher(Twist, topic_name, 10)

        twist = Twist()
        twist.linear.x = 0.5
        twist.angular.z = 0.0

        self.publisher.publish(twist)

class StartCarPublisher(Node):
    def __init__(self, car_id):
        super().__init__('start_car_publisher')
        self.publisher_ = self.create_publisher(String, 'start_car', 10)

        self.car_id = car_id
        self.timer_period = 0.5
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = self.car_id
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

        with open('variable.txt', 'w') as f:
            f.write(str(msg.data))

def selected_car():
    selected_car = comboBox.currentText()
    car_id = selected_car
    selector.select_car(car_id)
    start_car_publisher = StartCarPublisher(car_id)
    rclpy.spin(start_car_publisher)
    start_car_publisher.destroy_node()

app = QApplication([])
window = QWidget()
layout = QVBoxLayout()

rclpy.init()
selector = CarSelector()

comboBox = QComboBox()
comboBox.addItem("PR001")
comboBox.addItem("PR002")
layout.addWidget(comboBox)

button = QPushButton('Select Car')
button.clicked.connect(selected_car)
layout.addWidget(button)

window.setLayout(layout)
window.show()

app.exec_()
rclpy.shutdown()