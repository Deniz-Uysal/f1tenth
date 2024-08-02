import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

import numpy as np


class SafetyLayer(Node):
    def __init__(self):
        super().__init__("safety_layer")

        # Declaring real world data
        self.declare_parameter("threshold_ttc", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("frontal_distance_angle", rclpy.Parameter.Type.DOUBLE)

        self.threshold_time_to_collision = self.get_parameter("threshold_ttc").value
        self.frontal_distance_angle = self.get_parameter("frontal_distance_angle").value

        # Declaring offset parameters
        self.declare_parameter("std_dev_offset", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("mean_offset", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("maximum_weighted_value", rclpy.Parameter.Type.DOUBLE)

        self.std_dev_offset = self.get_parameter("std_dev_offset").value
        self.mean_offset = self.get_parameter("mean_offset").value
        self.maximum_weighted_value = self.get_parameter("maximum_weighted_value").value

        # Declaring topics
        self.declare_parameter("scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("safety_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("master_odom_topic", rclpy.Parameter.Type.STRING)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.safety_topic = self.get_parameter("safety_topic").value
        self.master_odom_topic = self.get_parameter("master_odom_topic").value

        # Data for processing
        self.odom_message_time = 0.0
        self.scan_message_time = 0.0

        self.clean_scan_data = None
        self.odom_data = None

        self.is_safe_message = Bool()
        self.is_safe_message.data = False

        self.init = True

        self.create_weighted_array()

        # Creating publishers and subscribers
        self.clean_scan_subscriber = self.create_subscription(
            LaserScan, self.scan_topic, self.clean_scan_callback, 10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry, self.master_odom_topic, self.odom_speed_callback, 10
        )

        self.is_safe_publisher = self.create_publisher(Bool, self.safety_topic, 10)

        self.is_safe_publisher.publish(self.is_safe_message)

    # Creating a weighted array to weigh distance measurements generated from the LiDAR scans
    def create_weighted_array(self):
        array_length = int(self.frontal_distance_angle * 8) + 1
        mean = array_length // 2
        std_dev = (1 / 5 * array_length) + self.std_dev_offset

        indices = np.arange(array_length)
        normalized_values = np.exp(
            -0.5 * ((indices - (mean + self.mean_offset)) / std_dev) ** 2
        )

        self.weighted_array = self.maximum_weighted_value - (
            self.maximum_weighted_value - 1
        ) * (normalized_values)

    def clean_scan_callback(self, message):
        self.scan_message_time = self.calculate_time(message.header)

        # Determining closest object in front of the car in a range of +- frontal_distance_angle degrees from car's travel direction
        index_range = int(self.frontal_distance_angle * 4)
        self.clean_scan_data = message.ranges[540 - index_range : 540 + index_range + 1]

        self.process_data()

    def odom_speed_callback(self, message):
        self.odom_message_time = self.calculate_time(message.header)

        # Due to weird physics on the simulator, car's speed is rounded to 5 decimal places
        self.odom_data = abs(round(message.twist.twist.linear.x, 5))

    def process_data(self):
        # At the start of the program, ensuring we have odom data before driving off
        if self.odom_data is not None:
            if self.init:
                self.is_safe_message.data = True
                self.init = False

            # Implementing liveness check for odom data
            if abs(self.scan_message_time - self.odom_message_time) >= 0.05:
                self.is_safe_message.data = False
                self.get_logger().info("Messages are outdated.")

        # Implementing the emergency braking mechanism
        if self.is_safe_message.data and self.odom_data:
            # Applying weights to the distances
            weighted_scan_data = np.min(self.clean_scan_data * self.weighted_array)
            time_to_collision = weighted_scan_data / self.odom_data

            if time_to_collision <= self.threshold_time_to_collision:
                self.is_safe_message.data = False
                self.get_logger().info("Danger detected ahead.")

            self.odom_data = None
            self.clean_scan_data = None

        self.is_safe_publisher.publish(self.is_safe_message)

    # Combining the seconds and nanoseconds component given the header of a message
    def calculate_time(self, header):
        seconds = header.stamp.sec
        nanoseconds = header.stamp.nanosec
        time = (seconds % 10e4) + (nanoseconds // 10e5) / 1000
        return time


def main(args=None):
    rclpy.init(args=args)
    safety_layer = SafetyLayer()

    try:
        rclpy.spin(safety_layer)
    except KeyboardInterrupt:
        print("SafetyLayer Node stopped gracefully.")
    finally:
        safety_layer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
