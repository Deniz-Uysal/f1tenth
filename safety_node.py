import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

import numpy as np
import time


class SafetyLayer(Node):
    def __init__(self):
        super().__init__("safety_layer")

        time.sleep(3)

        # Declaring topics
        self.declare_parameter("scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("safety_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("master_odom_topic", rclpy.Parameter.Type.STRING)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.safety_topic = self.get_parameter("safety_topic").value
        self.master_odom_topic = self.get_parameter("master_odom_topic").value

        # Declaring real world parameters
        self.declare_parameter("max_deceleration", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("safety_margin", rclpy.Parameter.Type.DOUBLE)

        self.max_deceleration = self.get_parameter("max_deceleration").value
        self.safety_margin = self.get_parameter("safety_margin").value

        # Declaring coefficients
        self.declare_parameter("cosine_coefficient", rclpy.Parameter.Type.DOUBLE)
        self.cosine_coefficient = self.get_parameter("cosine_coefficient").value

        # Data for processing
        self.scan_message = None
        self.master_odom_message = None

        self.is_safe_message = Bool()
        self.is_safe_message.data = False

        self.car_stopped = False
        self.init = True

        self.declare_parameter("num_scan_points", rclpy.Parameter.Type.INTEGER)
        self.num_scan_points = self.get_parameter("num_scan_points").value
        self.stopping_distance = np.zeros(self.num_scan_points, dtype=float)

        self.declare_parameter("liveness_threshold", rclpy.Parameter.Type.DOUBLE)
        self.liveness_threshold = self.get_parameter("liveness_threshold").value

        # Creating publishers and subscribers
        self.clean_scan_subscriber = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry, self.master_odom_topic, self.master_odom_callback, 10
        )

        self.is_safe_publisher = self.create_publisher(Bool, self.safety_topic, 10)
        self.is_safe_publisher.publish(self.is_safe_message)

    def scan_callback(self, message):
        self.scan_message = message

        self.process_data()

    def master_odom_callback(self, message):
        self.master_odom_message = message

        # Due to weird physics on the simulator, car's speed is rounded to 5 decimal places
        self.master_odom_message.twist.twist.linear.x = abs(
            round(self.master_odom_message.twist.twist.linear.x, 5)
        )

    def process_data(self):
        if self.master_odom_message is not None:
            # At the start of the program, ensuring we have odom data before driving off
            if self.init:
                self.is_safe_message.data = True
                self.init = False

            # Implementing liveness check for odom data
            if (
                abs(
                    self.get_time_in_seconds(self.scan_message.header)
                    - self.get_time_in_seconds(self.master_odom_message.header)
                )
                >= self.liveness_threshold
            ):
                self.is_safe_message.data = False
                self.get_logger().info("Messages are outdated.")
            else:
                if not self.car_stopped:
                    self.is_safe_message.data = True

            # The car's velocity vector
            car_velocity = self.master_odom_message.twist.twist.linear.x
            if self.is_safe_message.data and car_velocity:
                # Data points from the LiDAR scan
                scan_data = np.array(self.scan_message.ranges, dtype=float)

                # Corresponding angles for each data point from the car's central axis
                angles = np.arange(
                    self.scan_message.angle_min,
                    self.scan_message.angle_max,
                    self.scan_message.angle_increment,
                )

                # Velocity of car in each direction
                angular_car_velocities = car_velocity * np.cos(
                    self.cosine_coefficient * angles
                )
                above_zero_mask = angular_car_velocities > 0

                """
                # Calculate ttc with updated data
                self.ttc[above_zero_mask] = scan_data[above_zero_mask] / angular_car_velocities[above_zero_mask]
                """

                # Calculate threshold_ttc with updated data
                self.stopping_distance[above_zero_mask] = (
                    angular_car_velocities[above_zero_mask] ** 2
                ) / (2 * self.max_deceleration)
                self.stopping_distance[above_zero_mask] += self.safety_margin * np.cos(
                    self.cosine_coefficient * angles[above_zero_mask]
                )
                self.stopping_distance[~above_zero_mask] = 0.0

                """
                numerator = (self.stopping_distance[above_zero_mask] + self.safety_margin)
                denominator = angular_car_velocities[above_zero_mask]

                self.threshold_ttc[above_zero_mask] = numerator / denominator
                """

                # Implementing the emergency braking mechanism
                if np.any(scan_data < self.stopping_distance):
                    self.is_safe_message.data = False
                    self.car_stopped = True
                    self.get_logger().info("Danger detected ahead.")

                self.master_odom_message = None
                self.scan_message = None

        self.is_safe_publisher.publish(self.is_safe_message)

    # Combining the seconds and nanoseconds component given the header of a message
    def get_time_in_seconds(self, header):
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
        print("SafetyNode Node stopped gracefully.")
    finally:
        safety_layer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
