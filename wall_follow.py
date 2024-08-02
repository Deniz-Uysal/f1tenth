import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

import numpy as np


class WallFollow(Node):
    def __init__(self):
        super().__init__("wall_follow")

        # Declaring real world parameters
        self.declare_parameter("desired_distance_left", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("driving_speed", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("L", rclpy.Parameter.Type.DOUBLE)

        self.desired_distance_left = self.get_parameter("desired_distance_left").value
        self.driving_speed = self.get_parameter("driving_speed").value
        self.L = self.get_parameter("L").value

        # Declaring topics
        self.declare_parameter("scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("master_odom_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("drive_topic", rclpy.Parameter.Type.STRING)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.master_odom_topic = self.get_parameter("master_odom_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value

        # Declaring constants
        self.declare_parameter("theta_degrees", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter(
            "proportional_offset_constant", rclpy.Parameter.Type.DOUBLE
        )
        self.declare_parameter("Kp", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("Kd", rclpy.Parameter.Type.DOUBLE)

        self.theta_degrees = self.get_parameter(
            "theta_degrees"
        ).value  # Angle between two lidar scan data points a and b. Left as int as it would act as index later
        self.proportional_offset_constant = self.get_parameter(
            "proportional_offset_constant"
        ).value
        self.Kp = self.get_parameter("Kp").value
        self.Kd = self.get_parameter("Kd").value

        # Data for processing
        self.scan_message = None
        self.master_odom_message = None

        self.previous_error = 0.0
        self.error = 0.0

        # Creating publishers and subscribers
        self.clean_scan_subscriber = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry, self.master_odom_topic, self.master_odom_callback, 10
        )

        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

    def scan_callback(self, message):
        self.scan_message = message

        error = self.follow_wall()
        self.pid_control(error)

    def master_odom_callback(self, message):
        self.master_odom_message = message

    def follow_wall(self):
        self.previous_error = self.error

        scan_data = np.array(self.scan_message.ranges, dtype=float)

        b_index = 540 + 90 * 4
        a_index = b_index - self.theta_degrees * 4

        # Distances from the LiDAR scan
        b = scan_data[b_index]
        a = scan_data[a_index]

        # Calculating angle alpha
        theta_rad = np.radians(self.theta_degrees)
        numerator = a * np.cos(theta_rad) - b
        denominator = a * np.sin(theta_rad)

        alpha_rad = np.arctan(numerator / denominator)

        # Calculating present and future distances from wall
        Dt_0 = b * np.cos(alpha_rad)
        Dt_1 = Dt_0 + self.L * np.sin(alpha_rad)

        # self.get_logger().info(f"a: {round(a, 3)}, b: {round(b, 3)}, Alpha: {round(np.degrees(alpha_rad), 3)}, Dt+1: {round(Dt_1, 3)}, Desired distance from wall: {self.desired_distance_left} ")
        self.error = Dt_1 - self.desired_distance_left
        return self.error

    def pid_control(self, error):
        proportional_term = self.proportional_offset_constant * self.Kp * error
        derivative_term = self.Kd * (self.error - self.previous_error)
        desired_angle = proportional_term + derivative_term

        modified_drive = AckermannDriveStamped()
        modified_drive.drive.speed = self.calculate_driving_speed(abs(desired_angle))
        modified_drive.drive.steering_angle = desired_angle
        self.drive_publisher.publish(modified_drive)

    # The parameter 'angle' must be in radians.
    # It must be positive, i.e., the magnitude of the angle must be passed.
    def calculate_driving_speed(self, rad_angle):
        angle_degree = np.degrees(rad_angle)

        if angle_degree <= 10:
            return 1.5
        elif angle_degree <= 20:
            return 1.0
        else:
            return 0.5


def main(args=None):
    rclpy.init(args=args)
    wall_follow = WallFollow()

    try:
        rclpy.spin(wall_follow)
    except KeyboardInterrupt:
        print("WallFollow Node stopped gracefully.")
    finally:
        wall_follow.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
