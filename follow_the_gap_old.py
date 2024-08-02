import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray

import numpy as np


class GapFollow(Node):
    def __init__(self):
        super().__init__("gap_follow")

        # Declaring real world constants
        self.declare_parameter("car_width", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("driving_speed", rclpy.Parameter.Type.DOUBLE)

        self.car_width = self.get_parameter("car_width").value
        self.driving_speed = self.get_parameter("driving_speed").value

        # Declaring topics
        self.declare_parameter("scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("drive_topic", rclpy.Parameter.Type.STRING)

        self.scan_topic = self.get_parameter("scan_topic").value
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
        ).value  # Angular vision of car on 1 side from the front
        self.proportional_offset_constant = self.get_parameter(
            "proportional_offset_constant"
        ).value
        self.Kp = self.get_parameter("Kp").value
        self.Kd = self.get_parameter("Kd").value

        # Creating publishers and subscribers
        self.clean_scan_subscriber = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        # Data for processing
        self.scan_message = None

        self.previous_error = 0.0
        self.error = 0.0

        self.min_front_index, self.max_front_index = (
            540 - self.theta_degrees * 4,
            540 + self.theta_degrees * 4,
        )

    def scan_callback(self, message):
        self.scan_message = message

        self.follow_the_gap()

    def follow_the_gap(self):

        ranges = np.array(self.scan_message.ranges)
        front_scan_ranges = ranges[self.min_front_index : self.max_front_index]

        # Determining minimum distance from car
        min_distance_index = np.argmin(front_scan_ranges) + self.min_front_index
        min_distance = ranges[min_distance_index]

        # Create safety bubble around the index of the minimum distance
        bubble_radius = self.determine_bubble_radius(min_distance)
        ranges[
            min_distance_index - bubble_radius : min_distance_index + bubble_radius + 1
        ] = 0.0

        # Find furthest distance from the biggest availale gap between -90 and 90 degrees from the central axis
        max_distance_index = self.find_longest_distance_in_largest_gap(
            ranges, 180, 900, min_distance_index, bubble_radius
        )

        # Find angle in radians between max_distance_index and car's central index
        angle_difference = self.calculate_angular_disparity(max_distance_index)

        # Using a PID controller to obtain desired turning angle
        error = self.calculate_error(angle_difference, ranges[max_distance_index])
        self.pid_control(error)

    # Extending frontal vision in case there are further distances outside the current view
    def extend_frontal_vision(self, ranges):
        pass

    # Depending on the distance of the car from the nearest object, the bubble radious is changed to keep the real world distance constant
    def determine_bubble_radius(self, min_distance):
        radius_angle = np.arctan(self.car_width / min_distance)
        radius = int(np.ceil(np.degrees(radius_angle) * 4))
        return radius

    # Find furthest distance from the biggest availale gap
    def find_longest_distance_in_largest_gap(
        self,
        ranges,
        min_front_index,
        max_front_index,
        min_distance_index,
        bubble_radius,
    ):
        right_gap_size = (min_distance_index - bubble_radius) - min_front_index
        left_gap_size = max_front_index - (min_distance_index + bubble_radius)

        if left_gap_size < right_gap_size:
            largest_gap = ranges[min_front_index : min_distance_index - bubble_radius]
            index_offset = min_front_index
        else:
            largest_gap = ranges[
                min_distance_index + bubble_radius + 1 : max_front_index
            ]
            index_offset = min_distance_index + bubble_radius + 1

        # Checking if numpy array is empty
        if largest_gap is None or not np.any(largest_gap):
            return 0

        return np.argmax(largest_gap) + index_offset

    # Find angle between max_distance_index and car's central index
    def calculate_angular_disparity(self, max_distance_index):
        angle_degrees = (max_distance_index - 540) / 4
        angle_radians = np.radians(angle_degrees)
        return angle_radians

    # Determining the error term for the PID equation
    def calculate_error(self, angle_difference, max_distance):
        self.previous_error = self.error

        # Using the Law of Cosines to find error
        error_squared = 2 * max_distance**2 * (1 - np.cos(angle_difference))
        self.error = np.sqrt(error_squared)

        if angle_difference < 0:
            self.error = -self.error

        return self.error

    # Publish drive message with this angle
    def pid_control(self, error):
        proportional_term = self.proportional_offset_constant * self.Kp * error
        derivative_term = self.Kd * (self.error - self.previous_error)
        desired_angle = np.clip(
            proportional_term + derivative_term, a_min=-np.pi / 2, a_max=np.pi / 2
        )

        self.get_logger().info(f"{desired_angle}")

        modified_drive = AckermannDriveStamped()
        # speed_clamp = 0.9 * (np.cos(desired_angle) + 1)
        # modified_drive.drive.speed = speed_clamp * self.driving_speed

        modified_drive.drive.speed = 6.0

        modified_drive.drive.steering_angle = desired_angle

        self.drive_publisher.publish(modified_drive)


def main(args=None):
    rclpy.init(args=args)
    gap_follow = GapFollow()

    try:
        rclpy.spin(gap_follow)
    except KeyboardInterrupt:
        print("GapFollow Node stopped gracefully.")
    finally:
        gap_follow.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
