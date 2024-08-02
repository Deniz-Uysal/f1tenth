import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray

import numpy as np


class GapFollow(Node):
    def __init__(self):
        super().__init__("gap_follow")

        self.declare_parameter("scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("drive_topic", rclpy.Parameter.Type.STRING)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value

        # Creating publishers and subscribers
        self.clean_scan_subscriber = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )
        self.array_publisher = self.create_publisher(
            Float32MultiArray, "array_topic", 10
        )

        # Data for processing
        self.scan_message = None

        self.previous_error = 0.0
        self.error = 0.0

        self.index_min, self.index_max = 0, 0
        self.min_distance = 0.0
        self.car_width = 0.5
        self.radius = 0

        self.min_front_index = 400
        self.max_front_index = 680

        self.ranges = None
        self.front_scan_ranges = None

        # Declaring constants
        self.Kp = 200.0
        self.Kd = 0.03
        self.proportional_offset_constant = 0.05

    # All methods that have arrays as parameters are expected to take in numpy arrays
    # Create subarray front_scan_ranges which will hold values between 180 and 900 degrees
    def scan_callback(self, message):
        self.ranges = np.array(message.ranges)
        self.front_scan_ranges = self.ranges[
            self.min_front_index : self.max_front_index
        ]

        self.index_min, self.min_distance = self.find_min_distance()

        self.create_bubble()

        angle = self.calculate_steering_angle()

        error = self.calculate_error(angle, self.ranges[self.index_max])
        self.pid_control(error)

    # Use front_scan_ranges array to find minimum distance from car
    def find_min_distance(self):
        return (
            np.argmin(self.front_scan_ranges) + self.min_front_index,
            np.min(self.front_scan_ranges),
        )

    # Create bubble around index of minimum distance
    def create_bubble(self):
        self.radius = self.determine_radius()
        self.ranges[
            self.index_min - self.radius : self.index_min + self.radius + 1
        ] = 0.0
        a = Float32MultiArray()
        a.data = self.ranges.tolist()

        self.array_publisher.publish(a)

    def determine_radius(self):
        radius_angle = np.arctan(self.car_width / self.min_distance)
        radius = int(np.ceil(np.degrees(radius_angle) * 4))
        return radius

    # Determine maximum gap - check length of subarrays created on the left and right of the bubble
    # Determine the longest ray from the largest gap and get its index
    def determine_max_gap(self):
        right_gap_size = self.index_min - self.radius - self.min_front_index
        left_gap_size = self.max_front_index - (self.index_min + self.radius)

        largest_gap = []
        if left_gap_size < right_gap_size:
            largest_gap = self.ranges[
                self.min_front_index : self.index_min - self.radius
            ]
            index_offset = self.min_front_index
        else:
            largest_gap = self.ranges[
                self.index_min + self.radius + 1 : self.max_front_index
            ]
            index_offset = self.index_min + self.radius + 1

        return np.argmax(largest_gap) + index_offset

    # Use the index we just got to calculate angle between central axis and said index
    def calculate_steering_angle(self):
        self.index_max = self.determine_max_gap()
        # self.get_logger().info(f"{self.index_min}, {self.index_max}, {self.error}")
        angle_degrees = (self.index_max - 540) / 4
        angle_radians = np.radians(angle_degrees)
        return angle_radians

    def calculate_error(self, alpha_rad, max_distance):
        self.previous_error = self.error

        # Using the Law of Cosines to find error
        error_squared = 2 * max_distance**2 * (1 - np.cos(alpha_rad))
        self.error = np.sqrt(error_squared)

        if alpha_rad < 0:
            self.error = 0 - self.error

        return self.error

    # Publish drive message with this angle
    def pid_control(self, error):
        proportional_term = self.proportional_offset_constant * self.Kp * error
        derivative_term = self.Kd * (self.error - self.previous_error)
        desired_angle = proportional_term + derivative_term

        modified_drive = AckermannDriveStamped()
        modified_drive.drive.speed = 0.5
        modified_drive.drive.steering_angle = desired_angle
        self.drive_publisher.publish(modified_drive)

    # Method used to process data only in the front of the car
    def preprocess_lidar(self, ranges):
        # # Calculating averages for each value over some window
        # # Number of elements to one side of the current element
        # window_size = 5

        # # Indices for front scan array (going from right to left in terms of the car's perspective)
        # start_index = 540 - 360
        # end_index = 540 + 360

        # frontal_view = ranges[start_index: end_index+1]

        # for i in range(start_index, end_index + 1):
        #     total = 0
        #     for j in range(i-window_size, i + window_size + 1):
        #         total += ranges[j]
        #     frontal_view[i] = total/(2*window_size + 1)

        # ranges[start_index: end_index+1] = frontal_view

        return ranges


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
