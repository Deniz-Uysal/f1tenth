import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray
import numpy as np


class GapFollow(Node):
    def __init__(self):
        super().__init__("gap_follow")

        # self.pid_steering_angle_history = []

        # Declaring topics
        self.declare_parameter("scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("drive_topic", rclpy.Parameter.Type.STRING)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value

        # Declaring constants
        self.declare_parameter("theta_degrees", rclpy.Parameter.Type.INTEGER)
        self.theta_degrees = self.get_parameter("theta_degrees").value

        self.declare_parameter("Kp", rclpy.Parameter.Type.DOUBLE)
        self.Kp = self.get_parameter("Kp").value

        self.declare_parameter("Kd", rclpy.Parameter.Type.DOUBLE)
        self.Kd = self.get_parameter("Kd").value

        self.declare_parameter("straight_speed", rclpy.Parameter.Type.DOUBLE)
        self.straight_speed = self.get_parameter("straight_speed").value

        self.declare_parameter("corner_speed", rclpy.Parameter.Type.DOUBLE)
        self.corner_speed = self.get_parameter("corner_speed").value

        self.declare_parameter("straight_steering_angle", np.pi / 18)
        self.straight_steering_angle = self.get_parameter(
            "straight_steering_angle"
        ).value

        self.declare_parameter("best_point_conv_size", rclpy.Parameter.Type.INTEGER)
        self.best_point_conv_size = self.get_parameter("best_point_conv_size").value

        self.declare_parameter("max_lidar_distance", rclpy.Parameter.Type.INTEGER)
        self.max_lidar_distance = self.get_parameter("max_lidar_distance").value

        self.declare_parameter("preprocess_conv_size", rclpy.Parameter.Type.INTEGER)
        self.preprocess_conv_size = self.get_parameter("preprocess_conv_size").value

        self.declare_parameter("car_width", rclpy.Parameter.Type.DOUBLE)
        self.car_width = self.get_parameter("car_width").value

        # Creating publishers and subscribers
        self.clean_scan_subscriber = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        # Timer to control the rate of processing
        timer_period = 0.1  # This is in milliseconds adjust as needed
        self.timer = self.create_timer(timer_period / 1000, self.follow_the_gap)

        # Data for processing
        self.scan_message = None
        self.radians_per_elem = None

        self.previous_error = 0.0
        self.error = 0.0

    def scan_callback(self, message):
        self.scan_message = message

    def preprocess_lidar(self, ranges):
        # Preprocess the LiDAR scan array.
        self.radians_per_elem = (2 * np.pi) / len(ranges)

        # Only get the data from the front of the car
        proc_ranges = np.array(ranges[135:-135])

        # Sets each value to the mean over a given window
        proc_ranges = (
            np.convolve(proc_ranges, np.ones(self.preprocess_conv_size), "same")
            / self.preprocess_conv_size
        )
        proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_distance)

        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        # Mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)

        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]

        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl

        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        # Do a sliding window average over the data in the max gap, this will help the car to avoid hitting corners
        averaged_max_gap = (
            np.convolve(
                ranges[start_i:end_i], np.ones(self.best_point_conv_size), "same"
            )
            / self.best_point_conv_size
        )
        return averaged_max_gap.argmax() + start_i

    def pid_angle(self, range_index, range_len, min_distance):
        # Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        error = lidar_angle

        # Calculate PID terms
        d_error = (error - self.previous_error) / (min_distance + 0.0001)
        self.previous_error = error

        steering_angle = self.Kp * error + self.Kd * d_error

        distance_factor = 1 / (min_distance + 0.00001)
        steering_angle *= distance_factor
        steering_angle = np.clip(steering_angle, np.radians(-15), np.radians(15))

        return steering_angle

    def determine_bubble_radius(self, min_distance):
        radius_angle = np.arctan(self.car_width / (min_distance))
        radius = int(np.ceil(np.degrees(radius_angle) * 4))
        return radius

    def follow_the_gap(self):
        # This is the main function that is getting called from the simulation
        if self.scan_message is None:
            return

        # Preprocess the Lidar Information
        proc_ranges = self.preprocess_lidar(self.scan_message.ranges)

        masked_ranges = np.ma.masked_where(proc_ranges == 0, proc_ranges)

        # Find closest point to LiDAR
        closest = masked_ranges.argmin()

        self.bubble_radius = self.determine_bubble_radius(masked_ranges[closest])

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.bubble_radius
        max_index = closest + self.bubble_radius
        if min_index < 0:
            min_index = 0
        if max_index >= len(proc_ranges):
            max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Get the final steering angle and speed value
        steering_angle = self.pid_angle(best, len(proc_ranges), masked_ranges[closest])

        if abs(steering_angle) > self.straight_steering_angle:
            speed = self.corner_speed
        else:
            speed = self.straight_speed

        # Create a new AckermannDriveStamped message object
        modified_drive = AckermannDriveStamped()

        # Set the speed and steering angle of the drive message
        modified_drive.drive.speed = speed
        modified_drive.drive.steering_angle = steering_angle

        # Publish the modified drive message to the drive topic
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
