import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

import numpy as np


class KCLBase(Node):
    def __init__(self):
        super().__init__("kcl_base")
        """
        self.declare_parameter('pose_topic', rclpy.Parameter.Type.STRING)
        self.declare_parameter('master_pose_topic', rclpy.Parameter.Type.STRING)
        self.pose_topic = self.get_parameter('pose_topic').value
        self.master_pose_topic = self.get_parameter('master_pose_topic').value
        """

        # Declaring topics
        self.declare_parameter("master_scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("drive_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("master_drive_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("safety_topic", rclpy.Parameter.Type.STRING)

        self.master_scan_topic = self.get_parameter("master_scan_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value
        self.master_drive_topic = self.get_parameter("master_drive_topic").value
        self.safety_topic = self.get_parameter("safety_topic").value

        # Implementing publisher/subscribers
        self.master_scan_subscriber = self.create_subscription(
            LaserScan, self.master_scan_topic, self.scan_callback, 10
        )
        self.scan_publisher = self.create_publisher(LaserScan, self.scan_topic, 10)

        self.kcl_drive_subscriber = self.create_subscription(
            AckermannDriveStamped, self.drive_topic, self.kcl_drive_callback, 10
        )
        self.safety_subscriber = self.create_subscription(
            Bool, self.safety_topic, self.safety_callback, 10
        )

        self.master_drive_publisher = self.create_publisher(
            AckermannDriveStamped, self.master_drive_topic, 10
        )

        self.safety_message = None
        self.drive_message = None

    def scan_callback(self, message: AckermannDriveStamped):
        # Modifying the range attribute of the message to hold clean data
        message.ranges = np.clip(np.array(message.ranges), None, 7).tolist()
        self.scan_publisher.publish(message)

    def safety_callback(self, message: Bool):
        self.safety_message = message.data
        self.check_drive()

    def kcl_drive_callback(self, drive: AckermannDriveStamped):
        self.drive_message = drive

    def check_drive(self):
        if self.drive_message is not None:
            if not self.safety_message:
                new_drive_message = AckermannDriveStamped()
                new_drive_message.drive.speed = 0.0
                self.drive_message = new_drive_message

            self.master_drive_publisher.publish(self.drive_message)

            self.safety_message = None
            self.drive_message = None


def main(args=None):
    rclpy.init(args=args)
    kcl_base = KCLBase()

    try:
        rclpy.spin(kcl_base)
    except KeyboardInterrupt:
        print("KCL_base Node stopped gracefully.")
    finally:
        kcl_base.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
