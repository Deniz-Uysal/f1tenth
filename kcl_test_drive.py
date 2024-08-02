import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped


class KCLTestDrive(Node):
    def __init__(self):
        super().__init__("kcl_test_drive")
        # Declaring topics
        self.declare_parameter("drive_topic", rclpy.Parameter.Type.STRING)
        self.drive_topic = self.get_parameter("drive_topic").value

        # Declaring real world parameters
        self.declare_parameter("driving_speed", rclpy.Parameter.Type.DOUBLE)
        self.driving_speed = self.get_parameter("driving_speed").value

        # Creating publisher
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        # Commanding the car to move forward
        self.timer = self.create_timer(0.01, self.drive_callback)

    def drive_callback(self):
        modified_drive = AckermannDriveStamped()
        modified_drive.drive.speed = self.driving_speed
        self.drive_publisher.publish(modified_drive)


def main(args=None):
    rclpy.init(args=args)
    kcl_test_drive = KCLTestDrive()

    try:
        rclpy.spin(kcl_test_drive)
    except KeyboardInterrupt:
        print("KCL_Test_Drive Node stopped gracefully.")
    finally:
        kcl_test_drive.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
