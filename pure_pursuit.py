import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
from argparse import Namespace
from nav_msgs.msg import Odometry
from scipy.spatial import KDTree
import pandas as pd
import time
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def nearest_waypoint(point, data):
    kdtree = KDTree(data)
   
    distance, index = kdtree.query(point, k=1)
    closest_point = data[index]

    return closest_point, distance, index


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")

         # Declaring topics
        self.declare_parameter("master_odom_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("drive_topic", rclpy.Parameter.Type.STRING)

        self.odom_topic = self.get_parameter("master_odom_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value

        # Declaring constants
        self.declare_parameter("wheelbase", 1.0)
        # self.declare_parameter("config_path", rclpy.Parameter.Type.STRING.yaml)
        self.declare_parameter("theta_degrees", rclpy.Parameter.Type.DOUBLE)

        self.wheelbase = self.get_parameter("wheelbase").value
        # self.config_path = self.get_parameter("config_path").value
        self.theta_degrees = self.get_parameter("theta_degrees").value

        # Planner configuration
        self.declare_parameter("wpt_path", "data/maps/racelines/Spielberg_map.csv")
        self.declare_parameter("wpt_delim", ";")
        self.declare_parameter("wpt_rowskip", 0)
        self.declare_parameter("wpt_xind", 0)
        self.declare_parameter("wpt_yind", 1)
        self.declare_parameter("wpt_vind", 2)
        self.conf = Namespace(
            wpt_path=self.get_parameter("wpt_path").value,
            wpt_delim=self.get_parameter("wpt_delim").value,
            wpt_rowskip=self.get_parameter("wpt_rowskip").value,
            wpt_xind=self.get_parameter("wpt_xind").value,
            wpt_yind=self.get_parameter("wpt_yind").value,
            wpt_vind=self.get_parameter("wpt_vind").value
        )

        self.current_pose_x = 0.0
        self.current_pose_y = 0.0
        self.current_pose_theta = 0.0

        
        self.load_waypoints(self.conf)
        self.load_race_line(self.conf)
        self.max_reacquire = 2.0
        self.drawn_waypoints = []

        # self.kdtree = kdtree = KDTree(self.waypoints)

        # Creating publishers and subscribers
        self.odom_topic_subscriber = self.create_subscription(
            Odometry , self.odom_topic, self.odom_callback, 10
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        self.path_publisher = self.create_publisher(Path, 'race_line_path', 10)

        self.publish_race_line_path()

        # Timer to control the rate of processing
        timer_period = 0.1  # This is in milliseconds adjust as needed
        self.timer = self.create_timer(timer_period / 1000, self.pure_pursuit)

    def odom_callback(self, message):
        self.current_pose_x = message.pose.pose.position.x
        self.current_pose_y = message.pose.pose.position.y

        orientation_q = message.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.current_pose_theta = np.arctan2(siny_cosp, cosy_cosp)
    

    def load_waypoints(self, conf):
        self.waypoints = pd.read_csv(conf.wpt_path, skiprows=3, delimiter=';', names=["time_s", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"])

        # Extract the x and y coordinates for building the KDTree
        self.waypoints = self.waypoints[["x_m", "y_m"]].values
        
        self.waypoints *= 0.1

    def load_race_line(self, conf):
        self.race_line_data = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        self.race_line_data *= 0.1
    
    def publish_race_line_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'  # Ensure this matches your RViz frame


        for i, point in enumerate(self.race_line_data):
            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = path.header.frame_id
            pose.pose.position.x = float(point[1])
            pose.pose.position.y = float(point[2])
            pose.pose.position.z = 0.0  # Assuming the race line is on a 2D plane
            pose.pose.orientation.w = 1.0  # Neutral orientation
            path.poses.append(pose)

        self.path_publisher.publish(path)


    def calculate_steering_angle(self, current_position, target_position):
        dx = target_position[0] - current_position[0]
        dy = target_position[1] - current_position[1]
    
        # Calculate the angle to the target point relative to the global frame
        angle_to_target = np.arctan2(dy, dx)

    
        # Calculate the required steering angle
        alpha = angle_to_target - self.current_pose_theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        Lf = 2.5  # Adjust this lookahead distance if needed
        L = self.wheelbase
    
        steering_angle = np.arctan2(2 * L * np.sin(alpha) / Lf, 1.0)
    
        # Clamp the steering angle to the vehicle's steering limits
        max_steering_angle = np.radians(15)  # Example limit, adjust to your vehicle
        steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)
    
        return steering_angle


    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        query_point = (self.current_pose_x, self.current_pose_y)
        nearest_point, nearest_dist, i  = nearest_waypoint(query_point, self.waypoints)
    
        if nearest_dist > lookahead_distance:
            return nearest_point

        else:
            if i + 1 < len(self.waypoints):
                next_waypoint = self.waypoints[i + 1]
                return next_waypoint, i
            else:
                return self.waypoints[0], i+1


    def pure_pursuit(self):
        position = np.array([self.current_pose_x, self.current_pose_y])

        lookahead_point, i = self._get_current_waypoint(self.waypoints, 1 , position, self.current_pose_theta) 
        # change the lookahead distance value of this later it is 1 now
        
        drive_msg = AckermannDriveStamped()

        steering_angle = self.calculate_steering_angle(position, lookahead_point)
        speed = self.race_line_data[i, self.conf.wpt_vind]

        vgain = 1.0
        speed = vgain * speed

        drive_msg.drive.speed = 1.0
        drive_msg.drive.steering_angle = steering_angle

        self.drive_publisher.publish(drive_msg)
  

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit = PurePursuit()

    try:
        rclpy.spin(pure_pursuit)
    except KeyboardInterrupt:
        print("PurePursuit Node stopped gracefully.")
    finally:
        pure_pursuit.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
