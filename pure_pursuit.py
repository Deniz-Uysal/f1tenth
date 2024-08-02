import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray
import numpy as np
import yaml
from argparse import Namespace
from numba import njit
from nav_msgs.msg import Odometry
from scipy.spatial import KDTree

import pandas as pd
import time
from pyglet.gl import GL_POINTS

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


# @njit(fastmath=False, cache=True)
def nearest_waypoint(point, data):
    kdtree = KDTree(data)
    # kdtree = KDTree(data)
    # dists = np.sqrt(np.sum((trajectory - point) ** 2, axis=1))
    # min_dist_index = np.argmin(dists)
    # return trajectory[min_dist_index], dists[min_dist_index], min_dist_index

    # distance, index = kdtree.query(point)
    distance, index = kdtree.query(point, k=1)
    closest_point = data[index]

        # If the closest point is the last one, wrap around to the start of the data
    # next_index = (index - 1) % len(data)

    
        # Get the closest clockwise point
    # cw_point = data.iloc[next_index]
    
        # Calculate the Euclidean distance to the closest clockwise point
    # cw_distance = np.sqrt((point[0] - closest_point['x_m'])**2 + (point[1] - point['y_m'])**2)
    
        # Return the closest clockwise point and the distance to it
    return closest_point, distance, index



# @njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    # Extract the Waypoint information
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    # Define the radius of the arc to follow
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)

    # Calculate the steering angle based on the curvature of the arc to follow
    steering_angle = np.arctan(wheelbase / radius)

    return speed, steering_angle


# @njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):

    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

    

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

        # self.odom_publisher = self.create_publisher(Path, '/ego_racecar/odom', 10)

        # self.publish_current_position()

        # Timer to control the rate of processing
        timer_period = 0.1  # This is in milliseconds adjust as needed
        self.timer = self.create_timer(timer_period / 1000, self.pure_pursuit)

    # def scan_callback(self, message):
    #     self.scan_message = message

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
        # self.waypoints['x_m']=self.waypoints['x_m']/10
        # self.waypoints['y_m']=self.waypoints['y_m']/10
        
        self.waypoints *= 0.1

        # self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

        # self.waypoints *= 0.1
        

        self.get_logger().info(f"Publishing {len(self.waypoints)} race line points")

    def load_race_line(self, conf):
        self.race_line = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        self.race_line *= 0.1
    

    def publish_race_line_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'  # Ensure this matches your RViz frame


        for i, point in enumerate(self.race_line):
            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = path.header.frame_id
            pose.pose.position.x = float(point[1])
            pose.pose.position.y = float(point[2])
            pose.pose.position.z = 0.0  # Assuming the race line is on a 2D plane
            pose.pose.orientation.w = 1.0  # Neutral orientation
            path.poses.append(pose)

            # Log each point for debugging purposes
            self.get_logger().info(f"Race line point {i}: x={pose.pose.position.x}, y={pose.pose.position.y}")

        self.path_publisher.publish(path)
   

    # def nearest_waypoint(query_point, data):
    #     kdtree = KDTree(self.waypoints)

    #     # Find the index of the closest point to the query point
    #     distance, index = kdtree.query(query_point)

    #     # If the closest point is the last one, wrap around to the start of the data
    #     next_index = (index - 1) % len(data)
    
    #     # Get the closest clockwise point
    #     cw_point = data.iloc[next_index]
    
    #     # Calculate the Euclidean distance to the closest clockwise point
    #     cw_distance = np.sqrt((query_point[0] - cw_point['x_m'])**2 + (query_point[1] - cw_point['y_m'])**2)
    
    #     # Return the closest clockwise point and the distance to it
    #     return cw_point, cw_distance


    def render_waypoints(self, e):
        # points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        # wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        # nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
       
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T

        query_point = (self.current_pose_x, self.current_pose_y)
        nearest_point, nearest_dist, i  = nearest_waypoint(query_point, self.waypoints)
        
        
        
        # nearest_point, nearest_dist, i = self.nearest_waypoint(position, wpts)
    

        if nearest_dist < lookahead_distance:
            self.get_logger().info("AAA")
            # lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts,
            #                                                                         i + t, wrap=True)
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, wrap=True)
            if i2 == None:
                self.get_logger().info("AAA 2")

                return None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            self.get_logger().info("COFFEEEEEEEEEE")
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            self.get_logger().info("Pain and sufferring ... :(")
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        # Get the current Position of the car
        position = np.array([pose_x, pose_y])

        # Search for the next waypoint to track based on lookahead distance parameter
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        # Calculate the Actuation: Steering angle and speed
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)

        speed = vgain * speed

        # self.get_logger().info(f"{steering_angle}")


        return speed, steering_angle

    def pure_pursuit(self):
        speed, steering_angle = self.plan(self.current_pose_x, self.current_pose_y, self.current_pose_theta, lookahead_distance= 10, vgain=1.0)
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = speed
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

# The issue I think is that the nearest distance never gets updates with the car moving or that the nearest point it finds is the wrong one.











# @njit(fastmath=False, cache=True)
# def nearest_point_on_trajectory(point, trajectory):
    # diffs = trajectory[1:, :] - trajectory[:-1, :]
    # l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # # this is equivalent to the elementwise dot product
    # # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    # dots = np.empty((trajectory.shape[0] - 1,))
    # for i in range(dots.shape[0]):
    #     dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    # t = dots / l2s
    # t[t < 0.0] = 0.0
    # t[t > 1.0] = 1.0
    # # t = np.clip(dots / l2s, 0.0, 1.0)
    # projections = trajectory[:-1, :] + (t * diffs.T).T
    # # dists = np.linalg.norm(point - projections, axis=1)
    # dists = np.empty((projections.shape[0],))
    # for i in range(dists.shape[0]):
    #     temp = point - projections[i]
    #     dists[i] = np.sqrt(np.sum(temp * temp))
    # min_dist_segment = np.argmin(dists)
    # return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

# @njit(fastmath=False, cache=True)
# def nearest_waypoint(point, trajectory):
#     """
#     Return the nearest waypoint on the given trajectory.
#     point: size 2 numpy array
#     trajectory: Nx2 matrix of (x,y) trajectory waypoints
#     """
#     dists = np.sqrt(np.sum((trajectory - point) ** 2, axis=1))
#     min_dist_index = np.argmin(dists)
#     return trajectory[min_dist_index], dists[min_dist_index], min_dist_index
