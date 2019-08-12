#!/usr/bin/env python

'''
Performs an optimization to determine the best points for a 3 point
turn-around maneuver. The starting point for the maneuver is the ending point
for the obstacle avoidance path and the final point of the maneuver is the
starting point for the grasping path generation.
'''


import rospy
from geometry_msgs.msg import PoseStamped, Pose
from grasping.srv import OptimizeManeuver, OptimizeManeuverRequest, OptimizeManeuverResponse
from motion_testing.msg import PathWithGear
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf
#import numpy as np # currently using autograd
import autograd.numpy as np
from autograd import grad, jacobian, hessian
import pyipopt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math
from inPolygon import pointInPolygon
import time
from copy import copy, deepcopy


class Pose2D:
    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        rep = "Pose:\n   x: {0:0.04f}\n   y: {1:0.04f}\n   theta: {2:0.04f}".format(self.x, self.y, self.theta)
        return rep

def wrapToPi(angle):
    '''
    Wraps an angle in radians between [-pi, pi].

    Args:
    -----
    angle: angle in radians to be wrapped

    Returns:
    --------
    the angle now wrapped within the range [-pi, pi]
    '''
    return (angle + np.pi) % (2*np.pi) - np.pi

def rotZ2D(theta):
    '''
    Returns a 2D rotation matrix going from frame 2 to frame 1 where frame 2 is
    rotated about the frame 1 origin by theta radians
          x_2
         /
        /
       /
      /
     /\
    /  | theta
    ----------------- x_1

    x_1 = R*x_2

    Args:
    -----
    theta: angle frame 2 has been rotated about frame 1

    Returns:
    --------
    R: 2x2 rotation matrix as numpy array
    '''
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

    return R

class ManeuverPath:
    def __init__(self):
        #=============#
        # ROS Objects
        #=============#
        # ROS Parameters
        rospy.init_node("maneuver_path")
        self.approach_pose_offset = rospy.get_param("~approach_pose_offset", 8.0) # distance in meters used to offset the maneuver starting pose from the roll, this is to give a good initial value for the optimization
        self.roll_radius = rospy.get_param("/roll/radius", 0.20)
        self.resolution = rospy.get_param("~maneuver_resolution", 0.10) # resolution for determining number of waypoints in the maneuver paths
        self.rescale_factor = rospy.get_param("~rescale_factor", 0.5) # Decreases the resolution of the image along each axis by this fraction
        self.max_angle = rospy.get_param("/forklift/steering/max_angle", 65*(np.pi/180.)) # deg

        # Optimization Parameters
        print("Namespace: %s" % rospy.get_name())
        self.optimization_method = rospy.get_param("~optimization_method", 0) # 0 = use hardcoded, 1 = use scipy, 2 = use pyipopt
        self.start_x_s = rospy.get_param("~start_x_s", 11.5)
        self.start_y_s = rospy.get_param("~start_y_s", -6.8)
        self.start_theta_s = rospy.get_param("~start_theta_s", 0.2)
        self.start_r_1 = rospy.get_param("~start_r_1", -0.4)
        self.start_alpha_1 = rospy.get_param("~start_alpha_1", -1.6)
        self.start_r_2 = rospy.get_param("~start_r_2", 1.0)
        self.start_alpha_2 = rospy.get_param("~start_alpha_2", 1.4)

        self.target_x = None
        self.target_y = None
        self.target_approach_angle = None
        self.obstacles = None
        self.maneuver_path = PathWithGear()
        self.maneuver_path.path.header.frame_id = "/odom"
        self.optimization_success = False
        self.update_obstacle_end_pose = False
        self.current_pose = None
        self.rate = rospy.Rate(30)

        # Forklift dimensions
        self.base_to_clamp = rospy.get_param("/forklift/body/base_to_clamp", 1.4658) # {m}
        self.base_to_back = rospy.get_param("/forklift/body/base_to_back", 1.9472) # {m}
        self.width = rospy.get_param("/forklift/body/width", 1.3599) # {m}
        self.total_length = rospy.get_param("/forklift/body/total", 3.5659) # {m}
        self.buffer = 0.5 # {m} gap between edge of forklift and the bounding box

        '''
        Forklift Bounding Box
                                   L_3
                             |--------------|
                            ___             ___
                             | buffer        |
                            _|_              |
                         /                   |
                         \       /           |
                           \   /             | L_1
                     -----------------       |
                     |__     ^     __|       |
                     |  |    |    |  |       |
                     |__| <--o    |__|buffer---
                     |               |-----| |
                     |               |       |
                     |               |       |
                     |               |       | L_2
                     -----------------       |
                             | buffer        |
                            _|_             _|_


        '''
        self.L_1 = self.base_to_clamp + self.buffer # length from base to back end of box
        self.L_2 = self.base_to_back + self.buffer # length from base to front end of box
        self.L_3 = (self.width/2.) + self.buffer # length from base to side

        # Max turning radius
        self.axle_distance = 1.7249
        self.min_radius = 1.5*self.axle_distance/np.tan(self.max_angle) # add a scaling factor to give more space to make the turn

        # ROS Publishers and Subscribers
        self.path1_pub = rospy.Publisher("~path1", Path, queue_size=3, latch=True)
        self.path2_pub = rospy.Publisher("~path2", Path, queue_size=3, latch=True)
        self.path1_gear_pub = rospy.Publisher("~path1_gear", PathWithGear, queue_size=3, latch=True)
        self.path2_gear_pub = rospy.Publisher("~path2_gear", PathWithGear, queue_size=3, latch=True)
        self.approach_pose_pub = rospy.Publisher("/forklift/approach_pose", PoseStamped, queue_size=3)

        self.occupancy_grid_sub = rospy.Subscriber("/map", OccupancyGrid, self.occupancyGridCallback, queue_size=1)
        self.roll_pose_sub = rospy.Subscriber("/roll/pose", PoseStamped, self.rollCallback, queue_size=3)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odomCallback, queue_size=1)
        self.obstacle_path_sub = rospy.Subscriber("/obstacle_avoidance_path", Path, self.obstaclePathCallback, queue_size=1)

        # indicates whether the optimzation completed successfully or not, to know whether the path is usable
        self.optimize_maneuver_srv = rospy.Service("~optimize_maneuver", OptimizeManeuver, self.maneuverService)

    def spin(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def maneuverPoses(self, pose_s, r_1, alpha_1, r_2, alpha_2):
        '''
        Calculates the middle and final poses of the turn-around maneuver given
        the starting position and the desired turning radius and arc length of
        the two segments.

        Args:
        -----
        pose_s: Pose2D object containing (x,y) position and yaw angle, theta
        r_1: turning radius of the first section (positive means steering wheel
             is turned to the left, negative is turned to the right)
        alpha_1: arc angle of the first section (positive means traveling in the
                 forkward direction, negative is backwards))
        r_2: turning radius of the second section (should be positive, will be
             made the opposite sign of r_1)
        alpha_2: arc angle of the second section (should be positive, will be
                 made the oppposite sign of alpha_1)

        Returns:
        --------
        [pose_m, pose_f]: the middle and final pose values as Pose2D objects
        '''
        # Unpack pose values
        x_s = pose_s.x
        y_s = pose_s.y
        theta_s = pose_s.theta

        # Calculate parameters for the middle pose
        theta_m = theta_s + np.sign(r_1)*alpha_1
        pos_s = np.array([x_s, y_s])
        R1 = rotZ2D(theta_s)
        pos_m = pos_s + np.dot(R1, np.array([np.abs(r_1)*np.sin(alpha_1), r_1*(1 - np.cos(alpha_1))]))

        # Make second parameters opposite signs of the first
        r_2 = -np.sign(r_1)*r_2
        alpha_2 = -np.sign(alpha_1)*alpha_2

        # Calculate parameters for the final pose
        theta_f = theta_m + np.sign(r_2)*alpha_2
        R2 = rotZ2D(theta_m)
        pos_f = pos_m + np.dot(R2, np.array([np.abs(r_2)*np.sin(alpha_2), r_2*(1 - np.cos(alpha_2))]))

        # Save middle and final pose as Pose2D objects
        pose_m = Pose2D(pos_m[0], pos_m[1], theta_m)
        pose_f = Pose2D(pos_f[0], pos_f[1], theta_f)

        return [pose_m, pose_f]

    def maneuverLength(self, r_1, alpha_1, r_2, alpha_2):
        '''
        Calculates the length of the turn-around maneuver.

        Args:
        -----
        r_1: turning radius of the first section (can be positive or negative)
        alpha_1: arc length of the first section (positive only)
        r_2: turning radius of the second section (can be positive or negative)
        alpha_2: arc length of the second section (positive only)

        Returns:
        --------
        L: the length of the path
        '''
        L = np.abs(r_1)*np.abs(alpha_1) + np.abs(r_2)*np.abs(alpha_2)
        return L

    def maneuverSegmentPath(self, pose, r, alpha):
        '''
        Takes the length of the path and determines the number of points to add
        between the endpoints of the arc length defined by the radius, 'r', and
        arc angle, 'alpha'. Then it creates a list of points to be used as a
        path. Note: the starting point is not included in the path, this should
        be the endpoint of the previous path. This way the paths can be
        concatenated together without overlapping points.

        Args:
        -----
        pose: starting pose of arc as Pose2D object
        r: arc radius {m}
        alpha: arc angle {rad}

        Returns:
        --------
        path: list containing [x,y] pairs of points along the arc
        '''
        arc_length = abs(r*alpha)
        num_points = int(math.ceil(arc_length/self.resolution))
        if (num_points == 0):
            num_points = 1
        delta_alpha = float(alpha)/(num_points)

        start_position = np.array([pose.x, pose.y])
        R = rotZ2D(pose.theta)
        path = []
        for i in range(1, num_points+1):
            path_point = start_position + R.dot(np.array([np.abs(r)*np.sin(i*delta_alpha), r*(1 - np.cos(i*delta_alpha))]))
            path.append([path_point[0], path_point[1]])

        return path


    def forkliftBoundingBox(self, pose):
        '''
        Calculates the four corners of the forklift's bounding box and
        returns them as a 4x2 numpy array

        Args:
        -----
        pose: Pose2D object containing the forklifts position (x,y) and yaw
              angle, theta

        Returns:
        --------
        points: a 4x2 numpy array containing the bounding box points, first
                point is in the back right corner and goes counter-clockwise
        '''
        # Unpack the pose
        x = pose.x
        y = pose.y
        theta = pose.theta

        # Corner points in forklift frame
        corners = [np.array([-self.L_2,-self.L_3]),
                   np.array([ self.L_1,-self.L_3]),
                   np.array([ self.L_1, self.L_3]),
                   np.array([-self.L_2, self.L_3])]

        R_forklift = rotZ2D(theta)

        points = []
        for i in range(len(corners)):
            points.append(np.array([x, y]) + np.dot(R_forklift, corners[i]))

        return np.asarray(points)

    def maneuverBoundingBox(self, pose_s, pose_m, pose_f):
        # TODO: This code needs to be updated in the future to handle
        # situtations where the turn-around procedure may not be internal to the
        # endpoints. If the arc angle is large, then there will be points that
        # are not considered along the path with regards to obstacle collision.
        # Currently it just takes the three endpoints of the maneuver (start,
        # middle, final) and generates the convex hull of the set of forklift
        # bounding box points for each of those poses.
        '''
        Generates the convex hull for the set of bounding box points of the
        three maneuver end points (start, middle, final).

        Args:
        -----
        pose_s: starting pose as Pose2D object
        pose_m: middle pose as Pose2D object
        pose_f: final pose as Pose2D object

        Returns:
        --------
        points: mx2 numpy array containing all the points of the bounding box,
                size may vary based on poses, anticipate at least 6 points,
                Note: the starting point is added to the end of the array to
                make a "closed" polygon
        '''
        # Generate full list of bounding box points
        points_s = self.forkliftBoundingBox(pose_s)
        points_m = self.forkliftBoundingBox(pose_m)
        points_f = self.forkliftBoundingBox(pose_f)

        all_points = np.vstack((points_s, points_m, points_f))

        # Find the convex hull points
        hull = ConvexHull(all_points)
        points = hull.points[hull.vertices]

        # Add the starting point to the end of the array to make it a "closed"
        # polygon
        points = np.vstack((points, points[0,:]))

        return points

    def rowMajorIndex(self, i, width):
        '''
        Takes a single index 'i' of a 1D array representing a Row-Major matrix
        and determines the 2D indices. The top left cell is considered (0,0).

        Args:
        -----
        i: 1D index
        width: number of cells in a single row of the matrix

        Returns:
        --------
        [row,col]: 2D indices of the row-major matrix
        '''
        row = math.floor(i/width)
        col = i % width

        return [row, col]

    def gridMapToObstacles(self, grid):
        '''
        Converts an OccupancyGrid message into a numpy array of obstacle
        positions.

        Args:
        -----
        grid: an OccupancyGrid ROS message

        Returns:
        --------
        obstacles: mx2 numpy array of obstacle positions
        '''
        # Get map parameters
        width = grid.info.width
        resolution = grid.info.resolution
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y

        # Iterate through each cell in the grid and determine the obstacle probability
        obstacles = []
        for i in range(len(grid.data)):
            # If there is an obstacle, convert index into row & column
            if (grid.data[i] > 0):
                [row, col] = self.rowMajorIndex(i, width)

                # Convert row and column into a position in map frame
                # The map is in "image frame", meaning that the rows are along the Y
                # axis and the columns are along the X axis with the Z axis going
                # "into" the matrix.
                x = (col*resolution + resolution/2.0) + origin_x
                y = (row*resolution + resolution/2.0) + origin_y
                obstacles.append([x,y])

        obstacles = np.asarray(obstacles)

        return obstacles

    def obstaclesInManeuver(self, pose_s, obstacles):
        '''
        Outputs the number of obstacles present in the maneuver bounding box.

        Args:
        -----
        pose_s: starting pose as Pose2D object
        obstacles: mx2 numpy array of 2D points representing the position of
                   obstacles

        Returns:
        --------
        num_obstacles: number of obstacles within the maneuver bounding box
        '''
        #===== Box around single pose only =====#
        # Generate full list of bounding box points
        points_s = self.forkliftBoundingBox(pose_s)
        points_s = np.vstack((points_s, points_s[0,:]))
        # Determine which points are inside the boundaries
        [points_in, points_on] = pointInPolygon(obstacles, points_s)
        num_obstacles = np.sum(points_in)

        return num_obstacles

    def maneuverObjective(self, x, params):
        '''
        Optimization objective function that calculates the cost associated with
        generating the maneuver path for approaching the roll. Takes the design
        variables as well as the parameters. An additional wrapper function
        should be created that preassigns the parameters before passing the
        function to the optimizer.

        Args:
        -----
        x: [x_s, y_s]
        x_s: forklift starting 'x' position of maneuver
        y_s: forklist starting 'y' position of maneuver

        params: {"current_pose" : pose, "forklift_length" : L_f, "weights" :
                 [w_1, w_2, w_3]}
        "current_pose" : [x,y,yaw]: the forklift's current pose provided as list containing the (x,y) position and yaw angle, theta.
        "forklift_length" : L_f: length of the forklift from clamp tip to back.
        "weights" : [w_1, w_2]: weights for the cost, w_1 = weight for distance error between the desired approach radius and the current distance to the roll, w_2 = weight for the cross track error between the ideal approach line and the current position.

        Returns:
        --------
        J: the cost of the maneuver using the given design variables
        '''
        # Unpack variables and parameters
        # Starting position
        x_s = x[0]
        y_s = x[1]
        # Point the pose towards the center of the roll
        theta_s = np.arctan2(self.target_y - y_s, self.target_x - x_s)

        # Ideal approach position
        ideal_x = self.target_x + self.approach_pose_offset*np.cos(self.target_approach_angle)
        ideal_y = self.target_y + self.approach_pose_offset*np.sin(self.target_approach_angle)

        # Covnert state to pose
        pose_s = Pose2D(x_s, y_s, theta_s)

        # Calculate the radius error
        current_radius = np.sqrt((self.target_x - x_s)**2 + (self.target_y - y_s)**2)
        radius_error = (current_radius - self.approach_pose_offset)**2

        # Calculate the cross track error
        # Use law of cosines to get the angle between the starting point and the ideal approach line, then find the distance from the starting point to that line using Sine
        #                             (x_s, y_s)
        #                                 o
        #                              /  |\
        #                       B   /    c| \  A
        #                        /\      t|  \
        #                     /alpha     e|   \
        # (roll_x, roll_y)  o ---------------- o (ideal_x, ideal_y)
        #                             C
        #
        A = np.sqrt((ideal_x - x_s)**2 + (ideal_y - y_s)**2)
        B = np.sqrt((self.target_x - x_s)**2 + (self.target_y - y_s)**2)
        C = self.approach_pose_offset
        alpha = np.arccos((C**2 + B**2 - A**2)/(2*C*B))

        cross_track_error = (np.sin(alpha)*B)**2

        # Calculate the weighted cost
        J = params["weights"][0]*radius_error + params["weights"][1]*cross_track_error

        return J

    def maneuverIneqConstraints(self, x, params):
        '''
        Optimization inequality constraint function. scipy.optimize.minimize
        defines the constraints as c_j >= 0.

        subject to:
        1) -obstaclesInManeuver() >= 0
        2) abs(r_1) - min_radius >= 0

        Args:
        -----
        x: [x_s, y_s]
        x_s: forklift starting 'x' position of maneuver
        y_s: forklist starting 'y' position of maneuver

        params: {"obstacles" : obstacles, "min_radius" : min_radius}
        "obstacles" : obstacles: mx2 numpy array of obstacle locations (x,y)
        "min_radius" : min_radius: the minimum allowable turning radius

        Returns:
        --------
        C: number of obstacles in the bounding box
        '''
        # Unpack variables and parameters
        x_s = x[0]
        y_s = x[1]
        # Point the pose towards the center of the roll
        theta_s = np.arctan2(self.target_y - y_s, self.target_x - x_s)

        # Convert states to poses
        pose_s = Pose2D(x_s, y_s, theta_s)

        # Frist constraint
        C = -self.obstaclesInManeuver(pose_s, params["obstacles"])

        return C

    def gradManeuverIneqConstraints(self, x, params):
        '''
        Calculates the gradient for the constraints function using the finite
        differencing method.

        Args:
        -----
        x: input array for the constraints function, see
            'maneuverIneqConstraints' description for details

        Returns:
        --------
        g: 14x1 array containing the gradient of the constraints, the first 7
            elements are for the first constraint and the second 7 are for the
            second constraint
        '''
        g = np.zeros(1*x.size)
        delta = 0.00000001
        for i in range(x.size):
            dx = x
            dx[i] = dx[i] + delta
            dC = (self.maneuverIneqConstraints(dx, params) - self.maneuverIneqConstraints(x, params)) / delta
            g[i] = dC

        return g

    def optimizeManeuver(self):
        '''
        Sets up the optimization problem then calculates the optimal maneuver
        poses. Publishes the resulting path if the optimization is successful.

        Args:
        -----
        msg: ROS Bool message

        Returns:
        --------
        path: the optimal path as a ROS nav_msgs/Path message
        '''
        # Make sure a target pose exists
        if (self.target_x is not None):
            # Grab the current pose from the recent transform if there is no 'odom' topic being published to
            if (self.current_pose is None):
                # DEBUG:
                print("No 'odom' message received. Waiting for transform from 'odom' to 'base_link'...")
                listener = tf.TransformListener()
                try:
                    listener.waitForTransform('/odom', '/base_link', rospy.Time(0), rospy.Duration(10))
                    (trans, rot) = listener.lookupTransform('/odom', '/base_link', rospy.Time(0))

                    self.current_pose = Pose()
                    self.current_pose.position.x = trans[0]
                    self.current_pose.position.y = trans[1]
                    self.current_pose.position.z = trans[2]
                    self.current_pose.orientation.x = rot[0]
                    self.current_pose.orientation.y = rot[1]
                    self.current_pose.orientation.z = rot[2]
                    self.current_pose.orientation.w = rot[3]
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    return False, "Error looking up transform from 'odom' to 'base_link'"

            # DEBUG:
            print("Running maneuver optimization...")
            # Initial value for optimization
            x0 = [self.start_x_s, self.start_y_s]
            lower_bounds = [-2, -5]
            upper_bounds = [20, 16]

            # Set params
            # TODO: add the forklifts current pose from "/odom"
            current_pose2D = Pose2D()
            current_pose2D.x = self.current_pose.position.x
            current_pose2D.y = self.current_pose.position.y
            euler_angles = euler_from_quaternion([self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w])
            current_pose2D.theta = euler_angles[2]

            params = {"current_pose" : [current_pose2D.x,current_pose2D.y,current_pose2D.theta], "forklift_length" : (self.base_to_back + self.base_to_clamp), "weights" : [10, 1, 0.1, 1], "obstacles" : self.obstacles, "min_radius" : self.min_radius}

            print("Using optimization method: %d" % self.optimization_method)

            #==================================================================#
            # vvv Add Autograd gradient functions here if you get to it vvv
            #==================================================================#
            # Generate Gradient Functions
            self.grad_maneuverObjective = grad(lambda x: self.maneuverObjective(x, params))
            self.hessian_maneuverObjective = hessian(lambda x: self.maneuverObjective(x, params))
            self.jac_maneuverIneqConstraints = jacobian(lambda x: self.maneuverIneqConstraints(x, params))
            self.hessian_maneuverIneqConstraints = hessian(lambda x: self.maneuverIneqConstraints(x, params))

            # # Test Gradients against finite difference method
            # delta = 0.0000001
            # x = np.array([self.start_x_s, self.start_y_s], dtype=np.float)
            # dx = deepcopy(x)
            # dx[0] = x[0] + delta
            # print("Objective: ")
            # print(self.maneuverObjective(x, params))
            # print(self.maneuverObjective(dx, params))
            # print("Autograd:")
            # print(self.grad_maneuverObjective(x))
            # print("Finite Difference:")
            # print((self.maneuverObjective(dx, params) - self.maneuverObjective(x, params))/delta)
            # print("Hessian:")
            # print(self.hessian_maneuverObjective(x))
            # print("Autograd con:")
            # print(self.jac_maneuverIneqConstraints(x))
            # print("Constraint Jacobian:")
            # print(self.gradManeuverIneqConstraints(x, params))
            # print("Hessian con:")
            # print(self.hessian_maneuverIneqConstraints(x))
            #==================================================================#
            # ^^^ Add Autograd gradient functions here if you get to it ^^^
            #==================================================================#

            #==================================================================#
            # scipy.optimize.minimize optimizer
            #==================================================================#
            if (self.optimization_method == 1):
                # Set up optimization problem
                obj = lambda x: self.maneuverObjective(x, params)
                ineq_con = {'type': 'ineq',
                            'fun' : lambda x: self.maneuverIneqConstraints(x, params),
                            'jac' : self.jac_maneuverIneqConstraints}
                bounds = [(lower_bounds[0], upper_bounds[0]),
                          (lower_bounds[1], upper_bounds[1])]

                # Optimize
                tic = time.time()
                res = minimize(obj, x0, jac=self.grad_maneuverObjective, method='SLSQP', bounds=bounds, constraints=ineq_con)
                # res = minimize(obj, x0, method='BFGS', bounds=bounds, constraints=ineq_con)
                toc = time.time()

                # DEBUG:
                print("===== Optimization Results =====")
                print("time: %f(sec)" % (toc - tic))
                print("Success: %s" % res.success)
                print("Message: %s" % res.message)
                print("Results:\n  x: %f,  y: %f" % (res.x[0], res.x[1]))

                # Store result
                x_s = res.x[0]
                y_s = res.x[1]

                self.optimization_success = res.success
                message = res.message
            #==================================================================#
            # scipy.optimize.minimize optimizer
            #==================================================================#

            #==================================================================#
            # IPOPT Optimizer
            #==================================================================#
            if (self.optimization_method == 2):
                # Initial value for optimization
                x0_ip = np.array([x0[0], x0[1]])

                nvar = 2
                x_L = np.array(lower_bounds, dtype=np.float_)
                x_U = np.array(upper_bounds, dtype=np.float_)

                ncon = 1
                g_L = np.array([0], dtype=np.float_)
                g_U = np.array([0], dtype=np.float_)

                nnzj = nvar*ncon
                nnzh = nvar**2

                def eval_f(x):
                    return self.maneuverObjective(x, params)

                def eval_grad_f(x):
                    return self.grad_maneuverObjective(x)

                def eval_g(x):
                    return self.maneuverIneqConstraints(x, params)

                def eval_jac_g(x, flag):
                    if flag:
                        rows = np.concatenate((np.ones(nvar)*0, np.ones(nvar)*1))
                        cols = np.concatenate((np.linspace(0,nvar-1,nvar), np.linspace(nvar,2*nvar-1,nvar)))
                        return (rows, cols)
                    else:
                        return self.jac_maneuverIneqConstraints(x)

                def eval_h(x, lagrange, obj_factor, flag):
                    if flag:
                        rows = np.array([])
                        for i in range(nvar*ncon):
                            rows = np.concatenate((rows, np.ones(nvar)*i))
                        cols = np.array([])
                        for i in range(nvar*ncon):
                            cols = np.concatenate((cols, np.linspace(0,nvar-1,nvar)))
                        return (rows, cols)
                    else:
                        constraint_hessian = self.hessian_maneuverIneqConstraints(x)
                        constraint_sum = lagrange[0]*constraint_hessian[0,:,:]
                        constraint_sum = constraint_sum + lagrange[1]*constraint_hessian[1,:,:]
                        return obj_factor*self.hessian_maneuverObjective(x) + constraint_sum

                # Not using hessian, remove this line when using it
                nnzh = 0
                nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
                pyipopt.set_loglevel(0)

                tic = time.time()
                x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0_ip)
                nlp.close()
                toc = time.time()

                def print_variable(variable_name, value):
                    for i in range(len(value)):
                        print("{} {}".format(variable_name + "["+str(i)+"] =", value[i]))

                print("Solution of the primal variables, x")
                print_variable("x", x)
                #
                # print("Solution of the bound multipliers, z_L and z_U")
                # print_variable("z_L", zl)
                # print_variable("z_U", zu)
                #
                # print("Solution of the constraint multipliers, lambda")
                # print_variable("lambda", constraint_multipliers)
                #
                # print("Objective value")
                # print("f(x*) = {}".format(obj))

                # DEBUG:
                print("===== Optimization Results (IPOPT) =====")
                print("time: %f" % (toc - tic))
                print("Success: %s" % status)
                print("Message: %s" % "")
                print("Results:\n  x: %f,  y: %f" % (x[0], x[1]))

                # Store result
                x_s = x[0]
                y_s = x[1]

                self.optimization_success = (status > 0)
                message = "ipopt optimization finished with status: {0:d}".format(status)
            #==================================================================#
            # IPOPT Optimizer
            #==================================================================#

            #=================================================================#
            # Use hardcoded value
            #=================================================================#
            if (self.optimization_method == 0):
                x_s = x0[0]
                y_s = x0[1]

                self.optimization_success = 1
                message = "used hardcoded starting value"

            # Set initial pose angle for the forklift to be the direction facing the roll
            theta_s = np.arctan2(self.target_y - y_s, self.target_x - x_s)

            # Initialize path messages
            current_time = rospy.Time.now()
            path1_msg = Path()
            path2_msg = Path()
            path1_gear_msg = PathWithGear()
            path2_gear_msg = PathWithGear()
            path1_msg.header.stamp = current_time
            path1_msg.header.frame_id = "odom"
            path2_msg.header.stamp = current_time
            path2_msg.header.frame_id = "odom"
            path1_gear_msg.path.header.stamp = current_time
            path1_gear_msg.path.header.frame_id = "odom"
            path2_gear_msg.path.header.stamp = current_time
            path2_gear_msg.path.header.frame_id = "odom"

            # Publish first segment of maneuver
            # NOTE: Just set the path to be a single point at the current position. This will make the master controller work the same and quickly move through the two maneuver paths
            point = PoseStamped()
            point.header.frame_id = "odom"
            point.pose.position.x = self.current_pose.position.x
            point.pose.position.y = self.current_pose.position.y
            path1_msg.poses.append(point)
            path1_gear_msg.path.poses.append(point)
            # Set gear, positive alpha = forward gear
            self.path1_pub.publish(path1_msg)
            path1_gear_msg.gear = 1
            self.path1_gear_pub.publish(path1_gear_msg)

            # Publish second segment of maneuver
            point = PoseStamped()
            point.header.frame_id = "odom"
            point.pose.position.x = self.current_pose.position.x
            point.pose.position.y = self.current_pose.position.y
            path2_msg.poses.append(point)
            path2_gear_msg.path.poses.append(point)
            # Set gear, positive alpha = forward gear
            self.path2_pub.publish(path2_msg)
            path2_gear_msg.gear = 1
            self.path2_gear_pub.publish(path2_gear_msg)

            if (self.optimization_success):
                # If optimization was successful, publish the new target
                # position for the A* algorithm (you will want to make this a
                # separate "goal" value distinct from the roll target position)
                rospy.set_param("/control_panel_node/goal_x", float(x_s))
                rospy.set_param("/control_panel_node/goal_y", float(y_s))
                self.update_obstacle_end_pose = False

                # Publish the starting pose for the approach b-spline path
                approach_start_pose = PoseStamped()
                approach_start_pose.header.frame_id = "/odom"
                approach_start_pose.pose.position.x = x_s
                approach_start_pose.pose.position.y = y_s
                quat_forklift = quaternion_from_euler(0, 0, wrapToPi(theta_s))
                approach_start_pose.pose.orientation.x = quat_forklift[0]
                approach_start_pose.pose.orientation.y = quat_forklift[1]
                approach_start_pose.pose.orientation.z = quat_forklift[2]
                approach_start_pose.pose.orientation.w = quat_forklift[3]

                self.approach_pose_pub.publish(approach_start_pose)

            return self.optimization_success, message

        else:
            return False, "No target pose exists"


    def occupancyGridCallback(self, msg):
        '''
        Callback function for the OccupancyGrid created by the mapping node.
        Sets the member variable 'self.obstacles' as a vector of points
        representing the presence of obstacles on the map.

        Args:
        -----
        msg: nav_msgs/OccupancyGrid message

        Returns:
        None
        '''
        self.obstacles = self.gridMapToObstacles(msg)

    def rollCallback(self, msg):
        '''
        Reads in the rolls target pose. The (x,y) position are contained in the
        pose.position variable and orientation represents the surface direction
        towards which the vehicle should approach.

        Args:
        -----
        msg: PoseStamped object containing the roll position in
             msg.pose.position.x, msg.pose.position.y and the yaw angle in
             msg.pose.orientation.z

        Returns:
        --------
        None
        '''
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        euler_angles = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.target_approach_angle = euler_angles[2]

    def odomCallback(self, msg):
        '''
        Stores the forklift's current pose from odometry data.

        Args:
        -----
        msg: nav_msgs/Odometry object

        Returns:
        --------
        None
        '''
        self.current_pose = msg.pose.pose

    def obstaclePathCallback(self, msg):
        '''
        Stores the obstacle avoidance path generated by the path planner for
        determining where to update the final point of the path in order to
        accomodate the robots pose at the end of the obstacle path and the start
        of the maneuver's first segment.

        Args:
        -----
        msg: nav_msgs/Path object

        Returns:
        --------
        None
        '''
        if (self.update_obstacle_end_pose):
            # Calculate the pose from the last two points in the path
            x_1 = msg.poses[-2].pose.position.x
            y_1 = msg.poses[-2].pose.position.y
            x_2 = msg.poses[-1].pose.position.x
            y_2 = msg.poses[-1].pose.position.y
            theta_obstacle_path = np.arctan2((y_2 - y_1), (x_2 - x_1))

            # Find the heading error for the first maneuver segment
            theta_error = self.pose_s.theta - theta_obstacle_path
            direction = np.sign(theta_error)

            # Place end point at appropriate position along the minimum turning radius based on heading error
            center_point = np.array([self.pose_s.x, self.pose_s.y])
            R_s = rotZ2D(self.pose_s.theta)
            new_target = center_point + R_s.dot(np.array([self.min_radius*np.cos(-direction*(np.pi/2) - theta_error), self.min_radius*np.sin(-direction*(np.pi/2) - theta_error)]))

            # Update target position for obstacle avoidance path
            rospy.set_param("/control_panel_node/goal_x", float(new_target[0]))
            rospy.set_param("/control_panel_node/goal_y", float(new_target[1]))
            self.update_obstacle_end_pose = False

    def maneuverService(self, req):
        '''
        Receives the service request and runs the optimization function. Any
        other path generation logic should be implemented in here.

        Args:
        -----
        req: the service request message

        Returns:
        --------
        resp: the service response message, contains success status and a
            message
        '''
        #======================================================================#
        # Perform any preliminary logic conditions here
        #======================================================================#
        # This is where you can handle deciding whether an optimization is necessary or not or which options/conditiions to use.

        #===== Perform Optimization =====#
        success, message = self.optimizeManeuver()
        return OptimizeManeuverResponse(success, message)


if __name__ == "__main__":
    try:
        maneuver_path = ManeuverPath()
        maneuver_path.spin()
    except rospy.ROSInterruptException:
        pass
