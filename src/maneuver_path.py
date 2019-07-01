#!/usr/bin/env python

'''
Performs an optimization to determine the best points for a 3 point
turn-around maneuver. The starting point for the maneuver is the ending point
for the obstacle avoidance path and the final point of the maneuver is the
starting point for the grasping path generation.
'''


import rospy
import numpy as np
import math
from inPolygon import pointInPolygon


class Pose2D:
    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta

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
        # ROS Objects
        rospy.init_node("maneuver_path")
        self.rate = rospy.Rate(30)

        # Forklift dimensions
        self.base_to_clamp = 1.4658 # {m}
        self.base_to_back = 1.9472 # {m}
        self.width = 1.3599 # {m}
        self.buffer = 1.0 # {m} gap between edge of forklift and the bounding box

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
        self.max_angle = 70 # deg
        self.min_radius = self.axle_distance/np.tan(self.max_angle*(np.pi/180.0))

        self.resolution = 10 # map resolution, updates with each OccupancyGrid map callback

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
        pos_m = pos_s + R1.dot(np.array([np.abs(r_1)*np.sin(alpha_1), r_1*(1 - np.cos(alpha_1))]))

        # Make second parameters opposite signs of the first
        r_2 = -np.sign(r_1)*r_2
        alpha_2 = -np.sign(alpha_1)*alpha_2

        # Calculate parameters for the final pose
        theta_f = theta_m + np.sign(r_2)*alpha_2
        R2 = rotZ2D(theta_m)
        pos_f = pos_m + R2.dot(np.array([np.abs(r_2)*np.sin(alpha_2), r_2*(1 - np.cos(alpha_2))]))

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
        arc_length = r*alpha
        num_points = math.ceil(arc_length/self.resolution)
        delta_alpha = alpha/(num_points)

        start_position = np.array([pose.x, pose.y])
        R = rotZ2D(pose.theta)
        path = []
        for i in range(1, num_points+1)
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
            points.append(np.array([x, y]) + R_forklift.dot(corners[i]))

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
        points_s = forkliftBoundingBox(pose_s)
        points_m = forkliftBoundingBox(pose_m)
        points_f = forkliftBoundingBox(pose_f)

        all_points = np.vstack((points_s, points_m, points_f))

        # Find the convex hull points
        hull = ConvexHull(all_points)
        points = hull.points[hull.vertices]

        # Add the starting point to the end of the array to make it a "closed"
        # polygon
        points = np.vstack((points, points[0,:]))

        return points

    def rowMajorIndex(i, width):
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

    def gridMapToObstacles(grid):
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
        origin_x = grid.info.origin.x
        origin_y = grid.info.origin.y

        # Iterate through each cell in the grid and determine the obstacle probability
        obstacles = []
        for i in range(len(grid.data)):
            # If there is an obstacle, convert index into row & column
            if (grid.data[i] > 0):
                [row, col] = rowMajorIndex(i, width)

                # Convert row and column into a position in map frame
                # The map is in "image frame", meaning that the rows are along the Y
                # axis and the columns are along the X axis with the Z axis going
                # "into" the matrix.
                x = (col*resolution + resolution/2.0) + origin_x
                y = (row*resolution + resolution/2.0) + origin_y
                obstacles.append([x,y])

        obstacles = np.asarray(obstacles)

        return obstacles

    def obstaclesInManeuver(self, pose_s, pose_m, pose_f, obstacles):
        '''
        Outputs the number of obstacles present in the maneuver bounding box.

        Args:
        -----
        pose_s: starting pose as Pose2D object
        pose_m: middle pose as Pose2D object
        pose_f: final pose as Pose2D object
        obstacles: mx2 numpy array of 2D points representing the position of
                   obstacles

        Returns:
        --------
        num_obstacles: number of obstacles within the maneuver bounding box
        '''
        # Create polygon of points representing the bounding box
        polygon = maneuverBoundingBox(pose_s, pose_m, pose_f)

        # Determine which points are inside the boundary
        [points_in, points_on] = pointInPolygon(obstacles, polygon)
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
        x: [x_s, y_s, theta_s, r_1, alpha_1, r_2, alpha_2]
        x_s: forklift starting 'x' position of maneuver
        y_s: forklist starting 'y' position of maneuver
        theta_s: forklift starting 'yaw' angle of maneuver
        r_1: turning radius of initial maneuver segment (can be positive or
             negative, must be bounded by the radius of the max steering angle)
        alpha_1: arc angle of initial maneuver segment (can be positive or
                 negative)
        r_2: turning radius of second maneuver segment (must be positive so its
             sign can be adjusted to be opposite of r_1, only the magnitude of
             r_2 is important)
        alpha_2: arc angle of second maneuver segment (must be positive so its
                 sign can be adjusted to be opposite of alpha_1, only the
                 magnitude of alpha_2 is important)

        params: {"approach_point" : [x,y], "current_pose" : pose,
                 "forklift_length" : L_f, "weights" : [w_1, w_2, w_3]}
        "approach_point" : [x,y]: point 3 in the roll approach B-spline path
                           given as a list containing the (x,y) position.
        "current_pose" : [pose]: the forklift's current pose provided as Pose2D
                         object containing the (x,y) position and yaw angle, theta.
        "forklift_length" : L_f: length of the forklift from clamp tip to back.
        "weights" : [w_1, w_2, w_3]: weights for the cost, w_1 = weight for
                    distance error between the approach point and the point one
                    forklift length forward from the final pose of the
                    maneuver, w_2 = weight for the maneuver length, w_3 =
                    weight for the distance error between the maneuver starting
                    position and the forklift's current position.

        Returns:
        --------
        J: the cost of the maneuver using the given design variables
        '''
        # Unpack variables and parameters
        x_s = x[0]
        y_s = x[1]
        theta_s = x[2]
        r_1 = x[3]
        alpha_1 = x[4]
        r_2 = x[5]
        alpha_2 = x[6]
        approach_point = params["approach_point"]
        current_pose = params["current_pose"]
        L_f = params["forklift_length"]
        weights = params["weights"]

        # Get the maneuver poses based on the current design variables
        [x_m, y_m, theta_m, x_f, y_f, theta_f] = maneuverPoses(x_s, y_s, theta_s, r_1, alpha_1, r_2, alpha_2)

        # Calculate the maneuver length
        maneuver_length = maneuverLength(r_1, alpha_1, r_2, alpha_2)

        # Calculate the second point of the B-spline approach curve and get the distnace from the third point
        point2 = [L_f*np.cos(theta_f) + x_f, L_f*np.sin(theta_f) + y_f]
        approach_error = np.sqrt((approach_point[0] - point2[0])**2 + (approach_point[1] - point2[1])**2)

        # Calculate the distance between the maneuver start pose and the current forklift pose
        start_error = np.sqrt((current_pose[0] - x_s)**2 + (current_pose[1] - y_s)**2)

        # Calculate the weighted cost
        J = weights[0]*approach_error + weights[1]*maneuver_length + weights[2]*start_error

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
        x: [x_s, y_s, theta_s, r_1, alpha_1, r_2, alpha_2]
        x_s: forklift starting 'x' position of maneuver
        y_s: forklist starting 'y' position of maneuver
        theta_s: forklift starting 'yaw' angle of maneuver
        r_1: turning radius of initial maneuver segment (can be positive or
             negative, must be bounded by the radius of the max steering angle)
        alpha_1: arc angle of initial maneuver segment (can be positive or
                 negative)
        r_2: turning radius of second maneuver segment (must be positive so its
             sign can be adjusted to be opposite of r_1, only the magnitude of
             r_2 is important)
        alpha_2: arc angle of second maneuver segment (must be positive so its
                 sign can be adjusted to be opposite of alpha_1, only the
                 magnitude of alpha_2 is important)

        params: {"obstacles" : obstacles, "min_radius" : min_radius}
        "obstacles" : obstacles: mx2 numpy array of obstacle locations (x,y)
        "min_radius" : min_radius: the minimum allowable turning radius

        Returns:
        --------
        C: (2,) numpy array containing the constraint values
        '''
        # Unpack variables and parameters
        x_s = x[0]
        y_s = x[1]
        theta_s = x[2]
        r_1 = x[3]
        alpha_1 = x[4]
        r_2 = x[5]
        alpha_2 = x[6]

        # Initialize constraints
        C = np.zeros(2)

        # Get the maneuver poses based on the current design variables
        [x_m, y_m, theta_m, x_f, y_f, theta_f] = maneuverPoses(x_s, y_s, theta_s, r_1, alpha_1, r_2, alpha_2)

        # Frist constraint
        C[0] = -obstaclesInManeuver(x_s, y_s, theta_s, x_m, y_m, theta_m, x_f, y_f, theta_f, params["obstacles"])

        # Second constraint
        C[1] = abs(x[3]) - params["min_radius"]

        return C

    def optimizeManeuver(self):
        '''
        Sets up the optimization problem then calculates the optimal maneuver poses.

        Args:
        -----
        None

        Returns:
        --------
        path: the optimal path as a ROS nav_msgs/Path message
        '''
        # Set initial guess
        # TODO: need to determine a method for setting a good starting position. I think you should take the desired target postion and approach angle, then move the starting position out by X meters with two 90deg turns.
        x_s = 15
        y_s = 10
        theta_s = -(3*np.pi/4)
        r_1 = 2
        alpha_1 = -np.pi/2
        r_2 = 2
        alpha_2 = np.pi/2
        x0 = [x_s, y_s, theta_s, r_1, alpha_1, r_2, alpha_2]

        # Get obstacle vector
        obstacles = np.array([[10,10],[11,11],[12,12],[15,20],[14,16],[20,20]])

        # Set params
        params = {"approach_point" : [17,17], "current_pose" : [1,1,-3*np.pi/4], "forklift_length" : (base_to_back + base_to_clamp), "weights" : [10, 1, 1], "obstacles" : obstacles, "min_radius" : min_radius}

        # Set up optimization problem
        obj = lambda x: maneuverObjective(x, params)
        ineq_con = {'type': 'ineq',
                    'fun' : lambda x: maneuverIneqConstraints(x, params),
                    'jac' : None}
        bounds = [(None, None),
                  (None, None),
                  (-np.pi, np.pi),
                  (None, None),
                  (-2*np.pi, 2*np.pi),
                  (min_radius, None),
                  (0, 2*np.pi)]

        # Optimize
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=ineq_con)

        # Store result
        x_s = res.x[0]
        y_s = res.x[1]
        theta_s = res.x[2]
        r_1 = res.x[3]
        alpha_1 = res.x[4]
        r_2 = res.x[5]
        alpha_2 = res.x[6]

        # TODO: write function for getting a vector of points along the curvature of each path segment.
        # NOTE: remember to make the sign of r_2 opposite of r_1 and alpha_2 opposite of alpha_1
