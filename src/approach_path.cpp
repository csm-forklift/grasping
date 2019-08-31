/**
 * Generates a B-spline curve from a set of control points. The spline in this
 * code is intended to be cubic (p = 3) but can be changed to any order as long
 * as you have enough control points. The B-spline knot vector is defined wiht
 * the range [0,1] and uniformly split. The vector of points along the line
 * (defined as 'x') is also on the range of [0,1] and split uniformly, but the
 * resolution of the line can be defined. The path is generated using De Boor's
 * Algorithm (https://en.wikipedia.org/wiki/De_Boor%27s_algorithm) and then
 * converted to a ROS nav_msgs:Path message and published.
 */

// TODO: consider making your own message type with a vector of
// geometry_msgs::Points[] as the control points, an Int as the polynomial
// order, and another Int for the line resolution

// TODO: add a parameter that let's you set the frame that the path is in
// (odom/map/world/etc)

// TODO: publish an array of marker points for visualizing the control points,
// make this a debugging feature that you can turn on or off

#include <iostream>
#include <vector>
#include <algorithm> // find
#include <cmath> // used for M_PI (pi constant)
#include <Eigen>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <tf/tf.h>
#include <visualization_msgs/Marker.h>

using namespace std;
using namespace Eigen;

class GraspPath
{
private:
    // Path Variables
    vector<Vector2d> m_control_points; // Control points
    vector<double> m_knots; // Knot vector
    vector<double> m_x; // 'x' vector
    int m_x_size; // path resolution
    vector<int> m_k; // 'k' vector
    int m_p; // Polynomial order
    vector<Vector2d> m_path; // Vector path
    int path_type; // type of path to generate
    vector<int> available_path_types; // allowable path types (0 = full spline, 1 = from roll, 2 = straight line)

    // ROS Objects
    ros::NodeHandle nh_;
    ros::Subscriber sub_roll; // reads the roll position and approach angle
    ros::Subscriber sub_forklift; // reads the current forklift pose
    ros::Publisher pub_path;
    bool debug_control_points; // if true, publishes the control points as markers for rviz
    ros::Publisher pub_control_points;
    nav_msgs::Path ros_path;
    visualization_msgs::Marker ros_control_points;
    geometry_msgs::PoseStamped roll_pose; // contains the roll position and approach angle
    geometry_msgs::PoseStamped forklift_pose; // the forklifts current pose

    // Forklift Dimensions
    double body_length; // forklift body length
    double base_to_clamp; // length of long arm of clamp
    double total_length; // total length of the forklift + clamp

    // Roll parameters
    double roll_radius; // radius of the paper roll

public:
    GraspPath() : nh_("~"),available_path_types{0,1,2}
    {
        // Store parameters
        nh_.param<int>("path/type", path_type, 0);
        nh_.param<int>("path/polynomial_order", m_p, 3); // Define the polynomial order (prefer cubic, p = 3)
        nh_.param<int>("path/resolution", m_x_size, 20); // Define resolution of the line, this is the number of points on the path
        nh_.param<double>("/forklift/body/length", body_length, 2.5601);
        nh_.param<double>("/forklift/body/total", total_length, 3.5659);
        nh_.param<double>("/forklift/body/base_to_clamp", base_to_clamp, 1.0058);
        nh_.param<double>("/roll/radius", roll_radius, 0.20);

        // Parameter Input Checking
        vector<int>::iterator it;
        it = find(available_path_types.begin(), available_path_types.end(), path_type);
        if (it == available_path_types.end()) {
            std::string available_types = "";
            for (int i = 0; i < available_path_types.size(); ++i) {
                available_types += to_string(available_path_types.at(i));
                if (i < (available_path_types.size() - 1)) {
                    available_types += ", ";
                }
            }
            ROS_INFO("Invalid path_type: %d.\nSetting to default: 0\nAvailable types: %s", path_type, available_types.c_str());
            path_type = 0;
        }

        // Define ROS Objects
        pub_path = nh_.advertise<nav_msgs::Path>("path", 1);
        sub_roll = nh_.subscribe("/roll/pose", 1, &GraspPath::rollCallback, this);
        sub_forklift = nh_.subscribe("/forklift/approach_pose", 1, &GraspPath::forkliftCallback, this);

        // Publish the control points for debugging visualization
        debug_control_points = true;
        if (debug_control_points) {
            // Create publisher
            pub_control_points = nh_.advertise<visualization_msgs::Marker>("control_points", 1, true);

            // Initialize Marker message
            ros_control_points.header.frame_id = "odom";
            ros_control_points.id = 0;
            ros_control_points.type = visualization_msgs::Marker::SPHERE_LIST;
            ros_control_points.action = visualization_msgs::Marker::ADD;
            ros_control_points.scale.x = 0.1;
            ros_control_points.scale.y = 0.1;
            ros_control_points.scale.z = 0.1;
            ros_control_points.color.r = 1;
            ros_control_points.color.g = 0;
            ros_control_points.color.b = 0;
            ros_control_points.color.a = 1;
            ros_control_points.lifetime = ros::Duration(0);

            ros_control_points.pose.position.x = 0;
            ros_control_points.pose.position.y = 0;
            ros_control_points.pose.position.z = 0;
            ros_control_points.pose.orientation.x = 0;
            ros_control_points.pose.orientation.y = 0;
            ros_control_points.pose.orientation.z = 0;
            ros_control_points.pose.orientation.w = 1;

            updateControlPoints();
        }
    }

    void fullSplineControlPoints()
    {
        /**
         * Creates the control points for the full b-spline path that considers
         * the roll approach pose as well as the full forklift pose.
         */

        // Get roll position (x_r, y_r) and desired approach angle (alpha)
        double x_r = roll_pose.pose.position.x; // x location of roll
        double y_r = roll_pose.pose.position.y; // y location of roll
        tf::Quaternion roll_q(
            roll_pose.pose.orientation.x,
            roll_pose.pose.orientation.y,
            roll_pose.pose.orientation.z,
            roll_pose.pose.orientation.w
        );
        double roll_r, pitch_r, alpha;
        tf::Matrix3x3(roll_q).getRPY(roll_r, pitch_r, alpha);

        double x_f = forklift_pose.pose.position.x; // x location of forklift
        double y_f = forklift_pose.pose.position.y; // y location of forklift

        // Get yaw angle from quaternion
        tf::Quaternion q(
            forklift_pose.pose.orientation.x,
            forklift_pose.pose.orientation.y,
            forklift_pose.pose.orientation.z,
            forklift_pose.pose.orientation.w
        );
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // Calculate the control points for the desired path
        // Calculate distance to roll for determining which waypoints to use
        double roll_distance = sqrt(pow(x_f - x_r, 2) + pow(y_f - y_r, 2));
        m_control_points.clear();

        if (roll_distance < (base_to_clamp + roll_radius)) {
            m_control_points.push_back(Vector2d(x_f, y_f));
            m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));

            // Make polynomial order 1 (a line)
            m_p = 1;
        }
        else if (roll_distance < (2*base_to_clamp + roll_radius)) {
            m_control_points.push_back(Vector2d(x_f, y_f));
            m_control_points.push_back(Vector2d((roll_radius + base_to_clamp)*cos(alpha) + x_r, (roll_radius + base_to_clamp)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));

            // Make polynomial order 2 (quadratic)
            m_p = 2;
        }
        else if (roll_distance < (total_length + base_to_clamp + roll_radius)) {
            m_control_points.push_back(Vector2d(x_f, y_f));
            m_control_points.push_back(Vector2d((roll_radius + 2*base_to_clamp)*cos(alpha) + x_r, (roll_radius + 2*base_to_clamp)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d((roll_radius + base_to_clamp)*cos(alpha) + x_r, (roll_radius + base_to_clamp)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));
        }
        else if (roll_distance < (total_length + 2*base_to_clamp + roll_radius)) {
            m_control_points.push_back(Vector2d(x_f, y_f));
            m_control_points.push_back(Vector2d((roll_radius + base_to_clamp + total_length)*cos(alpha) + x_r, (roll_radius + base_to_clamp + total_length)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d((roll_radius + 2*base_to_clamp)*cos(alpha) + x_r, (roll_radius + 2*base_to_clamp)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d((roll_radius + base_to_clamp)*cos(alpha) + x_r, (roll_radius + base_to_clamp)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));
        }
        else {
            m_control_points.push_back(Vector2d(x_f, y_f));
            m_control_points.push_back(Vector2d(base_to_clamp*cos(yaw) + x_f, base_to_clamp*sin(yaw) + y_f));
            m_control_points.push_back(Vector2d((roll_radius + base_to_clamp + total_length)*cos(alpha) + x_r, (roll_radius + base_to_clamp + total_length)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d((roll_radius + 2*base_to_clamp)*cos(alpha) + x_r, (roll_radius + 2*base_to_clamp)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d((roll_radius + base_to_clamp)*cos(alpha) + x_r, (roll_radius + base_to_clamp)*sin(alpha) + y_r));
            m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));
        }
    }

    void fromRollControlPoints()
    {
        /**
         * Generates control points for a b-spline that does not consider
         * forklift orientation. It generates appropriate control points coming
         * out from the roll in the desired approach direction and then simply
         * uses the forklifts current (x,y) position as the final point.
         */

         // Get roll position (x_r, y_r) and desired approach angle (alpha)
         double x_r = roll_pose.pose.position.x; // x location of roll
         double y_r = roll_pose.pose.position.y; // y location of roll
         tf::Quaternion roll_q(
             roll_pose.pose.orientation.x,
             roll_pose.pose.orientation.y,
             roll_pose.pose.orientation.z,
             roll_pose.pose.orientation.w
         );
         double roll_r, pitch_r, alpha;
         tf::Matrix3x3(roll_q).getRPY(roll_r, pitch_r, alpha);

         double x_f = forklift_pose.pose.position.x; // x location of forklift
         double y_f = forklift_pose.pose.position.y; // y location of forklift

         // Calculate the control points for the desired path
         // Calculate distance to roll for determining which waypoints to use
         double roll_distance = sqrt(pow(x_f - x_r, 2) + pow(y_f - y_r, 2));
         m_control_points.clear();

         if (roll_distance < (base_to_clamp + roll_radius)) {
             m_control_points.push_back(Vector2d(x_f, y_f));
             m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));

             // Make polynomial order 1 (a line)
             m_p = 1;
         }
         else if (roll_distance < (2*base_to_clamp + roll_radius)) {
             m_control_points.push_back(Vector2d(x_f, y_f));
             m_control_points.push_back(Vector2d((roll_radius + base_to_clamp)*cos(alpha) + x_r, (roll_radius + base_to_clamp)*sin(alpha) + y_r));
             m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));

             // Make polynomial order 2 (quadratic)
             m_p = 2;
         }
         else if (roll_distance < (total_length + base_to_clamp + roll_radius)) {
             m_control_points.push_back(Vector2d(x_f, y_f));
             m_control_points.push_back(Vector2d((roll_radius + 2*base_to_clamp)*cos(alpha) + x_r, (roll_radius + 2*base_to_clamp)*sin(alpha) + y_r));
             m_control_points.push_back(Vector2d((roll_radius + base_to_clamp)*cos(alpha) + x_r, (roll_radius + base_to_clamp)*sin(alpha) + y_r));
             m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));
         }
         else  {
             m_control_points.push_back(Vector2d(x_f, y_f));
             m_control_points.push_back(Vector2d((roll_radius + base_to_clamp + total_length)*cos(alpha) + x_r, (roll_radius + base_to_clamp + total_length)*sin(alpha) + y_r));
             m_control_points.push_back(Vector2d((roll_radius + 2*base_to_clamp)*cos(alpha) + x_r, (roll_radius + 2*base_to_clamp)*sin(alpha) + y_r));
             m_control_points.push_back(Vector2d((roll_radius + base_to_clamp)*cos(alpha) + x_r, (roll_radius + base_to_clamp)*sin(alpha) + y_r));
             m_control_points.push_back(Vector2d(roll_radius*cos(alpha) + x_r, roll_radius*sin(alpha) + y_r));
         }
    }

    void straightLineControlPoints()
    {
        /**
         * Generates control points for a straight line from the forklift's
         * current position to the roll's current position.
         */

        // Get the two end points, then place two more points inbetween along a straight line. This allows the polynomial to remain 3rd order.
        // Get roll position (x_r, y_r) and desired approach angle (alpha)
        double x_r = roll_pose.pose.position.x; // x location of roll
        double y_r = roll_pose.pose.position.y; // y location of roll

        double x_f = forklift_pose.pose.position.x; // x location of forklift
        double y_f = forklift_pose.pose.position.y; // y location of forklift

        // Line Angle
        double theta = atan2(y_r - y_f, x_r - x_f);

        // Line length
        double roll_distance = sqrt(pow(x_f - x_r, 2) + pow(y_f - y_r, 2));

        // Add control points
        m_control_points.clear();
        m_control_points.push_back(Vector2d(x_f, y_f));
        m_control_points.push_back(Vector2d(x_f + roll_distance/3*cos(theta), y_f + roll_distance/3*sin(theta)));
        m_control_points.push_back(Vector2d(x_f + 2*roll_distance/3*cos(theta), y_f + 2*roll_distance/3*sin(theta)));
        m_control_points.push_back(Vector2d(x_r, y_r));
    }

    // De Boor's Algorithm Implementation
    Vector2d deBoors(int k, double x, const vector<double> &t, const vector<Vector2d> &c, int p)
    {
        /**
         * Evaluates the B-spline function at 'x'
         *
         * Args
         * -----
         * k: index of knot interval that contains x
         * x: position along curve (goes between lowest knot value to highest
         *    knot value)
         * t: array of knot positions, needs to be padded with 'p' extra
         *    endpoints
         * c: array of control points
         * p: degree of B-spline
         */

        vector< Vector2d > d(p+1);
        for (int j = 0; j <= p; j++) {
            d.at(j) = c.at(j+k-p);
        }
        for (int r = 1; r <= p; r++) {
            for (int j = p; j >= r; j--) {
                double alpha = (x - t.at(j+k-p)) / (t.at(j+1+k-r) - t.at(j+k-p));
                d.at(j) = (1 - alpha)*d.at(j-1) + alpha*d.at(j);
            }
        }

        return d.at(p);
    }

    // Generate vectors using the given control points
    void constructVectors()
    {
        /**
         * Generates the knot vector and the vectors for the position along the
         * path (m_x) and its corresponding knot vector interval (m_k). A check
         * is made to make sure the polynomial order does not exceed the number
         * of control points - 1.
         */

        // Check that the polynomial order does not exceed control points - 1
        if (m_p >= m_control_points.size()) {
            // DEBUG: print a warning
            cout << "[WARN]: The provided B-spline polynomial order of " << m_p << " requires at least " << (m_p + 1) << " control points.";
            cout << " Only " << m_control_points.size() << " control points were provided.";
            cout << " The polynomial order will be reduced to " << (m_control_points.size() - 1) << ".\n";

            m_p = m_control_points.size() - 1;
        }

        // Generate the knot vector based on the number of control points
        // Need at least (m_p - 3) more control points than internal knot points
        // (that's m_p - 1 more control points than internal knots + the end points, so you need to add 'm_p' repeated knot endpoints)
        int m_knot_size = m_control_points.size() - (m_p - 1);
        m_knots.resize(m_knot_size, 0.0);
        double div_size = 1.0/(m_knot_size - 1.0);
        for (int i = 0; i < m_knots.size(); i++) {
            m_knots.at(i) = i*div_size;
        }

        // Append the additional 'm_p' knots at each end
        m_knots.resize(m_knot_size + m_p, 1.0);
        m_knots.insert(m_knots.begin(), m_p, 0.0);

        // Generate the 'm_x' and 'm_k' vectors
        m_x.resize(m_x_size, 0.0);
        m_k.resize(m_x_size, 0);
        div_size = 1.0/(m_x_size - 1.0);
        for (int i = 0; i < m_x.size(); i++) {
            m_x.at(i) = i*div_size;
            for (int j = m_p; j < (m_knots.size()-1)-m_p; j++) {
                if (m_x.at(i) >= m_knots.at(j)) {
                    m_k.at(i) = j;
                }
            }
        }
    }

    // Iterate through all positions using DeBoor's algorithm to generate the path vector
    void generatePath()
    {
        /**
         * Iterates through each point in the 'm_x' vector and calculates the
         * corresponding path position using De Boor's algorithm.
         */
        m_path.resize(m_x.size());

        for (int i = 0; i < m_x.size(); i++) {
            Vector2d path_point = deBoors(m_k.at(i), m_x.at(i), m_knots, m_control_points, m_p);
            m_path.at(i) = path_point;
        }

        // Convert vector path into a ROS message path
        ros_path.header.frame_id = "odom";
        ros_path.poses.resize(m_path.size());
        geometry_msgs::PoseStamped pose;
        int path_seq = 0;
        for (int i = 0; i < m_path.size(); i++) {
            pose.pose.position.x = m_path.at(i)[0];
            pose.pose.position.y = m_path.at(i)[1];
            pose.header.seq = path_seq++;
            pose.header.frame_id = "odom";
            ros_path.poses.at(i) = pose;
        }
    }

    void rollCallback(const geometry_msgs::PoseStamped msg)
    {
        // Update the roll pose
        roll_pose = msg;

    }

    // This callback generates the desired path and publishes it once
    void forkliftCallback(const geometry_msgs::PoseStamped msg)
    {
        /**
         * This callback function receives the pose data for the forklift and
         * calculates the B-spline curve from the forklifts current position to
         * the roll at the desired approach angle. The approach angle is
         * contained in the 'z' component of the 'orientation' part of the
         * 'pose' in 'roll_pose'.
         *
         * The control points are calculated using the roll and forklift poses
         * considering the desired approach angle and stopping point. There are
         * four points placed coming out from the roll in the direction of the
         * approach angle. One on the roll surface. One a clamp's length away.
         * One two clamp's lengths away. And one a forklift length way from
         * the second point. This is so the baselink, which is the middle of
         * the front axle, will be roughly two clamp lengths away from the roll
         * while being aligned straight with the approach angle. This way the
         * tip of the clamp should be a full clamp length away and the
         * fine-tuned roll detection can be used for the final approach
         * distance.
         */
        // Update the forklift pose
        forklift_pose = msg;

        if (path_type == 2) {
            straightLineControlPoints();
        }
        else if (path_type == 1) {
            fromRollControlPoints();
        }
        else {
            fullSplineControlPoints();
        }

        updateControlPoints();
        publishControlPoints();
        constructVectors();
        generatePath();
        publishPath();
    }

    // Publish
    void publishPath()
    {
        pub_path.publish(ros_path);
    }

    // Convert control points vector into marker points
    void updateControlPoints()
    {
        /**
         * The control points used to generate the B-spline curve are stored in
         * 'm_control_points' as 'Vector2d' values. These points are converted
         * into a set of markers for a ROS message to visualize in RVIZ.
         */
        if (debug_control_points) {
            ros_control_points.points.resize(m_control_points.size());
            for (int i = 0; i < m_control_points.size(); i++) {
                ros_control_points.points.at(i).x = m_control_points.at(i)[0];
                ros_control_points.points.at(i).y = m_control_points.at(i)[1];
            }
        }
    }

    // Publish control points as markers for debugging
    void publishControlPoints()
    {
        if (debug_control_points) {
            // // Delete previous markers
            // visualization_msgs::Marker delete_markers;
            // delete_markers.action = visualization_msgs::Marker::DELETEALL;
            // Publish the values
            pub_control_points.publish(ros_control_points);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "approach_path");
    GraspPath grasp_path = GraspPath();

    ros::Rate rate(10);

    while (ros::ok()) {
        grasp_path.publishPath();
        //grasp_path.publishControlPoints();
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
