#include <math.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int8.h>
#include <tf/tf.h>
#include <sys/time.h>
#include <sensor_msgs/Joy.h>

using namespace std;


class ApproachController
{
private:
    //===== ROS Objects =====//
    // Publishers and Subscribers
    ros::NodeHandle nh_;
    ros::Subscriber stretch_sensor_sub; // reads boolean indicating whether the stretch sensor is over the threshold or not
    ros::Subscriber odom_sub; // read the forklifts current pose from the odom message
    ros::Subscriber roll_sub; // read paper roll's current pose
    ros::Subscriber grasp_success_sub; // reads whether the grasp was sucessful or not
    ros::Subscriber control_sub; // reads the control mode
    ros::Subscriber steering_angle_sub;
    ros::Subscriber joystick_override_sub;
    ros::Publisher velocity_pub; // publish linear velocity command
    ros::Publisher steering_angle_pub; // publish steering angle command
    ros::Rate rate; // publishing rate
    std_msgs::Float64 steering_angle_msg;
    std_msgs::Float64 velocity_msg;

    // DEBUG: check forklift target point
    ros::Publisher forklift_target_pub;

    //===== State Variables =====//
    // Desired location for the ideal roll placement relative to the forklift
    double forklift_target_x;
    double forklift_target_y;
    double prev_forklift_target_x;
    double prev_forklift_target_y;
    double grasp_angle; // angle of the grasping plate surface normal relative to the forklift frame
    double grasp_plate_offset_x; // x offset for the shortarm plate in forklift frame
    double grasp_plate_offset_y; // y offset for the shortarm plate in forklift frame

    // Ppaer roll target position
    double paperRoll_x;
    double paperRoll_y;
    double roll_radius;
    double approach_distance; // distance from the roll to the forklift's target point
    double approach_tolerance; // stops linear motion if distance is less than this value, can be positive or negative
    double backout_distance; // distance from the roll the forklift backs out to when the grasp is unsuccessful and a retry is necessary

    // Current forklift pose
    double forklift_x;
    double forklift_y;
    double forklift_heading;

    // Stetch sensor input
    bool stretch_on;

    // Velocity bounds
    double max_velocity;

    //===== Controller Variables =====//
    // Heading variables
    double theta_desired; // desired pose heading for the forklift
    double angle_tolerance; // heading error band

    // Proportional gains
    double proportional_control_angle;
    double proportional_control_linear;

    // Linear velocity command
    double linear_velocity; // velocity command
    double movement_velocity; // velocity command after bounding

    // Steering angle command of the back wheels, positive turns wheels to the left giving a right-hand turn and vis-versa
    double steering_angle, steering_angle_min, steering_angle_max, current_steering_angle;

    // Operational State
    int operation_mode; // 0 = get to starting angle, 1 = approach roll, 2 = back out for retry
    int control_mode; // determines whether this controller is active or not

    // Joystick Controller Variables
    double timeout, timeout_start;
    int autonomous_deadman_button, manual_deadman_button;
    bool autonomous_deadman_on, manual_deadman_on; // deadman switch


public:
    ApproachController() : nh_("~"), rate(30)
    {
        //===== Set up ROS Objects =====//
        // Read in parameters
        nh_.param<double>("target_x", paperRoll_x, 0.0);
        nh_.param<double>("target_y", paperRoll_y, 0.0);
        nh_.param<double>("/roll/radius", roll_radius, 0.20);
        nh_.param<double>("grasp_angle", grasp_angle, M_PI/4);
        nh_.param<double>("angle_tolerance", angle_tolerance, 0.05);
        nh_.param<double>("K_angle", proportional_control_angle, 1.0);
        nh_.param<double>("K_linear", proportional_control_linear, 1.0);
        nh_.param<double>("max_velocity", max_velocity, 1.0);
        nh_.param<double>("grasp_plate_offset_x", grasp_plate_offset_x, 0.36);
        nh_.param<double>("grasp_plate_offset_y", grasp_plate_offset_y, -0.18);
        nh_.param<double>("approach_offset", approach_distance, 3.0);
        nh_.param<double>("approach_tolerance", approach_tolerance, 0.01);
        nh_.param<double>("backout_distance", backout_distance, 1.0);
        nh_.param<double>("/forklift/steering/min_angle", steering_angle_min, -75*(M_PI/180.0));
        nh_.param<double>("/forklift/steering/max_angle", steering_angle_max, 75*(M_PI/180.0));

        nh_.param("manual_deadman", manual_deadman_button, 4);
        nh_.param("autonomous_deadman", autonomous_deadman_button, 5);
        nh_.param("timeout", timeout, 1.0);

        // Create publishers and subscribers
        stretch_sensor_sub = nh_.subscribe("/clamp_control/stretch", 1, &ApproachController::stretchSensorCallback, this);
        odom_sub = nh_.subscribe("/odom", 1, &ApproachController::odomCallback, this);
        roll_sub = nh_.subscribe("point", 1, &ApproachController::rollCallback, this);
        grasp_success_sub = nh_.subscribe("/grasp_successful", 1, &ApproachController::graspSuccessfulCallback, this);
        control_sub = nh_.subscribe("/control_mode", 1, &ApproachController::controlModeCallback, this);
        steering_angle_sub = nh_.subscribe("/steering_node/filtered_angle", 1, &ApproachController::steeringAngleCallback, this);
        joystick_override_sub = nh_.subscribe("/joy", 1, &ApproachController::joy_override, this);
        velocity_pub = nh_.advertise<std_msgs::Float64>("/velocity_node/velocity_setpoint", 1);
        steering_angle_pub = nh_.advertise<std_msgs::Float64>("/steering_node/angle_setpoint", 1);

        // DEBUG:
        forklift_target_pub = nh_.advertise<geometry_msgs::PointStamped>("forklift_target", 1);

        //===== Initialize States =====//
        forklift_x = 0;
        forklift_y = 0;
        forklift_heading = 0;
        stretch_on = false;

        //===== Initialize Controller Variables =====//
        theta_desired = 0;
        linear_velocity = 0.0;
        movement_velocity = 0.0;
        steering_angle = 0;
        current_steering_angle = 0.0;

        // Start in "Get to starting angle" mode
        operation_mode = 0;
        control_mode = 0;

        // Initialize joystick variables
        timeout_start = getWallTime();
        autonomous_deadman_on = false;
        manual_deadman_on = false;
    }

    void spin()
    {
        while (nh_.ok()) {
            // TODO: add 'if' statements around the publishers to make sure they do not publish if the control_mode is wrong
            if (control_mode == 2) {
                // TODO: add logic that checks if the clamp is lowered and open before beginning.
                if (operation_mode == 0) {
                    // DEBUG:
                    cout << "Initializing steering angle\n";
                    setInitialAngle();
                }
                if (operation_mode == 1) {
                    // DEBUG:
                    cout << "Beginning approach\n";
                    approach();
                }
                if (operation_mode == 2) {
                    backout();
                }
            }
            rate.sleep();
            ros::spinOnce();
        }
    }

    void setInitialAngle()
    {
        // Calculate initial steering angle
        theta_desired = atan2(paperRoll_y-forklift_target_y, paperRoll_x-forklift_target_x);
        steering_angle = -proportional_control_angle * (theta_desired - forklift_heading);

        // Set steering command
        // Bound steering angle
        steering_angle = min(steering_angle, steering_angle_max);
        steering_angle = max(steering_angle, steering_angle_min);

        movement_velocity = 0;

        // Give time for the steering angle to get to position
        // ros::Duration(1.0).sleep();
        while (abs(current_steering_angle-steering_angle) > angle_tolerance) {
            // Publish the new steering angle and velocity
            publishMessages();
            ros::spinOnce();
        }

        // Change operation mode to "approach"
        operation_mode = 1;
    }

    void approach()
    {
        // Control sequence for approaching the roll
        while (stretch_on == false && ros::ok()) {
            // Calculate linear velocity for forklift
            approach_distance = sqrt(pow(forklift_target_y-paperRoll_y,2)+pow(forklift_target_x-paperRoll_x,2));
            //linear_velocity = proportional_control_linear * approach_distance;

            cout << "Roll x: " << paperRoll_x << ", y: " << paperRoll_y << "\n";

            // Adjust wheel angle as necessary
            theta_desired = atan2(paperRoll_y-forklift_target_y, paperRoll_x-forklift_target_x);
            double theta_error = wrapToPi(theta_desired-forklift_heading);
            if(abs(theta_error) > angle_tolerance) {
                steering_angle = -proportional_control_angle * theta_error;
                //linear_velocity *= (M_PI/2 - theta_error)/(M_PI/2);
            }
            else {
                steering_angle = 0;
            }

            linear_velocity = approach_distance*cos(theta_error);

            if(linear_velocity > approach_tolerance){
                if (cos(theta_error) < 0) {
                    steering_angle = -steering_angle;
                    linear_velocity = linear_velocity*5*proportional_control_linear;
                }
                else {
                    linear_velocity = linear_velocity*proportional_control_linear;
                }
            }
            else{
                linear_velocity = 0;
            }

/*
            double prev_approach_distance = sqrt(pow(prev_forklift_target_y-paperRoll_y,2)+pow(prev_forklift_target_x-paperRoll_x,2));

            double dot = ((prev_forklift_target_x-paperRoll_x)/prev_approach_distance)*((forklift_target_x-paperRoll_x)/approach_distance) + ((prev_forklift_target_y-paperRoll_y)/prev_approach_distance)*((forklift_target_y-paperRoll_y)/approach_distance);

            // FIXME: additional idea for improvement
            // add a length*sin(theta_error) to the approach_tolerance and tune the length to give an extra tolerance when the heading is off
            if (approach_distance <= approach_tolerance || dot <= 0.999){
                linear_velocity = 0;

                // TODO:
                // When on the forklift, add backout command here if the distance comes less than a desired threshold and the stretch sensors has not triggered.
            }
*/
            // Send New steering angle
            // Bound steering angle
            steering_angle = min(steering_angle, steering_angle_max);
            steering_angle = max(steering_angle, steering_angle_min);

            // Set forklift velocity
            // Bound movement velocity
            movement_velocity = min(linear_velocity, max_velocity);

            // Publish new steering angle and velocity
            publishMessages();

            // DEBUG:
            printf("D: %0.03f, Steer: %0.03f, Theta: %0.03f, Angle: %0.03f, error: %0.03f, align: %0.03f, vel: %0.03e\n", approach_distance, steering_angle, theta_desired, forklift_heading, theta_error, cos(theta_error), movement_velocity);

            // Update states
            ros::spinOnce();
            ros::Duration(0.05).sleep();

            // Break the loop if the operation mode changes
            if (operation_mode != 1) {
                break;
            }
        }

        // Stop velocity
        movement_velocity = 0;
        //velocity_msg.data = movement_velocity;
        //velocity_pub.publish(velocity_msg);

        // Publish new velocity
        publishMessages();
    }

    void backout()
    {
        // Control sequence for backing out after a failed grasp
        // Until desired distance is achieved between the forklift and the roll, send the backout velocity command
        // DEBUG:

        while (approach_distance < backout_distance) {
            // Update approach distance
            approach_distance = sqrt(pow(forklift_target_y-paperRoll_y,2)+pow(forklift_target_x-paperRoll_x,2));

            // Set backup velocity
            movement_velocity = max_velocity/5;

            // Set backup steering angle to 0
            steering_angle = 0;

            // Publish new steering angle and velocity
            publishMessages();

            ros::spinOnce();
        }

        // Return to approach operation mode
        operation_mode = 1;
    }

    void stretchSensorCallback(const std_msgs::Bool &msg)
    {
        // Update stretch sensor state
        stretch_on = msg.data;
    }

    void odomCallback(const nav_msgs::Odometry &msg)
    {
        // Update previous position
        prev_forklift_target_x = forklift_target_x;
        prev_forklift_target_y = forklift_target_y;

        // Update forklift current position and heading
        forklift_x = msg.pose.pose.position.x;
        forklift_y = msg.pose.pose.position.y;
        tf::Quaternion forklift_q(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        );

        tf::Matrix3x3 m(forklift_q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        forklift_heading = yaw;

        // Calculate forklift's new target position in the odom frame
        double plate_x = forklift_x + grasp_plate_offset_x;
        double plate_y = forklift_y + grasp_plate_offset_y;
        forklift_target_x = plate_x + roll_radius*cos(grasp_angle + forklift_heading);
        forklift_target_y = plate_y + roll_radius*sin(grasp_angle + forklift_heading);

        geometry_msgs::PointStamped forklift_target;
        forklift_target.header.frame_id = "/odom";
        forklift_target.point.x = forklift_target_x;
        forklift_target.point.y = forklift_target_y;
        forklift_target_pub.publish(forklift_target);
    }

    void rollCallback(const geometry_msgs::PointStamped &msg)
    {
        // Update paper roll from cylinder detection
        paperRoll_x = msg.point.x;
        paperRoll_y = msg.point.y;
    }

    void graspSuccessfulCallback(const std_msgs::Bool &msg)
    {
        // If the grasp is successful we can go back to state 0 to prepare for the next round
        if (msg.data == true) {
            operation_mode = 0;
        }
        // If the grasp is unsuccessful, we need to back out and retry
        else {
            operation_mode = 2;
        }
    }

    void controlModeCallback(const std_msgs::Int8 &msg)
    {
        control_mode = msg.data;
    }

    void steeringAngleCallback(const std_msgs::Float64 &msg)
    {
        current_steering_angle = msg.data;
    }

    void joy_override(const sensor_msgs::Joy joy_msg)
    {
        // Update timeout time
        timeout_start = getWallTime();

        // Update deadman buttons
        if (joy_msg.buttons[manual_deadman_button] == 1) {
            manual_deadman_on = true;
        }
		else {
            manual_deadman_on = false;
        }

        if (joy_msg.buttons[autonomous_deadman_button] == 1) {
            autonomous_deadman_on = true;
        }
		else {
            autonomous_deadman_on = false;
        }
	}

    double wrapToPi(double angle)
    {
        angle = fmod(angle + M_PI, 2*M_PI);
        if (angle < 0) {
            angle += 2*M_PI;
        }
        return (angle - M_PI);
    }

    double getWallTime() {
        struct timeval time;
        if (gettimeofday(&time, NULL)) {
            return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec*0.000001;
    }

    void publishMessages() {

        if (manual_deadman_on and ((getWallTime() - timeout_start) < timeout)) {
            // Send no command
        }
        else if (autonomous_deadman_on and ((getWallTime() - timeout_start) < timeout)) {
            // Send desired steering angle through publisher
            steering_angle_msg.data = steering_angle;
            steering_angle_pub.publish(steering_angle_msg);

            // Send desired velocity through publisher
            velocity_msg.data = movement_velocity;
            velocity_pub.publish(velocity_msg);
        }
        else {
            // Joystick has timed out, send 0 velocity command
            // Do not send a steering angle command, so it remains where it is currently at.
            std_msgs::Float64 velocity_msg;
            velocity_msg.data = 0.0;
            velocity_pub.publish(velocity_msg);
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "approach_controller");
    ApproachController approach_controller = ApproachController();
    approach_controller.spin();

    return 0;
}