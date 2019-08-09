#include <math.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm> // min, max, find
#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int8.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <sys/time.h>
#include <sensor_msgs/Joy.h>

using namespace std;


class ApproachController
{
private:
    //===== ROS Objects =====//
    // Publishers and Subscribers
    ros::NodeHandle nh_;
    ros::Subscriber control_mode_sub; // reads the current controller mode
    ros::Subscriber stretch_sensor_sub; // reads boolean indicating whether the stretch sensor is over the threshold or not
    ros::Subscriber switch_plate_sub; // gets click status of limit switch on grasping plate
    ros::Subscriber odom_sub; // read the forklifts current pose from the odom message
    ros::Subscriber roll_sub; // read paper roll's current pose
    ros::Subscriber grasp_success_sub; // reads whether the grasp was sucessful or not
    ros::Subscriber grasp_finished_sub; // reads whether the entire grasp procedure is finished
    ros::Subscriber steering_angle_sub;
    ros::Subscriber joystick_override_sub;
    ros::Publisher velocity_pub; // publish linear velocity command
    ros::Publisher gear_pub; // publish the desired gear (1 = forward, 0 = neutral, -1 = reverse)
    ros::Publisher steering_angle_pub; // publish steering angle command
    ros::Rate rate; // publishing rate
    std_msgs::Float64 steering_angle_msg;
    std_msgs::Float64 velocity_msg;
    std_msgs::Int8 gear_msg;

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
    bool roll_seen; // turns to 'true' once a message has been received from cylinder detection
    double roll_radius;
    double approach_distance; // distance from the roll to the forklift's target point
    double approach_tolerance; // stops linear motion if distance is less than this value, can be positive or negative
    double backout_distance; // distance from the roll the forklift backs out to when the grasp is unsuccessful and a retry is necessary

    // Current forklift pose
    double forklift_x;
    double forklift_y;
    double forklift_heading;

    // Stetch sensor input
    bool plate_contact;

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
    int gear; // current gear direction of forklift: 1 = forward, 0 = neutral, -1 = reverse

    // Steering angle command of the back wheels, positive turns wheels to the left giving a right-hand turn and vis-versa
    double steering_angle, steering_angle_min, steering_angle_max, current_steering_angle;

    // Operational State
    int operation_mode; // 0 = get to starting angle, 1 = approach roll, 2 = back out for retry
    int control_mode; // determines whether this controller is active or not
    vector<int> available_control_modes; // a vector of integers representing which values 'control_mode' can be and allow this controller to operate

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
        roll_seen = false; // wait for this value to become true before sending commands
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
        control_mode_sub = nh_.subscribe("/control_mode", 1, &ApproachController::controlModeCallback, this);
        stretch_sensor_sub = nh_.subscribe("/clamp_control/clamp_plate_status", 1, &ApproachController::plateCheckCallback, this);
        odom_sub = nh_.subscribe("/odom", 1, &ApproachController::odomCallback, this);
        roll_sub = nh_.subscribe("point", 1, &ApproachController::rollCallback, this);
        grasp_success_sub = nh_.subscribe("/clamp_control/grasp_status", 1, &ApproachController::graspSuccessfulCallback, this);
        grasp_finished_sub = nh_.subscribe("/clamp_control/grasp_finished", 1, &ApproachController::graspFinishedCallback, this);
        steering_angle_sub = nh_.subscribe("/steering_node/filtered_angle", 1, &ApproachController::steeringAngleCallback, this);
        joystick_override_sub = nh_.subscribe("/joy", 1, &ApproachController::joy_override, this);

        velocity_pub = nh_.advertise<std_msgs::Float64>("/velocity_node/velocity_setpoint", 1);
        gear_pub = nh_.advertise<std_msgs::Int8>("/velocity_node/gear", 1);
        steering_angle_pub = nh_.advertise<std_msgs::Float64>("/steering_node/angle_setpoint", 1);

        // DEBUG:
        forklift_target_pub = nh_.advertise<geometry_msgs::PointStamped>("forklift_target", 1);

        //===== Initialize States =====//
        forklift_x = 0;
        forklift_y = 0;
        forklift_heading = 0;
        plate_contact = false;

        //===== Initialize Controller Variables =====//
        theta_desired = 0;
        linear_velocity = 0.0;
        movement_velocity = 0.0;
        gear = 0;
        steering_angle = 0;
        current_steering_angle = 0.0;

        // Start in "Get to starting angle" mode
        operation_mode = 0;
        control_mode = 0;

        //===== Print out possible values for control mode =====//
        // Pushback more numbers to allow this controller to operate in more
        // modes
        available_control_modes.push_back(3);
        string message = "Available control_modes for [" + ros::this_node::getName() + "]: ";
        for (int i = 0; i < available_control_modes.size(); ++i) {
            char msg_buffer[10]; // increase size if more digits are needed
            sprintf(msg_buffer, "%d", available_control_modes.at(i));
            message += msg_buffer;
            if (i != available_control_modes.size() - 1) {
                message += ", ";
            }
            else {
                message += '\n';
            }
        }
        ROS_INFO("%s", message.c_str());

        // Initialize joystick variables
        timeout_start = getWallTime();
        autonomous_deadman_on = false;
        manual_deadman_on = false;
    }

    void spin()
    {
        while (nh_.ok()) {
            // TODO: add 'if' statements around the publishers to make sure they do not publish if the control_mode is wrong
            if (checkControlMode(control_mode, available_control_modes) and roll_seen) {
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
        gear = 1;

        // Give time for the steering angle to get to position
        // ros::Duration(1.0).sleep();
        while (abs(current_steering_angle-steering_angle) > angle_tolerance && ros::ok()) {
            // Publish the new steering angle and velocity
            cout << "Initial angle - Current steering angle: " << current_steering_angle << ", Steering angle command: " << steering_angle << "\n";

            publishMessages();
            ros::spinOnce();
            rate.sleep();

            if (!checkControlMode(control_mode, available_control_modes)) {
                break;
            }
        }

        // Change operation mode to "approach"
        operation_mode = 1;
    }

    void approach()
    {
        // Control sequence for approaching the roll
        while (plate_contact == false && ros::ok()) {
            // Get pose of base_link on the forklift
            getForkliftPose();

            // Calculate forklift's new target position in the odom frame
            tf::TransformListener listener;
            tf::StampedTransform transform;
            listener.waitForTransform("/odom", "/clamp_short_arm", ros::Time(0), ros::Duration(1.0));
            listener.lookupTransform("/odom", "/clamp_short_arm", ros::Time(0), transform);
            double plate_x = transform.getOrigin().x();
            double plate_y = transform.getOrigin().y();
            forklift_target_x = plate_x + roll_radius*cos(grasp_angle + forklift_heading);
            forklift_target_y = plate_y + roll_radius*sin(grasp_angle + forklift_heading);

            geometry_msgs::PointStamped forklift_target;
            forklift_target.header.frame_id = "/odom";
            forklift_target.point.x = forklift_target_x;
            forklift_target.point.y = forklift_target_y;
            forklift_target_pub.publish(forklift_target);

            // Calculate linear velocity for forklift
            approach_distance = sqrt(pow(forklift_target_y-paperRoll_y,2)+pow(forklift_target_x-paperRoll_x,2));
            //linear_velocity = proportional_control_linear * approach_distance;

            cout << "Roll x: " << paperRoll_x << ", y: " << paperRoll_y << ", Forklift x: " << forklift_target_x << ", y: " << forklift_target_y << "\n";

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

            // 'linear_velocity' is used because it relates to the along track error through the 'cos(theta_error)' term, we don't want the direct aproach_distance because we need this comparison to be allowed to have a negative value to know when the roll as been passed
            if(linear_velocity > approach_tolerance){
                if (cos(theta_error) < 0) {
                    std::cout << "Need to go backwards!\n";
                    // Change gear to reverse

                    steering_angle = -steering_angle;
                    linear_velocity = linear_velocity*5*proportional_control_linear;

                    // Set forklift velocity
                    // Bound movement velocity
                    movement_velocity = max(linear_velocity, -max_velocity);

                    // Velocity setpoint is negative, change gear to reverse
                    gear = -1;
                }
                else {
                    linear_velocity = linear_velocity*proportional_control_linear;

                    // Set forklift velocity
                    // Bound movement velocity
                    movement_velocity = min(linear_velocity, max_velocity);
                    gear = 1;
                }
            }
            else{
                std::cout << "[" << ros::this_node::getName() << "]: approach tolerance reached.\n";
                linear_velocity = 0;

                // Wait 2 seconds for the stretch sensor to register
                ros::Duration(2.0).sleep();

                ros::spinOnce();

                // if (plate_contact == false) {
                //     // If the approach tolerance is reached and the stretch sensor has not been triggered, begin backup procedure
                //     operation_mode = 2;
                // }

                // Set forklift velocity
                // Bound movement velocity
                movement_velocity = min(linear_velocity, max_velocity);
                gear = 0;
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

            // Decrease velocity with steering angle
            movement_velocity = cos(steering_angle)*movement_velocity;

            // Publish new steering angle and velocity
            publishMessages();

            // DEBUG:
            printf("D: %0.03f, Steer: %0.03f, Theta: %0.03f, Angle: %0.03f, error: %0.03f, plate: %d, vel: %0.03e\n", approach_distance, steering_angle, theta_desired, forklift_heading, theta_error, plate_contact, movement_velocity);

            // Update states
            ros::spinOnce();
            ros::Duration(0.05).sleep();

            // Break the loop if the operation mode changes
            if (operation_mode != 1) {
                break;
            }
            if (!checkControlMode(control_mode, available_control_modes)) {
                break;
            }
        }

        // Stop velocity
        movement_velocity = 0;
        gear = 0;
        //velocity_msg.data = movement_velocity;
        //velocity_pub.publish(velocity_msg);

        // Publish new velocity
        publishMessages();
    }

    void backout()
    {
        // Control sequence for backing out after a failed grasp
        // Until desired distance is achieved between the forklift and the roll, send the backout velocity command

        // First wait for the system to reach a steering angle of 0
        steering_angle = 0;
        movement_velocity = 0;
        while (abs(current_steering_angle-steering_angle) > angle_tolerance && ros::ok()) {
            // Publish the new steering angle and velocity
            cout << "Backout - Current steering angle: " << current_steering_angle << ", Steering angle command: " << steering_angle << "\n";

            publishMessages();
            ros::spinOnce();
            rate.sleep();

            if (!checkControlMode(control_mode, available_control_modes)) {
                break;
            }
        }

        while (approach_distance < backout_distance && ros::ok()) {
            // Update approach distance
            approach_distance = sqrt(pow(forklift_target_y-paperRoll_y,2)+pow(forklift_target_x-paperRoll_x,2));

            // Set backup velocity
            movement_velocity = -max_velocity;
            gear = -1;

            // Set backup steering angle to 0
            steering_angle = 0;

            // Publish new steering angle and velocity
            publishMessages();

            ros::spinOnce();

            if (!checkControlMode(control_mode, available_control_modes)) {
                break;
            }
        }

        // Return to approach operation mode
        operation_mode = 1;
    }

    void plateCheckCallback(const std_msgs::Bool &msg)
    {
        // Update stretch sensor state
        plate_contact = msg.data;

        // // DEBUG:
        // std::cout << "[" << ros::this_node::getName() << "]: new plate sensor reading: " << plate_contact << "\n";
    }

    void getForkliftPose(void)
    {
        // Update previous position
        prev_forklift_target_x = forklift_target_x;
        prev_forklift_target_y = forklift_target_y;

        // Update forklift current position and heading
        tf::TransformListener listener;
        tf::StampedTransform transform;
        listener.waitForTransform("/odom", "/base_link", ros::Time(0), ros::Duration(1.0));
        listener.lookupTransform("/odom", "/base_link", ros::Time(0), transform);

        forklift_x = transform.getOrigin().x();
        forklift_y = transform.getOrigin().y();
        tf::Quaternion forklift_q(
            transform.getRotation().x(),
            transform.getRotation().y(),
            transform.getRotation().z(),
            transform.getRotation().w()
        );

        tf::Matrix3x3 m(forklift_q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        forklift_heading = yaw;
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
        tf::TransformListener listener;
        tf::StampedTransform transform;
        listener.waitForTransform("/odom", "/clamp_short_arm", ros::Time(0), ros::Duration(1.0));
        listener.lookupTransform("/odom", "/clamp_short_arm", ros::Time(0), transform);
        double plate_x = transform.getOrigin().x();
        double plate_y = transform.getOrigin().y();
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
        // TODO: check the distance between the currently read point and the forklift pose, if it is less than 1m (roughly the lidar window rejection length) then do not update the roll position in odom frame and just keep using the most recent point before that

        // Update paper roll from cylinder detection
        roll_seen = true;
        paperRoll_x = msg.point.x;
        paperRoll_y = msg.point.y;
    }

    void graspSuccessfulCallback(const std_msgs::Bool &msg)
    {
        // If the grasp is unsuccessful, we need to back out and retry
        if (msg.data == false) {
            operation_mode = 2;
        }

        // If the grasp was sucessful, keep in mode 1 until the grasp is finished
    }

    void graspFinishedCallback(const std_msgs::Bool &msg)
    {
        // If the grasp is finished, return to state 0
        if (msg.data == true) {
            operation_mode = 0;
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

            // Send desired gear
            gear_msg.data = gear;
            gear_pub.publish(gear_msg);

            // Send desired velocity through publisher
            //std::cout << "[" << ros::this_node::getName() << "]: publishing velocity: " << movement_velocity << "\n";
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

    bool checkControlMode(int mode, vector<int> vector_of_modes)
    {
        // Use 'find' on the vector to determine existence of 'mode'
        vector<int>::iterator it;
        it = find(vector_of_modes.begin(), vector_of_modes.end(), mode);
        if (it != vector_of_modes.end()) {
            return true;
        }
        else {
            return false;
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
