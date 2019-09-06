#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <stdlib.h>


// This code is to make a quick publisher for the demo for the cylinder detection demo
int main(int argc, char** argv){

	ros::init(argc,argv,"demo_roll_pub_node");
	ros::NodeHandle nh_;
	ros::Rate rate(0.5);
	double roll_x, roll_y;
	// THIS WHILE LOOP ASSUMES YOU WANT TO CHANGE THE TARGET WITHOUT RERUNNING THE NODE!
	// while(nh_.ok()){

	// These default parameters are assuming you meant it is in the "odom" frame NOT the "map" frame!
	nh_.param<double>("roll_pos_x", roll_x, -5.0);
	nh_.param<double>("roll_pos_y", roll_y, 0.0);

	// This is for POSE STAMPED
	ros::Publisher roll_pub = nh_.advertise<geometry_msgs::PoseStamped>("/roll/pose",1,true);
	geometry_msgs::PoseStamped temp_pose;
	temp_pose.header.frame_id = "odom";
	temp_pose.header.stamp = ros::Time::now();
	temp_pose.header.seq = 1;
	temp_pose.pose.position.x = roll_x;
	temp_pose.pose.position.y = roll_y;
	temp_pose.pose.position.z = 0;
	temp_pose.pose.orientation.x = 0;
	temp_pose.pose.orientation.y = 0;
	temp_pose.pose.orientation.z = 0;
	temp_pose.pose.orientation.w = 1;

	// This is for POSE
	//ros::Publisher roll_pub = nh_.advertise<geometry_msgs::Pose>("/roll/pose",1)

	/*geometry_msgs::Pose temp_pose;
	temp_pose.position.x = roll_x;
	temp_pose.position.y = roll_y;
	temp_pose.position.z = 0;
	temp_pose.orientation.x = 0;
	temp_pose.orientation.y = 0;
	temp_pose.orientation.z = 0;
	temp_pose.orientation.w = 1;
	*/

	// THIS WHILE LOOP IS ASSUMING YOU WILL NOT CHANGE THE TARGET ONCE IT IS SET
	// while(nh_.ok()){
	// 	roll_pub.publish(temp_pose);
	// 	rate.sleep();
	// }

    // Publish a single message
    roll_pub.publish(temp_pose);
    while(nh_.ok()) {
        ros::spinOnce();
        rate.sleep();
    }

	return 0;
}
