/*
 * Declares all functions for the cylinder detection class.
 */

#ifndef CYLINDER_DETECTION_H
#define CYLINDER_DETECTION_H

#include <iostream>
#include <string>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>


// Define the pointtype used with PointCloud data
typedef pcl::PointXYZ PointT;

// // DEBUG: Visualizers for debugging
// pcl::visualization::PCLVisualizer viewer("Debugging");

class CylinderDetector
{
private:
    //===== ROS Objects
    ros::NodeHandle nh_;
    ros::Subscriber pc_sub; // subscribes to pointcloud data
    ros::Subscriber control_mode_sub; // reads the current control mode set by the master controller
    ros::Subscriber roll_pose_sub; // reads the rolls new target position
    ros::Publisher cyl_pub; // publish the position of the cylinder as a pose
    ros::Publisher marker_pub; // publish cylinder marker
    ros::Publisher pc_debug_pub; // publish filtered pointcloud for debugging
    tf::TransformListener tf_listener;
    std::string sensor_name;
    std::string sensor_frame;
    std::string point_topic;
    std::string target_frame;
    std::string right_rotation_frame;
    std::string left_rotation_frame;
    int control_mode; // this node publishes only when this variable is a specific value
    std::vector<int> available_control_modes; // vector of possible numbers the 'control_mode' can be to turn this node on
    bool debug; // turns on some debugging features

    //===== Tuning Paramters
    float max_distance_x; // m
    float min_distance_x; // m
    float max_distance_y; // m
    float min_distance_y; // m
    float max_distance_z; // m
    float min_distance_z; // m
    float max_fixed_x; // m
    float min_fixed_x; // m
    float max_fixed_y; // m
    float min_fixed_y; // m
    float filter_z_low; // m
    float filter_z_high; // m
    float theta_min; // r
    float theta_max; // r
    float phi_min; // r
    float phi_max; // r
    double resolution; // pixels/m, 256 approx. = 1280 pixels / 5 m
    double rotation_resolution; // radians/section
    double circle_radius; // m
    int num_potentials; // number of potential points to check for selecting if/where a cylinder is present
    int threshold_low; // lower cutoff value for accepting a center point from the accumulator
    int threshold_high; // upper cutoff value for accepting a center point from the accumulator
    double variance_tolerance; // upper threshold for the variance
    double target_tolerance; // maximum distance from target to be accepted as a viable cylinder location, in meters
    double bound_offset; // determines the radial bound inside which the circle points will be considered for calculating the variance
    double upper_radius; // upper radius calculated using the bound_offset
    double lower_radius; // lower radius calculated using the bound offset

    //===== Filtering Options
    bool use_threshold_filter; // set to true to perform filtering using the accumulator low and high thresholds
    bool check_center_points; // set to true to remove potential locations which contain points within the cylinder surface
    bool use_variance_filter; // filters points based off the variance calculated using points within the vicinity of the expected circle
    bool use_location_filter; // Rejects all points which do not lie near the desired goal location (this should return only one point at most)
    int num_of_points_in_filter = 1; // For memory update points, need to start at 1 since we have a temp target point
    int interval_count = 1;
    std::vector<double> prior_points_x; // stores the x position read in from cylinder detection that is within the Mahalanobis distance threshold
    std::vector<double> prior_points_y; // stores the x position read in from cylinder detection that is within the Mahalanobis distance threshold
    double sigma_squared_x = 1, sigma_squared_y = 1;

    //===== Range and Resolution Data
    PointT min_pt; // minimum point in pointcloud
    PointT max_pt; // maximum point in pointcloud
    float x_max;
    float x_min;
    float y_max;
    float y_min;
    float z_max;
    float z_min;
    double x_range; // range for x data
    double y_range; // range for y data
    double sigma_squared;
    double mahalanobisDistance;
    double mahalanobisDistanceThreshold = 3;
    double sensor_threshold; // when the clamp to target distance is within this value, the mahalanobis distance threshold becomes very small
    double intervalStartTime;
    double intervalTimeCount;
    double currentWallTime;
    int x_pixels; // x resolution for image
    int y_pixels; // y resolution for image
    float x_pixel_delta; // length of each pixel in image frame 'x' direction
    float y_pixel_delta; // length of each pixel in image frame 'y' direction
    float y_mirror_min; // y minimum in sensor frame mirrored about x axis
    float y_mirror_max; // y maximum in sensor frame mirrored about x axis
    int radius_pixels; // number of pixels in the circle radius (based on resolution)
    int accum_x_pixels; // number of pixels in x dimension of accumulator matrix
    int accum_y_pixels; // number of pixels in y dimension of accumulator matrix

    //===== PCL Objects
    pcl::PointCloud<PointT>::Ptr scene_cloud_optical_frame;
    pcl::PointCloud<PointT>::Ptr scene_cloud_unfiltered;
    pcl::PointCloud<PointT>::Ptr scene_cloud;
    pcl::PointCloud<PointT>::Ptr scene_cloud_right_adjustment_frame;
    pcl::PointCloud<PointT>::Ptr scene_cloud_left_adjustment_frame;
    pcl::PointCloud<PointT>::Ptr scene_cloud_z_filter_frame;
    Eigen::Vector3f translation;
    Eigen::Quaternionf min_angle_rotation,max_angle_rotation, delta_angle_rotation;
    Eigen::Affine3f affine_transform;

    //===== OpenCV Objects
    cv::Mat top_image;
    //cv::Mat top_image_fixed; // For Debugging purposes
    //cv::Mat top_image_rgb; // For Debugging purposes
    cv::Mat accumulator;
    std::vector<cv::Point> potentials;
    cv::Point2d target_point;

    // nh_("~"),
    // scene_cloud_optical_frame(new pcl::PointCloud<PointT>),
    // scene_cloud_unfiltered(new pcl::PointCloud<PointT>),
    // scene_cloud_right_adjustment_frame(new pcl::PointCloud<PointT>),
    // scene_cloud_left_adjustment_frame(new pcl::PointCloud<PointT>),
    // scene_cloud_z_filter_frame(new pcl::PointCloud<PointT>),
    // scene_cloud(new pcl::PointCloud<PointT>)

public:
    CylinderDetector();

    void controlModeCallback(const std_msgs::Int8 &msg);

    void rollPoseCallback(const geometry_msgs::PoseStamped &msg);

    void pcCallback(const sensor_msgs::PointCloud2 &msg);

    void rosMsgToPCL(const sensor_msgs::PointCloud2& msg, pcl::PointCloud<PointT>::Ptr cloud);

    void pclToROSMsg(pcl::PointCloud<PointT>::Ptr cloud, sensor_msgs::PointCloud2& msg);

    cv::Mat generateFixedImage(cv::Mat &image, int &x_offset_pixels, int &y_offset_pixels);

    void midpointCircleAlgorithm(std::vector<cv::Point2i> &points, int radius, int x_center = 0, int y_center = 0);

    void trigCircleGeneration(std::vector<cv::Point2i> &points, int radius, double resolution, int x_center = 0, int y_center = 0);

    void generateAccumulatorUsingImage(cv::Mat &image, cv::Mat &accumulator);

    // FIXME: currently left in here to test it's speed against the new methods
    void generateAccumulatorUsingImage_OldMethod(cv::Mat &top_image_accum, cv::Mat accumulator);

    void findPointsDeletion(std::vector<cv::Point> &points, cv::Mat &accumulator);

    void findPointsLocalMax();

    void thresholdFilter(std::vector<cv::Point> &points, cv::Mat &accumulator);

    void checkCenterPoints(std::vector<cv::Point> &points, cv::Mat &top_image);

    void varianceFilter(std::vector<cv::Point> &points, std::vector<double> &variances, cv::Mat &top_image);

    double calculateRadius(double x, double y, double x_c, double y_c);

    void closestToTarget(std::vector<cv::Point> &points, cv::Point target);

    void targetFilter(std::vector<cv::Point> &points, cv::Point target);

    void sensorToImage(float sensor_x_in, float sensor_y_in, int& image_x_out, int& image_y_out);

    void imageToSensor(int image_x_in, int image_y_in, float& sensor_x_out, float& sensor_y_out);

    // Overloaded function to handle 'double' inputs
    void sensorToImage(double& sensor_x_in, double& sensor_y_in, int& image_x_out, int& image_y_out);

    // Overloaded function to handle 'double' inputs
    void imageToSensor(int image_x_in, int image_y_in, double& sensor_x_out, double& sensor_y_out);

    void imageToAccumulator(int image_x_in, int image_y_in, int& accum_x_out, int& accum_y_out);

    void accumulatorToImage(int accum_x_in, int accum_y_in, int& image_x_out, int& image_y_out);

    void sensorToTarget(double sensor_x_in, double sensor_y_in, double& target_x_out, double& target_y_out);

    void targetToSensor(double target_x_in, double target_y_in, double& sensor_x_out, double& sensor_y_out);

    double getWallTime();

    bool checkControlMode(int mode, std::vector<int> vector_of_modes);
};

#endif // CYLINDER_DETECTION_H
