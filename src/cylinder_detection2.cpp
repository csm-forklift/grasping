/* This code reads in point cloud data, compresses it along the Z axis into the
 * X,Y plane, then performs the circle version of the Hough transform to detect
 * circles from the points.
 *
 * Frames
 ****Draw frames here for reference***
 * image
 * o-->x-------------------------
 * |
 * y
 * |
 * |            x
 * |            |
 * ---------y<--o----------------
 *          target
 *
 * Note: the image frame has 'x' going to the right and 'y' going down. For the
 * matrices, the rows go down and the columns go right. So the rows represent
 * the 'y' dimension and the cols represent the 'x' dimension.
 */

// TODO: Ways to improve this code
/*
1) Instead of using the opencv image matrix to store where the points are, you could save them in a vector storing the (x,y) index of each point, then just iterate through the vector instead of the entire image. In other words, use a sparse matrix instead of dense. *** See if you can use Sparse Matrices for your images and if that makes it any faster ***
2) Add in the ability to check for circle of different radii. That way you could add a tolerance on the radius or you could search for multiple circle types.
3) Finish function that finds the local maxima to determine which points should be considered potentials (this should help to avoid placing cylinders next to walls, since the there should be many local maxima near a wall).
4) Filter out planes using PCL segmentation
5) Convert the 'points' vector into 'target' frame rather than leaving it in image frame whenever you find them. Then pass the target frame version to all of the filters.
 */

#include <iostream>
#include <string>
#include <cylinder_detection/cylinder_detection.hpp>


// Define the pointtype used with PointCloud data
typedef pcl::PointXYZ PointT;

// // DEBUG: Visualizers for debugging
// pcl::visualization::PCLVisualizer viewer("Debugging");

CylinderDetector::CylinderDetector() :
nh_("~"),
scene_cloud_optical_frame(new pcl::PointCloud<PointT>),
scene_cloud_unfiltered(new pcl::PointCloud<PointT>),
scene_cloud_right_adjustment_frame(new pcl::PointCloud<PointT>),
scene_cloud_left_adjustment_frame(new pcl::PointCloud<PointT>),
scene_cloud_z_filter_frame(new pcl::PointCloud<PointT>),
scene_cloud(new pcl::PointCloud<PointT>)
{
    //===== Load Parameters =====//
    nh_.param<std::string>("sensor", sensor_name, "sensor");
    nh_.param<std::string>("sensor_frame", sensor_frame, sensor_name + "_link");
    nh_.param<std::string>("point_cloud_topic", point_topic, "/"+sensor_name+"/depth/points");
    nh_.param<std::string>("target_frame", target_frame, sensor_frame);
    nh_.param<double>("target_x", target_point.x, 1.0);
    nh_.param<double>("target_y", target_point.y, 0.0);
    nh_.param<double>("/roll/radius", circle_radius, 0.200);
    nh_.param<double>("target_tolerance", target_tolerance, 2*circle_radius);
    nh_.param<double>("sensor_threshold", sensor_threshold, 1.0);
    nh_.param<bool>("debug", debug, false);

    //===== ROS Objects =====//
    cyl_pub = nh_.advertise<geometry_msgs::PointStamped>("point", 1);
    marker_pub = nh_.advertise<visualization_msgs::MarkerArray>("markers", 1);
    if (debug) {
        pc_debug_pub = nh_.advertise<sensor_msgs::PointCloud2>("filtered_points", 1);
    }
    //sensor_frame.insert(0, "/"); // sensor is assumed to be level with the ground, this frame must one where Z is up
    //target_frame.insert(0, "/"); // this frame should have the Z axis pointing upward
    ROS_INFO("Reading depth points from: %s", point_topic.c_str());
    ROS_INFO("Transforming cloud to '%s' frame", sensor_frame.c_str());
    pc_sub = nh_.subscribe(point_topic.c_str(), 1, &CylinderDetector::pcCallback, this);
    control_mode_sub = nh_.subscribe("/control_mode", 1, &CylinderDetector::controlModeCallback, this);
    roll_pose_sub = nh_.subscribe("/roll/pose", 1, &CylinderDetector::rollPoseCallback, this);


    //===== Print out possible values for control mode =====//
    // Pushback more numbers to allow this controller to operate in more
    // modes
    control_mode = 0; // start off with no controller operating
    available_control_modes.push_back(3);
    available_control_modes.push_back(4);
    std::string message = "Available control_modes for [" + ros::this_node::getName() + "]: ";
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

    //===== Tuning Parameters =====//
    // These parameters are all based on the sensor link frame (where Z axis is up)
    max_distance_x = 6; // m
    min_distance_x = 0; // m
    max_distance_y = 5; // m
    min_distance_y = -5; // m
    max_distance_z = 10; // m
    min_distance_z = -10; // m
    max_fixed_x = 6.0; // m
    min_fixed_x = 0.5; // m
    max_fixed_y = 2.5; // m
    min_fixed_y = -2.5; // m

    // Set Filter Parameters
    sigma_squared = 1.0;
    intervalTimeCount = 0.0;
    intervalStartTime = getWallTime();

    // Filter pointcloud height
    filter_z_low = -0.100; // m
    filter_z_high = 0.1500; // m
    resolution = 256.0; // pixels/m, 256 approx. = 1280 pixels / 5 m
    rotation_resolution = 0.01; // radians/section
    num_potentials = 5; // number of maximums to check in accumulator
    // Default minimum is number of pixels making 1/5 of a circle with a single pixel line
    threshold_low = (1.0/5)*(2*round(circle_radius*resolution) - 1);
    // Default maximum is number of pixels making half of a circle with 2 layers of pixels
    threshold_high = 2*(2*round(circle_radius*resolution) - 1);

    // Variance filter parameters
    bound_offset = 0.25*(2*circle_radius); // amount to add and subtract to radius to get the upper and lower bounds for accepting points to use in calculating the variance
    upper_radius = circle_radius + bound_offset;
    lower_radius = circle_radius - bound_offset;
    //============================//

    //===== Filtering Options =====//
    use_threshold_filter = true;
    check_center_points = true;
    use_variance_filter = true;
    use_location_filter = true;
    //=============================//

    // Get fixed rotations and translations for the point cloud transforms
    // Degrees are first variable of thetas. modify for desired angles of view off of center line
    theta_min = 41.0 * (M_PI/180.0); // convert degrees to radians
    theta_max = 36.0 * (M_PI/180.0); // convert degrees to radians
    phi_min = (M_PI/2) - theta_min;
    phi_max = (M_PI/2) - theta_max;

    translation.x() = 0.0;
    translation.y() = 0.0;
    translation.z() = 0.0;

    min_angle_rotation = Eigen::AngleAxisf(-1*phi_min, Eigen::Vector3f::UnitZ());

    max_angle_rotation = Eigen::AngleAxisf(-1*phi_max, Eigen::Vector3f::UnitZ());

    delta_angle_rotation = Eigen::AngleAxisf((phi_min+phi_max), Eigen::Vector3f::UnitZ());

}

void CylinderDetector::controlModeCallback(const std_msgs::Int8 &msg)
{
    control_mode = msg.data;
}

void CylinderDetector::rollPoseCallback(const geometry_msgs::PoseStamped &msg)
{
    // Update the roll's target point
    target_point.x = msg.pose.position.x;
    target_point.y = msg.pose.position.y;
}

void CylinderDetector::pcCallback(const sensor_msgs::PointCloud2 &msg)
{
    if (checkControlMode(control_mode, available_control_modes)) {
        //===== Convert PointCloud and Transform Data =====//
        // Convert ROS PointCloud2 message into PCL pointcloud
        //rosMsgToPCL(msg, scene_cloud_optical_frame);
        // Transform pointcloud into sensor frame
        //transformPointCloud(scene_cloud_optical_frame, scene_cloud_unfiltered, msg.header.frame_id, sensor_frame);

        rosMsgToPCL(msg, scene_cloud_unfiltered);

        // Make sure the pointcloud has points
        if (scene_cloud_unfiltered->size() == 0) {
            return;
        }

        //===== Preprocessing (filter, segment, etc.) =====//
        // Try to remove the ground layer (find the minimum z level and remove a few centimeters up)
        pcl::PassThrough<PointT> pass; // passthrough filter
        pass.setInputCloud(scene_cloud_unfiltered);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(filter_z_low, filter_z_high);

        pass.filter(*scene_cloud_z_filter_frame);

        // Make sure the pointcloud has points
        if (scene_cloud_z_filter_frame->size() == 0) {
            return;
        }

        pcl::transformPointCloud(*scene_cloud_z_filter_frame, *scene_cloud_right_adjustment_frame, translation, min_angle_rotation);

        // Filter all points to the right of minimum view angle
        pass.setInputCloud(scene_cloud_right_adjustment_frame);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(0.0, 120.0);

        pass.filter(*scene_cloud_right_adjustment_frame);

        // Make sure the pointcloud has points
        if (scene_cloud_right_adjustment_frame->size() == 0) {
            return;
        }

        pcl::transformPointCloud(*scene_cloud_right_adjustment_frame, *scene_cloud_left_adjustment_frame, translation, delta_angle_rotation);

        // Filter all points to the left of minimum view angle
        pass.setInputCloud(scene_cloud_left_adjustment_frame);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(0.0, 120.0);

        pass.filter(*scene_cloud_left_adjustment_frame);

        // Make sure the pointcloud has points
        if (scene_cloud_left_adjustment_frame->size() == 0) {
            return;
        }

        pcl::transformPointCloud(*scene_cloud_left_adjustment_frame, *scene_cloud, translation, max_angle_rotation);

        // Unnecessary filter required to apply transform????
        pass.setInputCloud(scene_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-120.0, 120.0);

        pass.filter(*scene_cloud);

        // If final pointcloud contains no points, exit the callback
        if (scene_cloud->size() == 0) {
            return;
        }

        // Get the bounds of the point cloud
        pcl::getMinMax3D(*scene_cloud, min_pt, max_pt);
        x_max = max_pt.x;
        x_min = min_pt.x;
        y_max = max_pt.y;
        y_min = min_pt.y;
        z_max = max_pt.z;
        z_min = min_pt.z;

        // DEBUG: Publish filtered pointcloud
        if (debug) {
            sensor_msgs::PointCloud2 pc_msg;
            pclToROSMsg(scene_cloud, pc_msg);
            pc_debug_pub.publish(pc_msg);
        }
        //=================================================//

        //===== Generate 2D Image =====//
        // Determine the grid size based on the desired resolution
        x_range = x_max - x_min;
        y_range = y_max - y_min;

        if (x_range == 0 && y_range == 0) {
            // Only the center point was received
            return;
        }

        if (!std::isfinite(x_range)) {
            x_range = 0;
        }
        if (!std::isfinite(y_range)) {
            y_range = 0;
        }

        // Remember that transforming from sensor frame to the image frame mirrors the axes, so x pixels depend on the y range and vis-versa
        x_pixels = round(y_range * resolution);
        y_pixels = round(x_range * resolution);
        x_pixel_delta = (y_range / x_pixels); // length of each pixel in image frame 'x' direction
        y_pixel_delta = (x_range / y_pixels); // length of each pixel in image frame 'y' direction

        // Calculate mirrored points for y-values in sensor frame (x direction in image frame)
        y_mirror_min = -y_max;
        y_mirror_max = -y_min;

        // Convert points into 2D image and display
        top_image = cv::Mat::zeros(y_pixels, x_pixels, CV_8U); // first index is rows which represent the y dimension of the image

        for (int i = 0; i < scene_cloud->size(); ++i) {
            // Transform sensor points to image indices
            int x_index;
            int y_index;
            sensorToImage(scene_cloud->points[i].x, scene_cloud->points[i].y, x_index, y_index);

            // Check that the values are within bounds
            if (x_index >= 0 && x_index <= (x_pixels - 1) && y_index >= 0 && y_index <= (y_pixels - 1)) {
                top_image.at<uint8_t>(cv::Point(x_index, y_index)) = 255; // make sure the type matches with the matrix value type, CV_*U = uint8_t
            }
        }

        //----- Find contours for blob shapes
        // First 'close' the image to fill in thin lines
        cv::Mat structure_element;
        structure_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15,15));
        cv::Mat top_image_close;
        cv::morphologyEx(top_image, top_image_close, cv::MORPH_CLOSE, structure_element);
        std::vector< std::vector<cv::Point> > contours; // stores the vectors of points making up the contours
        std::vector<cv::Vec4i> hierarchy; // vector storing vectors which represent the connection between contours ([i][0] = next, [i][1] = previous, [i][2] = first child, [i][3] = parent)
        findContours(top_image_close, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        cv::Mat top_image_contours = cv::Mat::zeros(top_image_close.size(), CV_8U);
        for (int i = 0; i < contours.size(); ++i) {
            drawContours(top_image_contours, contours, i, cv::Scalar(255, 255, 255), 1, 8, hierarchy, 0);
        }

        //----- Create an image fixed in place based on 'max/min_fixed_x' and 'max/min_fixed_y', for stable visualization
        int x_offset_pixels;
        int y_offset_pixels;
        //cv::Mat top_image_fixed = generateFixedImage(top_image, x_offset_pixels, y_offset_pixels);
        //=============================//

        //===== Circle Hough Transform =====//
        // Create an accumulator matrix that adds padding equal to the radius. This will allow the circle center to lie outside the actual image boundaries
        radius_pixels = round(circle_radius*resolution);
        accum_x_pixels = x_pixels + 2*radius_pixels;
        accum_y_pixels = y_pixels + 2*radius_pixels;
        cv::Mat accumulator = cv::Mat::zeros(accum_y_pixels, accum_x_pixels, CV_16U);

        //----- Hough Transform Iteration
        generateAccumulatorUsingImage(top_image_contours, accumulator);
        // generateAccumulatorUsingImage_OldMethod(top_image_contours, accumulator);
        // DEBUG: get the top cylinder position for drawing a circle on the image for debuggin
        // Scale accumulator matrix for better visualization
        double accum_max;
        double accum_min;
        cv::Point max_loc;
        cv::Point min_loc;
        cv::minMaxLoc(accumulator, &accum_min, &accum_max, &min_loc, &max_loc);
        cv::Mat accumulator_scaled = accumulator*(USHRT_MAX/accum_max); // if you change accumulator type you will need to change the max value
        // Get the maximum point of accumulator (center of the circle)
        int circle_index_x;
        int circle_index_y;
        accumulatorToImage(max_loc.x, max_loc.y, circle_index_x, circle_index_y);

        // Generate mask for removing max
        potentials.clear();
        findPointsDeletion(potentials, accumulator_scaled);

        // Filter using a accumulator threshold values
        if (use_threshold_filter) {
            thresholdFilter(potentials, accumulator);
        }

        // Check if the potential points could be valid cylinder points
        if (check_center_points) {
            checkCenterPoints(potentials, top_image);
        }

        // Filter potentials based on variance of points around circle
        std::vector<double> variances(potentials.size(), 0);
        if (use_variance_filter) {
            varianceFilter(potentials, variances, top_image);
        }

        // Find the point closest to the desired target
        // targetFilter(): deletes points outside 'target_tolerance' and returns the points sorted by distance from target
        // closestToTarget(): will only return the single closest point within range (otherwise no points returned)
        if (use_location_filter) {
            closestToTarget(potentials, target_point);
            //targetFilter(potentials, target_point);
        }
        //cout<<"Number of Points after filtering: " <<potentials.size() <<'\n';
        //cout << '\n';

        // If there are no potentials remaining, break out of the function
        if (potentials.size() == 0) {
            return;
        }


        //================================================================//
        // Mahalanobis Distance Filter with Bayesian Estimator
        //================================================================//
        float sensor_frame_x;
        float sensor_frame_y;
        double target_frame_x;
        double target_frame_y;

        double mean_x = 0.0;
        double mean_y = 0.0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double point_count = 5;

        std::vector<double> mahalanobis_vector;

        for (int i = 0; i < potentials.size(); ++i) {
            // Convert from image pixels(potentials) to meters(sensor_frame)
            imageToSensor(potentials.at(i).x, potentials.at(i).y, sensor_frame_x, sensor_frame_y);

            // Convert the sensor frame point into the target frame
            sensorToTarget(sensor_frame_x, sensor_frame_y, target_frame_x, target_frame_y);

            currentWallTime = getWallTime();

            // Calculate the mahalanobis distance using previous sigmas and target point values
            mahalanobisDistance = sqrt(pow(target_frame_x-target_point.x,2)/sigma_squared_x + pow(target_frame_y-target_point.y,2)/sigma_squared_y);

            mahalanobis_vector.push_back(mahalanobisDistance);
        }

        // Get index of the smallest mahalanobis distance
        int min_distance_index = 0;
        for (int i = 1; i < mahalanobis_vector.size(); ++i) {
            if (mahalanobis_vector.at(i) < mahalanobis_vector.at(min_distance_index)) {
                min_distance_index = i;
            }
        }

        // If the forklift clamp shortarm is within a certain distance from the target point, make the mahalanobis distance threshold very small to keep it from updating the point once the roll cannot be seen due to the lidar cutoff.
        tf::StampedTransform transform;
        // Get transform from sensor to target frame
        tf_listener.waitForTransform(target_frame.c_str(), "clamp_short_arm", ros::Time::now(), ros::Duration(0.051));
        try {
            tf_listener.lookupTransform(target_frame.c_str(), "/clamp_short_arm", ros::Time(0), transform);
        }
        catch(tf::TransformException& ex) {
            ROS_ERROR("Target Transform Exception: %s", ex.what());
        }
        double clamp_x = transform.getOrigin().x();
        double clamp_y = transform.getOrigin().y();
        // Get distance to target point
        double clamp_to_target = sqrt(pow(clamp_x - target_point.x, 2) + pow(clamp_y - target_point.y, 2));
        if (clamp_to_target < sensor_threshold) {
            // Make it a really small number so only very close values are used to update the target.
            mahalanobisDistanceThreshold = 0.01;
        }

        if (mahalanobis_vector.at(min_distance_index) < mahalanobisDistanceThreshold) {
            // Reset mahalanobis distance threshold
            mahalanobisDistanceThreshold = 3;

            /*
            // If a point is valid we need to update the measurement sum and point variance before we do bayesian update
            // this is becuase the first update is the frequentist approach, however, we need to then use this to do "memory update"
            num_of_points_in_filter++;
            */
            // Convert from image pixels(potentials) to meters(sensor_frame)
            imageToSensor(potentials.at(min_distance_index).x, potentials.at(min_distance_index).y, sensor_frame_x, sensor_frame_y);

            // Convert the sensor frame point into the target frame
            sensorToTarget(sensor_frame_x, sensor_frame_y, target_frame_x, target_frame_y);

            if ((prior_points_x.size() < point_count) && (prior_points_y.size() < point_count)) {
                num_of_points_in_filter++;
            } else {
                prior_points_x.erase(prior_points_x.begin());
                prior_points_y.erase(prior_points_y.begin());
            }

            prior_points_x.push_back(target_frame_x);
            prior_points_y.push_back(target_frame_y);
            double sum_x = target_point.x;
            double sum_y = target_point.y;
            for(int k=0; k<prior_points_x.size();k++){
                sum_x += prior_points_x[k];
                sum_y += prior_points_y[k];
            }
            double mean_x = sum_x/num_of_points_in_filter;
            double mean_y = sum_y/num_of_points_in_filter;
            double point_var_x = 0;
            double point_var_y = 0;
            for(int k =0; k<prior_points_x.size(); k++){
                point_var_x += (prior_points_x[k]-mean_x)*(prior_points_x[k]-mean_x);
                point_var_y += (prior_points_y[k]-mean_y)*(prior_points_y[k]-mean_y);
            }
            point_var_x /= num_of_points_in_filter;
            point_var_y /= num_of_points_in_filter;

            sigma_squared_x = 1;
            sigma_squared_y = 1;
            target_point.x = (sigma_squared_x*mean_x + point_var_x*target_point.x) / (sigma_squared_x+point_var_x);
            //sigma_squared_x = (point_var_x*sigma_squared_x) / (point_var_x + sigma_squared_x);

            // Calculate new target point y value
            target_point.y = (sigma_squared_y*mean_y + point_var_y*target_point.y) / (sigma_squared_y+point_var_y);
            //sigma_squared_y = (point_var_y*sigma_squared_y) / (point_var_y+ sigma_squared_y);

            intervalStartTime = getWallTime();
            interval_count++;

            geometry_msgs::PointStamped cylinder_point;
            cylinder_point.header = msg.header;
            cylinder_point.header.frame_id = target_frame.c_str();
            cylinder_point.point.x = target_point.x;
            cylinder_point.point.y = target_point.y;
            cylinder_point.point.z = 0;
            cyl_pub.publish(cylinder_point);

            visualization_msgs::MarkerArray cyl_markers;
            // Delete previous markers
            visualization_msgs::Marker delete_markers;
            delete_markers.action = visualization_msgs::Marker::DELETEALL;
            cyl_markers.markers.push_back(delete_markers);
            // DEBUG: Show cylinder marker
            // imageToSensor(target_point.x, target_point.y, sensor_frame_x, sensor_frame_y);
            visualization_msgs::Marker cyl_marker;
            cyl_marker.header = msg.header;
            cyl_marker.header.frame_id = target_frame.c_str();
            cyl_marker.id = 1;
            cyl_marker.type = visualization_msgs::Marker::CYLINDER;
            cyl_marker.pose.position.x = target_point.x;
            cyl_marker.pose.position.y = target_point.y;
            cyl_marker.pose.position.z = 0;
            cyl_marker.pose.orientation.x = 0;
            cyl_marker.pose.orientation.y = 0;
            cyl_marker.pose.orientation.z = 0;
            cyl_marker.pose.orientation.w = 1.0;
            cyl_marker.scale.x = 0.75*(2*circle_radius);
            cyl_marker.scale.y = 0.75*(2*circle_radius);
            cyl_marker.scale.z = 1.0;
            cyl_marker.color.a = 1.0;
            cyl_marker.color.r = 1.0;
            cyl_marker.color.g = 1.0;
            cyl_marker.color.b = 1.0;
            cyl_marker.lifetime = ros::Duration(1/100);
            cyl_markers.markers.push_back(cyl_marker);
            marker_pub.publish(cyl_markers);
        }
        else {
            mahalanobisDistanceThreshold *= 2;
        }
    }
}

void CylinderDetector::rosMsgToPCL(const sensor_msgs::PointCloud2& msg, pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
}

void CylinderDetector::pclToROSMsg(pcl::PointCloud<PointT>::Ptr cloud, sensor_msgs::PointCloud2& msg) {
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(*cloud, pcl_pc2);
    pcl_conversions::fromPCL(pcl_pc2, msg);
}

double CylinderDetector::getWallTime() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec*0.000001;
}

bool CylinderDetector::checkControlMode(int mode, std::vector<int> vector_of_modes)
{
    // Use 'find' on the vector to determine existence of 'mode'
    std::vector<int>::iterator it;
    it = std::find(vector_of_modes.begin(), vector_of_modes.end(), mode);
    if (it != vector_of_modes.end()) {
        return true;
    }
    else {
        return false;
    }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "cylinder_detection");
    CylinderDetector detector;
    ros::spin();

    return 0;
}
