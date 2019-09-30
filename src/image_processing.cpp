/*
 * Functions used to perform the image processing for the Hough Transform.
 */


cv::Mat generateFixedImage(cv::Mat &image, int &x_offset_pixels, int &y_offset_pixels)
{
    // Find the fixed range, pixels, and offset
    float x_fixed_range = max_fixed_x - min_fixed_x;
    float y_fixed_range = max_fixed_y - min_fixed_y;
    int x_fixed_pixels = round(y_fixed_range * resolution);
    int y_fixed_pixels = round(x_fixed_range * resolution);
    float x_offset = max_fixed_y - y_max;
    float y_offset = max_fixed_x - x_max;
    x_offset_pixels = round(x_offset * resolution);
    y_offset_pixels = round(y_offset * resolution);

    // Determine whether origin and max point for the relative image are within the fixed image bounds or not.
    // For each point:
    // -1 = relative is less than fixed minimum bounds
    //  0 = relative is within fixed bounds
    //  1 = relative is greater than fixed maxmimum bounds
    int rel_origin_x;
    int rel_origin_y;
    int rel_max_x;
    int rel_max_y;
    // Check location of relative image origin x coordinate
    if (x_offset_pixels < 0) { rel_origin_x = -1; }
    else if (x_offset_pixels < x_fixed_pixels) { rel_origin_x = 0; }
    else { rel_origin_x = 1; }
    // Check location of relative image origin y coordinate
    if (y_offset_pixels < 0) { rel_origin_y = -1; }
    else if (y_offset_pixels < y_fixed_pixels) { rel_origin_y = 0; }
    else { rel_origin_y = 1; }
    // Check location of relative image max x coordinate
    if (x_offset_pixels < -x_pixels) { rel_max_x = -1; }
    else if ((x_offset_pixels + x_pixels) < x_fixed_pixels) { rel_max_x = 0; }
    else { rel_max_x = 1; }
    // Check location of relative image max y coordinate
    if (y_offset_pixels < -y_pixels) { rel_max_y = -1; }
    else if ((y_offset_pixels + y_pixels) < y_fixed_pixels) { rel_max_y = 0; }
    else { rel_max_y = 1; }

    // Check if the relative image is completely outside the fixed image bounds (if it's out in one dimension, the whole image will be out)
    int x_lb_rel, x_ub_rel; // starting/ending x indices for relative image
    int x_lb_fixed, x_ub_fixed; // starting/ending x indices for fixed image
    int y_lb_rel, y_ub_rel; // starting/ending y indices for relative image
    int y_lb_fixed, y_ub_fixed; // starting/ending y indices for fixed image
    bool use_bounds = true; // if false, leaves fixed image blank
    if (rel_origin_x == 1 || rel_origin_y == 1 || rel_max_x == -1 || rel_max_y == -1) {
        use_bounds = false;
    }
    else {
        //----- X coordinate conditions
        // Origin is < bounds
        if (rel_origin_x == -1) {
            x_lb_fixed = 0;
            x_lb_rel = -x_offset_pixels;
        }
        // Origin is within bounds
        else {
            x_lb_fixed = x_offset_pixels;
            x_lb_rel = 0;
        }
        // Max is within bounds
        if (rel_max_x == 0) {
            x_ub_fixed = x_pixels + x_offset_pixels;
            x_ub_rel = x_pixels;
        }
        // Max is > bounds
        else {
            x_ub_fixed = x_fixed_pixels;
            x_ub_rel = x_fixed_pixels - x_offset_pixels;
        }
        //----- Y coordinate conditions
        // Origin is < bounds
        if (rel_origin_y == -1) {
            y_lb_fixed = 0;
            y_lb_rel = -y_offset_pixels;
        }
        // Origin is within bounds
        else {
            y_lb_fixed = y_offset_pixels;
            y_lb_rel = 0;
        }
        // Max is within bounds
        if (rel_max_y == 0) {
            y_ub_fixed = y_pixels + y_offset_pixels;
            y_ub_rel = y_pixels;
        }
        // Max is > bounds
        else {
            y_ub_fixed = y_fixed_pixels;
            y_ub_rel = y_fixed_pixels - y_offset_pixels;
        }
    }

    cv::Mat image_fixed = cv::Mat::zeros(y_fixed_pixels, x_fixed_pixels, CV_8U);

    // Copy points from the relative-size image translated into a fixed frame size image
    if (use_bounds) {
        // Assign ranges for copying image points
        cv::Range x_fixed_indices(x_lb_fixed, x_ub_fixed);
        cv::Range x_rel_indices(x_lb_rel, x_ub_rel);
        cv::Range y_fixed_indices(y_lb_fixed, y_ub_fixed);
        cv::Range y_rel_indices(y_lb_rel, y_ub_rel);

        // Create a pointer to the submatrix of interest
        cv::Mat subrange = image_fixed.colRange(x_fixed_indices).rowRange(y_fixed_indices);
        // Copy values into that submatrix
        image(y_rel_indices, x_rel_indices).copyTo(subrange);
    }

    // // DEBUG: Check indices and offsets
    // std::cout << "===== Matrix Info =====\n";
    // std::cout << "Total points seen: " << scene_cloud->points.size() << "\n";
    // std::cout << "Relative Image: (" << top_image.cols << ", " << top_image.rows << ")\n";
    // std::cout << "width: " << x_pixels << "\n";
    // std::cout << "height: " << y_pixels << "\n";
    // std::cout << "\nFixed Image: (" << top_image_fixed.cols << ", " << top_image_fixed.rows << ")\n";
    // std::cout << "width: " << x_fixed_pixels << "\n";
    // std::cout << "height: " << y_fixed_pixels << "\n";
    // std::cout << "=======================\n";
    // std::cout << std::endl;
    // std::cout << "===== Offset Info =====\n";
    // std::cout << "Relative position: \n";
    // std::cout << "(x,y): (" << x_max << ", " << y_max << ")\n";
    // std::cout << "Fixed position: \n";
    // std::cout << "(x,y): (" << max_fixed_x << ", " << max_fixed_y << ")\n";
    // std::cout << "Offset (pixels): (" << x_offset_pixels << ", " << y_offset_pixels << ")\n";
    // std::cout << "=======================\n";
    // std::cout << std::endl;
    // std::cout << "===== Range Info =====\n";
    // std::cout << "State: \n";
    // std::cout << "rel_origin_x: " << rel_origin_x << "\n";
    // std::cout << "rel_origin_y: " << rel_origin_y << "\n";
    // std::cout << "rel_max_x: " << rel_max_x << "\n";
    // std::cout << "rel_max_y: " << rel_max_y << "\n";
    // std::cout << "Relative: \n";
    // std::cout << "x start: " << x_lb_rel << "\n";
    // std::cout << "x end: " << x_ub_rel << "\n";
    // std::cout << "y start: " << y_lb_rel << "\n";
    // std::cout << "y end: " << y_ub_rel << "\n";
    // std::cout << "Fixed: \n";
    // std::cout << "x start: " << x_lb_fixed << "\n";
    // std::cout << "x end: " << x_ub_fixed << "\n";
    // std::cout << "y start: " << y_lb_fixed << "\n";
    // std::cout << "y end: " << y_ub_fixed << "\n";
    // std::cout << "======================\n";
    // std::cout << std::endl;

    return image_fixed;
}

void generateAccumulatorUsingImage(cv::Mat &image, cv::Mat &accumulator)
{
    // Iterate through each point in the image (the image should be of type 'CV_8U', if not change the template type from 'uint8_t' to whatever type matches the matrix image type), for each point that is not 0 create a circle and increment the pixel values in the accumulator along that circle.
    int num_rows = image.rows;
    int num_cols = image.cols;

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            if (!(image.at<uint8_t>(i,j) == 0)) {
                // Convert circle poinst from image index to accumulator index
                int accum_x;
                int accum_y;
                imageToAccumulator(j, i, accum_x, accum_y);

                // Increment around a circle and store votes
                std::vector<cv::Point2i> points;
                //--- Generates points using trigonometric functions and incrementing the angle by user-defined resolution
                // trigCircleGeneration(points, radius_pixels, rotation_resolution, accum_x, accum_y);
                //--- Generates points using midpoint circle algorithm, so only the necessary points are generated
                midpointCircleAlgorithm(points, radius_pixels, accum_x, accum_y);

                // accumulator.at<uint16_t>(cv::Point(accum_x, accum_y)) += 1; // type must match matrix type CV_16U = uint16_t
                for (int k = 0; k < points.size(); ++k) {
                    accumulator.at<uint16_t>(points.at(k)) += 1; // type must match matrix type, cv_16U = uint16_t
                }
            }
        }
    }
}

 // FIXME: currently left in here to test it's speed against the new methods
 void generateAccumulatorUsingImage_OldMethod(cv::Mat &top_image_accum, cv::Mat accumulator)
 {
     int num_rows = top_image_accum.rows;
     int num_cols = top_image_accum.cols;

     for (int i = 0; i < num_rows; ++i) {
         for (int j = 0; j < num_cols; ++j) {
             if (!(top_image_accum.at<uint8_t>(i,j) == 0)) {
                 // Increment around a circle by rotation_resolution
                 double theta = 0.0; /// current angle around the circle
                 while (theta < 2*M_PI) {
                     // Calculate the (x,y) position along the circle in the image frame
                     int a = j - radius_pixels*cos(theta);
                     int b = i - radius_pixels*sin(theta);
                     // Convert circle poinst from image index to accumulator index
                     int accum_x;
                     int accum_y;
                     imageToAccumulator(a, b, accum_x, accum_y);
                     accumulator.at<uint16_t>(cv::Point(accum_x, accum_y)) += 1; // type must match matrix type CV_16U = uint16_t
                     theta += rotation_resolution;
                 }
             }
         }
     }
 }

 void imageToSensor(int image_x_in, int image_y_in, float& sensor_x_out, float& sensor_y_out)
 {
     // Calculate the x position
     int y_index_flipped = (y_pixels - 1) - image_y_in;
     sensor_x_out = (y_index_flipped*y_pixel_delta) + (y_pixel_delta/2); // place at middle of pixel

     // Calculate the y position
     float y_translated = (image_x_in*x_pixel_delta) + (x_pixel_delta/2); // place at middle of pixel
     float y_mirror = y_translated + y_mirror_min;
     sensor_y_out = -y_mirror;
 }

 // Overloaded function to handle 'double' inputs
 void sensorToImage(double& sensor_x_in, double& sensor_y_in, int& image_x_out, int& image_y_out)
 {
     // Calculate the x index
     double y_mirror = -sensor_y_in;
     double y_translated = y_mirror - y_mirror_min;
     image_x_out = trunc(y_translated/x_pixel_delta);

     // Calculate the y index
     int y_index_flipped = trunc(sensor_x_in/y_pixel_delta); // sensor frame has positive going up, but image frame has positive going down, so find y with positive up first, then flip it
     image_y_out = (y_pixels - 1) - y_index_flipped;
 }
 // Overloaded function to handle 'double' inputs
 void imageToSensor(int image_x_in, int image_y_in, double& sensor_x_out, double& sensor_y_out)
 {
     // Calculate the x position
     int y_index_flipped = (y_pixels - 1) - image_y_in;
     sensor_x_out = (y_index_flipped*y_pixel_delta) + (y_pixel_delta/2); // place at middle of pixel

     // Calculate the y position
     double y_translated = (image_x_in*x_pixel_delta) + (x_pixel_delta/2); // place at middle of pixel
     double y_mirror = y_translated + y_mirror_min;
     sensor_y_out = -y_mirror;
 }

 void imageToAccumulator(int image_x_in, int image_y_in, int& accum_x_out, int& accum_y_out)
 {
     // Calculate the x index
     accum_x_out = image_x_in + radius_pixels;

     // Calculate the y index
     accum_y_out = image_y_in + radius_pixels;
 }

 void accumulatorToImage(int accum_x_in, int accum_y_in, int& image_x_out, int& image_y_out)
 {
     // Calculate the x index
     image_x_out = accum_x_in - radius_pixels;

     // Calculate the y index
     image_y_out = accum_y_in - radius_pixels;
 }

 void sensorToTarget(double sensor_x_in, double sensor_y_in, double& target_x_out, double& target_y_out)
 {
     /**
      * Transforms a sensor (x,y) point to an (x,y) point in the target
      * frame
      */

     // Sensor point
     geometry_msgs::PointStamped sensor_point;
     sensor_point.header.frame_id = sensor_frame.c_str();
     sensor_point.point.x = sensor_x_in;
     sensor_point.point.y = sensor_y_in;

     // Target point
     geometry_msgs::PointStamped target_point;

     // Get transform from sensor to target frame
     tf_listener.waitForTransform(sensor_frame.c_str(), target_frame.c_str(), ros::Time::now(), ros::Duration(0.051));
     try {
         // Transform point
         tf_listener.transformPoint(target_frame.c_str(), sensor_point, target_point);
     }
     catch(tf::TransformException& ex) {
         ROS_ERROR("Target Transform Exception: %s", ex.what());
     }

     // Extract target x and y
     target_x_out = target_point.point.x;
     target_y_out = target_point.point.y;
 }

 void targetToSensor(double target_x_in, double target_y_in, double& sensor_x_out, double& sensor_y_out)
 {
     /**
      * Transforms a point in the target frame to a the sensor frame
      */

     // Create ROS PointStamped for tf functions
     geometry_msgs::PointStamped sensor_point;
     geometry_msgs::PointStamped target_point;
     target_point.header.frame_id = target_frame.c_str();
     target_point.point.x = target_x_in;
     target_point.point.y = target_y_in;

     // Get transform from target to sensor frame
     tf_listener.waitForTransform(target_frame.c_str(), sensor_frame.c_str(), ros::Time(0), ros::Duration(0.051));
     tf::StampedTransform transform;
     try {
         // Transform point
         tf_listener.transformPoint(sensor_frame.c_str(), target_point, sensor_point);

         // // DEBUG: check transform
         // tf_listener.lookupTransform(sensor_frame.c_str(), target_frame.c_str(), ros::Time(0), transform);
         // std::cout << "Transform from " << transform.frame_id_ << " to " << transform.child_frame_id_ << ":\n";
         // std::cout << transform.getOrigin()[0] << std::endl;
         // std::cout << transform.getOrigin()[1] << std::endl;
         // std::cout << transform.getOrigin()[2] << std::endl;
         // std::cout << transform.getRotation()[0] << std::endl;
         // std::cout << transform.getRotation()[1] << std::endl;
         // std::cout << transform.getRotation()[2] << std::endl;
         // std::cout << transform.getRotation()[3] << std::endl;
     }
     catch(tf::TransformException& ex) {
         ROS_ERROR("Sensor Transform Exception: %s", ex.what());
     }

     // Extract (x,y) positions from PointStamped
     sensor_x_out = sensor_point.point.x;
     sensor_y_out = sensor_point.point.y;

     // // DEBUG: check tranform
     // std::cout << "Target point (" << target_frame << ")\n";
     // std::cout << "(" << target_x_in << ", " << target_y_in << ")" << std::endl;
     // std::cout << "Target point (" << sensor_frame << ")\n";
     // std::cout << "(" << sensor_x_out << ", " << sensor_y_out << ")" << std::endl;
 }
