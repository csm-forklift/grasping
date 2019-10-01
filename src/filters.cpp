/*
 * Functions used to filter the potential cylinder points.
 */

#include <cylinder_detection/cylinder_detection.hpp>

void CylinderDetector::findPointsDeletion(std::vector<cv::Point> &points, cv::Mat &accumulator)
{
    //----- Determines the top 'num_potentials' number of points by storing the maximum value in accumulator, setting the points within a circle of the desired radius in the image to 0, then refinding the maximum, deleting around that points, (repeat num_potentials times)
    // The returned 'points' vector is in image frame

    cv::Mat accum_iter = accumulator.clone();
    // cv::Mat accum_mask_debug = cv::Mat(accum_iter.rows, accum_iter.cols, CV_16U, USHRT_MAX);

    for (int i = 0; i < num_potentials; ++i) {
        // Grab Max
        double accum_max;
        double accum_min;
        cv::Point max_loc;
        cv::Point min_loc;
        cv::minMaxLoc(accum_iter, &accum_min, &accum_max, &min_loc, &max_loc);

        // // DEBUG: Print the max value
        // std::cout << i << ": " << accum_max << std::endl;

        // Save point
        int circle_index_x;
        int circle_index_y;
        accumulatorToImage(max_loc.x, max_loc.y, circle_index_x, circle_index_y);
        cv::Point point(circle_index_x, circle_index_y);
        points.push_back(point);

        // Delete accumulator values around previous max
        // Create mask
        cv::Mat accum_mask(accum_iter.rows, accum_iter.cols, CV_16U, USHRT_MAX);

        // Create structuring element as circle
        cv::Size mask_size(2*radius_pixels - 1, 2*radius_pixels - 1);
        cv::Mat circle_mask(mask_size, CV_16U);
        circle_mask = cv::getStructuringElement(cv::MORPH_ELLIPSE, mask_size);
        circle_mask.convertTo(circle_mask, CV_16U);
        circle_mask *= USHRT_MAX;

        bitwise_not(circle_mask, circle_mask);

        // Merge mask of 1's and circle mask
        int offset_x = 0;
        int offset_y = 0;
        int upper_x_accum = max_loc.x - (radius_pixels - 1);
        int upper_y_accum = max_loc.y - (radius_pixels - 1);
        int length_x = 2*radius_pixels - 1;
        int length_y = 2*radius_pixels - 1;
        int upper_x_mask = 0;
        int upper_y_mask = 0;

        // Make sure the edges of the masks are within the image bounds
        if (upper_x_accum < 0) {
            offset_x = upper_x_accum;
            upper_x_accum = 0;
            upper_x_mask = -offset_x;
        }
        else if ((upper_x_accum + length_x) > (accum_mask.cols - 1)) {
            offset_x = (accum_mask.cols - 1) - (upper_x_accum + length_x);
        }
        if (upper_y_accum < 0) {
            offset_y = upper_y_accum;
            upper_y_accum = 0;
            upper_y_mask = -offset_y;
        }
        else if ((upper_y_accum + length_y) > (accum_mask.rows - 1)) {
            offset_y = (accum_mask.rows - 1) - (upper_y_accum + length_y);
        }

        cv::Rect accum_area(upper_x_accum, upper_y_accum, length_x + offset_x, length_y + offset_y);
        cv::Rect mask_area(upper_x_mask, upper_y_mask, length_x + offset_x, length_y + offset_y);
        cv::Mat subrange_accum = accum_mask(accum_area);
        cv::Mat subrange_mask = circle_mask(mask_area);
        subrange_mask.copyTo(subrange_accum);

        // Bitwise AND to remove points inside circle
        if (i != (num_potentials - 1)) {
            bitwise_and(accum_iter, accum_mask, accum_iter);
        }
    }
}

void CylinderDetector::findPointsLocalMax()
{
    // See: https://stackoverflow.com/questions/5550290/find-local-maxima-in-grayscale-image-using-opencv
}

void CylinderDetector::thresholdFilter(std::vector<cv::Point> &points, cv::Mat &accumulator) {
    // Convert points to accumulator frame, then check if the value at that point is within the desired threshold
    int accum_x;
    int accum_y;
    std::vector<int> to_remove;
    for (int i = 0; i < points.size(); ++i) {
        imageToAccumulator(points.at(i).x, points.at(i).y, accum_x, accum_y);
        int value = accumulator.at<uint16_t>(cv::Point(accum_x, accum_y));
        if ((value < threshold_low) || (value > threshold_high)) {
            to_remove.push_back(i);
        }
    }

    // Now remove the points that are outside the threshold
    for (int i = to_remove.size() - 1; i >= 0; --i) {
        points.erase(points.begin() + to_remove.at(i));
    }
}

void CylinderDetector::checkCenterPoints(std::vector<cv::Point> &points, cv::Mat &top_image)
{
    // The points vector should be in the image frame (not accumulator frame)
    // Since the cylinder should only be producing points along the surface,
    // this function checks to see if there are any points within the
    // cirlce's radius. If there are, that means this is not a solid
    // cylinder and can be removed.

    // DEBUG: Visually check which points are being removed
    cv::Mat top_image_debug = top_image.clone();

    // Create structuring element as circle
    int scan_pixels = 0.5*radius_pixels;
    cv::Size mask_size(2*scan_pixels - 1, 2*scan_pixels - 1);
    cv::Mat circle_mask;
    circle_mask = cv::getStructuringElement(cv::MORPH_ELLIPSE, mask_size);
    circle_mask *= UCHAR_MAX;

    // Iterate through each point and check if there are any points in the top_image (the 2D compressed point cloud) that are within 75% of the circle radius
    std::vector<int> to_remove; // indices to remove from points
    for (int i = 0; i < points.size(); ++i) {
        // Create a mask to apply over top image
        cv::Mat top_mask = cv::Mat::zeros(top_image.size(), CV_8U);

        // Adjust the bounds for the mask if the edges lie outside the image bounds
        int offset_x = 0;
        int offset_y = 0;
        int upper_x_top = points.at(i).x - (scan_pixels - 1);
        int upper_y_top = points.at(i).y - (scan_pixels - 1);
        int length_x = 2*scan_pixels - 1;
        int length_y = 2*scan_pixels - 1;
        int upper_x_mask = 0;
        int upper_y_mask = 0;

        // Make sure the edges of the masks are within the image bounds
        if (upper_x_top < 0) {
            if (upper_x_top + length_x < 0) {
                upper_x_top = 0;
                length_x = 0;
            }
            else {
                offset_x = upper_x_top;
                upper_x_top = 0;
                upper_x_mask = -offset_x;
            }
        }
        else if ((upper_x_top + length_x) > (top_mask.cols - 1)) {
            if (upper_x_top > top_mask.cols - 1) {
                upper_x_top = 0;
                length_x = 0;
            }
            else {
                offset_x = (top_mask.cols - 1) - (upper_x_top + length_x);
            }
        }
        if (upper_y_top < 0) {
            if (upper_y_top + length_y < 0) {
                upper_y_top = 0;
                length_y = 0;
            }
            else {
                offset_y = upper_y_top;
                upper_y_top = 0;
                upper_y_mask = -offset_y;
            }
        }
        else if ((upper_y_top + length_y) > (top_mask.rows - 1)) {
            if (upper_y_top > top_mask.rows - 1) {
                upper_y_top = 0;
                length_y = 0;
            }
            else {
                offset_y = (top_mask.rows - 1) - (upper_y_top + length_y);
            }
        }

        // Add circle to top mask
        cv::Rect top_area(upper_x_top, upper_y_top, length_x + offset_x, length_y + offset_y);
        cv::Rect mask_area(upper_x_mask, upper_y_mask, length_x + offset_x, length_y + offset_y);
        cv::Mat subrange_top = top_mask(top_area);
        cv::Mat subrange_mask = circle_mask(mask_area);
        subrange_mask.copyTo(subrange_top);

        cv::Mat result_img;
        bitwise_and(top_image, top_mask, result_img);

        // If the max of the resulting image is not 0, there is a point within the perspective cylinder's radius
        double result_max;
        cv::minMaxLoc(result_img, NULL, &result_max, NULL, NULL);

        if (result_max != 0) {
            to_remove.push_back(i);
        }
    }

    // Now remove the points that are not cylinders, starting from the largest index
    for (int i = to_remove.size() - 1; i >= 0; --i) {
        points.erase(points.begin() + to_remove.at(i));
    }
}

void CylinderDetector::varianceFilter(std::vector<cv::Point> &points, std::vector<double> &variances, cv::Mat &top_image)
{
    // The points vector should be in the image frame (not accumulator frame)
    // 'variances' vector must be the same size as 'points'
    if (points.size() != variances.size()) {
        std::cout << "ERROR (varianceFilter): 'variances' and 'points' must be the same size." << std::endl;
    }

    std::vector<int> var_n(points.size(), 0); // stores the number of points being summed in the variance
    // Iterate through each point in top_image (i = x pixel, j = y pixel)
    for (int i = 0; i < top_image.cols; ++i) {
        for (int j = 0; j < top_image.rows; ++j) {
            // Check if point is not empty
            if (top_image.at<uint8_t>(cv::Point(i,j))) {
                for (int n = 0; n < points.size(); ++n) {
                    double x;
                    double y;
                    double x_c;
                    double y_c;
                    imageToSensor(points.at(n).x, points.at(n).y, x_c, y_c);
                    imageToSensor(i, j, x, y);
                    double r = calculateRadius(x, y, x_c, y_c);
                    // Check if radius is in bounds
                    if ((r < upper_radius) && (r > lower_radius)) {
                        variances.at(n) += pow((r - circle_radius), 2);
                        var_n.at(n) += 1;
                    }
                }
            }
        }
    }

    // Get average variance
    for (int i = 0; i < variances.size(); ++i) {
        variances.at(i) /= var_n.at(i);
    }

    std::vector<int> to_remove;
    for (int i = 0; i < points.size(); ++i) {
        // variance_tolerance = 0.001; // good baseline
        // This equation calculates a tolerance based on distance from the sensor. It was derived empirically. This assumes the target frame is the sensor frame.
        double x,y;
        imageToSensor(points.at(i).x, points.at(i).y, x, y);


        double distance = sqrt(pow(x,2) + pow(y,2));
        // variance_tolerance = 0.0003*distance + 0.0001;
        variance_tolerance = 0.3*distance + 0.1;

        if (variances.at(i) > variance_tolerance) {
            to_remove.push_back(i);
        }
    }


    // Now remove the points that are less than the threshold
    for (int i = to_remove.size() - 1; i >= 0; --i) {
        points.erase(points.begin() + to_remove.at(i));
    }
}

void CylinderDetector::closestToTarget(std::vector<cv::Point> &points, cv::Point target)
{
    // Checks all the potential points against a target point and accepts the closest one within a range. If no points are within the range, it returns an empty points vector.

    double min_distance_sq = 1000.0; // current minimum squared distance
    cv::Point closest_point; // the current closest point to the target

    // Before cycling through all the points, convert the target point into sensor frame.
    double target_cam_x;
    double target_cam_y;
    targetToSensor(target.x, target.y, target_cam_x, target_cam_y);

    for (int i = 0; i < points.size(); ++i) {
        // Convert points from image frame to target frame
        double potential_x_sensor;
        double potential_y_sensor;
        double potential_x;
        double potential_y;
        imageToSensor(points.at(i).x, points.at(i).y, potential_x, potential_y);

        // Convert from sensor frame to target frame
        //sensorToTarget(potential_x_sensor, potential_y_sensor, potential_x, potential_y);

        // Find minimum distance from target

        double distance_sq = pow(potential_x - target_cam_x, 2) + pow(potential_y - target_cam_y, 2);
        if (distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_point.x = points.at(i).x;
            closest_point.y = points.at(i).y;
        }
    }

    // Clear all the points
    points.clear();

    // Check if minimum distance is within range, if so store that point, if not leave the points vector empty
    // DEBUG: Show the current minimum distance seen
    //cout << "Min dist: " << min_distance_sq <<'\n';
    if (min_distance_sq < pow(target_tolerance, 2)) {
        points.push_back(closest_point);
    }
}

void CylinderDetector::targetFilter(std::vector<cv::Point> &points, cv::Point target)
{
    // Sorts the 'points' vector in order from closest to target to farthest away.

    // Create copy of points to iterate through and delete entries
    std::vector<cv::Point> points_copy(points);

    // Create vector for holding distances from target
    std::vector<double> distance_sq_from_target(points.size());

    // Calculate the distances
    for (int i = 0; i < points.size(); ++i) {
        double sensor_frame_x, sensor_frame_y, target_frame_x, target_frame_y;
        imageToSensor(points.at(i).x, points.at(i).y, sensor_frame_x, sensor_frame_y);
        sensorToTarget(sensor_frame_x, sensor_frame_y, target_frame_x, target_frame_y);
        distance_sq_from_target.push_back(sqrt(pow(target_frame_x - target.x, 2) + pow(target_frame_y - target.y, 2)));
    }

    // Remove points that are outside the target tolerance
    double target_tolerance_sq = pow(target_tolerance, 2);
    for (int i = distance_sq_from_target.size()-1; i >= 0; --i) {
        if (distance_sq_from_target.at(i) > target_tolerance_sq) {
            distance_sq_from_target.erase(distance_sq_from_target.begin() + i);
            points.erase(points.begin() + i);
        }
    }

    // Sort the 'points' vector by ascending distance from target
    for (int i = 0; i < points.size(); ++i) {
        // Get the index of the minimum distance
        int min_distance_index = std::distance(distance_sq_from_target.begin(), std::min_element(distance_sq_from_target.begin(), distance_sq_from_target.end()));

        // Add point to corresponding position
        points.at(i) = points_copy.at(min_distance_index);

        // Delete element from vectors
        points_copy.erase(points_copy.begin() + min_distance_index);
        distance_sq_from_target.erase(distance_sq_from_target.begin() + min_distance_index);
    }
}
