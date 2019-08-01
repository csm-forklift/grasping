#!/usr/bin/env python
'''
Tests receiving a grid map, converting the unknown cells into obstacle cells (value = 100), and then adjusts the image resolution to a specified value.
'''

import rospy
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np

class MapConverter:
    def __init__(self):
        rospy.init_node("map_resolution_test")

        self.rescale = 0.25

        self.low_res_map_pub = rospy.Publisher("~map", OccupancyGrid, queue_size=1)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)

    def map_callback(self, msg):
        # Convert the map data into an opencv image
        img_width = msg.info.width
        img_height = msg.info.height
        img = np.asarray(msg.data, dtype='uint8')
        # Convert all of the -1 spaces (value = 255 when a 'uint8') into obstacle value 100
        img[np.where(img == 255)[0]] = 100
        img = np.reshape(img, [img_height, img_width, 1])

        # Decrease the resolution of the map
        img_rescaled = cv2.resize(img, None, fx=self.rescale, fy=self.rescale)

        # Show the image for debugging
        # cv2.namedWindow('cartographer_map', 2)
        # cv2.imshow('cartographer_map', img_rescaled)
        # cv2.waitKey(0)

        # Convert image back into an occupancy grid message
        rescaled_height, rescaled_width = img_rescaled.shape
        rescaled_data = img_rescaled.flatten('C')
        new_map = OccupancyGrid()
        new_map.info.width = rescaled_width
        new_map.info.height = rescaled_height
        new_map.header.stamp = msg.header.stamp
        new_map.header.frame_id = msg.header.frame_id
        new_map.info.map_load_time = msg.info.map_load_time
        new_map.info.resolution = msg.info.resolution/self.rescale
        new_map.info.origin.position.x = self.rescale*msg.info.origin.position.x
        new_map.info.origin.position.y = self.rescale*msg.info.origin.position.y
        new_map.info.origin.orientation.x = msg.info.origin.orientation.x
        new_map.info.origin.orientation.y = msg.info.origin.orientation.y
        new_map.info.origin.orientation.z = msg.info.origin.orientation.z
        new_map.info.origin.orientation.w = msg.info.origin.orientation.w
        new_map.data = rescaled_data.astype('int8').tolist()

        # Publish new map
        self.low_res_map_pub.publish(new_map)


if __name__ == '__main__':
    try:
        map_converter = MapConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
