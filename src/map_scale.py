#!/usr/bin/env python
'''
Rescales map data of an OccupancyGrid message and publishes a new message.
'''


import rospy
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np

class MapScale:
    def __init__(self):
        rospy.init_node("map_scale")
        self.rescale_factor = rospy.get_param("~rescale_factor", 0.5)
        self.map_pub = rospy.Publisher("~scaled_map", OccupancyGrid, queue_size=1)
        self.map_sub = rospy.Subscriber("~input_map", OccupancyGrid, self.mapCallback, queue_size=1)

    def mapCallback(self, msg):
        # Convert the message into an opencv image
        img_width = msg.info.width
        img_height = msg.info.height
        img = np.asarray(msg.data, dtype='uint8')

        # Convert all of the -1 spces (value = 255 when an 'uint8'
        img[np.where(img == 255)[0]] = 100
        img = np.reshape(img, [img_height, img_width, 1])

        # Decrease the rsolution of the map
        img_scaled = cv2.resize(img, None, fx=self.rescale_factor, fy=self.rescale_factor)

        # Convert image back into an occupancy grid message
        scaled_height, scaled_width = img_scaled.shape
        scaled_data = img_scaled.flatten('C')
        scaled_map = OccupancyGrid()
        scaled_map.info.width = scaled_width
        scaled_map.info.height = scaled_height
        scaled_map.header.stamp = msg.header.stamp
        scaled_map.header.frame_id = msg.header.frame_id
        scaled_map.info.map_load_time = msg.info.map_load_time
        scaled_map.info.resolution = msg.info.resolution/self.rescale_factor
        scaled_map.info.origin.position.x = msg.info.origin.position.x
        scaled_map.info.origin.position.y = msg.info.origin.position.y
        scaled_map.info.origin.position.z = msg.info.origin.position.z
        scaled_map.info.origin.orientation.x = msg.info.origin.orientation.x
        scaled_map.info.origin.orientation.y = msg.info.origin.orientation.y
        scaled_map.info.origin.orientation.z = msg.info.origin.orientation.z
        scaled_map.info.origin.orientation.w = msg.info.origin.orientation.w
        scaled_map.data = scaled_data.astype('int8').tolist()

        # Publish rescaled map
        self.map_pub.publish(scaled_map)


if __name__ == '__main__':
    try:
        map_scale = MapScale()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
