#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class AxisSignSlices:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/lidar_points")
        base_ns = rospy.get_param("~pub_ns", "/axis_slices")

        # Publishers for ± along each axis
        self.pubs = {
            "x_pos": rospy.Publisher(f"{base_ns}/x_pos", PointCloud2, queue_size=1),
            "x_neg": rospy.Publisher(f"{base_ns}/x_neg", PointCloud2, queue_size=1),
            "y_pos": rospy.Publisher(f"{base_ns}/y_pos", PointCloud2, queue_size=1),
            "y_neg": rospy.Publisher(f"{base_ns}/y_neg", PointCloud2, queue_size=1),
            "z_pos": rospy.Publisher(f"{base_ns}/z_pos", PointCloud2, queue_size=1),
            "z_neg": rospy.Publisher(f"{base_ns}/z_neg", PointCloud2, queue_size=1),
        }

        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.cb_cloud, queue_size=1)
        rospy.loginfo("Listening to %s and publishing six half-clouds under %s/*", self.input_topic, base_ns)

    def cb_cloud(self, msg: PointCloud2):
        # Extract field indices
        field_names = [f.name for f in msg.fields]
        try:
            ix = field_names.index('x')
            iy = field_names.index('y')
            iz = field_names.index('z')
        except ValueError:
            rospy.logerr("Incoming PointCloud2 does not have x, y, z fields.")
            return

        # Prepare lists
        pos_x, neg_x, pos_y, neg_y, pos_z, neg_z = [], [], [], [], [], []

        # Separate points by sign
        for p in pc2.read_points(msg, field_names=None, skip_nans=True):
            x, y, z = p[ix], p[iy], p[iz]
            if x >= 0: pos_x.append(p)
            if y >= 0: pos_y.append(p)

        # Helper to publish if not empty
        def pub_if_any(points, topic):
            if not points: return
            msg_out = pc2.create_cloud(msg.header, msg.fields, points)
            self.pubs[topic].publish(msg_out)

        # Publish all six
        pub_if_any(pos_x, "x_pos")
        pub_if_any(pos_y, "y_pos")

def main():
    rospy.init_node("lidar_axis_sign_slices", anonymous=False)
    AxisSignSlices()
    rospy.spin()

if __name__ == "__main__":
    main()
