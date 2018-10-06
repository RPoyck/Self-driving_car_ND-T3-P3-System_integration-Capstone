#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO : Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
	def __init__(self):
		rospy.init_node('waypoint_updater')
		
		# ROS subscribers #
		rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
		rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

		# TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


		self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

		# TODO: Add other member variables you need below
		self.pose = None
		self.base_waypoints = None
		self.waypoints_2d = None
		self.waypoint_tree = None

		#rospy.spin()
		self.loop()



	#---------#
	# Methods #
	#---------#

	# When the complete list of waypoints have been received and stored, get and publish the closest waypoints with a frequency of 30 [hz] #
	def loop(self):
		rate = rospy.Rate(30)
		while not rospy.is_shutdown():
			if self.pose and self.base_waypoints:
				closest_waypoint_idx = self.get_closest_waypoint_idx()	# Get closest waypoint #
				self.publish_waypoints(closest_waypoint_idx)
				
			rate.sleep()


	def get_closest_waypoint_idx(self):
		x_veh = self.pose.pose.position.x
		y_veh = self.pose.pose.position.y
		# Get the ID of the one waypoint which is closest to the vehicle position #
		closest_idx = self.waypoint_tree.query([x_veh, y_veh], 1)[1]
		
		# Check if this closest waypoint is ahead or behind the vehicle #
		closest_coord = self.waypoints_2d[closest_idx]
		prev_coord = self.waypoints_2d[closest_idx-1]
		
		# Equation for hyperplane through closest_coords #
		cl_vect = np.array(closest_coord)
		prev_vect = np.array(prev_coord)
		pos_vect = np.array([x_veh, y_veh])
		
		# Dot product of the vectors of the two coords #
		val = np.dot( cl_vect - prev_vect, pos_vect - cl_vect )
		
		# If the val is positive, the waypoint is behind the vehicle, therefore the next waypoint is one index higher (with loop-back to the beginning of the waypoint list if necessary) #
		if val > 0:
			closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
		
		return closest_idx


	def publish_waypoints(self, closest_idx):
		lane = Lane()
		#lane.header = self.base_waypoints.header
		lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
		self.final_waypoints_pub.publish(lane)


	def get_waypoint_velocity(self, waypoint):
		return waypoint.twist.twist.linear.x


	def set_waypoint_velocity(self, waypoints, waypoint, velocity):
		waypoints[waypoint].twist.twist.linear.x = velocity


	def distance(self, waypoints, wp1, wp2):
		dist = 0
		dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
		for i in range(wp1, wp2+1):
			dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
			wp1 = i
		return dist


	#------------------#
	# Callback methods #
	#------------------#

	# Receives and stores the current pose of the vehicle #
	def pose_cb(self, msg):
		self.pose = msg


	# Receives the one-time message with all waypoints and saves it within a KDTree member variable for later use #
	def waypoints_cb(self, waypoints):
		self.base_waypoints = waypoints
		if not self.waypoints_2d:
			self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
			self.waypoint_tree = KDTree(self.waypoints_2d)
			
			
	def traffic_cb(self, msg):
		# TODO: Callback for /traffic_waypoint message. Implement
		pass
	

	def obstacle_cb(self, msg):
		# TODO: Callback for /obstacle_waypoint message. We will implement it later
		pass


if __name__ == '__main__':
	try:
		WaypointUpdater()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start waypoint updater node.')
