#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''

LOOKAHEAD_WPS = 150 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5 # [m/s^2]


class WaypointUpdater(object):
	def __init__(self):
		rospy.init_node('waypoint_updater')
		
		# ROS subscribers #
		rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
		rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
		rospy.Subscriber('/traffic_waypoint',Int32, self.traffic_cb)
		rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

		# ROS publishers #
		self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

		# Member variables #
		self.base_lane = None
		self.pose = None
		self.stopline_wp_idx = -1
		self.base_waypoints = None
		self.waypoints_2d = None
		self.waypoint_tree = None
		self.current_vel = None

		self.loop()


	#---------#
	# Methods #
	#---------#

	# When the complete list of waypoints have been received and stored, get and publish the closest waypoints with a frequency of 40 [hz] #
	def loop(self):
		rate = rospy.Rate(40)
		while not rospy.is_shutdown():
			if self.pose and self.base_waypoints:
				self.publish_waypoints()
			rate.sleep()


	def get_closest_waypoint_idx(self):
		x_veh = self.pose.pose.position.x
		y_veh = self.pose.pose.position.y
		
		# Get the ID of the one waypoint which is closest to the vehicle position #
		closest_idx = self.waypoint_tree.query([x_veh, y_veh], 1)[1]
		#print("Closest_idx: ", closest_idx, "\n")	# Testing and debug output #
		#print( "len(self.base_waypoints.waypoints): ", len(self.base_waypoints.waypoints) )	# Testing and debug output #
		
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


	def publish_waypoints(self):
		final_lane = self.generate_lane()
		self.final_waypoints_pub.publish(final_lane)


	def generate_lane(self):
		lane = Lane()
		
		# Get the base waypoints for the coming time-step #
		closest_idx = self.get_closest_waypoint_idx()
		farthest_idx = closest_idx + LOOKAHEAD_WPS
		
		# Check if the next waypoints would extend to beyond the range of base, and loopback to the beginning if this is the case #
		if len(self.base_waypoints.waypoints) <= farthest_idx:
			base_waypoints_section = self.base_waypoints.waypoints[ closest_idx : (len(self.base_waypoints.waypoints) - 1) ]
			for i in range( 0, LOOKAHEAD_WPS - (len(self.base_waypoints.waypoints) - 1 - closest_idx) ):
				base_waypoints_section.append(self.base_waypoints.waypoints[i])
		else:
			base_waypoints_section = self.base_waypoints.waypoints[closest_idx:farthest_idx]
		
		# If no traffic sign is detected or the detected sign is out of path planning reach, return the base waypoints # 
		if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx):
			lane.waypoints = base_waypoints_section
		# If a traffic sign is detected within range give the base waypoints to the waypoint manipulation function@
		else:
			lane.waypoints = self.decelerate_waypoints(base_waypoints_section, closest_idx)
		
		return lane


	def decelerate_waypoints(self, waypoints, closest_idx):
		print("Waypoints: ")
		# Make a new list of waypoints in order to store the manipulated base waypoints #
		temp = []
		# Two waypoints back from line so front of car stops approximately at the line #
		stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
		
		
		# Unused attempts at linear decelleration #
		
		# Calculate the distance between the first waypoint and the index at which the vehicle has to come to a standstill #
		dist_tot = self.distance(waypoints, 0, stop_idx)
		v_start = self.current_vel # self.pose.twist.twist.linear.x	# self.distance(waypoints, 0, stop_idx)
			# Calculate the time required to come to a standstill at the stop index assuming linear unlimited acceleration #
			#t_tot = 2.0 * dist_tot / v_start
			# Calculate the necessary acceleration ("deceleration") across the entire section #
			#accel = -1.0 * min(MAX_DECEL, ( v_start / t_tot ) )
			# Calculate the distance to come to a standstill #
			#dist_stop = v_start*v_start / ( 2.0 * (-1.0 * accel) )
			# Calculate the corresponding acceleration per meter #
			#accel_m = v_start / dist_stop
			
			
			#a_would_be = v_start * v_start / ( 2.0 * dist_tot )
			#if MAX_DECEL <= a_would_be:
				#for i in range(stop_idx, len(waypoints)):
					#d_would_be = self.distance(waypoints, 0, i)
					#a_would_be = v_start * v_start / ( 2.0 * d_would_be )
					#if a_would_be <= MAX_DECEL:
						#stop_idx = i
						#break
			#d_stop = v_start * v_start / ( 2.0 * a_would_be )
			#a_d = v_start / d_stop
		
		# -------------------------------------- #
		
		# Calculate the corresponding acceleration per waypoint #
		for i, wp in enumerate(waypoints):
			p = Waypoint()
			# The goal position and orientation of the vehicle at the waypoint remains the same #
			p.pose = wp.pose
			
			# Calculate the distance between this waypoint and the index at which the vehicle has to come to a standstill #
			dist = self.distance(waypoints, i, stop_idx)
			# Calculate the velocity the vehicle has to have at this waypoint to come to a standstill in time #
			#vel = math.sqrt(2.0 * MAX_DECEL * dist)
			#vel = v_start - ( (dist_tot - dist) * accel_m )
			#vel = v_start - ( a_d * (d_stop-dist) )
			delta_dist = dist_tot - dist
			if 0.0 < delta_dist:
				vel = v_start * ( 1 - (delta_dist / dist_tot) * (delta_dist / dist_tot) )
			else:
				vel = v_start
			
			if vel < 0.1:
				vel = 0.0
			
			# A safeguard to arrive at the destination despite of an over-enthusiastic controller #
			if i < (stop_idx-2) and v_start < 0.2:
				vel = 0.75
				#if i == stop_idx - 5:
					#vel = 0.3
				#else:
					#if i == stop_idx - 4:
						#vel = 1.0
					#else:
						#vel = 1.5
			else:
				if (stop_idx-2) < i:
					vel = 0.0
			
			# Use the calculated required waypoint velocity in stead of the original/base waypoint velocity, unless the base waypoint velocity is smaller than this value #
			p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
			temp.append(p)
		
		
		
		return temp


	#def get_waypoint_velocity(self, waypoint):
		#return waypoint.twist.twist.linear.x


	#def set_waypoint_velocity(self, waypoints, waypoint, velocity):
		#waypoints[waypoint].twist.twist.linear.x = velocity


	def distance(self, waypoints, wp1, wp2):
		dist = 0
		# Function to calculate the euclidean distance between two given waypoints #
		dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
		# Add the distance between all consecutive waypoints between wp1 and wp2 # 
		for i in range(wp1, wp2+1):
			dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
			wp1 = i
		return dist


	#------------------#
	# Callback methods #
	#------------------#


	def velocity_cb(self, msg):
		self.current_vel = msg.twist.linear.x


	# Receives and stores the current pose of the vehicle #
	def pose_cb(self, msg):
		self.pose = msg


	# Receives the one-time message with all waypoints and saves it within a KDTree member variable for later use #
	def waypoints_cb(self, waypoints):
		if not self.base_waypoints:
			self.base_waypoints = waypoints
		if not self.waypoints_2d:
			self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
			self.waypoint_tree = KDTree(self.waypoints_2d)


	# Receives and stores any incoming waypoints of detected traffic signs which are red #
	def traffic_cb(self, msg):
		self.stopline_wp_idx = msg.data


	#def obstacle_cb(self, msg):
		## TODO: Callback for /obstacle_waypoint message. We will implement it later
		#pass


if __name__ == '__main__':
	try:
		WaypointUpdater()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start waypoint updater node.')
