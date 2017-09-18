#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint

import tf
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', PoseStamped, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        # Current pose of the vehicle updated at an unknown rate
        self.currentPose = None

        # Part of the complete waypoints retrived from /base_waypoints
        # Note: In total /base_waypoints has 10902 data points
        self.map_waypoints = None

        #Boolean that specifies if the map has been loaded at least once
        self.map_loaded = False

        rospy.spin()

    def distance_from_waypoint(self, waypoint):
        """
            Compute the euclidian distance of the waipoint
            from the current pose of the vehicle
        """
        xw = waypoint.pose.pose.position.x
        yw = waypoint.pose.pose.position.y
        zw = waypoint.pose.pose.position.z

        xc = self.currentPose.position.x
        yc = self.currentPose.position.y
        zc = self.currentPose.position.z

        return math.sqrt((xw - xc)*(xw - xc) + 
                         (yw - yc)*(yw - yc) + 
                         (zw - zc)*(zw - zc))

    def closestWaypointId(self):
        """
            Find the closest point on the map to the current
            vehicle location
            NOTE: Currently iterate through the entire map
            => Need a more clever method for realtime operation
            Subsamplig on init + Sliding window, etc...
        """
        closestLen = 100000
        idClosest = 0

        #rospy.loginfo('[closestWaypointId] ==> ')
        # for i in range(len(self.map_waypoints)):
        #     waypoint = self.map_waypoints[i]
        #     dist = self.distance_from_waypoint(waypoint)
        #     if dist < closestLen:
        #         closestLen = dist
        #         idClosest = idClosest
            #rospy.loginfo('[closestWaypointId] - waypoint %s %s %s %s', 
            #              i,
            #              waypoint.pose.pose.position.x,
            #              waypoint.pose.pose.position.y,
            #              waypoint.pose.pose.position.z)

        return idClosest

    def nextWaypoint(self):
        """
            Find the next waypoint ahead of the vehicle
        """
        closestId = self.closestWaypointId()
        closestWaypoint = self.map_waypoints[closestId]

        map_x = closestWaypoint.pose.pose.position.x
        map_y = closestWaypoint.pose.pose.position.y

        car_x = self.currentPose.position.x
        car_y = self.currentPose.position.y

        heading = math.atan2((map_y - car_y), (map_x - car_x))

        _quaternion = (self.currentPose.orientation.x,
                      self.currentPose.orientation.y,
                      self.currentPose.orientation.z,
                      self.currentPose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(_quaternion)
        theta = euler[2]

        angle = math.fabs(theta - heading)

        if angle > (math.pi/4.0):
            closestId += 1

        return closestId

    def pose_cb(self, msg):
        """
            /current_pose callback funcion
            NB: Rate unknown => the processing may need to be done
            elswhere
        """
        # TODO: Implement
        self.currentPose = msg.pose

        if self.map_loaded:
            idNext = self.nextWaypoint()

            # Pose XYZ
            _nextWaypoint = self.map_waypoints[idNext]
            rospy.loginfo('[pose_cb_next] - id %s - %s %s %s' , 
                          idNext,
                          _nextWaypoint.pose.pose.position.x,
                          _nextWaypoint.pose.pose.position.y,
                          _nextWaypoint.pose.pose.position.z)

        # Pose XYZ
        rospy.loginfo('[pose_cb] - Pose xyz %s %s %s', 
                        self.currentPose.position.x, 
                        self.currentPose.position.y, 
                        self.currentPose.position.z)

        # rospy.loginfo('[msg_pose] - Pose xyz %s %s %s', 
        #                 msg.pose.position.x, 
        #                 msg.pose.position.y, 
        #                 msg.pose.position.z)

        # Quaternion 
        # rospy.loginfo('[pose_cb] - Quaternion xyzw %s %s %s - %s', 
        #                 self.currentPose.orientation.x, 
        #                 self.currentPose.orientation.y, 
        #                 self.currentPose.orientation.z,
        #                 self.currentPose.orientation.w)
        # pass

    def waypoints_cb(self, waypoints):
        """
            /base_waypoints Callback function
        """
        # TODO: Implement
        self.map_waypoints = waypoints.waypoints

        rospy.loginfo('[waypoints_cb] - map_length %s', len(self.map_waypoints))
        # rospy.loginfo('[waypoints_cb] - first %s %s %s', 
        #               self.map_waypoints[0].pose.pose.position.x,
        #               self.map_waypoints[0].pose.pose.position.y,
        #               self.map_waypoints[0].pose.pose.position.z)
        # _last = len(self.map_waypoints) - 1
        # rospy.loginfo('[waypoints_cb] - last %s %s %s', 
        #               self.map_waypoints[_last].pose.pose.position.x,
        #               self.map_waypoints[_last].pose.pose.position.y,
        #               self.map_waypoints[_last].pose.pose.position.z)
        # The map has been loaded

        rospy.loginfo('[waypoints_cb_list] ===> ')
        for _id in range(10):
            _mult = 100
            rospy.loginfo('[waypoints_cb_list] - id %s - %s %s %s' , 
                          _id,
                          self.map_waypoints[_mult*_id].pose.pose.position.x,
                          self.map_waypoints[_mult*_id].pose.pose.position.y,
                          self.map_waypoints[_mult*_id].pose.pose.position.z)
        self.map_loaded = True
        # rospy.loginfo('[waypoints_cb]')
        # pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
