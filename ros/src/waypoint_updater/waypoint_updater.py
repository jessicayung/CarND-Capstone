#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from operator import itemgetter

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

        # Current pose of the vehicle updated at an unknown rate
        self.pose = None

        # Part of the complete waypoints retrived from /base_waypoints
        # Note: In total /base_waypoints has 10902 data points
        self.base_waypoints = None

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

        xc = self.pose.position.x
        yc = self.pose.position.y
        zc = self.pose.position.z

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
        # for i in range(len(self.base_waypoints)):
        #     waypoint = self.base_waypoints[i]
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
        closestWaypoint = self.base_waypoints[closestId]

        map_x = closestWaypoint.pose.pose.position.x
        map_y = closestWaypoint.pose.pose.position.y

        car_x = self.pose.position.x
        car_y = self.pose.position.y

        heading = math.atan2((map_y - car_y), (map_x - car_x))

        _quaternion = (self.pose.orientation.x,
                      self.pose.orientation.y,
                      self.pose.orientation.z,
                      self.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(_quaternion)
        theta = euler[2]

        angle = math.fabs(theta - heading)

        if angle > (math.pi/4.0):
            closestId += 1

        return closestId

    def pose_cb(self, msg):
        """
            /current_pose callback function
            NB: Rate unknown => the processing may need to be done
            elsewhere
        """
        self.pose = msg.pose

    def waypoints_cb(self, lane):
        """
        /base_waypoints Callback function
        """
        self.base_waypoints = lane.waypoints

        if not self.pose:
            rospy.logwarn('no pose has been set')
            return None

        if not self.base_waypoints:
            rospy.logwarn('no waypoints have been set')
            return None

        # get closest waypoint
        c_x = self.pose.position.x
        c_y = self.pose.position.y
        c_o = self.pose.orientation
        c_q = (c_o.x, c_o.y, c_o.z, c_o.w)
        _, _, c_w = tf.transformations.euler_from_quaternion(c_q)

        waypoints_ahead = []
        for wp_i in range(len(self.base_waypoints)):
            # make sure waypoint is ahead of car
            w = self.base_waypoints[wp_i]
            w_x = w.pose.pose.position.x
            w_y = w.pose.pose.position.y
            w_ahead = ((w_x - c_x) * math.cos(c_w) +
                       (w_y - c_y) * math.sin(c_w)) > 0

            if not w_ahead:
                continue

            # calculate distance and store if closer than current
            w_d = math.sqrt((c_x - w_x)**2 + (c_y - w_y)**2)
            waypoints_ahead.append((w, w_d))

        waypoints_ahead = sorted(waypoints_ahead, key=itemgetter(1))[:LOOKAHEAD_WPS]
        wps = [wp[0] for wp in waypoints_ahead]
        lane = Lane()
        lane.waypoints = wps
        self.final_waypoints_pub.publish(lane)

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
