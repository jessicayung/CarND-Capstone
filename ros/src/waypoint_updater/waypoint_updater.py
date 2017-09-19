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

        #Select the base waypoints ahead of the vehicle
        self.waypoints_ahead = []

        #Id of the first waypoint ahead
        self.next_waypoint_id = None

        #Boolean that specifies if the map has been loaded at least once
        self.simulator_started = False

        self.main_loop()
        # rospy.spin()


    def main_loop(self):
        rate = rospy.Rate(20) #20Hz
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints: 
                # get the pose of the vehicle
                c_x = self.pose.position.x
                c_y = self.pose.position.y
                c_o = self.pose.orientation
                c_q = (c_o.x, c_o.y, c_o.z, c_o.w)
                _, _, c_t = tf.transformations.euler_from_quaternion(c_q)

                #Waipoints ahead of the car
                waypoints_ahead = []

                #Define the starting and ending index of the waypoints
                start = 0
                end = len(self.base_waypoints)
                upsampling = 100
                #if self.next_waypoint_id != None:
                    #start = self.next_waypoint_id
                    #end = min(start+LOOKAHEAD_WPS+1, len(self.base_waypoints))
                
                set_next_waypoint_id = True
                for wp_i in range(start, end, upsampling):
                    # make sure waypoint is ahead of car
                    w = self.base_waypoints[wp_i]
                    w_x = w.pose.pose.position.x
                    w_y = w.pose.pose.position.y
                    w_ahead = ((w_x - c_x) * math.cos(c_t) +
                               (w_y - c_y) * math.sin(c_t)) > 0.0

                    if not w_ahead:
                        continue

                    #Retrieve the Id of the first point ahead 
                    if set_next_waypoint_id:
                        self.next_waypoint_id = wp_i
                        set_next_waypoint_id = False

                    # calculate distance and store if closer than current
                    w_d = math.sqrt((c_x - w_x)**2 + (c_y - w_y)**2)
                    waypoints_ahead.append((w, w_d))

                    # if len(waypoints_ahead) >= LOOKAHEAD_WPS:
                    #     break

                waypoints_ahead = sorted(waypoints_ahead, key=itemgetter(1))[:LOOKAHEAD_WPS]
                # waypoints_ahead = sorted(waypoints_ahead, key=itemgetter(1))
                self.waypoints_ahead = waypoints_ahead
                wps = [wp[0] for wp in waypoints_ahead]
                lane = Lane()
                lane.waypoints = wps
                self.final_waypoints_pub.publish(lane)

            rate.sleep()

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


    def pose_cb(self, msg):
        """
            /current_pose callback function
            NB: Rate unknown => the processing may need to be done
            elsewhere
        """
        self.pose = msg.pose

        if not self.pose:
            rospy.logwarn('no pose has been set')
            return None

        if not self.base_waypoints:
            rospy.logwarn('no waypoints have been set')
            return None


    def waypoints_cb(self, lane):
        """
        /base_waypoints Callback function
        """
        self.base_waypoints = lane.waypoints
        rospy.logwarn('[waypoints_cb]')

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message.
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message.
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
