#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np
from scipy.misc import imresize

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        # /current_pose can be used to determine the vehicle's location
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)

        # /base_waypoints provides a complete list of waypoints for the course
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # /image_color provides the image stream from the car's camera
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        # /traffic_waypoint is the topic to publish to. It expects the index of the
        # waypoint for nearest upcoming red light
        self.redl_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        self.gt_sub = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)


        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.light_classifier = None
        self.light_classifier = TLClassifier()

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.redl_pub.publish(Int32(light_wp))
        else:
            self.redl_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            (float, int): tuple of the distance and index of
            closest waypoint in self.waypoints
        """
        x = pose.position.x
        y = pose.position.y

        closest_wp = (float('inf'), -1)
        for wp_i in range(len(self.waypoints)):
            wx = self.waypoints[wp_i].pose.pose.position.x
            wy = self.waypoints[wp_i].pose.pose.position.y
            dist = math.sqrt((x - wx)**2 + (y - wy)**2)
            if dist < closest_wp[0]:
                closest_wp = (dist, wp_i)

        rospy.logdebug("Closest waypoint distance is {}, "
                       "which is waypoint {}".format(closest_wp[0], closest_wp[1]))

        return closest_wp[1]

    def get_closest_stop_waypoint(self, position):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            (float, int): tuple of the distance and index of
            closest waypoint in self.waypoints
        """
        x = position[0]
        y = position[1]

        closest_wp = (float('inf'), -1)
        for wp_i in range(len(self.waypoints)):
            wx = self.waypoints[wp_i].pose.pose.position.x
            wy = self.waypoints[wp_i].pose.pose.position.y
            dist = math.sqrt((x - wx)**2 + (y - wy)**2)
            if dist < closest_wp[0]:
                closest_wp = (dist, wp_i)

        rospy.logdebug("Closest waypoint distance is {}, "
                       "which is waypoint {}".format(closest_wp[0], closest_wp[1]))
        return closest_wp[1]

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")
            return (0, 0)

        # Use tranform and rotation to calculate 2D position of light in image
        trans_m = self.listener.fromTranslationRotation(trans, rot)
        pt_world = np.array([[point_in_world.x],
                             [point_in_world.y],
                             [point_in_world.z],
                             [1.0]])
        cam_vec = np.dot(trans_m, pt_world)
        x = cam_vec[0][0]
        y = cam_vec[1][0]
        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image or not light:
            self.prev_light_loc = None
            rospy.logwarn("image {} and light {}".format(self.has_image, light))
            return TrafficLight.UNKNOWN

        # get image from msg into cv2
        try:
            # get rgb! cv handles different format
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        except:
            pass

        #TODO: use light location to zoom in on traffic light in image
        x, y = self.project_to_image_plane(light.pose.pose.position)
        #rospy.logerr("Project to image plane returned x={}, y={}".format(x, y))

        # Get classification
        image = imresize(cv_image, (224, 224, 3))
        return self.light_classifier.get_classification(image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # prevent from running function if values are not set
        idx, tl_state = -1, TrafficLight.UNKNOWN
        if not self.pose:
            rospy.logwarn('no pose has been set')
            return idx, tl_state

        if not self.waypoints:
            rospy.logwarn('no waypoints have been set')
            return idx, tl_state

        if not self.lights:
            rospy.logwarn('no lights have been set')
            return idx, tl_state

        if not self.light_classifier:
            rospy.logwarn('no classifier initialized')
            return idx, tl_state

        # get current position and yaw of the car
        c_wp = self.get_closest_waypoint(self.pose)
        c_x = self.pose.position.x
        c_y = self.pose.position.y
        c_o = self.pose.orientation
        c_q = (c_o.x, c_o.y, c_o.z, c_o.w)
        _, _, c_w = tf.transformations.euler_from_quaternion(c_q)

        # find closest light ahead
        l_wp = (float('inf'), -1, None)
        for i in range(len(self.lights)):
            l = self.lights[i]
            l_x = l.pose.pose.position.x
            l_y = l.pose.pose.position.y
            l_o = l.pose.pose.orientation
            l_q = (l_o.x, l_o.y, l_o.z, l_o.w)
            _, _, l_w = tf.transformations.euler_from_quaternion(l_q)

            # determine if light is ahead
            l_ahead = ((l_x - c_x) * math.cos(c_w) +
                       (l_y - c_y) * math.sin(c_w)) > 0
            if not l_ahead:
                rospy.logdebug("light not ahead " + str(self.get_closest_waypoint(l.pose.pose)))
                continue
            rospy.logdebug("light ahead " + str(self.get_closest_waypoint(l.pose.pose)))

            # determine if light is facing car
            l_facing_car = l_w * c_w > 0
            if not l_facing_car:
                rospy.logdebug("light not facing " + str(self.get_closest_waypoint(l.pose.pose)))
                continue
            rospy.logdebug("light facing " + str(self.get_closest_waypoint(l.pose.pose)))

            # calculate distance and store if closer than current
            l_d = math.sqrt((c_x - l_x)**2 + (c_y - l_y)**2)
            rospy.logdebug("Store light {} with distance {} and position {}, {}".format(i, l_d, l_x, l_y))
            if l_d < l_wp[0]:
                l_wp = l_d, self.get_closest_waypoint(l.pose.pose), l

        # waypoints that correspond to the stopping line positions
        stop_line_positions = self.config['stop_line_positions']
        rospy.logdebug("stop line positions ({})".format(stop_line_positions))
        s_wp = (float('inf'), -1)
        for stop_line in stop_line_positions:
            s_x = stop_line[0]
            s_y = stop_line[1]
            s_d = math.sqrt((c_x - s_x)**2 + (c_y - s_y)**2)
            if s_d < s_wp[0]:
                s_wp = (s_d, self.get_closest_stop_waypoint(stop_line))

        rospy.logdebug("closest light is " + str(l_wp[1]))
        rospy.logdebug("closest stop line is " + str(s_wp[1]))
        rospy.logdebug("light state is " + str(tl_state))

        idx = s_wp[1]
        tl_state = self.get_light_state(l_wp[2])

        return idx, tl_state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
