#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import threading
import cv2
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

PACKAGE_DIR = rospkg.RosPack().get_path("lab_vision_sim_task4")
SCRIPTS_DIR = os.path.join(PACKAGE_DIR, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import vision_core
MISSIONS = {
    "color": {
        "waypoints": [
            {
                "name": "red",
                "x": 1.0,
                "y": -1.0,
                "expected": "red",
            },
            {
                "name": "green",
                "x": 1.0,
                "y": 0.0,
                "expected": "green",
            },
            {
                "name": "blue",
                "x": 1.0,
                "y": 1.0,
                "expected": "blue",
            },
        ],
        "land_at": (1.0, 1.0),
        "scan_z": 0.9,
    },

    "qr": {
        "waypoints": [
            {
                "name": "qr",
                "x": 1.0,
                "y": 0.0,
                "expected": None,
            },
        ],
        "land_at": (1.0, 0.0),
        "scan_z": 1.0,
    },

    "apriltag": {
        "waypoints": [
            {
                "name": "apriltag",
                "x": 1.0,
                "y": 0.0,
                "expected": None,
            },
        ],
        "land_at": (1.0, 0.0),
        "scan_z": 0.9,
    },

    "ocr": {
        "waypoints": [
            {
                "name": "ocr",
                "x": 1.0,
                "y": 0.0,
                "expected": "AIRHUST",
            },
        ],
        "land_at": (1.0, 0.0),
        "scan_z": 0.8,
    },

    "logo": {
        "waypoints": [
            {
                "name": "logo",
                "x": 1.0,
                "y": 0.0,
                "expected": None,
            },
        ],
        "land_at": (1.0, 0.0),
        "scan_z": 0.9,
    },
}
class VisionMission:
    def __init__(self, mission_name):
        self.mission_name = mission_name
        self.config = MISSIONS[mission_name]

        self.err_max = float(
            rospy.get_param("~err_max", 0.15)
        )
        self.if_debug = int(
            rospy.get_param("~if_debug", 1)
        )
        self.scan_period = float(
            rospy.get_param("~scan_period", 0.6)
        )
        self.image_timeout = float(
            rospy.get_param("~image_timeout", 0.8)
        )

        self.state = "WAIT_CONNECTION"
        self.waypoint_index = 0

        self.mavros_state = State()
        self.odom = None

        self.latest_image = None
        self.latest_image_time = None
        self.image_lock = threading.Lock()

        self.stable_value = None
        self.stable_count = 0

        self.pending_result = None
        self.pending_lock = threading.Lock()
        self.scan_busy = False

        self.last_request_time = rospy.Time(0)
        self.land_start_time = None
        self.mission_results = []

        self.target = (0.0, 0.0, 0.6)

        self.bridge = CvBridge()

        self.create_interfaces()
        self.recognizer = self.create_recognizer()

    def create_interfaces(self):
        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_raw/local",
            PositionTarget,
            queue_size=10,
        )

        self.mode_client = rospy.ServiceProxy(
            "/mavros/set_mode",
            SetMode,
        )
        self.arming_client = rospy.ServiceProxy(
            "/mavros/cmd/arming",
            CommandBool,
        )

        rospy.Subscriber(
            "/mavros/state",
            State,
            self.state_callback,
            queue_size=10,
        )

        rospy.Subscriber(
            "/mavros/local_position/odom",
            Odometry,
            self.odom_callback,
            queue_size=10,
        )

        rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        self.setpoint = PositionTarget()
        self.setpoint.header.frame_id = "map"
        self.setpoint.coordinate_frame = (
            PositionTarget.FRAME_LOCAL_NED
        )
        self.setpoint.type_mask = (
            PositionTarget.IGNORE_VX
            | PositionTarget.IGNORE_VY
            | PositionTarget.IGNORE_VZ
            | PositionTarget.IGNORE_AFX
            | PositionTarget.IGNORE_AFY
            | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW_RATE
        )

    def create_recognizer(self):
        if self.mission_name == "color":
            return vision_core.recognize_color

        if self.mission_name == "qr":
            return vision_core.recognize_qr

        if self.mission_name == "apriltag":
            return lambda image: (
                vision_core.recognize_apriltag(
                    image,
                    family="36h11",
                )
            )

        if self.mission_name == "ocr":
            return vision_core.recognize_ocr

        weights = rospy.get_param(
            "~logo_weights",
            "",
        )
        if not weights:
            raise ValueError(
                "logo mission requires "
                "_logo_weights:=/path/to/best.pt"
            )

        return lambda image: (
            vision_core.recognize_logo(
                image,
                weights=weights,
                conf=float(
                    rospy.get_param("~logo_conf", 0.5)
                ),
                iou=float(
                    rospy.get_param("~logo_iou", 0.45)
                ),
            )
        )

    def state_callback(self, message):
        self.mavros_state = message

    def odom_callback(self, message):
        self.odom = message

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )

            with self.image_lock:
                self.latest_image = image
                self.latest_image_time = rospy.Time.now()

        except CvBridgeError as exc:
            rospy.logwarn_throttle(
                2.0,
                "cv_bridge error: %s",
                exc,
            )

    def publish_setpoint(self):
        x, y, z = self.target

        self.setpoint.header.stamp = rospy.Time.now()
        self.setpoint.position.x = x
        self.setpoint.position.y = y
        self.setpoint.position.z = z
        self.setpoint.yaw = 0.0

        self.setpoint_pub.publish(self.setpoint)

    def reached(self, target=None):
        if self.odom is None:
            return False

        if target is None:
            target = self.target

        x, y, z = target
        position = self.odom.pose.pose.position

        return (
            abs(position.x - x) < self.err_max
            and abs(position.y - y) < self.err_max
            and abs(position.z - z) < self.err_max
        )

    def request_control(self):
        now = rospy.Time.now()

        if now - self.last_request_time < rospy.Duration(1.0):
            return

        self.last_request_time = now

        try:
            if self.mavros_state.mode != "OFFBOARD":
                if self.if_debug == 1:
                    response = self.mode_client(
                        custom_mode="OFFBOARD"
                    )
                    rospy.loginfo(
                        "OFFBOARD request: %s",
                        response.mode_sent,
                    )
                else:
                    rospy.loginfo(
                        "Waiting for remote OFFBOARD"
                    )

            elif not self.mavros_state.armed:
                response = self.arming_client(value=True)
                rospy.loginfo(
                    "Arm request: %s",
                    response.success,
                )

        except rospy.ServiceException as exc:
            rospy.logwarn(
                "MAVROS service error: %s",
                exc,
            )

    def current_waypoint(self):
        return self.config["waypoints"][
            self.waypoint_index
        ]

    def current_waypoint_target(self):
        waypoint = self.current_waypoint()

        return (
            waypoint["x"],
            waypoint["y"],
            self.config["scan_z"],
        )

    def accept_result(self, value):
        waypoint = self.current_waypoint()
        expected = waypoint["expected"]

        if expected is not None and value != expected:
            rospy.logwarn_throttle(
                1.0,
                "expected=%s recognized=%s",
                expected,
                value,
            )
            self.stable_value = None
            self.stable_count = 0
            return

        rospy.loginfo(
            "MISSION_RESULT mission=%s waypoint=%s value=%s",
            self.mission_name,
            waypoint["name"],
            value,
        )

        self.mission_results.append(
            {
                "waypoint": waypoint["name"],
                "value": value,
            }
        )

        self.waypoint_index += 1
        self.stable_value = None
        self.stable_count = 0

        if self.waypoint_index < len(
            self.config["waypoints"]
        ):
            self.target = self.current_waypoint_target()
            self.state = "CRUISE"
            return

        land_x, land_y = self.config["land_at"]
        self.target = (
            land_x,
            land_y,
            self.config["scan_z"],
        )
        self.state = "TO_LAND"

    def consume_pending_result(self):
        with self.pending_lock:
            pending = self.pending_result
            self.pending_result = None

        if pending is None:
            return

        waypoint_index, value = pending

        if waypoint_index != self.waypoint_index:
            return

        if not value:
            self.stable_value = None
            self.stable_count = 0
            return

        if value == self.stable_value:
            self.stable_count += 1
        else:
            self.stable_value = value
            self.stable_count = 1

        rospy.loginfo_throttle(
            1.0,
            "scan value=%s stable=%d/2",
            value,
            self.stable_count,
        )

        if self.stable_count >= 2:
            self.accept_result(value)

    def advance_state(self):
        if self.state == "WAIT_CONNECTION":
            if self.mavros_state.connected:
                rospy.loginfo("MAVROS connected")
                self.state = "ARM_OFFBOARD"

        elif self.state == "ARM_OFFBOARD":
            if (
                self.mavros_state.mode == "OFFBOARD"
                and self.mavros_state.armed
            ):
                rospy.loginfo("OFFBOARD and armed")
                self.state = "TAKEOFF"

                self.target = (
                    0.0,
                    0.0,
                    self.config["scan_z"],
                )
            else:
                self.request_control()

        elif self.state == "TAKEOFF":
            if self.reached():
                rospy.loginfo("takeoff reached")
                self.target = self.current_waypoint_target()
                self.state = "CRUISE"

        elif self.state == "CRUISE":
            if self.reached():
                waypoint = self.current_waypoint()
                rospy.loginfo(
                    "waypoint %s reached; inspecting",
                    waypoint["name"],
                )
                self.stable_value = None
                self.stable_count = 0
                self.state = "INSPECT"

        elif self.state == "INSPECT":
            self.consume_pending_result()

        elif self.state == "TO_LAND":
            if self.reached():
                rospy.loginfo(
                    "landing position reached"
                )
                self.target = (
                    self.target[0],
                    self.target[1],
                    -0.15,
                )
                self.land_start_time = rospy.Time.now()
                self.state = "LAND"

        elif self.state == "LAND":
            if (
                self.land_start_time is not None
                and rospy.Time.now()
                - self.land_start_time
                > rospy.Duration(5.0)
            ):
                rospy.loginfo("landing complete")
                rospy.loginfo(
                    "mission results: %s",
                    self.mission_results,
                )
                self.state = "DONE"

        elif self.state == "DONE":
            rospy.signal_shutdown("mission complete")

    def control_loop(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            self.publish_setpoint()

            try:
                self.advance_state()
            except Exception as exc:
                rospy.logerr_throttle(
                    1.0,
                    "state error: %s",
                    exc,
                )

            rate.sleep()

    def scan_timer(self, _event):
        if self.state != "INSPECT":
            return

        if self.scan_busy:
            return

        with self.image_lock:
            image = self.latest_image
            image_time = self.latest_image_time

        if image is None:
            rospy.loginfo_throttle(
                1.0,
                "scan: waiting for camera image",
            )
            return

        if image_time is None:
            rospy.loginfo_throttle(
                1.0,
                "scan: image has no timestamp",
            )
            return

        image_age = rospy.Time.now() - image_time

        if image_age > rospy.Duration(self.image_timeout):
            rospy.loginfo_throttle(
                1.0,
                "scan: stale image age=%.2f",
                image_age.to_sec(),
            )
            return

        self.scan_busy = True
        waypoint_index = self.waypoint_index
        image = image.copy()

        try:
            value = self.recognizer(image)

            rospy.loginfo_throttle(
                0.8,
                "scan: result=%r",
                value,
            )

            if value is None and self.mission_name == "color":
                height, width = image.shape[:2]
                cx = width // 2
                cy = height // 2

                patch = image[
                    cy - 5:cy + 6,
                    cx - 5:cx + 6,
                ]

                b, g, r = patch.reshape(-1, 3).mean(axis=0)

                rospy.loginfo_throttle(
                    0.8,
                    "scan: center BGR=(%.1f, %.1f, %.1f)",
                    b,
                    g,
                    r,
                )

        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "recognizer error: %s",
                exc,
            )
            value = None

        finally:
            self.scan_busy = False

        with self.pending_lock:
            self.pending_result = (
                waypoint_index,
                value,
            )
    def run(self):
        self.control_thread = threading.Thread(
            target=self.control_loop,
            daemon=True,
        )
        self.control_thread.start()

        self.scan_timer_handle = rospy.Timer(
            rospy.Duration(self.scan_period),
            self.scan_timer,
        )

        rospy.loginfo(
            "mission=%s waypoints=%s",
            self.mission_name,
            self.config["waypoints"],
        )

        rospy.spin()


def main():
    arguments = rospy.myargv(argv=sys.argv)

    if len(arguments) < 2:
        print(
            "usage: vision_mission.py "
            "color|qr|apriltag|ocr|logo"
        )
        return 1

    mission_name = arguments[1].strip().lower()

    if mission_name not in MISSIONS:
        print("unknown mission:", mission_name)
        return 1

    rospy.init_node("vision_mission")

    mission = VisionMission(mission_name)
    mission.run()

    return 0
if __name__=="__main__":
    sys.exit(main())
