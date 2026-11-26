#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import threading

import rospy
from cv_bridge import CvBridge, CvBridgeError
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

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

