import logging
import cv2
import numpy as np

from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import ui
from methods import normalize

from pupil_detector_plugins import PupilDetectorPlugin
from pupil_detector_plugins.detector_base_plugin import (
    DetectorPropertyProxy,
    PupilDetectorPlugin,
)
from pupil_detector_plugins.visualizer_2d import draw_pupil_outline
import zmq

logger = logging.getLogger(__name__)

# ZeroMQ Context and Socket (reuse them or create them inside your plugin)
context = zmq.Context()
socket = context.socket(zmq.PULL)  # PULL socket to receive frames
socket.connect("tcp://192.168.137.172:5550")  # Connect to sender's address

# Poller setup
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)


class CustomDetector(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)

    label = "Custom Detector"
    identifier = "2d"
    order = 0.9

    @property
    def pretty_class_name(self):
        return "Custom Detector"

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.__detector_2d

    def __init__(self, g_pool=None):
        super().__init__(g_pool=g_pool)
        self.__detector_2d = Detector2D({})
        self._stop_other_pupil_detectors()

        # Store the last known, scaled pupil info for each eye
        # Default to (x, y, major_axis, minor_axis, angle) = (0, 0, 0, 0, 0)
        self._pupil_info = {
            0: (0, 0, 0, 0, 0),
            1: (0, 0, 0, 0, 0)
        }

        # Original vs new image size (to scale data properly)
        self.original_width = 320
        self.original_height = 240
        self.new_width = 400
        self.new_height = 400

        # Precompute scaling factors
        self.scale_x = self.new_width / self.original_width
        self.scale_y = self.new_height / self.original_height

        # Keep references to socket/poller if you wish
        self._socket = socket
        self._poller = poller

    def _stop_other_pupil_detectors(self):
        plugin_list = self.g_pool.plugins
        for plugin in plugin_list:
            if isinstance(plugin, PupilDetectorPlugin) and plugin is not self:
                plugin.alive = False
        plugin_list.clean()

    def detect(self, frame, **kwargs):
        """
        Called every frame to detect pupil. Returns dict with ellipse, center, etc.
        """
        result = {
            'ellipse': None,
            'diameter': None,
            'location': None,
            'confidence': 0,
            'id': 0,
            'topic': None,
            'method': None,
            'timestamp': None,
            'norm_pos': None
        }

        eye_id = self.g_pool.eye_id
        (x, y, w, h, temp_angle) = self.receive_info(eye_id)

        result['ellipse'] = {
            'center': (x, y),
            'axes': (w, h),
            'angle': temp_angle
        }
        result['diameter'] = w
        result['location'] = (x, y)
        result['confidence'] = 1  # Hard-coded for demonstration

        result["id"] = eye_id
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        result["method"] = "custom-2d"
        result["timestamp"] = frame.timestamp

        if result['location'] is not None:
            result["norm_pos"] = normalize(
                result["location"], (frame.width, frame.height), flip_y=True
            )

        return result

    def receive_info(self, eye_id: int):
        """
        - Polls the ZMQ socket (non-blocking).
        - If new data is available, updates self._pupil_info for both eyes.
        - Returns the last known pupil info for the given eye_id.
        """
        default_result = (0, 0, 0, 0, 0)
        events = dict(self._poller.poll(0))  # Non-blocking poll

        if self._socket in events:
            # We have new data
            message = self._socket.recv_json()

            # For each eye (0 and 1), update if info is present
            for i in [0, 1]:
                key = f"info{i}"
                if key in message and message[key] is not None:
                    raw_center = message[key][0]  # (cx, cy)
                    raw_axes   = message[key][1]  # (major_axis, minor_axis)
                    raw_angle  = message[key][2]  # angle

                    # Scale the data
                    scaled_cx = raw_center[0] * self.scale_x
                    scaled_cy = raw_center[1] * self.scale_y
                    scaled_maj = raw_axes[0] * self.scale_y
                    scaled_min = raw_axes[1] * self.scale_x
                    angle = raw_angle

                    # Store in the plugin instance
                    self._pupil_info[i] = (scaled_cx, scaled_cy, scaled_maj, scaled_min, angle)

        # Return last known data for this eye
        return self._pupil_info.get(eye_id, default_result)

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text("Custom 2D Pupil Detector Plugin")
        self.menu.append(info)

    def gl_display(self):
        if self._recent_detection_result:
            # Visualize the detected ellipse if desired
            from pupil_detector_plugins.visualizer_2d import draw_pupil_outline
            draw_pupil_outline(self._recent_detection_result, color_rgb=(0.3, 1.0, 0.1))
