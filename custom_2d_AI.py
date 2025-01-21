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

# ZeroMQ Context and Socket
context = zmq.Context()
socket = context.socket(zmq.PULL)  # PULL socket to receive frames
socket.connect("tcp://192.168.137.172:5550")  # Connect to sender's address

# Poller setup
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

logger = logging.getLogger(__name__)


class CustomDetector(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)

    label = "Custom Detector"

    # Use the same identifier as the built-in 2D pupil detector
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

    def _stop_other_pupil_detectors(self):
        plugin_list = self.g_pool.plugins

        # Deactivate other PupilDetectorPlugin instances
        for plugin in plugin_list:
            if isinstance(plugin, PupilDetectorPlugin) and plugin is not self:
                plugin.alive = False

        # Force Plugin_List to remove deactivated plugins
        plugin_list.clean()

    def detect(self, frame, **kwargs):

        debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None

        result = {'ellipse': None, 'diameter': None, 'location': None, 'confidence': 0, 'id': 0, 'topic': None,
                  'method': None, 'timestamp': None, 'norm_pos': None}

        eye_id = self.g_pool.eye_id

        if eye_id == 0:
            x, y, w, h, angle = self.receive_info()
            result['ellipse'] = {'center': (x, y),
                                 'axes': (w, h),
                                 'angle': angle}
            result['diameter'] = w * 2  # The diameter of the circle
            result['location'] = (x, y)  # The center of the circle
            result['confidence'] = 1  # Confidence is set to 1 for now
            # print(x, y, w, h, angle)

        result["id"] = eye_id
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        result["method"] = "custom-2d"
        result["timestamp"] = frame.timestamp
        if result['location'] is not None:
            result["norm_pos"] = normalize(result["location"], (frame.width, frame.height), flip_y=True)

        ##with open(r'C:\Users\L1303\Desktop\pupilSource\output.txt', 'w') as f:
          ##  f.write(str(result))

        return result

    def receive_info(self):
        # TODO the information scaled to network target size(320,240). Need to scale original image size
        # Original and new resolutions
        original_width, original_height = 320, 240
        new_width, new_height = 400, 400

        # Scaling factors
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # Poll the socket with a timeout of 1000ms (1 second)
        events = dict(poller.poll(100))

        if socket in events:
            # Receive JSON data from the sender
            message = socket.recv_json()

            # Extract data from the received message
            center_x = message['x']*scale_x
            center_y = message['y']*scale_y
            major_axis = message['w']*scale_x
            minor_axis = message['h']*scale_y
            angle = message['angle']
            # Print the received data
            # print("Received Ellipse Data:")
            # print(f"Center: ({center_x}, {center_y})")
            # print(f"Major Axis Length: {major_axis}")
            # print(f"Minor Axis Length: {minor_axis}")
            # print(f"Angle: {angle}\n")
            return center_x, center_y, major_axis, minor_axis, angle
        else:
            # No data received within the timeout period
            # print("Waiting for data...")
            return 0, 0, 0, 0, 0

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text("Custom 2D Pupil Detector Plugin")
        self.menu.append(info)

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(self._recent_detection_result, color_rgb=(0.3, 1.0, 0.1))
