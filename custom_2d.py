import logging
import cv2
import numpy as np

from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import ui

from methods import normalize

from pupil_detector_plugins import available_detector_plugins
from pupil_detector_plugins.detector_base_plugin import (
    PupilDetectorPlugin,
)
from pupil_detector_plugins.visualizer_2d import draw_pupil_outline

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

        frame_data = np.asarray(bytearray(frame.jpeg_buffer), dtype=np.uint8)
        frame_bgr = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Apply a blur to reduce noise
        blurred_frame = cv2.GaussianBlur(frame_gray, (7, 7), 0)

        # Use Canny Edge Detection
        edges = cv2.Canny(blurred_frame, 100, 200)

        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = {'ellipse': None, 'diameter': None, 'location': None, 'confidence': 0, 'id': 0, 'topic': None,
                  'method': None, 'timestamp': None, 'norm_pos': None}

        if contours:
            for contour in contours:
                # Calculate the area of the contour
                area = cv2.contourArea(contour)

                # Filter out very small contours based on their area
                if area > 100:
                    # Fit a circle to the contour
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)

                    # Draw the circle on the image
                    cv2.circle(frame_bgr, center, radius, (0, 255, 0), 2)

                    # Update the result
                    result['ellipse'] = {'center': (x, y), 'axes': (radius, radius), 'angle': 0}
                    result['diameter'] = radius * 2  # The diameter of the circle
                    result['location'] = (x, y)  # The center of the circle
                    result['confidence'] = 1  # Confidence is set to 1 for now

        eye_id = self.g_pool.eye_id

        result["id"] = eye_id
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        result["method"] = "custom-2d"
        result["timestamp"] = frame.timestamp
        if result['location'] is not None:
            result["norm_pos"] = normalize(result["location"], (frame.width, frame.height), flip_y=True)

        ##with open(r'C:\Users\L1303\Desktop\pupilSource\output.txt', 'w') as f:
          ##  f.write(str(result))

        return result

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text("Custom 2D Pupil Detector Plugin")
        self.menu.append(info)

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(self._recent_detection_result, color_rgb=(0.3, 1.0, 0.1))
