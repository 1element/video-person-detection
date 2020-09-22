import cv2

from videopersondetection.zone_filter import ZoneFilter


class FrameWriter:
    def __init__(self, config):
        self.config = config
        self.zone_filter = ZoneFilter(config)

    def _draw_zone_contour(self, frame, camera_id):
        zone_contour = self.zone_filter.zone_contour(camera_id)
        if zone_contour is not None:
            color_red = (0, 0, 255)
            line_thickness = 2
            cv2.drawContours(frame, [zone_contour], -1, color_red, line_thickness)

    def _draw_bounding_box(self, frame, bounding_box):
        color_green = (0, 255, 0)
        line_thickness = 2
        cv2.rectangle(frame, (bounding_box.xmin, bounding_box.ymin), (bounding_box.xmax, bounding_box.ymax), color_green, line_thickness)

    def _draw_bounding_box_center_circle(self, frame, bounding_box):
        color_green = (0, 255, 0)
        radius = 7
        filled = -1
        cv2.circle(frame, bounding_box.center, radius, color_green, filled)

    def _write_image(self, frame, frame_index):
        cv2.imwrite(self.config.get('debug_output_path', 'debug_output/') + 'frame-' + str(frame_index) + '.png', frame)

    def debug_output_frame(self, frame, frame_index, camera_id, result):
        if self.config.get('debug_output_frames_enabled', False):
            self._draw_zone_contour(frame, camera_id)
            self._draw_bounding_box(frame, result.bounding_box)
            self._draw_bounding_box_center_circle(frame, result.bounding_box)
            self._write_image(frame, frame_index)
