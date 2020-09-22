import cv2
import numpy as np


class ZoneFilter:
    def __init__(self, config):
        self.config = config

    def _zone_coordinates(self, camera_id):
        return self.config.get('camera', {}).get(camera_id, {}).get('ignore_zone_coordinates', None)

    def zone_contour(self, camera_id):
        zone_coordinates = self._zone_coordinates(camera_id)
        if zone_coordinates is not None:
            zone_points = zone_coordinates.split(',')
            return np.array([[int(zone_points[i]), int(zone_points[i+1])] for i in range(0, len(zone_points), 2)])

    def inside_ignore_zone(self, camera_id, bounding_box):
        zone_contour = self.zone_contour(camera_id)
        return zone_contour is not None and cv2.pointPolygonTest(zone_contour, bounding_box.center, False) > 0
