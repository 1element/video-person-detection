
class AreaFilter:
    def __init__(self, config):
        self.config = config

    def _min_area_threshold(self, camera_id):
        return self.config.get('camera', {}).get(camera_id, {}).get('min_area_threshold', 0)

    def below_min_area_threshold(self, camera_id, bounding_box):
        return bounding_box.area < self._min_area_threshold(camera_id)
