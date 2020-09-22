import logging

from videopersondetection.tflite_person_detector import TFLitePersonDetector
from videopersondetection.image_helper import ImageHelper
from videopersondetection.area_filter import AreaFilter
from videopersondetection.zone_filter import ZoneFilter
from videopersondetection.video_reader import VideoReader
from videopersondetection.frame_writer import FrameWriter


class VideoPersonDetector:
    def __init__(self, config):
        self.config = config
        self.zone_filter = ZoneFilter(config)
        self.area_filter = AreaFilter(config)
        self.frame_writer = FrameWriter(config)
        self.image_helper = ImageHelper()
        self.tflite_person_detector = TFLitePersonDetector()

    def detect_person(self, video_file_path, camera_id):
        video_reader = VideoReader(video_file_path, self.config)

        highest_person_confidence_result = 0
        while True:
            has_frame, frame = video_reader.next_frame()

            if not has_frame:
                break

            image = self.image_helper.create_from_frame(frame)
            result = self.tflite_person_detector.detect_person(image)
            logging.debug('Inference #1 for frame %s: %s', video_reader.current_frame_index(), result)

            if result is None:
                continue

            if result.confidence < self.config.get('min_confidence_threshold', 0.1):
                logging.debug('Skipping frame, confidence is below threshold.')
                continue

            self.frame_writer.debug_output_frame(frame, video_reader.current_frame_index(), camera_id, result)

            if self.area_filter.below_min_area_threshold(camera_id, result.bounding_box):
                logging.debug('Skipping frame, area %s is below threshold.', result.bounding_box.area)
                continue

            if self.zone_filter.inside_ignore_zone(camera_id, result.bounding_box):
                logging.debug('Skipping frame, bounding box is inside ignore zone.')
                continue

            if result.confidence < self.config.get('second_run_threshold', 0.5):
                cropped_image = self.image_helper.crop(image, result.bounding_box)
                result = self.tflite_person_detector.detect_person(cropped_image)
                logging.debug('Inference #2 for frame %s: %s', video_reader.current_frame_index(), result)

            if result and result.confidence > highest_person_confidence_result:
                highest_person_confidence_result = result.confidence

            if highest_person_confidence_result >= self.config.get('skip_further_processing_threshold', 0.9):
                logging.debug('Skip further processing, confidence is high enough.')
                break

        video_reader.release()
        return highest_person_confidence_result
