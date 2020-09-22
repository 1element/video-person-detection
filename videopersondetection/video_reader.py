import logging
import cv2

SKIP_FRAMES_DEFAULT = 15


class VideoReader:
    def __init__(self, video_file_path, config):
        self.config = config
        self._init_video_capture(video_file_path)

    def _init_video_capture(self, video_file_path):
        self.frame_index = 0
        self.video_capture = cv2.VideoCapture(video_file_path)
        if not self.video_capture.isOpened():
            raise ValueError('Video file could not be read: ' + video_file_path)
        self.total_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.debug('Total frame count: %s', self.total_frame_count)

    def current_frame_index(self):
        current_frame_index = self.frame_index - self.config.get('skip_frames', SKIP_FRAMES_DEFAULT)
        return max(current_frame_index, 0)

    def next_frame(self):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        has_frame, frame = self.video_capture.read()
        if has_frame and self.frame_index <= self.total_frame_count:
            self.frame_index += self.config.get('skip_frames', SKIP_FRAMES_DEFAULT)
            return has_frame, frame

        return False, None

    def release(self):
        self.video_capture.release
