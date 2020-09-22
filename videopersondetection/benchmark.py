from timeit import default_timer as timer
from statistics import mean
from videopersondetection.video_reader import VideoReader
from videopersondetection.tflite_person_detector import TFLitePersonDetector
from videopersondetection.image_helper import ImageHelper

SKIP_FRAMES = 15
VIDEO_FILE = 'benchmark.mkv'

image_helper = ImageHelper()
tflite_person_detector = TFLitePersonDetector()


def _processing_time(with_inference):
    start = timer()
    video_reader = VideoReader(VIDEO_FILE, {'skip_frames': SKIP_FRAMES})
    while True:
        has_frame, frame = video_reader.next_frame()

        if not has_frame:
            break

        image = image_helper.create_from_frame(frame)
        if with_inference:
            tflite_person_detector.detect_person(image)

    video_reader.release()
    return timer() - start


def _average_inference_time():
    video_reader = VideoReader(VIDEO_FILE, {})
    has_frame, frame = video_reader.next_frame()

    inference_times = []
    if has_frame:
        image = image_helper.create_from_frame(frame)
        for i in range(0, 5):
            start = timer()
            tflite_person_detector.detect_person(image)
            inference_times.append(timer() - start)

    video_reader.release()
    return mean(inference_times)


def main():
    print("Executing simple benchmark...")

    duration = _processing_time(False)
    print(f"Processing video file with SKIP_FRAMES = {SKIP_FRAMES} took {duration:.2f} seconds (without inference).")

    duration = _processing_time(True)
    print(f"Processing video file with SKIP_FRAMES = {SKIP_FRAMES} took {duration:.2f} seconds (with inference).")

    average_inference_time = _average_inference_time()
    print(f"Average frame inference time: {average_inference_time*1000:.2f}ms.")


if __name__ == '__main__':
    main()
