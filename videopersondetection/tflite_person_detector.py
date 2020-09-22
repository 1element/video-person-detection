import numpy as np
import collections
import tflite_runtime.interpreter as tflite
import platform

from PIL import Image
from videopersondetection.bounding_box import BoundingBox

CPU_MODEL_PATH = "models/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
TPU_MODEL_PATH = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
PERSON_LABEL_ID = 0

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

Result = collections.namedtuple('Result', ['confidence', 'bounding_box'])


class TFLitePersonDetector:
    def __init__(self):
        self.interpreter = self._create_interpreter()
        self.interpreter.allocate_tensors()

    def _create_interpreter(self):
        edge_tpu_delegate = None
        try:
            edge_tpu_delegate = tflite.load_delegate(EDGETPU_SHARED_LIB)
        except Exception:
            print("EdgeTPU initialization failed. Using CPU instead.")

        if edge_tpu_delegate is None:
            return tflite.Interpreter(model_path=CPU_MODEL_PATH)
        else:
            return tflite.Interpreter(model_path=TPU_MODEL_PATH, experimental_delegates=[edge_tpu_delegate])

    def _input_size(self):
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
        return width, height

    def _prepare_input_data(self, image):
        input_width, input_height = self._input_size()
        resized_image = image.resize((input_width, input_height), Image.ANTIALIAS)
        return np.expand_dims(resized_image, axis=0)

    def _output_tensor(self, i):
        tensor = self.interpreter.tensor(self.interpreter.get_output_details()[i]['index'])()
        return np.squeeze(tensor)

    def _output_tensors(self):
        boxes = self._output_tensor(0)
        classes = self._output_tensor(1)
        scores = self._output_tensor(2)
        return boxes, classes, scores

    def _is_person(self, class_id):
        return PERSON_LABEL_ID == class_id

    def _person_result(self, boxes, classes, scores, image_width, image_height):
        result = None
        highest_person_confidence_result = 0

        for i in range(len(scores)):
            confidence = float(scores[i])
            class_id = int(classes[i])

            if self._is_person(class_id) and confidence > highest_person_confidence_result:
                highest_person_confidence_result = confidence
                ymin, xmin, ymax, xmax = boxes[i]
                result = Result(
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax).scale(image_width, image_height).map(int))

        return result

    def detect_person(self, image):
        input_data = self._prepare_input_data(image)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)

        self.interpreter.invoke()
        boxes, classes, scores = self._output_tensors()
        image_width, image_height = image.size

        return self._person_result(boxes, classes, scores, image_width, image_height)
