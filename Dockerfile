FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y curl python3 python3-pip
RUN pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_x86_64.whl

COPY . /opt/video-person-detection
WORKDIR /opt/video-person-detection

RUN pip3 install -r requirements.txt

# install Edge TPU runtime
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y libedgetpu1-std

# download object detection models
RUN curl -o models/ssd_mobilenet_v2_coco_quant_postprocess.tflite https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite && \
    curl -o models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite

VOLUME ["/opt/video-person-detection/config", "/opt/video-person-detection/video_data", "/opt/video-person-detection/debug_output_data"]
EXPOSE 8000

CMD ["uvicorn", "videopersondetection.main:app", "--host", "0.0.0.0"]
