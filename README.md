# Video Person Detection

REST API server that can detect the presence of humans in a video file. 
Intended for video surveillance to post-process IP camera footage and reduce 
false positives.

Video frames are analyzed using TensorFlow Lite. This deep learning-based 
approach will reduce false motion alerts usually caused by weather conditions 
(e.g. clouds changing the lightning) or insects crawling across the camera lens.

The Google Coral USB Accelerator (TPU) is optionally supported to speed up 
inference time. See performance measurements below.

Sample request:

```
curl -d '{"videoFilePath":"video-file.mp4", "cameraId":"backyard"}' -H "Content-Type: application/json" -X POST http://localhost:8000/detect/person
```

Response:

```
{
  "personDetected": true,
  "personConfidence": 0.89
}
```


## How to run

The preferred way is to run video-person-detection in a docker container. 
If you don't want to use docker, you can install python and all dependencies
to run the uvicorn server on your own.


### With docker

Clone the repository and run:

```
sudo docker-compose build
```

This will build the docker image.

Afterwards start the container:

```
sudo docker-compose up
```

You should end up with a webserver running on port 8000.

Refer to the configuration section on how to configure the application.


### Without docker

If you don't want to use a docker container you can install all dependencies
on your own.

These instructions have been tested on Ubuntu Server 20.04 LTS.

```
# install python 3.8 and virtual environment support
sudo apt-get install python3 python3-venv

# clone the repository
git clone https://github.com/1element/video-person-detection.git .

# create a new python virtual environment in the cloned repository directory
python3 -m venv .venv 

# activate the virtual environment
source .venv/bin/activate
```

Install the appropriate TensorFlow Lite version for your platform according to
[https://www.tensorflow.org/lite/guide/python](https://www.tensorflow.org/lite/guide/python)

For example:

```
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_x86_64.whl
```

Install the project dependencies (Pillow, fastapi, uvicorn, 
opencv-python-headless):

```
pip3 install -r requirements.txt
```

If you have the Coral USB accelerator you should [install the Edge TPU runtime](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime)
as well:

```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

Download the TensorFlow Lite models:

```
curl -o models/ssd_mobilenet_v2_coco_quant_postprocess.tflite https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite
curl -o models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
```

You should now be able to start the webserver by executing:

```
uvicorn videopersondetection.main:app --host 0.0.0.0
```


## Configuration

A yaml configuration file `config/config.yml` must exist. Beside different 
thresholds you can configure how many frames should be skipped. So only 
every n-th frame will be processed. This can increase the performance 
significantly, but might also affect accurancy.

A camera specific configuration is possible to set a minimum required area of
the bounding box for the detected person. You can also define a zone contour
to ignore if the center of the bounding box for the detected person is inside
it.

Make sure the property `video_file_base_path` is set to the directory
containing the video files you want to analyze.

Refer to the provided [example configuration](https://github.com/1element/video-person-detection/blob/master/config/config.yml)
for details.


## Usage

Sample request via cURL:

```
curl -d '{"videoFilePath":"video-file.mp4", "cameraId":"backyard"}' -H "Content-Type: application/json" -X POST http://localhost:8000/detect/person
```

The `videoFilePath` property must contain the relative path to the video file 
to analyze (based upon the configured `video_file_base_path` in 
`config/config.yml`). The application must have read access to this location.

The `cameraId` property must match one of the configured cameras in 
`config/config.yml`.


## Performance

A simple benchmark script is included in this repository.

To download the benchmark video file run:

```
curl -o benchmark.mkv http://www.jell.yfish.us/media/jellyfish-3-mbps-hd-h264.mkv
```

Run the benchmark (`benchmark.py`):

```
python3 -m videopersondetection.benchmark
```

Here are some reports for the downloaded jellyfish videofile
(h264, 1920x1080, 30fps, 30 seconds) to give you an idea of the performance.

On Intel Core i3-7100U with EdgeTPU (Coral USB Accelerator):

```
Processing video file with SKIP_FRAMES = 15 took 7.66 seconds (without inference).
Processing video file with SKIP_FRAMES = 15 took 10.20 seconds (with inference).
Average frame inference time: 39.09ms.
```

On Intel Core i3-7100U (without TPU):

```
Processing video file with SKIP_FRAMES = 15 took 7.66 seconds (without inference).
Processing video file with SKIP_FRAMES = 15 took 32.04 seconds (with inference).
Average frame inference time: 406.05ms.
```

On Intel Core i5-6200U (without TPU):

```
Processing video file with SKIP_FRAMES = 15 took 8.89 seconds (without inference).
Processing video file with SKIP_FRAMES = 15 took 52.36 seconds (with inference).
Average frame inference time: 691.09ms.
```

On Raspberry Pi 3 (without TPU):

```
Processing video file with SKIP_FRAMES = 15 took 91.96 seconds (without inference).
Processing video file with SKIP_FRAMES = 15 took 179.71 seconds (with inference).
Average frame inference time: 1475.31ms.
```

The performance does increase for smaller video resolutions. Here are some 
reports for a different videofile (mpeg4, 896x672, 30fps, 30 seconds) to have
a comparison.

Intel Core i3-7100U with EdgeTPU (Coral USB Accelerator):

```
Processing video file with SKIP_FRAMES = 15 took 2.59 seconds (without inference).
Processing video file with SKIP_FRAMES = 15 took 3.77 seconds (with inference).
Average frame inference time: 22.47ms.
```

Intel Core i3-7100U (without TPU):

```
Processing video file with SKIP_FRAMES = 15 took 2.59 seconds (without inference).
Processing video file with SKIP_FRAMES = 15 took 21.73 seconds (with inference).
Average frame inference time: 390.82ms.
```


## License

This project is licensed under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

For more information, see [LICENSE](https://github.com/1element/video-person-detection/blob/master/LICENSE).
