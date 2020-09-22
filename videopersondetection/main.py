import logging
import sys
import os
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from videopersondetection.video_person_detector import VideoPersonDetector


class DetectionRequest(BaseModel):
    video_file_path: str = Field(alias='videoFilePath')
    camera_id: str = Field(alias='cameraId')


with open('config/config.yml') as file:
    config = yaml.safe_load(file)

logging.basicConfig(level=logging.getLevelName(config.get('log_level', 'DEBUG')), format='%(asctime)s [%(levelname)s]: %(message)s')

app = FastAPI()
videoPersonDetector = VideoPersonDetector(config)

def validated_video_file_path(detection_request):
    base_path = config.get('video_file_base_path')
    video_file_path = base_path + detection_request.video_file_path
    logging.info('Request for: %s', video_file_path)
    if os.path.commonprefix((os.path.realpath(video_file_path), base_path)) != base_path:
        raise ValueError('Requested video file not within configured base path.')

    return video_file_path

@app.post("/detect/person")
async def detect_person(detection_request: DetectionRequest):
    try:
        video_file_path = validated_video_file_path(detection_request)
        confidence_result = videoPersonDetector.detect_person(video_file_path, detection_request.camera_id)

        return {
            'personDetected': confidence_result > 0,
            'personConfidence': confidence_result
        }

    except:
        exception = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(exception))
