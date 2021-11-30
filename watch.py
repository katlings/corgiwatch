#!/usr/bin/env python3

from imageai.Detection import VideoObjectDetection
import os


def detect():
    detector = VideoObjectDetection()
#    detector.setModelTypeAsYOLOv3()
#    detector.setModelPath('models/yolo.h5')
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath('models/resnet50_coco_best_v2.1.0.h5')
    detector.loadModel()


    path = detector.detectObjectsFromVideo(
        input_file_path='videos/corgis.mp4',
        output_file_path='videos/detected-resnet',
        frames_per_second=24,
        log_progress=True
    )


    print(path)


if __name__ == '__main__':
    detect()
