#!/usr/bin/env python
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

def main():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, 'models', "yolo.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, 'images', "corgis.jpg"), output_image_path=os.path.join(execution_path, 'output', "corgis-labeled.jpg"), minimum_percentage_probability=30)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")

if __name__ == '__main__':
    main()
