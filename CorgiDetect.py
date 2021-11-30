#!/usr/bin/env python
import os

import click
from imageai.Detection import ObjectDetection
from imageai.Classification import ImageClassification

execution_path = os.getcwd()

def detect():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, 'models', "yolo.h5"))
    detector.loadModel()

    dogs = detector.CustomObjects(dog=True)
    _, detections, extracted_images = detector.detectCustomObjectsFromImage(custom_objects=dogs, input_image=os.path.join(execution_path, 'images', "winona.jpg"), output_type='array', minimum_percentage_probability=30, extract_detected_objects=True)

    print(detections)
    return extracted_images


def identify(images):
    prediction = ImageClassification()
    prediction.setModelTypeAsInceptionV3()
    prediction.setModelPath(os.path.join(execution_path, 'models', "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
    prediction.loadModel()
    
    for image in images:
        predictions, probabilities = prediction.classifyImage(image, input_type='array', result_count=5)
        print(predictions)
        print(probabilities)
        for pred, prob in zip(predictions, probabilities):
            print(f'{pred} : {prob}')


def main():
    filenames = detect()
    identify(filenames)


if __name__ == '__main__':
    main()
