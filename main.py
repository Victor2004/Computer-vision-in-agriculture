# Документация 
# Detection Classes: https://imageai.readthedocs.io/en/latest/detection/index.html
# Video and Live-Feed Detection and Analysis: https://imageai.readthedocs.io/en/latest/video/index.html
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=30)

detections = detector.detectObjectsFromImage(input_image=r"image.jpg", output_image_path=r"imagenew.jpg")
# detections = detector.detectObjectsFromImage(input_image=r"image.jpg", output_image_path=r"imagenew.jpg", minimum_percentage_probability=50)

for eachObject in detections:
    # print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    # print("--------------------------------")
    print(eachObject["name"] + " - "
          +str(eachObject["percentage_probability"]),
          eachObject["box_points"])